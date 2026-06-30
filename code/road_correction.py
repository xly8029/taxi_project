# -*- coding: utf-8 -*-
"""
06 路网校正与地图选点
======================
依据：06-路网校正与地图选点.html

功能：
    1. 加载深圳驾车路网（优先 pkl，回退 graphml）
    2. GPS 点最近邻匹配到道路节点
    3. 最短路径拼接校正轨迹
    4. 记录加载方式、耗时、校正效果与失败样例
"""

import os
import sys
import json
import pickle
import time
import logging
import math
import hashlib
from datetime import datetime

import networkx as nx
import osmnx as ox
import pandas as pd
from pyproj import Transformer
from shapely.geometry import LineString, Point
from shapely.geometry import box
from shapely.ops import substring

from road_correction_cache import (
    build_cache_slice_key,
    build_correction_cache_key,
    cache_file_exists,
    cache_mode_token,
    correction_cache_path,
    get_single_day_window,
    is_full_day_range,
    load_correction_cache,
    load_vehicle_cache_store,
    normalize_cache_time,
    slice_cached_result_by_time_range,
    write_correction_cache,
)

# ========================= 配置区 =========================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROAD_NETWORK_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PKL_PATH = os.path.join(ROAD_NETWORK_DIR, "shenzhen_drive.pkl")
GRAPHML_PATH = os.path.join(ROAD_NETWORK_DIR, "shenzhen_drive.graphml")
LOG_PATH = os.path.join(PROJECT_ROOT, "docs", "road_correction_log.txt")
CORRECTION_CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "cache", "road_correction")

# 相邻 GPS 点直线距离超过此值（米）时，不做最短路径拼接，直接保留原始点
MAX_GPS_SEGMENT_METERS = 2000

# GPS 点距离最近道路过远时，认为不可靠，不参与路网吸附
MAX_SNAP_DISTANCE_METERS = 80
MAX_SNAP_DISTANCE_HARD_METERS = 120

# 如果路网路径相对 GPS 直线距离绕行过大，则拒绝该段校正
MAX_ROUTE_DETOUR_RATIO = 4.0
MAX_ROUTE_EXTRA_METERS = 500
# 候选搜索半径与候选数适度放宽：深圳主辅路/立交并行路段较多，过窄的
# 搜索范围或过少的候选数容易让真正应匹配的道路漏出候选集之外。
MAX_CANDIDATE_SEARCH_METERS = 150
MAX_CANDIDATES_PER_POINT = 8
MAX_SEGMENT_TIME_GAP_SECONDS = 90
MAX_SEGMENT_SPEED_MPS = 35
EMISSION_SIGMA_METERS = 20
TRANSITION_BETA_METERS = 50
# 方向容差适度收紧：75度过于宽松，容易让方向上明显不对的候选边
# （比如垂直路口处的横向道路）混入同等竞争，导致吸附到错误道路。
MAX_DIRECTION_DIFF_DEGREES = 55
REVERSE_DIRECTION_PENALTY = 120
MIN_HEADING_SPEED_KMH = 5
MIN_HEADING_DISTANCE_METERS = 20
STATIONARY_MERGE_DISTANCE_METERS = 12
SHORT_JUMP_POINTS = 2
SEGMENT_DEGRADE_REJECT_RATIO = 0.4

# 样例车辆与时间窗口（小范围演示，避免全量处理）
SAMPLE_VEHICLES = [22223, 22224, 22225]
SAMPLE_START_TIME = "2013-10-22 00:00:00"
SAMPLE_END_TIME = "2013-10-22 00:20:00"

# 全局路网缓存，避免重复加载
_GRAPH_CACHE = {
    "G": None,
    "G_projected": None,
    "edges_projected": None,
    "source": None,
    "load_seconds": None,
    "to_projected": None,
    "to_latlon": None,
}

# ===============================================================================

logger = logging.getLogger("road_correction")


def _vehicle_cache_file(vehicle_id):
    return os.path.join(PROJECT_ROOT, "data", "cache", "vehicle", f"{vehicle_id}.csv")


def _network_cache_paths():
    return [PKL_PATH, GRAPHML_PATH]


def _normalize_cache_time(value):
    return normalize_cache_time(value)


def _cache_mode_token(kwargs):
    return cache_mode_token(kwargs)


def _is_full_day_range(start_time, end_time):
    return is_full_day_range(start_time, end_time)


def _get_single_day_window(start_time, end_time):
    return get_single_day_window(start_time, end_time)


def _build_correction_cache_key(vehicle_id, start_time=None, end_time=None, **kwargs):
    return build_correction_cache_key(
        vehicle_id,
        start_time,
        end_time,
        vehicle_cache_file=_vehicle_cache_file(vehicle_id),
        network_paths=_network_cache_paths(),
        algo_version=ALGO_VERSION,
        kwargs=kwargs,
    )


def _correction_cache_path(vehicle_id):
    return correction_cache_path(CORRECTION_CACHE_DIR, vehicle_id)


def _build_cache_slice_key(start_time=None, end_time=None):
    return build_cache_slice_key(start_time, end_time)


def _load_vehicle_cache_store(vehicle_id):
    return load_vehicle_cache_store(CORRECTION_CACHE_DIR, vehicle_id, logger)


def _load_correction_cache(cache_key, vehicle_id=None, start_time=None, end_time=None, **kwargs):
    return load_correction_cache(CORRECTION_CACHE_DIR, cache_key, vehicle_id=vehicle_id, logger=logger)


def _write_correction_cache(cache_key, result, vehicle_id=None, start_time=None, end_time=None, **kwargs):
    return write_correction_cache(
        CORRECTION_CACHE_DIR,
        cache_key,
        result,
        vehicle_id=vehicle_id,
        algo_version=ALGO_VERSION,
        logger=logger,
    )


def _cache_file_exists(vehicle_id, start_time=None, end_time=None, **kwargs):
    return cache_file_exists(
        CORRECTION_CACHE_DIR,
        vehicle_id,
        start_time,
        end_time,
        vehicle_cache_file=_vehicle_cache_file(vehicle_id),
        network_paths=_network_cache_paths(),
        algo_version=ALGO_VERSION,
        logger=logger,
        kwargs=kwargs,
    )


def _slice_cached_result_by_time_range(result, df, start_time=None, end_time=None):
    return slice_cached_result_by_time_range(
        result,
        df,
        start_time,
        end_time,
        rebuild_coords_fn=_rebuild_coords_from_timed_pieces,
    )


def _prepare_cached_result(result, df, vehicle_id, source, load_sec, cache_hit):
    prepared = dict(result)
    prepared["vehicle_id"] = vehicle_id
    prepared["df"] = df
    prepared["network_source"] = source
    prepared["network_load_seconds"] = load_sec
    prepared["cache_hit"] = cache_hit
    return prepared


def _rebuild_coords_from_timed_pieces(timed_pieces):
    corrected_coords = []
    corrected_segments = []
    current_segment = []

    def _flush_segment():
        nonlocal current_segment
        if len(current_segment) >= 2:
            corrected_segments.append(current_segment)
        current_segment = []

    for piece in timed_pieces:
        coords = piece.get("coords", [])
        if piece.get("break_before"):
            _flush_segment()
        for coord in coords:
            _append_coord(corrected_coords, coord)
            _append_coord(current_segment, coord)

    _flush_segment()
    return corrected_coords, corrected_segments


def _setup_logger():
    """配置日志：同时输出到文件和控制台。"""
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def _haversine_meters(lat1, lon1, lat2, lon2):
    """计算两点间球面距离（米）。"""
    from math import radians, sin, cos, sqrt, atan2

    r = 6371000
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * r * atan2(sqrt(a), sqrt(1 - a))


def _bearing_degrees(lat1, lon1, lat2, lon2):
    """计算从点1到点2的方向角(0-360度)。"""
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def _angle_diff_degrees(a, b):
    """返回两个方向角的最小夹角。"""
    diff = abs(a - b) % 360
    return diff if diff <= 180 else 360 - diff


def _movement_bearing(df, idx):
    """优先使用原始方向角_HEAD，缺失时再用相邻点现算。"""
    if "speed" in df.columns and pd.notna(df.at[idx, "speed"]) and float(df.at[idx, "speed"]) < MIN_HEADING_SPEED_KMH:
        return None

    if idx < len(df) - 1:
        seg_dist = _haversine_meters(
            df.at[idx, "lati"], df.at[idx, "long"],
            df.at[idx + 1, "lati"], df.at[idx + 1, "long"],
        )
        if seg_dist < MIN_HEADING_DISTANCE_METERS:
            return None

    if "方向角_HEAD" in df.columns and pd.notna(df.at[idx, "方向角_HEAD"]):
        # 低速/短位移时原始方向角抖动很大，宁可不用方向角，也不要把车辆
        # 从近距离主路硬拉到远处支路/对向车道。
        speed_ok = True
        if "speed" in df.columns and pd.notna(df.at[idx, "speed"]):
            speed_ok = float(df.at[idx, "speed"]) >= MIN_HEADING_SPEED_KMH
            if speed_ok and idx < len(df) - 1 and pd.notna(df.at[idx + 1, "speed"]):
                speed_ok = float(df.at[idx + 1, "speed"]) >= MIN_HEADING_SPEED_KMH
            if speed_ok and idx > 0 and pd.notna(df.at[idx - 1, "speed"]):
                speed_ok = float(df.at[idx - 1, "speed"]) >= MIN_HEADING_SPEED_KMH

        dist_ok = True
        if idx < len(df) - 1:
            seg_dist = _haversine_meters(
                df.at[idx, "lati"], df.at[idx, "long"],
                df.at[idx + 1, "lati"], df.at[idx + 1, "long"],
            )
            dist_ok = seg_dist >= MIN_HEADING_DISTANCE_METERS
        elif idx > 0:
            seg_dist = _haversine_meters(
                df.at[idx - 1, "lati"], df.at[idx - 1, "long"],
                df.at[idx, "lati"], df.at[idx, "long"],
            )
            dist_ok = seg_dist >= MIN_HEADING_DISTANCE_METERS

        if speed_ok and dist_ok:
            return float(df.at[idx, "方向角_HEAD"])

    if idx < len(df) - 1:
        if "speed" in df.columns:
            curr_speed = float(df.at[idx, "speed"]) if pd.notna(df.at[idx, "speed"]) else 0.0
            next_speed = float(df.at[idx + 1, "speed"]) if pd.notna(df.at[idx + 1, "speed"]) else 0.0
            if min(curr_speed, next_speed) < MIN_HEADING_SPEED_KMH:
                return None
        next_dist = _haversine_meters(
            df.at[idx, "lati"], df.at[idx, "long"],
            df.at[idx + 1, "lati"], df.at[idx + 1, "long"],
        )
        if next_dist < MIN_HEADING_DISTANCE_METERS:
            return None
        return _bearing_degrees(
            df.at[idx, "lati"], df.at[idx, "long"],
            df.at[idx + 1, "lati"], df.at[idx + 1, "long"],
        )
    if idx > 0:
        if "speed" in df.columns:
            prev_speed = float(df.at[idx - 1, "speed"]) if pd.notna(df.at[idx - 1, "speed"]) else 0.0
            curr_speed = float(df.at[idx, "speed"]) if pd.notna(df.at[idx, "speed"]) else 0.0
            if min(prev_speed, curr_speed) < MIN_HEADING_SPEED_KMH:
                return None
        prev_dist = _haversine_meters(
            df.at[idx - 1, "lati"], df.at[idx - 1, "long"],
            df.at[idx, "lati"], df.at[idx, "long"],
        )
        if prev_dist < MIN_HEADING_DISTANCE_METERS:
            return None
        return _bearing_degrees(
            df.at[idx - 1, "lati"], df.at[idx - 1, "long"],
            df.at[idx, "lati"], df.at[idx, "long"],
        )
    return None


def load_road_network(force_reload=False):
    """
    加载深圳驾车路网。优先 pkl，其次 graphml。

    Returns:
        tuple: (MultiDiGraph, source_label, load_seconds)
    """
    _setup_logger()

    if _GRAPH_CACHE["G"] is not None and not force_reload:
        return _GRAPH_CACHE["G"], _GRAPH_CACHE["source"], _GRAPH_CACHE["load_seconds"]

    t0 = time.perf_counter()
    source = None
    G = None

    if os.path.exists(PKL_PATH):
        with open(PKL_PATH, "rb") as f:
            G = pickle.load(f)
        source = "pkl"
    elif os.path.exists(GRAPHML_PATH):
        G = ox.load_graphml(GRAPHML_PATH)
        source = "graphml"
    else:
        raise FileNotFoundError(
            f"未找到路网文件，请将 shenzhen_drive.pkl 放到:\n  {PKL_PATH}\n"
            f"或 shenzhen_drive.graphml 放到:\n  {GRAPHML_PATH}"
        )

    elapsed = time.perf_counter() - t0
    _GRAPH_CACHE["G"] = G
    _GRAPH_CACHE["source"] = source
    _GRAPH_CACHE["load_seconds"] = elapsed

    node_count = G.number_of_nodes()
    edge_count = G.number_of_edges()
    logger.info(
        "路网加载完成 | 方式=%s | 耗时=%.2fs | 节点=%d | 边=%d",
        source, elapsed, node_count, edge_count,
    )
    return G, source, elapsed


def load_projected_road_network(force_reload=False):
    """加载投影后的路网，并缓存坐标转换器。"""
    G, source, elapsed = load_road_network(force_reload=force_reload)
    if _GRAPH_CACHE["G_projected"] is None or force_reload:
        G_projected = ox.project_graph(G)
        projected_crs = G_projected.graph["crs"]
        geographic_crs = G.graph.get("crs", "EPSG:4326")
        _GRAPH_CACHE["G_projected"] = G_projected
        _GRAPH_CACHE["to_projected"] = Transformer.from_crs(geographic_crs, projected_crs, always_xy=True)
        _GRAPH_CACHE["to_latlon"] = Transformer.from_crs(projected_crs, geographic_crs, always_xy=True)
    return G, _GRAPH_CACHE["G_projected"], source, elapsed


def load_projected_edges(force_reload=False):
    """加载投影后边表，供候选边搜索使用。"""
    G, G_projected, source, elapsed = load_projected_road_network(force_reload=force_reload)
    if _GRAPH_CACHE["edges_projected"] is None or force_reload:
        edges = ox.graph_to_gdfs(G_projected, nodes=False, edges=True).reset_index()
        _GRAPH_CACHE["edges_projected"] = edges
    return G, G_projected, _GRAPH_CACHE["edges_projected"], source, elapsed


def match_gps_to_nodes(G, lats, lngs):
    """
    将 GPS 点匹配到最近道路节点。

    注意：nearest_nodes 参数顺序为 (G, 经度, 纬度)。
    """
    return ox.distance.nearest_nodes(G, lngs, lats)


def _node_lat_lng(G, node):
    """返回节点 (lat, lng)。"""
    return G.nodes[node]["y"], G.nodes[node]["x"]


def _path_to_coords(G, path):
    """将节点路径转为 folium 可用的 [lat, lng] 列表。"""
    return [[G.nodes[n]["y"], G.nodes[n]["x"]] for n in path]


def _geometry_to_latlngs(geometry):
    """将 shapely 几何转为 [lat, lng] 坐标序列。"""
    coords = list(geometry.coords)
    return [[lat, lng] for lng, lat in coords]


def _best_edge_geometry_between_nodes(G, u, v):
    """在多重边中选择最短的那条边几何。"""
    edge_bundle = G.get_edge_data(u, v)
    if not edge_bundle:
        return LineString([
            (G.nodes[u]["x"], G.nodes[u]["y"]),
            (G.nodes[v]["x"], G.nodes[v]["y"]),
        ])
    best_key = min(edge_bundle, key=lambda item: edge_bundle[item].get("length", float("inf")))
    geometry = edge_bundle[best_key].get("geometry")
    if geometry is not None:
        return geometry
    return LineString([
        (G.nodes[u]["x"], G.nodes[u]["y"]),
        (G.nodes[v]["x"], G.nodes[v]["y"]),
    ])


def _select_edge_by_weight(G, u, v, weight="length"):
    """按权重从平行道路边中选择本次路径应使用的道路边。"""
    edge_bundle = G.get_edge_data(u, v)
    if not edge_bundle:
        return None, None

    valid = [
        (key, data)
        for key, data in edge_bundle.items()
        if data.get(weight) is not None
    ]
    if not valid:
        valid = list(edge_bundle.items())

    return min(
        valid,
        key=lambda item: float(item[1].get(weight, float("inf"))),
    )


def _route_to_edge_sequence(G, path, weight="length"):
    """将节点路径展开为 (u, v, key) 序列，避免多重边几何选错。"""
    if not path or len(path) < 2:
        return []

    try:
        route_gdf = ox.routing.route_to_gdf(G, path, weight=weight)
        return [tuple(index) for index in route_gdf.index.tolist()]
    except Exception:
        edge_sequence = []
        for u, v in zip(path[:-1], path[1:]):
            key, _ = _select_edge_by_weight(G, u, v, weight=weight)
            edge_sequence.append((u, v, key))
        return edge_sequence


def _slice_edge_geometry_by_progress(G, u, v, key, start_progress, end_progress):
    """按边上的归一化位置裁剪局部几何，避免吸附点到节点的直连飞线。"""
    geometry = _edge_geometry(G, u, v, key) if key is not None else _best_edge_geometry_between_nodes(G, u, v)
    start_progress = min(max(float(start_progress), 0.0), 1.0)
    end_progress = min(max(float(end_progress), 0.0), 1.0)
    if geometry.length <= 0:
        return geometry

    clipped = substring(geometry, start_progress, end_progress, normalized=True)
    if clipped.geom_type == "Point":
        return LineString([clipped.coords[0], clipped.coords[0]])
    return clipped


def _edge_segment_to_coords(G, edge, start_progress, end_progress):
    """把边上的局部区间转为 folium 坐标，并按 edge 方向统一。"""
    u, v, key = edge
    geometry = _slice_edge_geometry_by_progress(G, u, v, key, start_progress, end_progress)
    segment_coords = _geometry_to_latlngs(geometry)
    if not segment_coords:
        return []

    u_coord = _node_lat_lng(G, u)
    if _haversine_meters(segment_coords[0][0], segment_coords[0][1], u_coord[0], u_coord[1]) > \
            _haversine_meters(segment_coords[-1][0], segment_coords[-1][1], u_coord[0], u_coord[1]):
        segment_coords.reverse()
    return segment_coords


def _route_to_geometry_coords(G, path, start_snap=None, end_snap=None, route_edges=None,
                              start_edge=None, end_edge=None, start_progress=None, end_progress=None,
                              route_start_node=None, route_end_node=None):
    """按真实道路边几何输出路线坐标，避免节点直连造成飞线。"""
    if not path:
        return []

    coords = []

    if len(path) == 1:
        shared_node = path[0]
        if start_edge is not None and start_progress is not None:
            start_u, start_v, _ = start_edge
            if shared_node in {start_u, start_v}:
                exit_progress = 0.0 if shared_node == start_u else 1.0
                for coord in _edge_segment_to_coords(G, start_edge, start_progress, exit_progress):
                    _append_coord(coords, coord)

        if not coords:
            if start_snap is not None:
                _append_coord(coords, start_snap)
            else:
                _append_coord(coords, _node_lat_lng(G, shared_node))

        if end_edge is not None and end_progress is not None:
            end_u, end_v, _ = end_edge
            if shared_node in {end_u, end_v}:
                enter_progress = 0.0 if shared_node == end_u else 1.0
                for coord in _edge_segment_to_coords(G, end_edge, enter_progress, end_progress):
                    _append_coord(coords, coord)
    else:
        edge_sequence = route_edges or _route_to_edge_sequence(G, path, weight="length")

        if start_edge is not None and start_progress is not None and route_start_node is not None:
            start_u, start_v, _ = start_edge
            if route_start_node in {start_u, start_v} and (not edge_sequence or edge_sequence[0] != start_edge):
                exit_progress = 0.0 if route_start_node == start_u else 1.0
                for coord in _edge_segment_to_coords(G, start_edge, start_progress, exit_progress):
                    _append_coord(coords, coord)

        total_edges = len(edge_sequence)
        for idx, (u, v, key) in enumerate(edge_sequence):
            if total_edges == 1 and start_edge == (u, v, key) and end_edge == (u, v, key) and \
                    start_progress is not None and end_progress is not None:
                geometry = _slice_edge_geometry_by_progress(G, u, v, key, start_progress, end_progress)
            elif idx == 0 and start_edge == (u, v, key) and start_progress is not None:
                exit_progress = 0.0 if route_start_node == u else 1.0
                geometry = _slice_edge_geometry_by_progress(G, u, v, key, start_progress, exit_progress)
            elif idx == total_edges - 1 and end_edge == (u, v, key) and end_progress is not None:
                enter_progress = 0.0 if route_end_node == u else 1.0
                geometry = _slice_edge_geometry_by_progress(G, u, v, key, enter_progress, end_progress)
            else:
                geometry = _edge_geometry(G, u, v, key) if key is not None else _best_edge_geometry_between_nodes(G, u, v)
            segment_coords = _geometry_to_latlngs(geometry)
            if segment_coords:
                u_coord = _node_lat_lng(G, u)
                if _haversine_meters(segment_coords[0][0], segment_coords[0][1], u_coord[0], u_coord[1]) > \
                        _haversine_meters(segment_coords[-1][0], segment_coords[-1][1], u_coord[0], u_coord[1]):
                    segment_coords.reverse()
                for coord in segment_coords:
                    _append_coord(coords, coord)

        if end_edge is not None and end_progress is not None and route_end_node is not None:
            end_u, end_v, _ = end_edge
            if route_end_node in {end_u, end_v} and (not edge_sequence or edge_sequence[-1] != end_edge):
                enter_progress = 0.0 if route_end_node == end_u else 1.0
                for coord in _edge_segment_to_coords(G, end_edge, enter_progress, end_progress):
                    _append_coord(coords, coord)

    if not coords and start_snap is not None:
        _append_coord(coords, start_snap)
    if end_snap is not None:
        _append_coord(coords, end_snap)
    return coords


def _edge_geometry(G, u, v, key):
    """返回边的几何线；若缺失 geometry，则退化为两端点直线。"""
    edge_data = G.get_edge_data(u, v, key)
    geometry = edge_data.get("geometry") if edge_data else None
    if geometry is not None:
        return geometry
    return LineString([
        (G.nodes[u]["x"], G.nodes[u]["y"]),
        (G.nodes[v]["x"], G.nodes[v]["y"]),
    ])


def _snap_gps_point_to_edge(G, G_projected, lat, lng, edge=None):
    """将单个 GPS 点吸附到最近道路边，并返回吸附结果。"""
    to_projected = _GRAPH_CACHE["to_projected"]
    to_latlon = _GRAPH_CACHE["to_latlon"]
    x, y = to_projected.transform(lng, lat)
    u, v, key = edge if edge is not None else ox.distance.nearest_edges(G_projected, x, y)

    geometry = _edge_geometry(G_projected, u, v, key)
    point = Point(x, y)
    offset = geometry.project(point)
    snapped_point = geometry.interpolate(offset)
    snap_distance_m = point.distance(snapped_point)
    edge_length_m = max(geometry.length, 1e-6)
    snapped_lng, snapped_lat = to_latlon.transform(snapped_point.x, snapped_point.y)

    # 局部切线方向：采样窗口按边长自适应（边长的10%，且夹在2米~15米之间），
    # 避免固定5米窗口在很短的边上把首尾两个采样点压成几乎同一个点（导致
    # bearing 噪声很大），也避免在长直道上窗口过小放大 GPS/几何抖动误差。
    half_window = max(2.0, min(15.0, edge_length_m * 0.1))
    edge_start = geometry.interpolate(max(offset - half_window, 0))
    edge_end = geometry.interpolate(min(offset + half_window, geometry.length))
    if edge_start.distance(edge_end) < 1e-6:
        # 仍然退化（极短边或吸附点恰好在端点），直接用整条边的首尾点。
        edge_start = geometry.interpolate(0)
        edge_end = geometry.interpolate(geometry.length)
    edge_start_lng, edge_start_lat = to_latlon.transform(edge_start.x, edge_start.y)
    edge_end_lng, edge_end_lat = to_latlon.transform(edge_end.x, edge_end.y)
    edge_bearing = _bearing_degrees(edge_start_lat, edge_start_lng, edge_end_lat, edge_end_lng)

    edge_data = G.get_edge_data(u, v, key) or {}
    oneway = bool(edge_data.get("oneway", False))
    osmid = edge_data.get("osmid")
    if isinstance(osmid, (list, tuple)):
        osmid = frozenset(osmid)

    return {
        "edge": (u, v, key),
        "osmid": osmid,
        "snapped": [snapped_lat, snapped_lng],
        "snap_distance_m": snap_distance_m,
        "distance_to_u_m": offset,
        "distance_to_v_m": edge_length_m - offset,
        "edge_progress": offset / edge_length_m,
        "edge_bearing": edge_bearing,
        "oneway": oneway,
    }


def _snap_gps_points_to_edges(G, G_projected, lats, lngs):
    """批量将 GPS 点吸附到最近道路边，避免逐点查询过慢。"""
    to_projected = _GRAPH_CACHE["to_projected"]
    projected = [to_projected.transform(lng, lat) for lat, lng in zip(lats, lngs)]
    xs = [item[0] for item in projected]
    ys = [item[1] for item in projected]
    edges = ox.distance.nearest_edges(G_projected, X=xs, Y=ys)
    return [
        _snap_gps_point_to_edge(G, G_projected, lat, lng, edge=edge)
        for (lat, lng, edge) in zip(lats, lngs, edges)
    ]


def _append_coord(coords, coord):
    """避免重复追加相同坐标。"""
    normalized = [float(coord[0]), float(coord[1])]
    if not coords or coords[-1] != normalized:
        coords.append(normalized)


def _candidate_from_edge(G, G_projected, lat, lng, edge):
    """基于指定边生成一个候选吸附结果。"""
    return _snap_gps_point_to_edge(G, G_projected, lat, lng, edge=edge)


def _build_candidates_for_point(G, G_projected, edges_projected, lat, lng,
                                search_radius_m=MAX_CANDIDATE_SEARCH_METERS,
                                max_candidates=MAX_CANDIDATES_PER_POINT):
    """为单个 GPS 点搜索多个候选道路边。"""
    to_projected = _GRAPH_CACHE["to_projected"]
    x, y = to_projected.transform(lng, lat)
    bbox = box(x - search_radius_m, y - search_radius_m, x + search_radius_m, y + search_radius_m)
    match_idx = list(edges_projected.sindex.intersection(bbox.bounds))
    if not match_idx:
        nearest = ox.distance.nearest_edges(G_projected, X=x, Y=y)
        return [_candidate_from_edge(G, G_projected, lat, lng, nearest)]

    nearby = edges_projected.iloc[match_idx].copy()
    point = Point(x, y)
    nearby["candidate_distance_m"] = nearby.geometry.distance(point)
    nearby = nearby[nearby["candidate_distance_m"] <= search_radius_m]
    if nearby.empty:
        nearest = ox.distance.nearest_edges(G_projected, X=x, Y=y)
        return [_candidate_from_edge(G, G_projected, lat, lng, nearest)]

    nearby = nearby.nsmallest(max_candidates, "candidate_distance_m")
    candidates = []
    seen = set()
    for row in nearby.itertuples(index=False):
        edge = (int(row.u), int(row.v), int(row.key))
        if edge in seen:
            continue
        seen.add(edge)
        candidates.append(_candidate_from_edge(G, G_projected, lat, lng, edge))
    return candidates


def _build_candidate_sets(G, G_projected, edges_projected, lats, lngs,
                          search_radius_m=MAX_CANDIDATE_SEARCH_METERS,
                          max_candidates=MAX_CANDIDATES_PER_POINT):
    """为整个轨迹构建多候选边集合。"""
    return [
        _build_candidates_for_point(
            G, G_projected, edges_projected, lat, lng,
            search_radius_m=search_radius_m,
            max_candidates=max_candidates,
        )
        for lat, lng in zip(lats, lngs)
    ]


LINK_HIGHWAY_TYPES = {
    "motorway_link", "trunk_link", "primary_link",
    "secondary_link", "tertiary_link",
}

# 匝道/环形道路天然弧长远大于弦长，绕行比例和绕行余量都应放宽，
# 否则真实的弯道路径会被系统性地判定为"绕行过大"而拒绝。
RAMP_DETOUR_RATIO_MULTIPLIER = 2.5
RAMP_DETOUR_EXTRA_MULTIPLIER = 2.5

# 同一条 OSM 道路（osmid）连续匹配时给予的转移代价折扣，抑制在立交处
# 因为候选边密集、方向相近而在相邻匝道之间反复跳变。
SAME_ROAD_CONTINUITY_BONUS = 15


def _edge_highway_type(G, u, v, key):
    """返回边的 highway 标签（osmnx 里可能是字符串或列表，取首个）。"""
    edge_data = G.get_edge_data(u, v, key) or {}
    highway = edge_data.get("highway")
    if isinstance(highway, (list, tuple)):
        return highway[0] if highway else None
    return highway


def _edge_osmid(G, u, v, key):
    """返回边的 osmid（可能是单值或列表），用于判断是否为同一条道路的延续。"""
    edge_data = G.get_edge_data(u, v, key) or {}
    osmid = edge_data.get("osmid")
    if isinstance(osmid, (list, tuple)):
        return frozenset(osmid)
    return osmid


def _same_physical_edge(prev_snap, curr_snap):
    """判断两个吸附结果是否落在同一条双向物理道路边上。"""
    prev_u, prev_v, _ = prev_snap["edge"]
    curr_u, curr_v, _ = curr_snap["edge"]
    if prev_snap.get("oneway") or curr_snap.get("oneway"):
        return False
    same_nodes = {prev_u, prev_v} == {curr_u, curr_v}
    if not same_nodes:
        return False

    prev_osmid = prev_snap.get("osmid")
    curr_osmid = curr_snap.get("osmid")
    if prev_osmid is not None and curr_osmid is not None:
        return prev_osmid == curr_osmid
    return True


def _maybe_repair_prev_snap(G_directed, G_undirected, prev_snap, curr_snap,
                            prev_candidates, gps_bearing, use_undirected=False):
    """修复明显选错的上一帧候选，避免整段绿线被带到旁路上。"""
    if gps_bearing is None:
        return prev_snap, None

    prev_dir_diff = _angle_diff_degrees(gps_bearing, prev_snap["edge_bearing"])
    curr_dir_diff = _angle_diff_degrees(gps_bearing, curr_snap["edge_bearing"])
    prev_wrong_way = prev_snap.get("oneway", False) and prev_dir_diff > MAX_DIRECTION_DIFF_DEGREES
    curr_wrong_way = curr_snap.get("oneway", False) and curr_dir_diff > MAX_DIRECTION_DIFF_DEGREES
    if not prev_wrong_way or curr_wrong_way:
        return prev_snap, None

    repaired = None
    best_route = None
    best_score = None
    for cand in prev_candidates:
        if cand["edge"] == prev_snap["edge"]:
            continue

        cand_dir_diff = _angle_diff_degrees(gps_bearing, cand["edge_bearing"])
        if cand.get("oneway", False) and cand_dir_diff > MAX_DIRECTION_DIFF_DEGREES:
            continue

        # 只接受距离相近的替代候选，避免为了修方向跳到很远的道路。
        if cand["snap_distance_m"] - prev_snap["snap_distance_m"] > 20:
            continue

        route = _choose_best_route(
            G_directed, G_undirected, cand, curr_snap, use_undirected=use_undirected
        )
        if route is None or route["graph_used"] == "undirected_fallback":
            continue

        score = (
            abs(route["total_cost_m"]),
            0 if cand.get("osmid") == curr_snap.get("osmid") else 1,
            cand_dir_diff,
            cand["snap_distance_m"],
        )
        if best_score is None or score < best_score:
            best_score = score
            repaired = cand
            best_route = route

    if repaired is None:
        return prev_snap, None
    return repaired, best_route


def _maybe_repair_curr_snap(G_directed, G_undirected, prev_snap, curr_snap,
                            curr_candidates, gps_bearing, use_undirected=False):
    """修复明显选错的当前帧候选，避免跳到对向车道后整串带偏。"""
    if gps_bearing is None:
        return curr_snap, None

    curr_dir_diff = _angle_diff_degrees(gps_bearing, curr_snap["edge_bearing"])
    curr_wrong_way = curr_snap.get("oneway", False) and curr_dir_diff > MAX_DIRECTION_DIFF_DEGREES
    if not curr_wrong_way:
        return curr_snap, None

    repaired = None
    best_route = None
    best_score = None
    for cand in curr_candidates:
        if cand["edge"] == curr_snap["edge"]:
            continue

        cand_dir_diff = _angle_diff_degrees(gps_bearing, cand["edge_bearing"])
        if cand.get("oneway", False) and cand_dir_diff > MAX_DIRECTION_DIFF_DEGREES:
            continue

        if cand["snap_distance_m"] - curr_snap["snap_distance_m"] > 20:
            continue

        route = _choose_best_route(
            G_directed, G_undirected, prev_snap, cand, use_undirected=use_undirected
        )
        if route is None or route["graph_used"] == "undirected_fallback":
            continue

        score = (
            abs(route["total_cost_m"]),
            0 if cand.get("osmid") == prev_snap.get("osmid") else 1,
            cand_dir_diff,
            cand["snap_distance_m"],
        )
        if best_score is None or score < best_score:
            best_score = score
            repaired = cand
            best_route = route

    if repaired is None:
        return curr_snap, None
    return repaired, best_route


def _maybe_repair_unreachable_curr_snap(G_directed, G_undirected, prev_snap, curr_snap,
                                        curr_candidates, route, use_undirected=False):
    """当前候选若只能靠无向图连通，尝试替换为附近可沿正确方向连续行驶的候选。"""
    if route is None or route.get("graph_used") != "undirected_fallback":
        return curr_snap, route

    repaired = None
    best_route = route
    best_score = None
    for cand in curr_candidates:
        if cand["edge"] == curr_snap["edge"]:
            continue
        if cand["snap_distance_m"] - curr_snap["snap_distance_m"] > 20:
            continue

        cand_route = _choose_best_route(
            G_directed, G_undirected, prev_snap, cand, use_undirected=use_undirected
        )
        if cand_route is None or cand_route["graph_used"] == "undirected_fallback":
            continue

        continuity_bonus = 0
        if cand["edge"] == prev_snap["edge"] or _same_physical_edge(prev_snap, cand):
            continuity_bonus = -1

        score = (
            continuity_bonus,
            cand_route["total_cost_m"],
            cand["snap_distance_m"],
        )
        if best_score is None or score < best_score:
            best_score = score
            repaired = cand
            best_route = cand_route

    if repaired is None:
        return curr_snap, route
    return repaired, best_route


def _edge_progress_in_reference_orientation(snap, ref_edge):
    """将吸附点在线上的位置换算到参考边方向，便于双向同路段裁切几何。"""
    ref_u, ref_v, _ = ref_edge
    u, v, _ = snap["edge"]
    progress = snap.get("edge_progress")
    if progress is None:
        return None
    if (u, v) == (ref_u, ref_v):
        return progress
    if (u, v) == (ref_v, ref_u):
        return 1.0 - progress
    return progress


def _path_ramp_ratio(G, path):
    """路径中匝道类边（highway=*_link）的长度占比，用于放宽绕行判定。"""
    if not path or len(path) < 2:
        return 0.0
    total_len = 0.0
    ramp_len = 0.0
    for u, v in zip(path[:-1], path[1:]):
        edge_bundle = G.get_edge_data(u, v)
        if not edge_bundle:
            continue
        best_key = min(edge_bundle, key=lambda item: edge_bundle[item].get("length", float("inf")))
        edge_data = edge_bundle[best_key]
        length = edge_data.get("length", 0.0) or 0.0
        total_len += length
        highway = edge_data.get("highway")
        if isinstance(highway, (list, tuple)):
            highway = highway[0] if highway else None
        if highway in LINK_HIGHWAY_TYPES:
            ramp_len += length
    if total_len <= 0:
        return 0.0
    return ramp_len / total_len


def _detour_threshold_m(gps_dist_m, ramp_ratio):
    """
    计算"判定为绕行过大"的距离阈值，按匝道占比放宽。

    环形/匝道道路的弧长本来就远大于两点间弦长（gps_dist_m），这是道路
    几何的固有特性，不是匹配错误。若路径主要由匝道类道路构成
    （ramp_ratio 接近1），按比例放宽 ratio 和 extra 两项余量；
    普通道路（ramp_ratio=0）维持原有较严格的阈值，避免放过真正的
    错误匹配。
    """
    ratio = MAX_ROUTE_DETOUR_RATIO * (1 + (RAMP_DETOUR_RATIO_MULTIPLIER - 1) * ramp_ratio)
    extra = MAX_ROUTE_EXTRA_METERS * (1 + (RAMP_DETOUR_EXTRA_MULTIPLIER - 1) * ramp_ratio)
    return max(gps_dist_m * ratio, gps_dist_m + extra)


def _choose_best_route(G_directed, G_undirected, prev_snap, curr_snap, use_undirected=False):
    """
    在边端点组合中选择总代价最小的路网路径。

    关键修正：
    1. 同一条边内部移动（prev/curr 吸附到同一条边）必须走"沿边推进"，
       不能退化成端点间的最短路径搜索——否则在有向图上会被允许从 v 端
       重新出发，产生不存在的转向/绕路。
    2. 在有向图上，一条有向边 u->v 只能从 v 继续向外走（如果车辆正沿
       该边方向行驶），不能把 u 也当作"下一段路径的起点"，否则等于
       允许车辆瞬间穿越/逆行回到边的入口去找近路。只有在该边非单行时
       才把 u 也纳入候选起点。
    """
    # 情形 1：两点吸附在同一条边上，直接按边上的先后顺序处理，不做图搜索。
    if prev_snap["edge"] == curr_snap["edge"]:
        if curr_snap["distance_to_u_m"] >= prev_snap["distance_to_u_m"]:
            same_edge_len = abs(curr_snap["distance_to_u_m"] - prev_snap["distance_to_u_m"])
            u, v, key = prev_snap["edge"]
            highway = _edge_highway_type(G_directed, u, v, key)
            return {
                "path": [u, v],
                "route_edges": [(u, v, key)],
                "start_node": u,
                "end_node": v,
                "graph_used": "same_edge",
                "route_length_m": same_edge_len,
                "total_cost_m": same_edge_len,
                "ramp_ratio": 1.0 if highway in LINK_HIGHWAY_TYPES else 0.0,
            }

    primary_graph = G_undirected if use_undirected else G_directed
    graph_choices = [(primary_graph, "directed" if not use_undirected else "undirected")]
    if not use_undirected:
        graph_choices.append((G_undirected, "undirected_fallback"))

    candidates = []
    for graph, graph_used in graph_choices:
        is_directed_pass = graph_used == "directed"

        if is_directed_pass:
            # 有向图上，只允许从"沿边方向上的下一个节点"出发：
            # 若该边非单行，u/v 都可作为合法的离开点；若单行，只能从 v 出发。
            prev_nodes = [(prev_snap["edge"][1], prev_snap["distance_to_v_m"])]
            if not prev_snap.get("oneway", False):
                prev_nodes.append((prev_snap["edge"][0], prev_snap["distance_to_u_m"]))

            # 到达 curr 边时，若非单行，可以从 u 或 v 任一端进入；
            # 若单行，只能从 u 端进入（沿边方向行驶到吸附点之前）。
            curr_nodes = [(curr_snap["edge"][0], curr_snap["distance_to_u_m"])]
            if not curr_snap.get("oneway", False):
                curr_nodes.append((curr_snap["edge"][1], curr_snap["distance_to_v_m"]))
        else:
            # 无向图/兜底场景不存在方向限制，两端都可尝试。
            prev_nodes = [
                (prev_snap["edge"][0], prev_snap["distance_to_u_m"]),
                (prev_snap["edge"][1], prev_snap["distance_to_v_m"]),
            ]
            curr_nodes = [
                (curr_snap["edge"][0], curr_snap["distance_to_u_m"]),
                (curr_snap["edge"][1], curr_snap["distance_to_v_m"]),
            ]

        found_in_this_graph = False
        for start_node, start_cost in prev_nodes:
            for end_node, end_cost in curr_nodes:
                try:
                    route_length = nx.shortest_path_length(graph, start_node, end_node, weight="length")
                    route_path = nx.shortest_path(graph, start_node, end_node, weight="length")
                    route_edges = _route_to_edge_sequence(G_directed, route_path, weight="length")
                    candidates.append({
                        "path": route_path,
                        "route_edges": route_edges,
                        "start_node": start_node,
                        "end_node": end_node,
                        "graph_used": graph_used,
                        "route_length_m": route_length,
                        "total_cost_m": start_cost + route_length + end_cost,
                        "ramp_ratio": _path_ramp_ratio(G_directed, route_path),
                    })
                    found_in_this_graph = True
                except nx.NetworkXNoPath:
                    continue

        # 只有在主图（有向图）完全找不到路径时才进入无向图兜底，
        # 避免有向图里本来有解却被无向图"更短但违反方向"的路径抢走。
        if found_in_this_graph:
            break

    if not candidates:
        return None
    return min(candidates, key=lambda item: item["total_cost_m"])


def _viterbi_match_candidates(candidate_sets, movement_bearings, lats, lngs, G_directed, G_undirected,
                              use_undirected=False, max_snap_distance_m=MAX_SNAP_DISTANCE_METERS):
    """
    使用 HMM/Viterbi 在多候选边之间选择全局最优序列。

    关键修正：原实现一旦某一帧的所有候选都无法从上一帧任何候选转移过来
    （dp_costs 整行变为 inf），后续每一帧的 backpointer 都会因为"上一帧
    全 inf"而继续传播 None，最终整条轨迹被判定为匹配失败，退回到完全不
    做方向/转移约束的粗糙最近边吸附——哪怕断点只发生在中间一个点。
    现在改为：当某一帧整行变为 inf 时，该帧"重新起跳"，只用当帧自身的
    发射概率作为代价（允许从这一帧重新开始一条新的匹配链），并记录该帧
    与前一帧的连接断裂，但不影响断点之外的其它点继续做完整的方向/转移
    约束匹配。
    """
    route_cache = {}
    emissions = []
    for candidates in candidate_sets:
        emission_row = []
        for cand in candidates:
            if cand["snap_distance_m"] > max_snap_distance_m:
                emission_row.append(float("inf"))
            else:
                z = cand["snap_distance_m"] / EMISSION_SIGMA_METERS
                emission_row.append(0.5 * z * z + math.log(EMISSION_SIGMA_METERS))
        emissions.append(emission_row)

    dp_costs = [emissions[0][:]]
    backpointers = [[None] * len(candidate_sets[0])]
    restart_flags = [all(cost == float("inf") for cost in dp_costs[0])]

    for i in range(1, len(candidate_sets)):
        gps_dist = _haversine_meters(lats[i - 1], lngs[i - 1], lats[i], lngs[i])
        gps_bearing = movement_bearings[i - 1]
        row_costs = [float("inf")] * len(candidate_sets[i])
        row_back = [None] * len(candidate_sets[i])

        prev_row_all_inf = all(cost == float("inf") for cost in dp_costs[i - 1])

        for curr_idx, curr_cand in enumerate(candidate_sets[i]):
            if emissions[i][curr_idx] == float("inf"):
                continue
            for prev_idx, prev_cand in enumerate(candidate_sets[i - 1]):
                if dp_costs[i - 1][prev_idx] == float("inf"):
                    continue
                cache_key = (i - 1, prev_idx, i, curr_idx)
                if cache_key not in route_cache:
                    route = _choose_best_route(
                        G_directed, G_undirected, prev_cand, curr_cand, use_undirected=use_undirected
                    )
                    route_cache[cache_key] = route
                route = route_cache[cache_key]
                if route is None:
                    continue

                route_cost = route["total_cost_m"]
                detour = abs(route_cost - gps_dist)
                ramp_ratio = route.get("ramp_ratio", 0.0)
                if route_cost > _detour_threshold_m(gps_dist, ramp_ratio):
                    continue

                transition_cost = 0.5 * (detour / TRANSITION_BETA_METERS) ** 2 + math.log(TRANSITION_BETA_METERS)

                # 同路连续性奖励：若 prev/curr 吸附到同一条 OSM 道路
                # （osmid 相同，含同一道路被拆成多段 edge 的情况），说明
                # 车辆很可能一直沿着这条路行驶，给予代价折扣；这能在立交
                # 处候选边密集、彼此距离相近时，抑制 Viterbi 在不同匝道
                # 间反复跳变所导致的锯齿状轨迹。
                prev_osmid = prev_cand.get("osmid")
                curr_osmid = curr_cand.get("osmid")
                if prev_osmid is not None and prev_osmid == curr_osmid:
                    transition_cost = max(transition_cost - SAME_ROAD_CONTINUITY_BONUS, 0.0)

                if gps_bearing is not None:
                    curr_dir_diff = _angle_diff_degrees(gps_bearing, curr_cand["edge_bearing"])
                    prev_dir_diff = _angle_diff_degrees(gps_bearing, prev_cand["edge_bearing"])
                    if curr_cand["oneway"] and curr_dir_diff > MAX_DIRECTION_DIFF_DEGREES:
                        continue
                    # 上一帧候选边的局部切线方向可能因为刚经过路口、匝道汇入或
                    # 候选点靠近边端点而与整段运动方向短暂不一致。这里若直接把
                    # prev 候选硬性判死，会把后续已经回到正确主路的整段路径一并
                    # 淘汰，网页上就只剩一条红色回退直线。对 prev 方向冲突改为
                    # 高额惩罚，由 Viterbi 自行权衡；curr 方向冲突仍保持硬拒绝。
                    prev_wrong_way = prev_cand["oneway"] and prev_dir_diff > MAX_DIRECTION_DIFF_DEGREES
                    # 非单行道不再硬性拒绝，但仍应让方向不一致的候选在
                    # 竞争中付出代价——否则在双向路/辅路并行处，路网距离
                    # 略短就足以让 Viterbi 选中方向完全拧反的边。
                    dir_penalty = (
                        (curr_dir_diff / 180.0) ** 2 + (prev_dir_diff / 180.0) ** 2
                    ) * REVERSE_DIRECTION_PENALTY
                    if prev_wrong_way:
                        dir_penalty += REVERSE_DIRECTION_PENALTY * 2
                    transition_cost += dir_penalty
                if route["graph_used"] == "undirected_fallback":
                    continue
                total_cost = dp_costs[i - 1][prev_idx] + transition_cost + emissions[i][curr_idx]
                if total_cost < row_costs[curr_idx]:
                    row_costs[curr_idx] = total_cost
                    row_back[curr_idx] = prev_idx

        row_all_inf = all(cost == float("inf") for cost in row_costs)
        if row_all_inf:
            # 只要当前帧无法从上一帧任何候选转移过来，就必须允许它以
            # 自身发射代价重新起跳。否则一旦前一帧已经是全 inf，后续帧
            # 会永远继承这个状态，整段轨迹都无法恢复。
            finite_emissions = [cost for cost in emissions[i] if cost != float("inf")]
            if finite_emissions:
                row_costs = emissions[i][:]
                row_back = [None] * len(candidate_sets[i])
                restart_flags.append(True)
            else:
                restart_flags.append(False)
        else:
            restart_flags.append(False)

        dp_costs.append(row_costs)
        backpointers.append(row_back)

    last_row = dp_costs[-1]
    if all(cost == float("inf") for cost in last_row):
        return None, route_cache

    last_idx = min(range(len(last_row)), key=lambda idx: last_row[idx])
    path_indices = [last_idx]
    for i in range(len(candidate_sets) - 1, 0, -1):
        if restart_flags[i]:
            # 该帧是重新起跳点，没有可回溯的上一帧链接，
            # 直接沿用同一帧索引向前结束回溯（链条在此截断）。
            break
        prev_idx = backpointers[i][path_indices[-1]]
        if prev_idx is None:
            return None, route_cache
        path_indices.append(prev_idx)
    path_indices.reverse()

    # 因为存在"重新起跳"截断，path_indices 可能比 candidate_sets 短；
    # 对于截断点之前的帧，直接用该帧候选里发射概率最高（snap_distance_m
    # 最小）的候选兜底，保证每一帧都有输出，不丢点。
    if len(path_indices) < len(candidate_sets):
        filled = [None] * len(candidate_sets)
        offset = len(candidate_sets) - len(path_indices)
        for j, idx in enumerate(path_indices):
            filled[offset + j] = idx
        for j in range(offset):
            candidates = candidate_sets[j]
            best_idx = min(
                range(len(candidates)),
                key=lambda k: candidates[k]["snap_distance_m"],
            )
            filled[j] = best_idx
        path_indices = filled

    selected = [candidate_sets[i][cand_idx] for i, cand_idx in enumerate(path_indices)]
    return {"selected": selected, "path_indices": path_indices}, route_cache


def _split_trajectory_segments(df, max_gap_seconds=MAX_SEGMENT_TIME_GAP_SECONDS,
                               max_jump_m=MAX_GPS_SEGMENT_METERS,
                               max_speed_mps=MAX_SEGMENT_SPEED_MPS):
    """按时间间隔、距离跳变和异常速度将轨迹切成多个稳定片段。"""
    if df.empty:
        return []

    segments = []
    start = 0
    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]
        gap_seconds = (curr["time"] - prev["time"]).total_seconds()
        dist_m = _haversine_meters(prev["lati"], prev["long"], curr["lati"], curr["long"])
        speed_mps = dist_m / max(gap_seconds, 1)
        if gap_seconds > max_gap_seconds or dist_m > max_jump_m or speed_mps > max_speed_mps:
            segments.append((start, i))
            start = i
    segments.append((start, len(df)))
    return segments


def _compress_stationary_points(df, merge_distance_m=STATIONARY_MERGE_DISTANCE_METERS):
    """压缩连续停顿/近停顿点，避免静止状态下方向角抖动污染匹配。"""
    if df.empty:
        return df

    keep_rows = [0]
    for i in range(1, len(df)):
        prev_idx = keep_rows[-1]
        prev = df.iloc[prev_idx]
        curr = df.iloc[i]
        same_vehicle = curr["id"] == prev["id"]
        dist_m = _haversine_meters(prev["lati"], prev["long"], curr["lati"], curr["long"])
        prev_speed = float(prev["speed"]) if "speed" in df.columns and pd.notna(prev["speed"]) else 0.0
        curr_speed = float(curr["speed"]) if "speed" in df.columns and pd.notna(curr["speed"]) else 0.0
        if same_vehicle and dist_m < merge_distance_m and prev_speed < MIN_HEADING_SPEED_KMH and curr_speed < MIN_HEADING_SPEED_KMH:
            keep_rows[-1] = i
        else:
            keep_rows.append(i)
    return df.iloc[keep_rows].reset_index(drop=True)


def _smooth_debug_backtracks(corrected_coords, debug_segments):
    """为短跳路保留后处理入口；当前主要返回坐标本身，避免过度平滑误杀真实转弯。"""
    if len(corrected_coords) <= 2:
        return corrected_coords
    return corrected_coords


def _remove_short_backtracks(coords, max_gap=12):
    """保留原坐标，不再做回折裁剪。"""
    return coords


def _remove_small_loops(coords, revisit_gap=80, revisit_radius_m=8, min_loop_len_m=60):
    """保留原坐标，不再做小环裁剪。"""
    return coords


def _cleanup_corrected_segments(corrected_segments):
    """仅清理重复点，不再改写轨迹形状。"""
    cleaned = []
    for seg in corrected_segments:
        deduped = []
        for coord in seg:
            _append_coord(deduped, coord)
        if len(deduped) >= 2:
            cleaned.append(deduped)
    return cleaned


def _cleanup_piece_coords(coords):
    """仅清理重复点，不再改写单段几何形状。"""
    if not coords:
        return []
    cleaned = []
    for coord in coords:
        _append_coord(cleaned, coord)
    return cleaned


def correct_trajectory(df, G=None, use_undirected=False,
                       max_gps_segment_m=MAX_GPS_SEGMENT_METERS,
                       max_snap_distance_m=MAX_SNAP_DISTANCE_METERS):
    """
    对轨迹 DataFrame 做路网校正。

    Args:
        df: 含 lati, long, time 列的轨迹
        G: 路网图，None 时自动加载
        use_undirected: True 时全程使用无向图
        max_gps_segment_m: GPS 点间距超过此值则跳过最短路径

    Returns:
        dict: corrected_coords, original_coords, stats, matched_nodes
    """
    _setup_logger()
    if G is None:
        G, G_projected, edges_projected, _, _ = load_projected_edges()
    else:
        _, G_projected, edges_projected, _, _ = load_projected_edges()

    if df.empty or len(df) < 2:
        coords = df[["lati", "long"]].values.tolist() if not df.empty else []
        return {
            "corrected_coords": coords,
            "original_coords": coords,
            "matched_nodes": [],
            "stats": {"total_segments": 0, "success_segments": 0, "failed_segments": 0, "skipped_segments": 0, "failures": []},
        }

    df = df.sort_values("time").reset_index(drop=True)
    df = _compress_stationary_points(df)
    original_coords = df[["lati", "long"]].values.tolist()

    G_directed = G
    G_undirected = G.to_undirected()
    corrected = []
    corrected_segments = []
    current_corrected_segment = []
    matched_nodes = []
    debug_segments = []
    timed_pieces = []
    stats = {
        "total_segments": 0,
        "success_segments": 0,
        "failed_segments": 0,
        "skipped_segments": 0,
        "rejected_segments": 0,
        "degraded_segments": 0,
        "offroad_points": 0,
        "undirected_fallback": 0,
        "split_segments": 0,
        "failures": [],
    }

    segments = _split_trajectory_segments(df, max_jump_m=max_gps_segment_m)
    stats["split_segments"] = len(segments)

    def _append_corrected(coord):
        _append_coord(corrected, coord)
        _append_coord(current_corrected_segment, coord)

    def _record_timed_piece(start_row_idx, end_row_idx, coords, break_before=False):
        if start_row_idx is None or end_row_idx is None or not coords:
            return
        coords = _cleanup_piece_coords(coords)
        if not coords:
            return
        timed_pieces.append({
            "start_time": str(df.iloc[start_row_idx]["time"]),
            "end_time": str(df.iloc[end_row_idx]["time"]),
            "start_row_idx": int(start_row_idx),
            "end_row_idx": int(end_row_idx),
            "coords": [[float(lat), float(lng)] for lat, lng in coords],
            "break_before": bool(break_before),
        })

    def _break_corrected_segment():
        nonlocal current_corrected_segment
        if len(current_corrected_segment) >= 2:
            corrected_segments.append(current_corrected_segment)
        current_corrected_segment = []

    for seg_index, (start_idx, end_idx) in enumerate(segments):
        _break_corrected_segment()
        segment_df = df.iloc[start_idx:end_idx].reset_index(drop=True)
        seg_lats = segment_df["lati"].tolist()
        seg_lngs = segment_df["long"].tolist()
        seg_bearings = [_movement_bearing(segment_df, idx) for idx in range(len(segment_df))]
        if not seg_lats:
            continue

        if len(seg_lats) == 1:
            coord = [seg_lats[0], seg_lngs[0]]
            _append_corrected(coord)
            _record_timed_piece(start_idx, start_idx, [coord], break_before=True)
            continue

        candidate_sets = _build_candidate_sets(G, G_projected, edges_projected, seg_lats, seg_lngs)
        matched, route_cache = _viterbi_match_candidates(
            candidate_sets, seg_bearings, seg_lats, seg_lngs, G_directed, G_undirected,
            use_undirected=use_undirected,
            max_snap_distance_m=max_snap_distance_m,
        )
        if matched is None:
            logger.warning("片段 %d 动态规划匹配失败，回退到最近边吸附策略", seg_index)
            snapped_points = _snap_gps_points_to_edges(G, G_projected, seg_lats, seg_lngs)
            path_indices = None
        else:
            snapped_points = matched["selected"]
            path_indices = matched["path_indices"]

        segment_corrected_start = len(corrected)
        segment_debug_start = len(debug_segments)
        seg_total_before = stats["total_segments"]
        seg_success_before = stats["success_segments"]
        seg_failed_before = stats["failed_segments"]
        seg_skipped_before = stats["skipped_segments"]
        seg_rejected_before = stats["rejected_segments"]

        matched_nodes.extend([snap["edge"][0] for snap in snapped_points])
        _append_corrected(snapped_points[0]["snapped"])

        for i in range(1, len(snapped_points)):
            prev_snap = snapped_points[i - 1]
            curr_snap = snapped_points[i]
            lat_prev, lng_prev = seg_lats[i - 1], seg_lngs[i - 1]
            lat_curr, lng_curr = seg_lats[i], seg_lngs[i]

            prev_too_far = prev_snap["snap_distance_m"] > MAX_SNAP_DISTANCE_HARD_METERS
            curr_too_far = curr_snap["snap_distance_m"] > MAX_SNAP_DISTANCE_HARD_METERS
            if prev_too_far or curr_too_far:
                _break_corrected_segment()
                fallback_coord = [lat_curr, lng_curr]
                _append_corrected(fallback_coord)
                _record_timed_piece(start_idx + i, start_idx + i, [fallback_coord], break_before=True)
                stats["offroad_points"] += int(prev_too_far) + int(curr_too_far)
                stats["skipped_segments"] += 1
                debug_segments.append({
                    "type": "offroad",
                    "coords": [[lat_prev, lng_prev], [lat_curr, lng_curr]],
                    "start_time": str(segment_df.iloc[i - 1]["time"]),
                    "end_time": str(segment_df.iloc[i]["time"]),
                    "message": f"片段{seg_index} 段{i}: 点距道路过远，回退原始GPS",
                })
                continue

            stats["total_segments"] += 1
            dist_m = _haversine_meters(lat_prev, lng_prev, lat_curr, lng_curr)
            gps_bearing = seg_bearings[i - 1]

            if dist_m > max_gps_segment_m:
                _break_corrected_segment()
                fallback_coord = [lat_curr, lng_curr]
                _append_corrected(fallback_coord)
                _record_timed_piece(start_idx + i, start_idx + i, [fallback_coord], break_before=True)
                stats["skipped_segments"] += 1
                debug_segments.append({
                    "type": "jump",
                    "coords": [[lat_prev, lng_prev], [lat_curr, lng_curr]],
                    "start_time": str(segment_df.iloc[i - 1]["time"]),
                    "end_time": str(segment_df.iloc[i]["time"]),
                    "message": f"片段{seg_index} 段{i}: GPS跳点过大，回退原始GPS",
                })
                logger.warning(
                    "片段 %d 段 %d GPS 间距过大(%.0fm)，跳过最短路径 | (%f,%f)->(%f,%f)",
                    seg_index, i, dist_m, lat_prev, lng_prev, lat_curr, lng_curr,
                )
                continue

            same_edge = prev_snap["edge"] == curr_snap["edge"]
            same_physical_edge = _same_physical_edge(prev_snap, curr_snap)
            if same_edge or same_physical_edge:
                curr_dir_diff = _angle_diff_degrees(gps_bearing, curr_snap["edge_bearing"]) if gps_bearing is not None else None
                if curr_snap["oneway"] and curr_dir_diff is not None and curr_dir_diff > MAX_DIRECTION_DIFF_DEGREES:
                    _break_corrected_segment()
                    _append_corrected(curr_snap["snapped"])
                    _record_timed_piece(start_idx + i, start_idx + i, [curr_snap["snapped"]], break_before=True)
                    stats["rejected_segments"] += 1
                    debug_segments.append({
                        "type": "direction_warning",
                        "coords": [[lat_prev, lng_prev], [lat_curr, lng_curr]],
                        "start_time": str(segment_df.iloc[i - 1]["time"]),
                        "end_time": str(segment_df.iloc[i]["time"]),
                        "message": f"片段{seg_index} 段{i}: 同边但单行方向冲突({curr_dir_diff:.0f}°)，回退吸附点",
                    })
                    continue

                base_edge = prev_snap["edge"]
                same_edge_coords = _route_to_geometry_coords(
                    G,
                    [base_edge[0], base_edge[1]],
                    start_snap=prev_snap["snapped"],
                    end_snap=curr_snap["snapped"],
                    route_edges=[base_edge],
                    start_edge=base_edge,
                    end_edge=base_edge,
                    start_progress=_edge_progress_in_reference_orientation(prev_snap, base_edge),
                    end_progress=_edge_progress_in_reference_orientation(curr_snap, base_edge),
                    route_start_node=base_edge[0],
                    route_end_node=base_edge[1],
                )
                for coord in same_edge_coords:
                    _append_corrected(coord)
                _record_timed_piece(start_idx + i - 1, start_idx + i, same_edge_coords, break_before=False)
                stats["success_segments"] += 1
                continue

            route = None
            if path_indices is not None:
                prev_idx = path_indices[i - 1]
                curr_idx = path_indices[i]
                route = route_cache.get((i - 1, prev_idx, i, curr_idx))
            if route is None:
                route = _choose_best_route(G_directed, G_undirected, prev_snap, curr_snap, use_undirected=use_undirected)

            repaired_prev_snap, repaired_route = _maybe_repair_prev_snap(
                G_directed,
                G_undirected,
                prev_snap,
                curr_snap,
                candidate_sets[i - 1],
                gps_bearing,
                use_undirected=use_undirected,
            )
            if repaired_prev_snap is not prev_snap:
                prev_snap = repaired_prev_snap
                snapped_points[i - 1] = prev_snap
                route = repaired_route

            repaired_curr_snap, repaired_route = _maybe_repair_curr_snap(
                G_directed,
                G_undirected,
                prev_snap,
                curr_snap,
                candidate_sets[i],
                gps_bearing,
                use_undirected=use_undirected,
            )
            if repaired_curr_snap is not curr_snap:
                curr_snap = repaired_curr_snap
                snapped_points[i] = curr_snap
                route = repaired_route

            repaired_curr_snap, repaired_route = _maybe_repair_unreachable_curr_snap(
                G_directed,
                G_undirected,
                prev_snap,
                curr_snap,
                candidate_sets[i],
                route,
                use_undirected=use_undirected,
            )
            if repaired_curr_snap is not curr_snap:
                curr_snap = repaired_curr_snap
                snapped_points[i] = curr_snap
                route = repaired_route

            if route is None:
                _break_corrected_segment()
                _append_corrected(curr_snap["snapped"])
                _record_timed_piece(start_idx + i, start_idx + i, [curr_snap["snapped"]], break_before=True)
                stats["failed_segments"] += 1
                failure = {
                    "segment_index": i,
                    "split_segment_index": seg_index,
                    "from_edge": list(prev_snap["edge"]),
                    "to_edge": list(curr_snap["edge"]),
                    "from_gps": [lat_prev, lng_prev],
                    "to_gps": [lat_curr, lng_curr],
                    "gps_distance_m": round(dist_m, 1),
                }
                stats["failures"].append(failure)
                debug_segments.append({
                    "type": "route_failed",
                    "coords": [[lat_prev, lng_prev], [lat_curr, lng_curr]],
                    "start_time": str(segment_df.iloc[i - 1]["time"]),
                    "end_time": str(segment_df.iloc[i]["time"]),
                    "message": f"片段{seg_index} 段{i}: 路网路径失败，回退原始GPS",
                })
                logger.warning(
                    "最短路径失败 | 片段=%d 段=%d | GPS (%.5f,%.5f)->(%.5f,%.5f)",
                    seg_index, i, lat_prev, lng_prev, lat_curr, lng_curr,
                )
                continue

            route_ramp_ratio = route.get("ramp_ratio", 0.0)
            if route["total_cost_m"] > _detour_threshold_m(dist_m, route_ramp_ratio):
                _append_coord(corrected, curr_snap["snapped"])
                stats["rejected_segments"] += 1
                prev_hwy = _edge_highway_type(G_directed, *prev_snap["edge"])
                curr_hwy = _edge_highway_type(G_directed, *curr_snap["edge"])
                debug_segments.append({
                    "type": "detour",
                    "coords": [[lat_prev, lng_prev], [lat_curr, lng_curr]],
                    "message": f"片段{seg_index} 段{i}: 绕行过大，回退吸附点 | prev_edge={prev_snap['edge']}(hwy={prev_hwy},osmid={prev_snap.get('osmid')}) curr_edge={curr_snap['edge']}(hwy={curr_hwy},osmid={curr_snap.get('osmid')})",
                })
                logger.warning(
                    "片段 %d 段 %d 绕行过大(路网=%.0fm, GPS=%.0fm) | prev_edge=%s hwy=%s | curr_edge=%s hwy=%s | (%f,%f)->(%f,%f)",
                    seg_index, i, route["total_cost_m"], dist_m,
                    prev_snap["edge"], prev_hwy, curr_snap["edge"], curr_hwy,
                    lat_prev, lng_prev, lat_curr, lng_curr,
                )
                continue

            if route["graph_used"] == "undirected_fallback":
                _break_corrected_segment()
                _append_corrected(curr_snap["snapped"])
                _record_timed_piece(start_idx + i, start_idx + i, [curr_snap["snapped"]], break_before=True)
                stats["undirected_fallback"] += 1
                stats["rejected_segments"] += 1
                debug_segments.append({
                    "type": "undirected_fallback",
                    "coords": [[lat_prev, lng_prev], [lat_curr, lng_curr]],
                    "start_time": str(segment_df.iloc[i - 1]["time"]),
                    "end_time": str(segment_df.iloc[i]["time"]),
                    "message": f"片段{seg_index} 段{i}: 仅无向图可达，回退吸附点",
                })
                continue

            curr_dir_diff = _angle_diff_degrees(gps_bearing, curr_snap["edge_bearing"]) if gps_bearing is not None else None
            prev_dir_diff = _angle_diff_degrees(gps_bearing, prev_snap["edge_bearing"]) if gps_bearing is not None else None
            prev_wrong_way = (
                gps_bearing is not None and
                prev_snap["oneway"] and
                prev_dir_diff > MAX_DIRECTION_DIFF_DEGREES
            )
            curr_wrong_way = (
                gps_bearing is not None and
                curr_snap["oneway"] and
                curr_dir_diff > MAX_DIRECTION_DIFF_DEGREES
            )
            if curr_wrong_way:
                _break_corrected_segment()
                _append_corrected(curr_snap["snapped"])
                _record_timed_piece(start_idx + i, start_idx + i, [curr_snap["snapped"]], break_before=True)
                stats["rejected_segments"] += 1
                debug_segments.append({
                    "type": "direction_warning",
                    "coords": [[lat_prev, lng_prev], [lat_curr, lng_curr]],
                    "start_time": str(segment_df.iloc[i - 1]["time"]),
                    "end_time": str(segment_df.iloc[i]["time"]),
                    "message": f"片段{seg_index} 段{i}: 当前单行方向冲突(prev={prev_dir_diff:.0f}°, curr={curr_dir_diff:.0f}°)，回退吸附点",
                })
                continue

            path_coords = _route_to_geometry_coords(
                G,
                route["path"],
                start_snap=prev_snap["snapped"],
                end_snap=curr_snap["snapped"],
                route_edges=route.get("route_edges"),
                start_edge=prev_snap["edge"],
                end_edge=curr_snap["edge"],
                start_progress=prev_snap.get("edge_progress"),
                end_progress=curr_snap.get("edge_progress"),
                route_start_node=route.get("start_node"),
                route_end_node=route.get("end_node"),
            )

            if prev_wrong_way:
                for coord in path_coords:
                    _append_corrected(coord)
                _record_timed_piece(start_idx + i - 1, start_idx + i, path_coords, break_before=False)
                stats["rejected_segments"] += 1
                debug_segments.append({
                    "type": "direction_warning",
                    "coords": path_coords,
                    "start_time": str(segment_df.iloc[i - 1]["time"]),
                    "end_time": str(segment_df.iloc[i]["time"]),
                    "message": f"片段{seg_index} 段{i}: 上一单行边方向偏差较大(prev={prev_dir_diff:.0f}°, curr={curr_dir_diff:.0f}°)，保留道路路径",
                })
                continue

            stats["success_segments"] += 1
            for coord in path_coords:
                _append_corrected(coord)
            _record_timed_piece(start_idx + i - 1, start_idx + i, path_coords, break_before=False)

        seg_total = stats["total_segments"] - seg_total_before
        seg_success = stats["success_segments"] - seg_success_before
        seg_failed = stats["failed_segments"] - seg_failed_before
        seg_skipped = stats["skipped_segments"] - seg_skipped_before
        seg_rejected = stats["rejected_segments"] - seg_rejected_before
        seg_problem_ratio = (seg_rejected + seg_failed) / max(seg_total, 1)
        if seg_total > 0 and seg_problem_ratio >= SEGMENT_DEGRADE_REJECT_RATIO:
            corrected = corrected[:segment_corrected_start]
            _break_corrected_segment()
            for coord in segment_df[["lati", "long"]].values.tolist():
                _append_corrected(coord)

            debug_segments = debug_segments[:segment_debug_start]
            debug_segments.append({
                "type": "segment_degraded",
                "coords": segment_df[["lati", "long"]].values.tolist(),
                "start_time": str(segment_df.iloc[0]["time"]),
                "end_time": str(segment_df.iloc[-1]["time"]),
                "message": f"片段{seg_index}: 拒绝/失败比例过高({seg_rejected + seg_failed}/{seg_total})，整段降级为原始GPS",
            })

            timed_pieces = [
                piece for piece in timed_pieces
                if piece["end_row_idx"] < start_idx or piece["start_row_idx"] >= end_idx
            ]
            segment_coords = segment_df[["lati", "long"]].values.tolist()
            for local_idx, coord in enumerate(segment_coords):
                row_idx = start_idx + local_idx
                _record_timed_piece(row_idx, row_idx, [coord], break_before=(local_idx == 0))

            stats["success_segments"] -= seg_success
            stats["failed_segments"] -= seg_failed
            stats["skipped_segments"] -= seg_skipped
            stats["rejected_segments"] -= seg_rejected
            stats["degraded_segments"] += 1

    _break_corrected_segment()
    corrected_segments = _cleanup_corrected_segments(corrected_segments)
    corrected_coords = [coord for seg in corrected_segments for coord in seg]
    corrected_coords = [[lat, lng] for lat, lng in _smooth_debug_backtracks(corrected_coords, debug_segments)]

    success_rate = (
        stats["success_segments"] / stats["total_segments"] * 100
        if stats["total_segments"] else 100.0
    )
    logger.info(
        "轨迹校正完成 | 点数=%d | 片段=%d | 路段=%d | 成功=%d | 失败=%d | 跳过=%d | 拒绝=%d | 成功率=%.1f%%",
        len(df), stats["split_segments"], stats["total_segments"], stats["success_segments"],
        stats["failed_segments"], stats["skipped_segments"], stats["rejected_segments"], success_rate,
    )
    if stats["failures"]:
        logger.info("失败样例（最多3条）: %s", stats["failures"][:3])

    return {
        "corrected_coords": corrected_coords,
        "corrected_segments": corrected_segments,
        "original_coords": original_coords,
        "matched_nodes": matched_nodes,
        "stats": stats,
        "debug_segments": debug_segments,
        "timed_pieces": timed_pieces,
    }

# ========================= 算法版本自动指纹 =========================
# 缓存命中与否取决于这些函数的源码内容：只要其中任何一个函数的逻辑被
# 修改（哪怕只改一行），ALGO_VERSION 就会自动变化，旧缓存文件名不再
# 匹配，自然不会被命中，也不需要手动维护一个数字版本号去记"改到第几版了"。
_ALGO_FINGERPRINT_FUNCS = [
    _choose_best_route,
    _viterbi_match_candidates,
    _route_to_geometry_coords,
    _snap_gps_point_to_edge,
    _build_candidates_for_point,
    correct_trajectory,
]


def _compute_algo_version():
    import inspect
    source_blob = "".join(inspect.getsource(fn) for fn in _ALGO_FINGERPRINT_FUNCS)
    return hashlib.sha1(source_blob.encode("utf-8")).hexdigest()[:10]


ALGO_VERSION = _compute_algo_version()

def correct_vehicle_trajectory(vehicle_id, start_time=None, end_time=None, **kwargs):
    """从车辆缓存读取轨迹并校正。"""
    from map_visualization import load_vehicle_trajectory

    requested_start = start_time
    requested_end = end_time
    requested_df = load_vehicle_trajectory(vehicle_id, requested_start, requested_end)
    if requested_df.empty:
        raise ValueError(f"车辆 {vehicle_id} 在指定时间段内无数据")

    day_window = _get_single_day_window(start_time, end_time)
    query_start = start_time
    query_end = end_time
    if day_window is not None and not _is_full_day_range(start_time, end_time):
        query_start = day_window["full_start"]
        query_end = day_window["full_end"]

    df = requested_df if (query_start == requested_start and query_end == requested_end) else load_vehicle_trajectory(vehicle_id, query_start, query_end)

    cache_key = _build_correction_cache_key(vehicle_id, query_start, query_end, **kwargs)
    cache_path = _correction_cache_path(vehicle_id)
    cached_result = _load_correction_cache(
        cache_key,
        vehicle_id=vehicle_id,
        start_time=query_start,
        end_time=query_end,
        **kwargs,
    )
    G, source, load_sec = load_road_network()
    if cached_result is not None:
        result_to_return = cached_result
        response_df = df
        if day_window is not None and not _is_full_day_range(requested_start, requested_end):
            result_to_return = _slice_cached_result_by_time_range(cached_result, requested_df, requested_start, requested_end)
            result_to_return["cache_source"] = {
                "type": "vehicle_day_slice",
                "day": day_window["day"],
                "slice_key": _build_cache_slice_key(requested_start, requested_end),
                "mode": _cache_mode_token(kwargs),
            }
            response_df = requested_df
        logger.info(
            "命中轨迹校正缓存 | vehicle=%s | %s -> %s | file=%s",
            vehicle_id,
            _normalize_cache_time(query_start),
            _normalize_cache_time(query_end),
            os.path.basename(cache_path),
        )
        return _prepare_cached_result(result_to_return, response_df, vehicle_id, source, load_sec, cache_hit=True)

    result = correct_trajectory(df, G=G, **kwargs)
    result["cache_source"] = {
        "type": "vehicle_day" if day_window is not None else "exact_range",
        "day": day_window["day"] if day_window is not None else None,
        "slice_key": _build_cache_slice_key(query_start, query_end),
        "mode": _cache_mode_token(kwargs),
    }
    _write_correction_cache(
        cache_key,
        result,
        vehicle_id=vehicle_id,
        start_time=query_start,
        end_time=query_end,
        **kwargs,
    )
    logger.info(
        "写入轨迹校正缓存 | vehicle=%s | %s -> %s | file=%s",
        vehicle_id,
        _normalize_cache_time(query_start),
        _normalize_cache_time(query_end),
        os.path.basename(cache_path),
    )

    result_to_return = result
    response_df = df
    if day_window is not None and not _is_full_day_range(requested_start, requested_end):
        result_to_return = _slice_cached_result_by_time_range(result, requested_df, requested_start, requested_end)
        result_to_return["cache_source"] = {
            "type": "vehicle_day_slice",
            "day": day_window["day"],
            "slice_key": _build_cache_slice_key(requested_start, requested_end),
            "mode": _cache_mode_token(kwargs),
        }
        response_df = requested_df
    return _prepare_cached_result(result_to_return, response_df, vehicle_id, source, load_sec, cache_hit=False)


def warmup_vehicle_correction_cache(vehicle_id, day, **kwargs):
    """预生成某辆车某一天的路网校正缓存，供 web 查询直接命中。"""
    day_ts = pd.to_datetime(day)
    start_time = day_ts.strftime("%Y-%m-%d 00:00:00")
    end_time = (day_ts + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S")
    result = correct_vehicle_trajectory(vehicle_id, start_time, end_time, **kwargs)
    return {
        "vehicle_id": vehicle_id,
        "day": day_ts.strftime("%Y-%m-%d"),
        "start_time": start_time,
        "end_time": end_time,
        "points": len(result["df"]),
        "cache_hit": bool(result.get("cache_hit")),
        "cache_file": os.path.basename(_correction_cache_path(vehicle_id)),
        "stats": result["stats"],
    }


def warmup_all_vehicle_day_caches(day, limit=None, skip_existing=True, **kwargs):
    """按车辆缓存目录批量生成某一天的整天校正缓存。"""
    vehicle_dir = os.path.join(PROJECT_ROOT, "data", "cache", "vehicle")
    if not os.path.exists(vehicle_dir):
        raise FileNotFoundError(f"车辆缓存目录不存在: {vehicle_dir}")

    day_ts = pd.to_datetime(day)
    start_time = day_ts.strftime("%Y-%m-%d 00:00:00")
    end_time = (day_ts + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S")

    vehicle_ids = []
    for name in sorted(os.listdir(vehicle_dir)):
        if not name.endswith('.csv'):
            continue
        stem = os.path.splitext(name)[0]
        if stem.isdigit():
            vehicle_ids.append(int(stem))

    if limit is not None:
        vehicle_ids = vehicle_ids[:limit]

    summaries = []
    for index, vehicle_id in enumerate(vehicle_ids, start=1):
        exists, cache_path, _ = _cache_file_exists(vehicle_id, start_time, end_time, **kwargs)
        if skip_existing and exists:
            logger.info(
                "跳过已存在缓存 | %d/%d | vehicle=%s | file=%s",
                index, len(vehicle_ids), vehicle_id, os.path.basename(cache_path)
            )
            summaries.append({
                "vehicle_id": vehicle_id,
                "status": "skipped",
                "cache_file": os.path.basename(cache_path),
            })
            continue

        logger.info("开始生成整天缓存 | %d/%d | vehicle=%s", index, len(vehicle_ids), vehicle_id)
        try:
            summary = warmup_vehicle_correction_cache(vehicle_id, day, **kwargs)
            summary["status"] = "generated" if not summary["cache_hit"] else "hit"
            summaries.append(summary)
        except Exception as exc:
            logger.error("生成整天缓存失败 | vehicle=%s | %s", vehicle_id, exc)
            summaries.append({
                "vehicle_id": vehicle_id,
                "status": "failed",
                "error": str(exc),
            })

    return {
        "day": day_ts.strftime("%Y-%m-%d"),
        "total": len(vehicle_ids),
        "generated": sum(1 for item in summaries if item["status"] == "generated"),
        "hit": sum(1 for item in summaries if item["status"] == "hit"),
        "skipped": sum(1 for item in summaries if item["status"] == "skipped"),
        "failed": sum(1 for item in summaries if item["status"] == "failed"),
        "items": summaries,
    }


def _parse_cli_args(argv):
    if len(argv) < 2:
        return {"command": "help"}
    if len(argv) >= 2 and argv[1] == "warmup-day":
        if len(argv) < 4:
            raise ValueError("用法: python code/road_correction.py warmup-day <vehicle_id> <YYYY-MM-DD> [--use-undirected]")
        vehicle_id = int(argv[2])
        day = argv[3]
        use_undirected = "--use-undirected" in argv[4:]
        return {
            "command": "warmup-day",
            "vehicle_id": vehicle_id,
            "day": day,
            "use_undirected": use_undirected,
        }
    if len(argv) >= 3 and argv[1] == "warmup-all-day":
        day = argv[2]
        use_undirected = "--use-undirected" in argv[3:]
        limit = None
        if "--limit" in argv[3:]:
            limit_idx = argv.index("--limit")
            if limit_idx + 1 >= len(argv):
                raise ValueError("--limit 后必须跟数量")
            limit = int(argv[limit_idx + 1])
        skip_existing = "--force" not in argv[3:]
        return {
            "command": "warmup-all-day",
            "day": day,
            "use_undirected": use_undirected,
            "limit": limit,
            "skip_existing": skip_existing,
        }
    if argv[1] == "sample":
        return {"command": "sample"}
    raise ValueError(
        "用法:\n"
        "  python code/road_correction.py sample\n"
        "  python code/road_correction.py warmup-day <vehicle_id> <YYYY-MM-DD> [--use-undirected]\n"
        "  python code/road_correction.py warmup-all-day <YYYY-MM-DD> [--use-undirected] [--limit N] [--force]"
    )


def run_sample_correction():
    """对 1-3 辆样例车做路网校正并写日志。"""
    _setup_logger()
    logger.info("=" * 60)
    logger.info("开始样例路网校正 | 车辆=%s | %s ~ %s", SAMPLE_VEHICLES, SAMPLE_START_TIME, SAMPLE_END_TIME)

    summaries = []
    for vid in SAMPLE_VEHICLES:
        try:
            result = correct_vehicle_trajectory(vid, SAMPLE_START_TIME, SAMPLE_END_TIME)
            s = result["stats"]
            summaries.append({
                "vehicle_id": vid,
                "points": len(result["df"]),
                "success_rate": round(s["success_segments"] / max(s["total_segments"], 1) * 100, 1),
                "failures": len(s["failures"]),
            })
        except FileNotFoundError as exc:
            logger.error("车辆 %s: %s", vid, exc)
        except Exception as exc:
            logger.error("车辆 %s 校正失败: %s", vid, exc)

    logger.info("样例校正汇总: %s", summaries)

    # 生成路网校正对比地图（原始 + 校正后轨迹）
    try:
        from map_visualization import plot_corrected_trajectory, MAP_OUTPUT_DIR
        os.makedirs(MAP_OUTPUT_DIR, exist_ok=True)
        m_corr = plot_corrected_trajectory(
            SAMPLE_VEHICLES,
            start_time=SAMPLE_START_TIME,
            end_time=SAMPLE_END_TIME,
            enable_correction=True,
        )
        corr_path = os.path.join(MAP_OUTPUT_DIR, "06_road_corrected_trajectory.html")
        m_corr.save(corr_path)
        logger.info("路网校正对比地图已保存: %s", corr_path)
    except Exception as exc:
        logger.error("生成校正对比地图失败: %s", exc)

    # 生成地图选点工具
    try:
        from map_visualization import create_point_picker_map, MAP_OUTPUT_DIR
        os.makedirs(MAP_OUTPUT_DIR, exist_ok=True)
        m_pick = create_point_picker_map()
        pick_path = os.path.join(MAP_OUTPUT_DIR, "06_point_picker.html")
        m_pick.save(pick_path)
        logger.info("地图选点工具已保存: %s", pick_path)
    except Exception as exc:
        logger.error("生成地图选点工具失败: %s", exc)

    return summaries


if __name__ == "__main__":
    cli_args = _parse_cli_args(sys.argv)
    if cli_args["command"] == "help":
        print(
            "用法:\n"
            "  python code/road_correction.py sample\n"
            "  python code/road_correction.py warmup-day <vehicle_id> <YYYY-MM-DD> [--use-undirected]\n"
            "  python code/road_correction.py warmup-all-day <YYYY-MM-DD> [--use-undirected] [--limit N] [--force]"
        )
    elif cli_args["command"] == "warmup-day":
        summary = warmup_vehicle_correction_cache(
            cli_args["vehicle_id"],
            cli_args["day"],
            use_undirected=cli_args["use_undirected"],
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif cli_args["command"] == "warmup-all-day":
        summary = warmup_all_vehicle_day_caches(
            cli_args["day"],
            limit=cli_args["limit"],
            skip_existing=cli_args["skip_existing"],
            use_undirected=cli_args["use_undirected"],
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif cli_args["command"] == "sample":
        run_sample_correction()
