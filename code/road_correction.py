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
import pickle
import time
import logging
import math
from datetime import datetime

import networkx as nx
import osmnx as ox
import pandas as pd
from pyproj import Transformer
from shapely.geometry import LineString, Point
from shapely.geometry import box

# ========================= 配置区 =========================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROAD_NETWORK_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PKL_PATH = os.path.join(ROAD_NETWORK_DIR, "shenzhen_drive.pkl")
GRAPHML_PATH = os.path.join(ROAD_NETWORK_DIR, "shenzhen_drive.graphml")
LOG_PATH = os.path.join(PROJECT_ROOT, "docs", "road_correction_log.txt")

# 相邻 GPS 点直线距离超过此值（米）时，不做最短路径拼接，直接保留原始点
MAX_GPS_SEGMENT_METERS = 2000

# GPS 点距离最近道路过远时，认为不可靠，不参与路网吸附
MAX_SNAP_DISTANCE_METERS = 80

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
        return float(df.at[idx, "方向角_HEAD"])

    if idx < len(df) - 1:
        return _bearing_degrees(
            df.at[idx, "lati"], df.at[idx, "long"],
            df.at[idx + 1, "lati"], df.at[idx + 1, "long"],
        )
    if idx > 0:
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


def _route_to_geometry_coords(G, path, start_snap=None, end_snap=None):
    """按真实道路边几何输出路线坐标，避免节点直连造成飞线。"""
    if not path:
        return []

    coords = []
    if start_snap is not None:
        _append_coord(coords, start_snap)

    if len(path) == 1:
        _append_coord(coords, _node_lat_lng(G, path[0]))
    else:
        for u, v in zip(path[:-1], path[1:]):
            geometry = _best_edge_geometry_between_nodes(G, u, v)
            segment_coords = _geometry_to_latlngs(geometry)
            if segment_coords:
                u_coord = _node_lat_lng(G, u)
                if _haversine_meters(segment_coords[0][0], segment_coords[0][1], u_coord[0], u_coord[1]) > \
                        _haversine_meters(segment_coords[-1][0], segment_coords[-1][1], u_coord[0], u_coord[1]):
                    segment_coords.reverse()
                for coord in segment_coords:
                    _append_coord(coords, coord)

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
                    candidates.append({
                        "path": route_path,
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
                    if prev_cand["oneway"] and prev_dir_diff > MAX_DIRECTION_DIFF_DEGREES:
                        continue
                    # 非单行道不再硬性拒绝，但仍应让方向不一致的候选在
                    # 竞争中付出代价——否则在双向路/辅路并行处，路网距离
                    # 略短就足以让 Viterbi 选中方向完全拧反的边。
                    dir_penalty = (
                        (curr_dir_diff / 180.0) ** 2 + (prev_dir_diff / 180.0) ** 2
                    ) * REVERSE_DIRECTION_PENALTY
                    transition_cost += dir_penalty
                if route["graph_used"] == "undirected_fallback":
                    continue
                total_cost = dp_costs[i - 1][prev_idx] + transition_cost + emissions[i][curr_idx]
                if total_cost < row_costs[curr_idx]:
                    row_costs[curr_idx] = total_cost
                    row_back[curr_idx] = prev_idx

        row_all_inf = all(cost == float("inf") for cost in row_costs)
        if row_all_inf and not prev_row_all_inf:
            # 本帧与上一帧之间彻底连不通（很可能是 GPS 噪声导致的临时
            # 候选不匹配）。不让整条链就此终止：以当帧自身发射概率重新
            # 起跳，相当于在这里截断一条新的匹配子链，断点之外的点仍可
            # 正常参与完整的方向/转移约束匹配。
            row_costs = emissions[i][:]
            row_back = [None] * len(candidate_sets[i])
            restart_flags.append(True)
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
    matched_nodes = []
    debug_segments = []
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

    for seg_index, (start_idx, end_idx) in enumerate(segments):
        segment_df = df.iloc[start_idx:end_idx].reset_index(drop=True)
        seg_lats = segment_df["lati"].tolist()
        seg_lngs = segment_df["long"].tolist()
        seg_bearings = [_movement_bearing(segment_df, idx) for idx in range(len(segment_df))]
        if not seg_lats:
            continue

        if len(seg_lats) == 1:
            _append_coord(corrected, [seg_lats[0], seg_lngs[0]])
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
        _append_coord(corrected, snapped_points[0]["snapped"])

        for i in range(1, len(snapped_points)):
            prev_snap = snapped_points[i - 1]
            curr_snap = snapped_points[i]
            lat_prev, lng_prev = seg_lats[i - 1], seg_lngs[i - 1]
            lat_curr, lng_curr = seg_lats[i], seg_lngs[i]

            if prev_snap["snap_distance_m"] > max_snap_distance_m or curr_snap["snap_distance_m"] > max_snap_distance_m:
                _append_coord(corrected, [lat_curr, lng_curr])
                stats["offroad_points"] += int(prev_snap["snap_distance_m"] > max_snap_distance_m) + int(curr_snap["snap_distance_m"] > max_snap_distance_m)
                stats["skipped_segments"] += 1
                debug_segments.append({
                    "type": "offroad",
                    "coords": [[lat_prev, lng_prev], [lat_curr, lng_curr]],
                    "message": f"片段{seg_index} 段{i}: 点距道路过远，回退原始GPS",
                })
                continue

            stats["total_segments"] += 1
            dist_m = _haversine_meters(lat_prev, lng_prev, lat_curr, lng_curr)
            gps_bearing = seg_bearings[i - 1]

            if dist_m > max_gps_segment_m:
                _append_coord(corrected, [lat_curr, lng_curr])
                stats["skipped_segments"] += 1
                debug_segments.append({
                    "type": "jump",
                    "coords": [[lat_prev, lng_prev], [lat_curr, lng_curr]],
                    "message": f"片段{seg_index} 段{i}: GPS跳点过大，回退原始GPS",
                })
                logger.warning(
                    "片段 %d 段 %d GPS 间距过大(%.0fm)，跳过最短路径 | (%f,%f)->(%f,%f)",
                    seg_index, i, dist_m, lat_prev, lng_prev, lat_curr, lng_curr,
                )
                continue

            same_edge = prev_snap["edge"] == curr_snap["edge"]
            if same_edge:
                curr_dir_diff = _angle_diff_degrees(gps_bearing, curr_snap["edge_bearing"]) if gps_bearing is not None else None
                if curr_snap["oneway"] and curr_dir_diff is not None and curr_dir_diff > MAX_DIRECTION_DIFF_DEGREES:
                    _append_coord(corrected, curr_snap["snapped"])
                    stats["rejected_segments"] += 1
                    debug_segments.append({
                        "type": "direction_warning",
                        "coords": [[lat_prev, lng_prev], [lat_curr, lng_curr]],
                        "message": f"片段{seg_index} 段{i}: 同边但单行方向冲突({curr_dir_diff:.0f}°)，回退吸附点",
                    })
                    continue
                _append_coord(corrected, curr_snap["snapped"])
                stats["success_segments"] += 1
                continue

            route = None
            if path_indices is not None:
                prev_idx = path_indices[i - 1]
                curr_idx = path_indices[i]
                route = route_cache.get((i - 1, prev_idx, i, curr_idx))
            if route is None:
                route = _choose_best_route(G_directed, G_undirected, prev_snap, curr_snap, use_undirected=use_undirected)
            if route is None:
                _append_coord(corrected, curr_snap["snapped"])
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
                debug_segments.append({
                    "type": "detour",
                    "coords": [[lat_prev, lng_prev], [lat_curr, lng_curr]],
                    "message": f"片段{seg_index} 段{i}: 绕行过大，回退吸附点",
                })
                logger.warning(
                    "片段 %d 段 %d 绕行过大(路网=%.0fm, GPS=%.0fm)，拒绝校正 | (%f,%f)->(%f,%f)",
                    seg_index, i, route["total_cost_m"], dist_m, lat_prev, lng_prev, lat_curr, lng_curr,
                )
                continue

            if route["graph_used"] == "undirected_fallback":
                _append_coord(corrected, curr_snap["snapped"])
                stats["undirected_fallback"] += 1
                stats["rejected_segments"] += 1
                debug_segments.append({
                    "type": "undirected_fallback",
                    "coords": [[lat_prev, lng_prev], [lat_curr, lng_curr]],
                    "message": f"片段{seg_index} 段{i}: 仅无向图可达，回退吸附点",
                })
                continue

            curr_dir_diff = _angle_diff_degrees(gps_bearing, curr_snap["edge_bearing"]) if gps_bearing is not None else None
            prev_dir_diff = _angle_diff_degrees(gps_bearing, prev_snap["edge_bearing"]) if gps_bearing is not None else None
            if gps_bearing is not None and (
                (prev_snap["oneway"] and prev_dir_diff > MAX_DIRECTION_DIFF_DEGREES) or
                (curr_snap["oneway"] and curr_dir_diff > MAX_DIRECTION_DIFF_DEGREES)
            ):
                _append_coord(corrected, curr_snap["snapped"])
                stats["rejected_segments"] += 1
                debug_segments.append({
                    "type": "direction_warning",
                    "coords": [[lat_prev, lng_prev], [lat_curr, lng_curr]],
                    "message": f"片段{seg_index} 段{i}: 单行方向冲突(prev={prev_dir_diff:.0f}°, curr={curr_dir_diff:.0f}°)，回退吸附点",
                })
                continue

            stats["success_segments"] += 1
            path_coords = _route_to_geometry_coords(
                G,
                route["path"],
                start_snap=prev_snap["snapped"],
                end_snap=curr_snap["snapped"],
            )
            for coord in path_coords:
                _append_coord(corrected, coord)

        seg_total = stats["total_segments"] - seg_total_before
        seg_success = stats["success_segments"] - seg_success_before
        seg_failed = stats["failed_segments"] - seg_failed_before
        seg_skipped = stats["skipped_segments"] - seg_skipped_before
        seg_rejected = stats["rejected_segments"] - seg_rejected_before
        seg_problem_ratio = (seg_rejected + seg_failed) / max(seg_total, 1)
        if seg_total > 0 and seg_problem_ratio >= SEGMENT_DEGRADE_REJECT_RATIO:
            corrected = corrected[:segment_corrected_start]
            for coord in segment_df[["lati", "long"]].values.tolist():
                _append_coord(corrected, coord)

            debug_segments = debug_segments[:segment_debug_start]
            debug_segments.append({
                "type": "segment_degraded",
                "coords": segment_df[["lati", "long"]].values.tolist(),
                "message": f"片段{seg_index}: 拒绝/失败比例过高({seg_rejected + seg_failed}/{seg_total})，整段降级为原始GPS",
            })

            stats["success_segments"] -= seg_success
            stats["failed_segments"] -= seg_failed
            stats["skipped_segments"] -= seg_skipped
            stats["rejected_segments"] -= seg_rejected
            stats["degraded_segments"] += 1

    corrected_coords = [[lat, lng] for lat, lng in _smooth_debug_backtracks(corrected, debug_segments)]

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
        "original_coords": original_coords,
        "matched_nodes": matched_nodes,
        "stats": stats,
        "debug_segments": debug_segments,
    }


def correct_vehicle_trajectory(vehicle_id, start_time=None, end_time=None, **kwargs):
    """从车辆缓存读取轨迹并校正。"""
    from map_visualization import load_vehicle_trajectory

    df = load_vehicle_trajectory(vehicle_id, start_time, end_time)
    if df.empty:
        raise ValueError(f"车辆 {vehicle_id} 在指定时间段内无数据")

    G, source, load_sec = load_road_network()
    result = correct_trajectory(df, G=G, **kwargs)
    result["vehicle_id"] = vehicle_id
    result["df"] = df
    result["network_source"] = source
    result["network_load_seconds"] = load_sec
    return result


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
    return summaries


if __name__ == "__main__":
    run_sample_correction()