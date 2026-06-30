# -*- coding: utf-8 -*-
"""
07 校正结果复用与最短最快路线
==============================
依据：07-校正结果复用与最短最快路线.html

功能：
    1. 为 OD 端点补充路网信息（pickup/dropoff 校正坐标和节点）
    2. 建立道路基准速度缓存（全日历史平均速度 + 静态通行成本）
    3. 将 route_cost 写入路网
    4. 计算最短距离路线和基准最快路线
    5. 生成双路线地图（支持 Web 接口选点 / 历史 OD 端点）
"""

import os
import sys
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox
import folium
from folium import plugins

# ========================= 路径配置 =========================
PROJECT_ROOT = Path(__file__).parent.parent
ROAD_NETWORK_PKL = PROJECT_ROOT / "data" / "raw" / "shenzhen_drive.pkl"
TRACK_CACHE = PROJECT_ROOT / "data" / "cache" / "matched_trajectory_stage7.parquet"
OD_CACHE_SRC = PROJECT_ROOT / "data" / "cache" / "od" / "od_cache.csv"
OD_CACHE_OUT = PROJECT_ROOT / "data" / "cache" / "od" / "od_cache_stage7.parquet"
SPEED_CACHE = PROJECT_ROOT / "data" / "cache" / "edge_baseline_speed.parquet"
GRAPH_CACHE = PROJECT_ROOT / "data" / "cache" / "shenzhen_drive_stage7.pkl"
MAP_OUTPUT_DIR = PROJECT_ROOT / "maps"

# ========================= highway 默认速度 =========================
DEFAULT_SPEED_KPH = {
    "motorway": 80.0,
    "motorway_link": 40.0,
    "trunk": 60.0,
    "trunk_link": 35.0,
    "primary": 50.0,
    "primary_link": 30.0,
    "secondary": 40.0,
    "secondary_link": 25.0,
    "tertiary": 35.0,
    "tertiary_link": 25.0,
    "unclassified": 30.0,
    "residential": 25.0,
    "living_street": 15.0,
}
FALLBACK_SPEED_KPH = 30.0


def normalize_highway(value):
    """统一 highway 字段格式"""
    if isinstance(value, list):
        value = value[0] if value else "road"
    return str(value or "road")


# ========================= 步骤一：加载路网 =========================
def load_road_network():
    """加载路网图"""
    print("[1] 加载路网...")
    t0 = time.perf_counter()
    with open(ROAD_NETWORK_PKL, "rb") as f:
        G = pickle.load(f)
    print(f"    节点={G.number_of_nodes()}, 边={G.number_of_edges()}, "
          f"耗时={time.perf_counter()-t0:.2f}s")
    return G


# ========================= 步骤二：为 OD 端点补充路网信息 =========================
def enrich_od_endpoints(G):
    """
    为 OD 上下车点补充校正坐标和路网节点
    优先从校正轨迹关联，无法关联时使用 nearest_nodes
    """
    print("\n[2] 为 OD 端点补充路网信息...")

    # 读取 OD 缓存
    od = pd.read_csv(OD_CACHE_SRC, encoding="utf-8")
    # 统一列名
    od.columns = [
        "id", "pickup_time", "pickup_lon", "pickup_lat",
        "dropoff_time", "dropoff_lon", "dropoff_lat",
        "distance_km", "duration_s", "duration_min",
        "avg_speed_kmh", "heading", "start_date", "start_hour", "is_valid"
    ]
    od = od[od["is_valid"] == True].copy()
    od["pickup_time"] = pd.to_datetime(od["pickup_time"])
    od["dropoff_time"] = pd.to_datetime(od["dropoff_time"])
    print(f"    有效 OD 数量: {len(od):,}")

    # 读取校正轨迹
    print("    加载校正轨迹...")
    track = pd.read_parquet(TRACK_CACHE)
    track["time"] = pd.to_datetime(track["time"])

    # 构建车辆分组索引
    track_by_car = {
        car_id: group.sort_values("time").reset_index(drop=True)
        for car_id, group in track.groupby("id", sort=False)
    }

    def find_nearest_track_record(car_track, target_time, tolerance_s=10):
        """在校正轨迹中查找时间最近的记录"""
        if car_track is None or car_track.empty:
            return None
        delta = (car_track["time"] - target_time).abs()
        min_idx = delta.idxmin()
        if delta.loc[min_idx].total_seconds() > tolerance_s:
            return None
        return car_track.loc[min_idx]

    # 批量处理
    pickup_nodes = []
    pickup_matched_lons = []
    pickup_matched_lats = []
    pickup_snap_dists = []
    dropoff_nodes = []
    dropoff_matched_lons = []
    dropoff_matched_lats = []
    dropoff_snap_dists = []
    match_sources = []

    for _, row in od.iterrows():
        car_track = track_by_car.get(row["id"])

        # 上车点
        pickup_rec = find_nearest_track_record(car_track, row["pickup_time"])
        if pickup_rec is not None:
            pickup_nodes.append(int(pickup_rec["matched_node"]))
            pickup_matched_lons.append(float(pickup_rec["matched_lon"]))
            pickup_matched_lats.append(float(pickup_rec["matched_lat"]))
            # 计算吸附距离
            dist = _haversine(
                row["pickup_lat"], row["pickup_lon"],
                pickup_rec["matched_lat"], pickup_rec["matched_lon"]
            )
            pickup_snap_dists.append(dist)
            p_source = "track"
        else:
            # 使用 nearest_nodes
            node = ox.distance.nearest_nodes(G, row["pickup_lon"], row["pickup_lat"])
            node_data = G.nodes[node]
            pickup_nodes.append(int(node))
            pickup_matched_lons.append(float(node_data["x"]))
            pickup_matched_lats.append(float(node_data["y"]))
            dist = _haversine(
                row["pickup_lat"], row["pickup_lon"],
                node_data["y"], node_data["x"]
            )
            pickup_snap_dists.append(dist)
            p_source = "nearest"

        # 下车点
        dropoff_rec = find_nearest_track_record(car_track, row["dropoff_time"])
        if dropoff_rec is not None:
            dropoff_nodes.append(int(dropoff_rec["matched_node"]))
            dropoff_matched_lons.append(float(dropoff_rec["matched_lon"]))
            dropoff_matched_lats.append(float(dropoff_rec["matched_lat"]))
            dist = _haversine(
                row["dropoff_lat"], row["dropoff_lon"],
                dropoff_rec["matched_lat"], dropoff_rec["matched_lon"]
            )
            dropoff_snap_dists.append(dist)
            d_source = "track"
        else:
            node = ox.distance.nearest_nodes(G, row["dropoff_lon"], row["dropoff_lat"])
            node_data = G.nodes[node]
            dropoff_nodes.append(int(node))
            dropoff_matched_lons.append(float(node_data["x"]))
            dropoff_matched_lats.append(float(node_data["y"]))
            dist = _haversine(
                row["dropoff_lat"], row["dropoff_lon"],
                node_data["y"], node_data["x"]
            )
            dropoff_snap_dists.append(dist)
            d_source = "nearest"

        match_sources.append(f"{p_source}/{d_source}")

    od["pickup_node"] = pickup_nodes
    od["pickup_matched_lon"] = pickup_matched_lons
    od["pickup_matched_lat"] = pickup_matched_lats
    od["pickup_snap_distance_m"] = pickup_snap_dists
    od["dropoff_node"] = dropoff_nodes
    od["dropoff_matched_lon"] = dropoff_matched_lons
    od["dropoff_matched_lat"] = dropoff_matched_lats
    od["dropoff_snap_distance_m"] = dropoff_snap_dists
    od["match_source"] = match_sources

    # 保存
    OD_CACHE_OUT.parent.mkdir(parents=True, exist_ok=True)
    od.to_parquet(OD_CACHE_OUT, engine="pyarrow", compression="zstd", index=False)

    track_pct = match_sources.count("track/track") / len(match_sources) * 100
    print(f"    完成！track/track 匹配率: {track_pct:.1f}%")
    print(f"    吸附距离中位数: pickup={np.median(pickup_snap_dists):.1f}m, "
          f"dropoff={np.median(dropoff_snap_dists):.1f}m")
    print(f"    保存到: {OD_CACHE_OUT}")

    return od


# ========================= 步骤三：建立道路基准速度缓存 =========================
def build_edge_baseline_speed(G):
    """
    从校正轨迹统计每条道路的全日平均速度
    """
    print("\n[3] 建立道路基准速度缓存...")

    track = pd.read_parquet(TRACK_CACHE)
    print(f"    校正轨迹: {len(track):,} 条记录")

    # 筛选有效记录
    valid = track[
        track["edge_u"].notna()
        & track["edge_v"].notna()
        & track["edge_key"].notna()
        & track["speed"].between(1, 120)
    ].copy()
    print(f"    有效记录（有边且速度在 1-120 km/h）: {len(valid):,}")

    # 按道路边统计
    speed_stats = (
        valid.groupby(["edge_u", "edge_v", "edge_key"])
        .agg(
            avg_speed=("speed", "mean"),
            sample_count=("speed", "size"),
            vehicle_count=("id", "nunique"),
        )
        .reset_index()
    )
    print(f"    有速度数据的道路边: {len(speed_stats):,}")

    # 获取道路信息
    edge_info = []
    for u, v, key, data in G.edges(keys=True, data=True):
        edge_info.append({
            "edge_u": u,
            "edge_v": v,
            "edge_key": key,
            "length": data.get("length", 0),
            "highway": normalize_highway(data.get("highway")),
        })
    edge_df = pd.DataFrame(edge_info)

    # 合并
    speed_stats["edge_u"] = speed_stats["edge_u"].astype("int64")
    speed_stats["edge_v"] = speed_stats["edge_v"].astype("int64")
    speed_stats["edge_key"] = speed_stats["edge_key"].astype("int64")
    edge_df["edge_u"] = edge_df["edge_u"].astype("int64")
    edge_df["edge_v"] = edge_df["edge_v"].astype("int64")
    edge_df["edge_key"] = edge_df["edge_key"].astype("int64")

    speed_stats = speed_stats.merge(
        edge_df, on=["edge_u", "edge_v", "edge_key"], how="left"
    )

    # 计算通行成本（秒）
    speed_stats["route_cost"] = (
        speed_stats["length"] / speed_stats["avg_speed"] * 3.6
    )

    # 可靠样本的 highway 中位数速度
    reliable = speed_stats[speed_stats["sample_count"] >= 3].copy()
    highway_median_speed = (
        reliable.groupby("highway")["avg_speed"]
        .median()
        .to_dict()
    )
    print(f"    highway 中位数速度类型数: {len(highway_median_speed)}")

    # 保存
    SPEED_CACHE.parent.mkdir(parents=True, exist_ok=True)
    speed_stats.to_parquet(SPEED_CACHE, engine="pyarrow", compression="zstd", index=False)
    print(f"    保存到: {SPEED_CACHE}")

    return speed_stats, highway_median_speed


# ========================= 步骤四：把通行成本写入路网 =========================
def apply_baseline_cost(G, speed_stats, highway_median_speed):
    """
    为路网所有边写入 baseline_speed_kph 和 route_cost
    """
    print("\n[4] 写入路网静态通行成本...")

    # 构建已观测速度字典
    observed_speed = {}
    for row in speed_stats.itertuples():
        if row.sample_count >= 3 and pd.notna(row.avg_speed) and row.avg_speed > 0:
            observed_speed[(int(row.edge_u), int(row.edge_v), int(row.edge_key))] = float(row.avg_speed)

    print(f"    有效观测速度: {len(observed_speed):,} 条边")

    source_counts = {"observed": 0, "highway_median": 0, "default": 0, "fallback": 0}

    for u, v, key, data in G.edges(keys=True, data=True):
        length = float(data.get("length", 0.0))
        road_type = normalize_highway(data.get("highway"))

        speed = observed_speed.get((u, v, key))
        if speed:
            source_counts["observed"] += 1
        else:
            speed = highway_median_speed.get(road_type)
            if speed:
                source_counts["highway_median"] += 1
            else:
                speed = DEFAULT_SPEED_KPH.get(road_type)
                if speed:
                    source_counts["default"] += 1
                else:
                    speed = FALLBACK_SPEED_KPH
                    source_counts["fallback"] += 1

        speed = max(float(speed), 1.0)
        data["baseline_speed_kph"] = speed
        data["route_cost"] = length / speed * 3.6  # 秒

    # 验证
    missing_cost = sum(
        "route_cost" not in data
        for _, _, _, data in G.edges(keys=True, data=True)
    )
    assert missing_cost == 0, f"仍有 {missing_cost} 条道路缺少 route_cost"

    print(f"    速度来源分布:")
    for src, cnt in source_counts.items():
        print(f"      {src}: {cnt:,}")

    # 保存带成本的路网
    GRAPH_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(GRAPH_CACHE, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"    路网已保存: {GRAPH_CACHE}")

    return G


# ========================= 步骤五：路线计算 =========================
def compute_routes(G, origin_lon, origin_lat, destination_lon, destination_lat):
    """
    计算最短距离路线和基准最快路线

    Args:
        G: 带 route_cost 的路网
        origin_lon/lat: 起点经纬度
        destination_lon/lat: 终点经纬度

    Returns:
        dict with shortest/fastest route info, or error message
    """
    # 查找最近节点
    origin_node = ox.distance.nearest_nodes(G, origin_lon, origin_lat)
    dest_node = ox.distance.nearest_nodes(G, destination_lon, destination_lat)

    if origin_node == dest_node:
        return {"error": "起点和终点吸附到同一节点"}

    # 最短距离路线
    try:
        shortest_route = nx.shortest_path(G, origin_node, dest_node, weight="length")
    except nx.NodeNotFound:
        return {"error": "起点或终点节点不在路网中"}
    except nx.NetworkXNoPath:
        return {"error": "起终点之间无可达路径（最短距离）"}

    # 基准最快路线
    try:
        fastest_route = nx.shortest_path(G, origin_node, dest_node, weight="route_cost")
    except nx.NetworkXNoPath:
        return {"error": "起终点之间无可达路径（最快路线）"}

    # 提取路线几何和统计
    shortest_gdf = ox.routing.route_to_gdf(G, shortest_route, weight="length")
    fastest_gdf = ox.routing.route_to_gdf(G, fastest_route, weight="route_cost")

    shortest_distance = shortest_gdf["length"].sum()
    fastest_distance = fastest_gdf["length"].sum()
    fastest_cost = fastest_gdf["route_cost"].sum()
    shortest_cost = shortest_gdf["route_cost"].sum()

    return {
        "shortest": {
            "route": shortest_route,
            "gdf": shortest_gdf,
            "distance_m": float(shortest_distance),
            "cost_s": float(shortest_cost),
            "edges": len(shortest_route) - 1,
        },
        "fastest": {
            "route": fastest_route,
            "gdf": fastest_gdf,
            "distance_m": float(fastest_distance),
            "cost_s": float(fastest_cost),
            "edges": len(fastest_route) - 1,
        },
        "origin_node": origin_node,
        "dest_node": dest_node,
    }


# ========================= 步骤六：绘制双路线地图 =========================
def plot_dual_route_map(G, result, origin_latlon, dest_latlon, save_path=None):
    """
    绘制最短路线（蓝色）和最快路线（绿色）的地图
    """
    mid_lat = (origin_latlon[0] + dest_latlon[0]) / 2
    mid_lon = (origin_latlon[1] + dest_latlon[1]) / 2

    m = folium.Map(location=[mid_lat, mid_lon], zoom_start=13)

    # 最短路线（蓝色）
    shortest_gdf = result["shortest"]["gdf"]
    for geom in shortest_gdf.geometry:
        if geom is None:
            continue
        coords = [(lat, lon) for lon, lat in geom.coords]
        folium.PolyLine(
            coords, color="blue", weight=5, opacity=0.7,
            tooltip=f"最短路线: {result['shortest']['distance_m']:.0f}m"
        ).add_to(m)

    # 最快路线（绿色）
    fastest_gdf = result["fastest"]["gdf"]
    for geom in fastest_gdf.geometry:
        if geom is None:
            continue
        coords = [(lat, lon) for lon, lat in geom.coords]
        folium.PolyLine(
            coords, color="green", weight=5, opacity=0.7,
            tooltip=f"最快路线: {result['fastest']['cost_s']:.0f}s"
        ).add_to(m)

    # 起点标记
    folium.Marker(
        origin_latlon, icon=folium.Icon(color="red", icon="play"),
        tooltip="起点"
    ).add_to(m)

    # 终点标记
    folium.Marker(
        dest_latlon, icon=folium.Icon(color="black", icon="stop"),
        tooltip="终点"
    ).add_to(m)

    # 图例
    legend_html = """
    <div style="position:fixed; bottom:30px; left:30px; z-index:1000;
                background:white; padding:12px 16px; border-radius:8px;
                box-shadow:0 2px 6px rgba(0,0,0,.2); font-size:13px;">
      <b>路线图例</b><br>
      <span style="color:blue;">&#9644;</span> 最短距离路线: {sd:.0f}m, {sc:.0f}s<br>
      <span style="color:green;">&#9644;</span> 基准最快路线: {fd:.0f}m, {fc:.0f}s
    </div>
    """.format(
        sd=result["shortest"]["distance_m"],
        sc=result["shortest"]["cost_s"],
        fd=result["fastest"]["distance_m"],
        fc=result["fastest"]["cost_s"],
    )
    m.get_root().html.add_child(folium.Element(legend_html))

    # 图层控制
    folium.LayerControl().add_to(m)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        m.save(save_path)
        print(f"    地图已保存: {save_path}")

    return m


# ========================= 工具函数 =========================
def _haversine(lat1, lon1, lat2, lon2):
    """计算两点距离（米）"""
    from math import radians, sin, cos, sqrt, atan2
    R = 6371000
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))


# ========================= Web 接口路线选点页面 =========================
def create_route_picker_html(G, save_path=None):
    """
    生成可交互的路线选点地图页面（Leaflet + 后端接口）
    支持点击地图选择起终点，自动请求路线
    """
    html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>07 最短/最快路线选点</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  body { margin: 0; font-family: "Microsoft YaHei", sans-serif; }
  #map { width: 100%; height: 100vh; }
  .info-panel {
    position: fixed; top: 10px; right: 10px; z-index: 1000;
    background: white; padding: 14px 18px; border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,.2); max-width: 320px;
    font-size: 13px; line-height: 1.8;
  }
  .info-panel h3 { margin: 0 0 8px; font-size: 15px; }
  .info-panel .hint { color: #666; }
  .info-panel .result { margin-top: 10px; }
  .info-panel button {
    margin-top: 8px; padding: 6px 14px; border: none;
    border-radius: 6px; background: #0f6cbd; color: white;
    cursor: pointer; font-size: 13px;
  }
  .info-panel button:hover { background: #0a4f8a; }
  .legend { margin-top: 10px; }
  .legend span { display: inline-block; width: 30px; height: 4px; margin-right: 6px; vertical-align: middle; }
</style>
</head>
<body>
<div id="map"></div>
<div class="info-panel">
  <h3>路线规划</h3>
  <p class="hint" id="hint">点击地图选择<b>起点</b></p>
  <div class="result" id="result" style="display:none;"></div>
  <button id="reset-btn" style="display:none;" onclick="resetAll()">重新选点</button>
  <div class="legend">
    <span style="background:blue;"></span>最短距离路线<br>
    <span style="background:green;"></span>基准最快路线
  </div>
</div>

<script>
const map = L.map('map').setView([22.6, 114.1], 11);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution: '&copy; OpenStreetMap'
}).addTo(map);

let points = [];
let markers = [];
let shortestLayer = null;
let fastestLayer = null;

function resetAll() {
  points = [];
  markers.forEach(m => map.removeLayer(m));
  markers = [];
  if (shortestLayer) { map.removeLayer(shortestLayer); shortestLayer = null; }
  if (fastestLayer) { map.removeLayer(fastestLayer); fastestLayer = null; }
  document.getElementById('hint').innerHTML = '点击地图选择<b>起点</b>';
  document.getElementById('result').style.display = 'none';
  document.getElementById('reset-btn').style.display = 'none';
}

map.on('click', async function(e) {
  if (points.length >= 2) return;

  const pt = { lat: e.latlng.lat, lon: e.latlng.lng };
  points.push(pt);

  const label = points.length === 1 ? '起点' : '终点';
  const color = points.length === 1 ? 'red' : 'black';
  const marker = L.marker([pt.lat, pt.lon]).addTo(map).bindTooltip(label).openTooltip();
  markers.push(marker);

  if (points.length === 1) {
    document.getElementById('hint').innerHTML = '点击地图选择<b>终点</b>';
    return;
  }

  // 两点齐全，请求路线
  document.getElementById('hint').innerHTML = '正在计算路线...';

  try {
    const resp = await fetch('/api/routes', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ origin: points[0], destination: points[1] })
    });
    const data = await resp.json();

    if (data.error) {
      document.getElementById('hint').innerHTML = '<span style="color:red;">' + data.error + '</span>';
      document.getElementById('reset-btn').style.display = 'block';
      return;
    }

    // 绘制路线
    shortestLayer = L.geoJSON(data.shortest, {
      style: { color: 'blue', weight: 5, opacity: 0.7 }
    }).addTo(map);

    fastestLayer = L.geoJSON(data.fastest, {
      style: { color: 'green', weight: 5, opacity: 0.7 }
    }).addTo(map);

    map.fitBounds(shortestLayer.getBounds().extend(fastestLayer.getBounds()));

    // 显示结果
    const s = data.summary;
    document.getElementById('result').innerHTML =
      '<b>最短距离路线</b>: ' + (s.shortest_distance_m/1000).toFixed(2) + ' km, ' +
      (s.shortest_cost_s/60).toFixed(1) + ' min<br>' +
      '<b>基准最快路线</b>: ' + (s.fastest_distance_m/1000).toFixed(2) + ' km, ' +
      (s.fastest_cost_s/60).toFixed(1) + ' min';
    document.getElementById('result').style.display = 'block';
    document.getElementById('hint').innerHTML = '计算完成';
    document.getElementById('reset-btn').style.display = 'block';

  } catch (err) {
    document.getElementById('hint').innerHTML = '<span style="color:red;">请求失败: ' + err.message + '</span>';
    document.getElementById('reset-btn').style.display = 'block';
  }
});
</script>
</body>
</html>"""

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"    选点页面已保存: {save_path}")

    return html


# ========================= 主流程 =========================
def run_pipeline():
    """执行完整的阶段 7 流水线"""
    print("=" * 60)
    print("07 校正结果复用与最短最快路线")
    print("=" * 60)

    # 步骤 1: 加载路网
    G = load_road_network()

    # 步骤 2: OD 端点补充
    print("\n  [提示] OD 端点补充需要遍历 48 万条记录，预计 5-15 分钟...")
    od = enrich_od_endpoints(G)

    # 步骤 3: 道路基准速度
    speed_stats, highway_median_speed = build_edge_baseline_speed(G)

    # 步骤 4: 写入路网
    G = apply_baseline_cost(G, speed_stats, highway_median_speed)

    # 步骤 5: 示例路线计算
    print("\n[5] 示例路线计算...")
    # 从 OD 中取一个样例
    sample_od = od.iloc[0]
    result = compute_routes(
        G,
        origin_lon=sample_od["pickup_matched_lon"],
        origin_lat=sample_od["pickup_matched_lat"],
        destination_lon=sample_od["dropoff_matched_lon"],
        destination_lat=sample_od["dropoff_matched_lat"],
    )

    if "error" in result:
        print(f"    路线计算失败: {result['error']}")
    else:
        print(f"    最短路线: {result['shortest']['distance_m']:.0f}m, "
              f"{result['shortest']['edges']} 条边, "
              f"{result['shortest']['cost_s']:.0f}s")
        print(f"    最快路线: {result['fastest']['distance_m']:.0f}m, "
              f"{result['fastest']['edges']} 条边, "
              f"{result['fastest']['cost_s']:.0f}s")

        # 步骤 6: 绘制地图
        print("\n[6] 绘制双路线地图...")
        map_path = str(MAP_OUTPUT_DIR / "07_dual_route.html")
        plot_dual_route_map(
            G, result,
            origin_latlon=(sample_od["pickup_matched_lat"], sample_od["pickup_matched_lon"]),
            dest_latlon=(sample_od["dropoff_matched_lat"], sample_od["dropoff_matched_lon"]),
            save_path=map_path,
        )

    # 生成选点页面
    print("\n[7] 生成交互选点页面...")
    picker_path = str(MAP_OUTPUT_DIR / "07_route_picker.html")
    create_route_picker_html(G, save_path=picker_path)

    print("\n" + "=" * 60)
    print("阶段 7 完成！")
    print("=" * 60)
    print(f"\n生成文件:")
    print(f"  - OD 缓存:      {OD_CACHE_OUT}")
    print(f"  - 速度缓存:     {SPEED_CACHE}")
    print(f"  - 带成本路网:   {GRAPH_CACHE}")
    print(f"  - 双路线地图:   {MAP_OUTPUT_DIR / '07_dual_route.html'}")
    print(f"  - 交互选点页面: {MAP_OUTPUT_DIR / '07_route_picker.html'}")
    print(f"\n启动 Web 接口（支持选点）:")
    print(f"  python code/stage7_web.py")


def run_speed_only():
    """仅生成速度缓存和路网（跳过 OD）"""
    print("=" * 60)
    print("07 校正结果复用 - 仅速度 + 路网")
    print("=" * 60)

    G = load_road_network()
    speed_stats, highway_median_speed = build_edge_baseline_speed(G)
    G = apply_baseline_cost(G, speed_stats, highway_median_speed)

    print("\n完成！")


def run_demo_route(origin_lon, origin_lat, dest_lon, dest_lat):
    """计算并展示单条路线"""
    print("加载带成本路网...")
    if GRAPH_CACHE.exists():
        with open(GRAPH_CACHE, "rb") as f:
            G = pickle.load(f)
    else:
        print("带成本路网不存在，请先运行 python code/stage7_route.py 生成")
        return

    result = compute_routes(G, origin_lon, origin_lat, dest_lon, dest_lat)
    if "error" in result:
        print(f"路线计算失败: {result['error']}")
        return

    print(f"最短路线: {result['shortest']['distance_m']/1000:.2f}km, "
          f"{result['shortest']['cost_s']/60:.1f}min")
    print(f"最快路线: {result['fastest']['distance_m']/1000:.2f}km, "
          f"{result['fastest']['cost_s']/60:.1f}min")

    map_path = str(MAP_OUTPUT_DIR / "07_dual_route.html")
    plot_dual_route_map(
        G, result,
        origin_latlon=(origin_lat, origin_lon),
        dest_latlon=(dest_lat, dest_lon),
        save_path=map_path,
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "speed":
            run_speed_only()
        elif cmd == "route" and len(sys.argv) >= 6:
            run_demo_route(
                float(sys.argv[2]), float(sys.argv[3]),
                float(sys.argv[4]), float(sys.argv[5]),
            )
        elif cmd == "help":
            print("用法:")
            print("  python code/stage7_route.py         # 完整流水线")
            print("  python code/stage7_route.py speed   # 仅速度+路网")
            print("  python code/stage7_route.py route <起点经度> <起点纬度> <终点经度> <终点纬度>")
        else:
            print("未知命令，使用 help 查看帮助")
    else:
        run_pipeline()
