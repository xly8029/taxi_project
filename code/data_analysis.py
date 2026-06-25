# -*- coding: utf-8 -*-
"""
出租车GPS数据分析及可视化模块
================================
覆盖阶段05《热力图与统计分析》主要需求：
    1. 制作车辆位置静态热力图和上车点静态热力图
    2. 统计每小时订单数量、载客车辆数量和载客率变化
    3. 尝试用 DBSCAN 聚类上车点，输出中心点、热力值和时间字段
    4. 制作动态热力图，支持按分钟、15分钟、30分钟、60分钟聚合
    5. 完成短途、中途、长途订单数量和占比统计
    6. 完成车辆全天载客率、总里程、载客里程、空载里程统计，并导出 CSV

依赖数据：
    - data/cache/minute/      分钟缓存（车辆位置热力图、载客车辆数）
    - data/cache/od/od_cache.csv   正常OD缓存（上车点热力图、订单统计、DBSCAN）
    - data/cache/vehicle/     车辆缓存（车辆全天里程与载客率统计）
"""

import json
import os
from math import cos, radians

import folium
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from folium.plugins import HeatMap, HeatMapWithTime
from sklearn.cluster import DBSCAN


# ========================= 配置区 =========================
BASE_DIR = os.path.dirname(__file__)
VEHICLE_CACHE_DIR = os.path.join(BASE_DIR, "../data/cache/vehicle")
MINUTE_CACHE_DIR = os.path.join(BASE_DIR, "../data/cache/minute/2013-10-22")
OD_CACHE_PATH = os.path.join(BASE_DIR, "../data/cache/od/od_cache.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "../analysis")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
MAP_DIR = os.path.join(OUTPUT_DIR, "maps")
TABLE_DIR = os.path.join(OUTPUT_DIR, "tables")

SHENZHEN_CENTER = [22.52847, 114.05454]

TRIP_NEAR_KM = 4
TRIP_MIDDLE_KM = 8

DBSCAN_EPS_DEG = 0.005
DBSCAN_MIN_SAMPLES = 8
HEATMAP_GRADIENT = {
    0.05: '#2b83ba',
    0.20: '#abdda4',
    0.45: '#ffffbf',
    0.70: '#fdae61',
    1.00: '#d7191c',
}
# ==========================================================


plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False


def ensure_output_dirs():
    for path in [OUTPUT_DIR, FIG_DIR, MAP_DIR, TABLE_DIR]:
        os.makedirs(path, exist_ok=True)


def create_base_map(title: str):
    m = folium.Map(location=SHENZHEN_CENTER, zoom_start=11, tiles='OpenStreetMap', control_scale=True)
    title_html = f'''
    <div style="position: fixed; top: 10px; left: 50px; width: 420px; height: 48px;
                background-color: white; border:2px solid #8ea8bf; z-index:9999;
                font-size:18px; text-align:center; padding:10px;">
      <b>{title}</b>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    return m


def scale_heat_weight(series: pd.Series, floor: float = 0.05) -> pd.Series:
    """把热力权重压缩到可见区间，避免弱点完全看不见。"""
    max_weight = series.max()
    if max_weight <= 0:
        return pd.Series([floor] * len(series), index=series.index)

    normalized = series / max_weight

    # 使用平方增强高密度区域差异
    normalized = normalized ** 2
    return floor + (1 - floor) * normalized


def build_weighted_heat_points(df: pd.DataFrame, lat_col: str, lon_col: str, precision: int = 4):
    """按栅格近似聚合热力点，输出 [lat, lon, weight]。"""
    points = df[[lat_col, lon_col]].dropna().copy()
    if points.empty:
        return []

    points[lat_col] = points[lat_col].round(precision)
    points[lon_col] = points[lon_col].round(precision)
    grouped = points.groupby([lat_col, lon_col]).size().reset_index(name='weight')
    grouped['weight'] = scale_heat_weight(grouped['weight'])
    return grouped[[lat_col, lon_col, 'weight']].values.tolist()


def load_od_cache() -> pd.DataFrame:
    df = pd.read_csv(OD_CACHE_PATH)
    df['开始时间'] = pd.to_datetime(df['开始时间'])
    df['结束时间'] = pd.to_datetime(df['结束时间'])
    return df


def load_minute_snapshot(time_str: str) -> pd.DataFrame:
    dt = pd.to_datetime(time_str)
    cache_file = os.path.join(MINUTE_CACHE_DIR, f"{dt.strftime('%H-%M')}.csv")
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"分钟缓存不存在：{cache_file}")
    df = pd.read_csv(cache_file)
    df['time'] = pd.to_datetime(df['time'])
    return df


def haversine_km_series(lon1, lat1, lon2, lat2):
    import numpy as np

    r = 6371.0
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a.clip(0, 1)))


def load_all_vehicle_stats() -> pd.DataFrame:
    vehicle_files = [
        os.path.join(VEHICLE_CACHE_DIR, file_name)
        for file_name in os.listdir(VEHICLE_CACHE_DIR)
        if file_name.endswith('.csv')
    ]

    records = []
    for file_path in vehicle_files:
        df = pd.read_csv(file_path)
        if df.empty or len(df) < 2:
            continue

        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').reset_index(drop=True)
        df['segment_km'] = haversine_km_series(
            df['long'].shift(1), df['lati'].shift(1), df['long'], df['lati']
        )
        df['segment_km'] = df['segment_km'].fillna(0)
        df['prev_status'] = df['status'].shift(1).fillna(df['status'])

        total_km = float(df['segment_km'].sum())
        occupied_km = float(df.loc[df['prev_status'] == 1, 'segment_km'].sum())
        empty_km = float(df.loc[df['prev_status'] == 0, 'segment_km'].sum())
        occupied_ratio = float((df['status'] == 1).mean())

        records.append({
            '车辆id': int(df['id'].iloc[0]),
            '总里程_km': round(total_km, 3),
            '载客里程_km': round(occupied_km, 3),
            '空载里程_km': round(empty_km, 3),
            '全天载客率': round(occupied_ratio, 4),
            '记录点数': int(len(df)),
        })

    return pd.DataFrame(records)


def build_static_vehicle_heatmap(time_str='2013-10-22 08:00', output_name='01_static_vehicle_heatmap.html'):
    df = load_minute_snapshot(time_str)
    heat_data = build_weighted_heat_points(df, 'lati', 'long')

    m = create_base_map(f'车辆位置静态热力图（{time_str}）')
    HeatMap(
        heat_data,
        radius=18,
        blur=14,
        min_opacity=0.30,
        max_zoom=16,
        gradient=HEATMAP_GRADIENT,
    ).add_to(m)
    out_path = os.path.join(MAP_DIR, output_name)
    m.save(out_path)
    return out_path


def build_static_pickup_heatmap(start_time='2013-10-22 08:00:00', end_time='2013-10-22 09:00:00', output_name='02_static_pickup_heatmap.html'):
    df = load_od_cache()
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)
    df = df[(df['开始时间'] >= start_time) & (df['开始时间'] <= end_time)]

    heat_data = build_weighted_heat_points(df, '开始纬度', '开始经度')
    m = create_base_map(f'上车点静态热力图（{start_time} - {end_time}）')
    HeatMap(
        heat_data,
        radius=20,
        blur=16,
        min_opacity=0.28,
        max_zoom=16,
        gradient=HEATMAP_GRADIENT,
    ).add_to(m)
    out_path = os.path.join(MAP_DIR, output_name)
    m.save(out_path)
    return out_path


def _build_heat_time_slices(df: pd.DataFrame, time_col: str, lat_col: str, lon_col: str, freq: str):
    df = df.copy()
    df['time_slice'] = df[time_col].dt.floor(freq)
    grouped = df.groupby('time_slice')
    time_labels = []
    data = []
    for time_slice, group in grouped:
        points = group[[lat_col, lon_col]].dropna().copy()
        if points.empty:
            continue
        counts = points.copy()
        # 动态热力图进一步粗粒度聚合，降低前端每个时间片的点数，避免浏览器渲染失败
        counts[lat_col] = counts[lat_col].round(4)
        counts[lon_col] = counts[lon_col].round(4)
        counts = counts.groupby([lat_col, lon_col]).size().reset_index(name='weight')
        counts['weight'] = scale_heat_weight(counts['weight'])
        data.append(counts[[lat_col, lon_col, 'weight']].values.tolist())
        time_labels.append(time_slice.strftime('%Y-%m-%d %H:%M'))
    return data, time_labels


def build_dynamic_pickup_heatmap(freq='15min', output_name='03_dynamic_pickup_heatmap_15min.html'):
    df = load_od_cache()
    data, time_labels = _build_heat_time_slices(df, '开始时间', '开始纬度', '开始经度', freq)
    out_path = os.path.join(MAP_DIR, output_name)
    build_custom_dynamic_heatmap_html(
        title=f'上车点动态热力图（{freq}）',
        data=data,
        time_labels=time_labels,
        output_path=out_path,
        radius=0.018,
        blur=0.85,
    )
    return out_path


def build_dynamic_vehicle_heatmap(freq='60min', output_name='04_dynamic_vehicle_heatmap_60min.html'):
    minute_files = sorted([f for f in os.listdir(MINUTE_CACHE_DIR) if f.endswith('.csv')])
    records = []
    for file_name in minute_files:
        dt = pd.to_datetime(f"2013-10-22 {file_name.replace('-', ':').replace('.csv', '')}")
        df = pd.read_csv(os.path.join(MINUTE_CACHE_DIR, file_name))
        df['time'] = dt
        records.append(df[['time', 'lati', 'long']])
    df_all = pd.concat(records, ignore_index=True)

    data, time_labels = _build_heat_time_slices(df_all, 'time', 'lati', 'long', freq)
    out_path = os.path.join(MAP_DIR, output_name)
    build_custom_dynamic_heatmap_html(
        title=f'车辆位置动态热力图（{freq}）',
        data=data,
        time_labels=time_labels,
        output_path=out_path,
        radius=0.016,
        blur=0.82,
    )
    return out_path


def build_custom_dynamic_heatmap_html(title: str, data, time_labels, output_path: str, radius: float, blur: float):
    html = f"""<!doctype html>
<html lang=\"zh-CN\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>{title}</title>
  <link rel=\"stylesheet\" href=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.css\"/>
  <script src=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.js\"></script>
  <script src=\"https://unpkg.com/leaflet.heat/dist/leaflet-heat.js\"></script>
  <style>
    html, body {{ margin: 0; height: 100%; font-family: Microsoft YaHei, sans-serif; }}
    #map {{ width: 100%; height: 100vh; }}
    .title {{
      position: fixed; top: 10px; left: 50px; z-index: 9999;
      background: white; border: 2px solid #8ea8bf; padding: 10px 16px; font-size: 18px;
    }}
    .control-box {{
      position: fixed; left: 50px; bottom: 28px; z-index: 9999;
      width: 420px; background: rgba(255,255,255,0.96); border: 1px solid #d7e0ea;
      border-radius: 14px; box-shadow: 0 10px 30px rgba(31,60,96,0.12); padding: 14px 16px;
    }}
    .row {{ display: flex; align-items: center; gap: 10px; margin-top: 10px; }}
    button {{ border: 0; background: #0f6cbd; color: white; border-radius: 8px; padding: 8px 12px; cursor: pointer; }}
    input[type=range] {{ flex: 1; }}
    #time-label {{ font-size: 16px; color: #0f172a; font-weight: 700; margin-top: 8px; }}
    .helper {{ font-size: 12px; color: #64748b; margin-top: 6px; }}
  </style>
</head>
<body>
  <div class=\"title\"><b>{title}</b></div>
  <div id=\"map\"></div>
  <div class=\"control-box\">
    <div><b>时间片控制</b></div>
    <div id=\"time-label\">{time_labels[0] if time_labels else ''}</div>
    <div class=\"row\">
      <button id=\"play-btn\" onclick=\"togglePlay()\">播放</button>
      <button onclick=\"stepFrame(-1)\">上一帧</button>
      <button onclick=\"stepFrame(1)\">下一帧</button>
      <input id=\"slider\" type=\"range\" min=\"0\" max=\"{max(len(time_labels) - 1, 0)}\" value=\"0\" oninput=\"renderFrame(parseInt(this.value))\">
    </div>
    <div class=\"helper\">拖动滑块或使用上一帧 / 下一帧按钮切换时间片；点击播放可自动轮播。</div>
  </div>
  <script>
    const heatFrames = {json.dumps(data, ensure_ascii=False)};
    const timeLabels = {json.dumps(time_labels, ensure_ascii=False)};
    const map = L.map('map').setView([{SHENZHEN_CENTER[0]}, {SHENZHEN_CENTER[1]}], 11);
    L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{ maxZoom: 19, attribution: '&copy; OpenStreetMap contributors' }}).addTo(map);
    let heatLayer = L.heatLayer([], {{ radius: 12, blur: 10, maxZoom: 16, minOpacity: 0.25, gradient: {{0.05:'#2b83ba',0.2:'#abdda4',0.45:'#ffffbf',0.7:'#fdae61',1.0:'#d7191c'}} }}).addTo(map);
    let timer = null;
    let currentIndex = 0;

    function convertFrame(frame) {{
      return frame.map(item => [item[0], item[1], item[2]]);
    }}

    function renderFrame(index) {{
      if (!heatFrames.length) return;
      if (index < 0) index = heatFrames.length - 1;
      if (index >= heatFrames.length) index = 0;
      currentIndex = index;
      const frame = heatFrames[index] || [];
      heatLayer.setLatLngs(convertFrame(frame));
      document.getElementById('slider').value = index;
      document.getElementById('time-label').textContent = timeLabels[index] || '';
    }}

    function stepFrame(direction) {{
      renderFrame(currentIndex + direction);
    }}

    function togglePlay() {{
      const btn = document.getElementById('play-btn');
      if (timer) {{
        clearInterval(timer);
        timer = null;
        btn.textContent = '播放';
        return;
      }}
      btn.textContent = '暂停';
      timer = setInterval(() => {{
        currentIndex = (currentIndex + 1) % heatFrames.length;
        renderFrame(currentIndex);
      }}, 900);
    }}

    renderFrame(0);
  </script>
</body>
</html>
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)


def run_pickup_dbscan(output_name='pickup_dbscan_clusters.csv'):
    df = load_od_cache().copy()
    df['time_slice'] = df['开始时间'].dt.floor('15min')
    cluster_rows = []

    for time_slice, group in df.groupby('time_slice'):
        coords = group[['开始纬度', '开始经度']].dropna().values
        if len(coords) < DBSCAN_MIN_SAMPLES:
            continue

        labels = DBSCAN(eps=DBSCAN_EPS_DEG, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(coords)
        group = group.loc[group[['开始纬度', '开始经度']].dropna().index].copy()
        group['cluster'] = labels

        for cluster_id, cluster_group in group[group['cluster'] != -1].groupby('cluster'):
            cluster_rows.append({
                'lat': round(cluster_group['开始纬度'].mean(), 6),
                'lng': round(cluster_group['开始经度'].mean(), 6),
                'count': int(len(cluster_group)),
                'time': time_slice.strftime('%Y-%m-%d %H:%M:%S'),
                'cluster_id': int(cluster_id),
            })

    out_df = pd.DataFrame(cluster_rows)
    out_path = os.path.join(TABLE_DIR, output_name)
    out_df.to_csv(out_path, index=False, encoding='utf-8-sig')
    return out_df, out_path


def build_dbscan_cluster_map(cluster_df: pd.DataFrame, output_name='05_pickup_dbscan_clusters.html'):
    import numpy as np
    m = create_base_map('DBSCAN 上车点聚类中心图')

    if cluster_df.empty:
        out_path = os.path.join(MAP_DIR, output_name)
        m.save(out_path)
        return out_path

    max_count = cluster_df['count'].max()
    for _, row in cluster_df.iterrows():
        # 用 log 缩放半径，避免大聚类圆圈过大互相叠压，固定范围 4~12px
        norm = np.log1p(row['count']) / np.log1p(max_count)
        radius = 4 + norm * 8
        folium.CircleMarker(
            location=[row['lat'], row['lng']],
            radius=radius,
            color='#a00000',
            fill=True,
            fillColor='#e53935',
            fillOpacity=0.7,
            weight=1.5,
            popup=(
                f"聚类中心<br>时间片: {row['time']}<br>"
                f"权重: {int(row['count'])}<br>簇ID: {int(row['cluster_id'])}"
            )
        ).add_to(m)

    out_path = os.path.join(MAP_DIR, output_name)
    m.save(out_path)
    return out_path


def analyze_hourly_orders_and_occupied(df_od: pd.DataFrame):
    order_stats = df_od.copy()
    order_stats['小时'] = order_stats['开始时间'].dt.hour
    hourly_orders = order_stats.groupby('小时')['车辆id'].count().rename('订单数量').reset_index()

    occupied_records = []
    minute_files = sorted([f for f in os.listdir(MINUTE_CACHE_DIR) if f.endswith('.csv')])
    for file_name in minute_files:
        hh = int(file_name[:2])
        df = pd.read_csv(os.path.join(MINUTE_CACHE_DIR, file_name))
        occupied_count = int((df['status'] == 1).sum())
        occupied_records.append({'小时': hh, '载客车辆数_分钟值': occupied_count})

    occupied_df = pd.DataFrame(occupied_records)
    hourly_occupied = occupied_df.groupby('小时')['载客车辆数_分钟值'].mean().round(2).rename('平均载客车辆数').reset_index()

    merged = pd.merge(hourly_orders, hourly_occupied, on='小时', how='outer').fillna(0)
    merged['载客率'] = (merged['平均载客车辆数'] / merged['平均载客车辆数'].max()).round(4)
    return merged


def plot_hourly_stats(hourly_stats: pd.DataFrame, output_name='01_hourly_order_occupied_ratio.png'):
    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=180)
    ax1.bar(hourly_stats['小时'], hourly_stats['订单数量'], color='#7db7e8', label='订单数量')
    ax1.set_xlabel('小时')
    ax1.set_ylabel('订单数量')
    ax1.set_xticks(range(24))

    ax2 = ax1.twinx()
    ax2.plot(hourly_stats['小时'], hourly_stats['平均载客车辆数'], color='#d1495b', marker='o', label='平均载客车辆数')
    ax2.plot(hourly_stats['小时'], hourly_stats['载客率'], color='#2a9d8f', marker='s', label='载客率')
    ax2.set_ylabel('载客车辆数 / 载客率')

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
    ax1.set_title('每小时订单数量、载客车辆数量和载客率变化')

    out_path = os.path.join(FIG_DIR, output_name)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    return out_path


def analyze_trip_distance_levels(df_od: pd.DataFrame):
    df = df_od.copy()
    distance_col = '轨迹距离_km' if '轨迹距离_km' in df.columns else '订单距离_km'
    df['day'] = df['开始时间'].dt.strftime('%Y%m%d').astype(int)
    df['near'] = (df[distance_col] < TRIP_NEAR_KM).astype(int)
    df['middle'] = ((df[distance_col] >= TRIP_NEAR_KM) & (df[distance_col] <= TRIP_MIDDLE_KM)).astype(int)
    df['far'] = (df[distance_col] > TRIP_MIDDLE_KM).astype(int)

    result = df.groupby('day')[['near', 'middle', 'far']].sum().reset_index()
    result['near_ratio'] = (result['near'] / (result['near'] + result['middle'] + result['far'])).round(4)
    result['middle_ratio'] = (result['middle'] / (result['near'] + result['middle'] + result['far'])).round(4)
    result['far_ratio'] = (result['far'] / (result['near'] + result['middle'] + result['far'])).round(4)
    return result


def export_tables(df_od: pd.DataFrame):
    hourly_stats = analyze_hourly_orders_and_occupied(df_od)
    hourly_path = os.path.join(TABLE_DIR, 'hourly_order_occupied_ratio.csv')
    hourly_stats.to_csv(hourly_path, index=False, encoding='utf-8-sig')

    distance_stats = analyze_trip_distance_levels(df_od)
    distance_path = os.path.join(TABLE_DIR, 'trip_distance_levels.csv')
    distance_stats.to_csv(distance_path, index=False, encoding='utf-8-sig')

    vehicle_stats = load_all_vehicle_stats()
    vehicle_path = os.path.join(TABLE_DIR, 'vehicle_daily_stats.csv')
    vehicle_stats.to_csv(vehicle_path, index=False, encoding='utf-8-sig')

    return {
        'hourly_stats': (hourly_stats, hourly_path),
        'distance_stats': (distance_stats, distance_path),
        'vehicle_stats': (vehicle_stats, vehicle_path),
    }


def plot_duration_distribution(df_od: pd.DataFrame, output_name='02_order_duration_boxplot.png'):
    df = df_od.copy()
    df['小时'] = df['开始时间'].dt.hour
    df['订单时长_分钟'] = (df['结束时间'] - df['开始时间']).dt.total_seconds() / 60

    fig = plt.figure(figsize=(8, 4.5), dpi=180)
    ax = plt.subplot(111)
    sns.boxplot(x='小时', y='订单时长_分钟', data=df, ax=ax)
    plt.ylim(0, 60)
    plt.title('各时段订单时长分布')
    plt.xlabel('小时')
    plt.ylabel('订单时长（分钟）')

    out_path = os.path.join(FIG_DIR, output_name)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    return out_path


def export_dynamic_heatmap_data(freq='15min', output_name='dynamic_pickup_heatmap_15min.json'):
    df = load_od_cache()
    data, labels = _build_heat_time_slices(df, '开始时间', '开始纬度', '开始经度', freq)
    output = [{'time': label, 'points': points} for label, points in zip(labels, data)]
    out_path = os.path.join(TABLE_DIR, output_name)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False)
    return out_path


def main():
    ensure_output_dirs()
    df_od = load_od_cache()

    static_vehicle_map = build_static_vehicle_heatmap()
    static_pickup_map = build_static_pickup_heatmap()
    dynamic_pickup_map = build_dynamic_pickup_heatmap(freq='15min')
    dynamic_vehicle_map = build_dynamic_vehicle_heatmap(freq='60min')

    cluster_df, cluster_path = run_pickup_dbscan()
    dbscan_map = build_dbscan_cluster_map(cluster_df)
    table_outputs = export_tables(df_od)

    hourly_plot = plot_hourly_stats(table_outputs['hourly_stats'][0])
    duration_plot = plot_duration_distribution(df_od)
    dynamic_json = export_dynamic_heatmap_data(freq='15min')

    print('阶段05分析完成，输出如下：')
    print(f'- 车辆位置静态热力图: {static_vehicle_map}')
    print(f'- 上车点静态热力图: {static_pickup_map}')
    print(f'- 上车点动态热力图: {dynamic_pickup_map}')
    print(f'- 车辆位置动态热力图: {dynamic_vehicle_map}')
    print(f'- DBSCAN聚类中心地图: {dbscan_map}')
    print(f'- DBSCAN聚类结果: {cluster_path}（{len(cluster_df)} 行）')
    print(f'- 小时订单/载客率图: {hourly_plot}')
    print(f'- 订单时长箱线图: {duration_plot}')
    print(f'- 动态热力图时间片JSON: {dynamic_json}')
    print(f'- 每小时统计表: {table_outputs['hourly_stats'][1]}')
    print(f'- 短中长途统计表: {table_outputs['distance_stats'][1]}')
    print(f'- 车辆全天统计表: {table_outputs['vehicle_stats'][1]}')


if __name__ == '__main__':
    main()