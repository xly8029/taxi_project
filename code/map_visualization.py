# -*- coding: utf-8 -*-
"""
04 地图基础与轨迹查询
======================
依据：04-地图基础与轨迹查询.html

本模块基于阶段03的缓存数据，实现：
    1. 车辆轨迹查询与展示（载客/空载状态区分）
    2. 按分钟查询所有车辆位置
    3. 上车点、下车点标注
    4. 单车动画轨迹（车辆图标沿轨迹移动，速度可视化）
    5. 为后续路网校正和ETA预留地图选点功能

技术栈：
    - folium：地图展示
    - JavaScript：动画控制
"""

import os
import json
import pandas as pd
import folium
from folium import plugins
from datetime import datetime, timedelta

# ========================= 配置区 =========================
VEHICLE_CACHE_DIR = os.path.join(os.path.dirname(__file__), "../data/cache/vehicle")
MINUTE_CACHE_DIR = os.path.join(os.path.dirname(__file__), "../data/cache/minute")
OD_CACHE_PATH = os.path.join(os.path.dirname(__file__), "../data/cache/od/od_cache.csv")
BOUNDARY_FILE = os.path.join(os.path.dirname(__file__), "../data/raw/深圳市.json")

MAP_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../maps")

# 深圳地图中心点和缩放级别
SHENZHEN_CENTER = [22.52847, 114.05454]
DEFAULT_ZOOM = 12

# 颜色配置
COLOR_OCCUPIED = '#FF4444'  # 载客状态：红色
COLOR_EMPTY = '#4444FF'  # 空载状态：蓝色
COLOR_PICKUP = '#00CC00'  # 上车点：绿色
COLOR_DROPOFF = '#FF8800'  # 下车点：橙色
TRAJECTORY_BASE_COLOR = '#1f2937'  # 轨迹主线颜色：深灰蓝
GLOW_OCCUPIED = '#ffd9e2'  # 载客浅色光晕边
MULTI_TRAJECTORY_COLORS = [
    '#d90429', '#005f73', '#7b2cbf', '#fb5607', '#1982c4',
    '#2b9348', '#ff006e', '#6c757d', '#8338ec', '#f77f00'
]


# ===============================================================================


def _load_boundary_geojson():
    """加载项目内的深圳行政边界 GeoJSON。"""
    if os.path.exists(BOUNDARY_FILE):
        with open(BOUNDARY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def add_boundary_layer(m):
    """给地图添加深圳行政边界图层。"""
    boundary_geojson = _load_boundary_geojson()
    if not boundary_geojson:
        return m

    folium.GeoJson(
        boundary_geojson,
        name='深圳行政边界',
        style_function=lambda _: {
            'color': '#1f6f8b',
            'weight': 2,
            'fillColor': '#74c0fc',
            'fillOpacity': 0.05,
        },
        tooltip=folium.GeoJsonTooltip(fields=['name'], aliases=['区域'])
    ).add_to(m)
    return m


# --------------------------- 1. 基础地图创建 ---------------------------
def create_base_map(center=SHENZHEN_CENTER, zoom=DEFAULT_ZOOM, title="深圳出租车GPS分析",
                    show_boundary=True, prefer_canvas=False):
    """创建基础地图"""
    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles='OpenStreetMap',
        control_scale=True,
        prefer_canvas=prefer_canvas,
    )
    if prefer_canvas:
        # 当前 folium 版本不会稳定透传 prefer_canvas，这里直接补到底层 Leaflet 配置。
        m.options['preferCanvas'] = True

    # 添加标题
    title_html = f'''
    <div style="position: fixed; 
                top: 10px; left: 50px; width: 400px; height: 50px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:18px; text-align: center; padding: 10px;">
        <b>{title}</b>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    if show_boundary:
        add_boundary_layer(m)

    return m


def fit_map_to_points(m, df, lat_col='lati', lon_col='long', padding=(20, 20), max_zoom=14):
    """根据点集自动调整地图视野，并限制最大放大级别。"""
    points = df[[lat_col, lon_col]].dropna()
    if points.empty:
        return m

    south_west = [points[lat_col].min(), points[lon_col].min()]
    north_east = [points[lat_col].max(), points[lon_col].max()]

    # 如果范围过小，手动扩一个最小包围盒，避免地图放得过近看起来跳动太剧烈
    lat_span = max(points[lat_col].max() - points[lat_col].min(), 0.012)
    lon_span = max(points[lon_col].max() - points[lon_col].min(), 0.012)
    center_lat = points[lat_col].mean()
    center_lon = points[lon_col].mean()
    south_west = [center_lat - lat_span / 2, center_lon - lon_span / 2]
    north_east = [center_lat + lat_span / 2, center_lon + lon_span / 2]

    m.fit_bounds([south_west, north_east], padding=padding, max_zoom=max_zoom)
    return m


# --------------------------- 2. 车辆轨迹查询与展示 ---------------------------
def load_vehicle_trajectory(vehicle_id, start_time=None, end_time=None):
    """
    从车辆缓存读取指定车辆的轨迹数据

    Args:
        vehicle_id: 车辆ID
        start_time: 开始时间（字符串或datetime）
        end_time: 结束时间（字符串或datetime）

    Returns:
        DataFrame: 轨迹数据
    """
    cache_file = os.path.join(VEHICLE_CACHE_DIR, f"{vehicle_id}.csv")

    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"车辆 {vehicle_id} 的缓存文件不存在")

    df = pd.read_csv(cache_file)
    df['time'] = pd.to_datetime(df['time'])

    # 时间过滤
    if start_time:
        start_time = pd.to_datetime(start_time)
        df = df[df['time'] >= start_time]

    if end_time:
        end_time = pd.to_datetime(end_time)
        df = df[df['time'] <= end_time]

    return df.reset_index(drop=True)


def plot_vehicle_trajectory(vehicle_id, start_time=None, end_time=None,
                            show_status=True, show_markers=True):
    """
    绘制车辆轨迹地图

    Args:
        vehicle_id: 车辆ID
        start_time: 开始时间
        end_time: 结束时间
        show_status: 是否区分载客/空载状态
        show_markers: 是否显示起终点标记

    Returns:
        folium.Map: 地图对象
    """
    df = load_vehicle_trajectory(vehicle_id, start_time, end_time)

    if df.empty:
        raise ValueError(f"车辆 {vehicle_id} 在指定时间段内无数据")

    # 创建地图，中心点为轨迹中心
    center_lat = df['lati'].mean()
    center_lon = df['long'].mean()
    m = create_base_map(
        center=[center_lat, center_lon],
        title=f"车辆 {vehicle_id} 轨迹 ({df['time'].min()} 至 {df['time'].max()})"
    )

    # 绘制轨迹
    if show_status:
        # 按载客状态分段绘制
        df['status_change'] = (df['status'] != df['status'].shift()).cumsum()

        for _, group in df.groupby('status_change'):
            if len(group) < 2:
                continue

            points = group[['lati', 'long']].values.tolist()
            status = group['status'].iloc[0]
            color = COLOR_OCCUPIED if status == 1 else COLOR_EMPTY
            label = "载客" if status == 1 else "空载"

            folium.PolyLine(
                points,
                color=color,
                weight=3,
                opacity=0.8,
                popup=f"{label}状态轨迹"
            ).add_to(m)
    else:
        # 统一颜色绘制
        points = df[['lati', 'long']].values.tolist()
        folium.PolyLine(
            points,
            color='blue',
            weight=3,
            opacity=0.8,
            popup="轨迹"
        ).add_to(m)

    # 标记起终点
    if show_markers:
        start_point = df.iloc[0]
        end_point = df.iloc[-1]

        folium.Marker(
            [start_point['lati'], start_point['long']],
            popup=f"起点<br>时间: {start_point['time']}<br>速度: {start_point['speed']}km/h",
            icon=folium.Icon(color='green', icon='play')
        ).add_to(m)

        folium.Marker(
            [end_point['lati'], end_point['long']],
            popup=f"终点<br>时间: {end_point['time']}<br>速度: {end_point['speed']}km/h",
            icon=folium.Icon(color='red', icon='stop')
        ).add_to(m)

    fit_map_to_points(m, df, padding=(30, 30), max_zoom=14)
    return m


def plot_multi_vehicle_trajectory(vehicle_ids, start_time=None, end_time=None, show_markers=True):
    """
    在同一张地图上叠加多辆车轨迹（区分载客/空载状态）。

    Args:
        vehicle_ids: 车辆ID列表
        start_time: 开始时间
        end_time: 结束时间
        show_markers: 是否显示每辆车的起终点标记
    """
    all_frames = []
    valid_vehicle_ids = []
    for vehicle_id in vehicle_ids:
        try:
            df = load_vehicle_trajectory(vehicle_id, start_time, end_time)
            if not df.empty:
                all_frames.append(df.assign(query_vehicle_id=vehicle_id))
                valid_vehicle_ids.append(vehicle_id)
        except FileNotFoundError:
            continue

    if not all_frames:
        raise ValueError('所有车辆在指定时间段内都没有可用轨迹数据')

    merged = pd.concat(all_frames, ignore_index=True)
    center_lat = merged['lati'].mean()
    center_lon = merged['long'].mean()
    m = create_base_map(
        center=[center_lat, center_lon],
        title=f"多车辆轨迹查询（共 {len(valid_vehicle_ids)} 辆）"
    )

    for index, vehicle_id in enumerate(valid_vehicle_ids):
        color = MULTI_TRAJECTORY_COLORS[index % len(MULTI_TRAJECTORY_COLORS)]
        vehicle_df = merged[merged['query_vehicle_id'] == vehicle_id].sort_values('time').reset_index(drop=True)
        full_points = vehicle_df[['lati', 'long']].values.tolist()

        # 先铺一层白色描边，避免和彩色底图道路混在一起
        folium.PolyLine(
            full_points,
            color='#ffffff',
            weight=7,
            opacity=0.95,
            popup=None
        ).add_to(m)

        # 按载客状态分段绘制，载客段更粗更实，空载段更细更透
        vehicle_df['status_change'] = (vehicle_df['status'] != vehicle_df['status'].shift()).cumsum()

        for _, group in vehicle_df.groupby('status_change'):
            if len(group) < 2:
                continue

            points = group[['lati', 'long']].values.tolist()
            status = group['status'].iloc[0]
            status_text = "载客" if status == 1 else "空载"

            if status == 1:
                weight = 5
                opacity = 0.98
            else:
                weight = 3
                opacity = 0.55

            folium.PolyLine(
                points,
                color=color,
                weight=weight,
                opacity=opacity,
                popup=f"车辆 {vehicle_id} - {status_text}"
            ).add_to(m)

        if show_markers and len(vehicle_df) > 0:
            start_point = vehicle_df.iloc[0]
            end_point = vehicle_df.iloc[-1]
            start_status_text = "载客" if start_point['status'] == 1 else "空载"
            end_status_text = "载客" if end_point['status'] == 1 else "空载"
            folium.CircleMarker(
                [start_point['lati'], start_point['long']],
                radius=5,
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.9,
                popup=f"车辆 {vehicle_id} 起点<br>时间: {start_point['time']}<br>状态: {start_status_text}<br>速度: {start_point['speed']}km/h"
            ).add_to(m)
            folium.CircleMarker(
                [end_point['lati'], end_point['long']],
                radius=5,
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.5,
                popup=f"车辆 {vehicle_id} 终点<br>时间: {end_point['time']}<br>状态: {end_status_text}<br>速度: {end_point['speed']}km/h"
            ).add_to(m)

    return m


# --------------------------- 3. 按分钟查询车辆位置 ---------------------------
def load_minute_snapshot(time_str):
    """
    从分钟缓存读取指定时刻所有车辆位置

    Args:
        time_str: 时间字符串，格式 "YYYY-MM-DD HH:MM"

    Returns:
        DataFrame: 车辆位置数据
    """
    dt = pd.to_datetime(time_str)
    date_str = dt.strftime('%Y-%m-%d')
    hhmm_str = dt.strftime('%H-%M')

    cache_file = os.path.join(MINUTE_CACHE_DIR, date_str, f"{hhmm_str}.csv")

    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"时间 {time_str} 的分钟缓存文件不存在")

    df = pd.read_csv(cache_file)
    df['time'] = pd.to_datetime(df['time'])

    return df


def _build_trajectory_lookup_js(map_js_name):
    """
    生成"点击车辆标记 -> 弹窗确认 -> 当前页面逐点动画播放该车后续轨迹"的JS脚本。

    流程：
        1. 标记的popup里有一个"查看后续轨迹"按钮，点击后调用 window.queryVehicleTrajectoryAfter
        2. 该函数先弹出确认框（confirm），确认后调用后端 /api/vehicle_trajectory 接口
           （这个接口在 map_web_app.py 里已经实现，专门返回"指定车辆从某时间点开始的后续轨迹"）
        3. 拿到轨迹点数据后，不是一次性画出整条线，而是像单车动画轨迹那样"一点一点播放出来"：
           - 一个车辆图标marker沿轨迹移动
           - 轨迹线按载客/空载状态分段：载客段白色描边、空载段黑色描边，内部主线统一红色
           - 右上角有一个"动画速度倍数"输入框（跟单车动画轨迹页面的输入框一样，可以自己输入
             任意数字，默认200），动画播放过程中随时可以改，下一帧就会用上新速度，不用重新查询
        4. 每次点新的车辆前，会先清掉上一次的动画轨迹和计时器，避免画面里叠了一堆线、
           或者上一次动画还在后台偷偷继续跑
        5. 注意：点击"查看后续轨迹"后用的是 fitBounds 把视野缩放到这条轨迹范围，
           原来分钟快照里的其他车辆标记并没有被删除，只是因为视野缩小/平移而看不见了
           （"消失"是视觉上的，不是真的被移除）。所以加一个"返回"按钮：
           记录第一次进入时的地图中心点和缩放级别，点击返回按钮时把视野恢复回去，
           同时清掉这条后续轨迹和动画，回到最初的分钟快照效果。
    """
    return f"""
    <div id="traj-speed-panel" style="display:none; position:fixed; top:70px; right:10px; z-index:9999;
                background:white; padding:10px 12px; border-radius:8px;
                box-shadow:0 2px 8px rgba(0,0,0,0.3); font-size:13px; font-family:sans-serif;">
      <div style="margin-bottom:6px; color:#333;">动画速度倍数</div>
      <input id="traj-speed-input" type="number" min="1" step="1" value="200" placeholder="200"
             style="width:90px;padding:5px 8px;border:1px solid #ccc;border-radius:6px;font-size:13px;">
      <button id="traj-back-btn"
              style="margin-top:10px;width:100%;padding:6px 0;border:0;border-radius:6px;
                     background:#475569;color:#fff;cursor:pointer;font-size:13px;">
        ← 返回原来的显示
      </button>
    </div>
    <script>
    (function() {{
        var mapRef = null;
        var lookupLayers = [];
        var lookupMarker = null;
        var lookupTimer = null;
        var lookupIndex = 0;
        var lookupCurrentStatus = null;
        var lookupCurrentCoords = [];
        var lookupOuterLine = null;
        var lookupInnerLine = null;
        var originalCenter = null;
        var originalZoom = null;
        var savedMarkers = [];  // 保存分钟快照的全部车辆标记

        window.trajectoryAnimSpeed = 200;  // 默认倍速，和单车动画轨迹的默认值保持一致

        function waitForMap(callback) {{
            if (typeof {map_js_name} !== 'undefined' && {map_js_name}) {{
                mapRef = {map_js_name};
                if (originalCenter === null) {{
                    // 只在第一次拿到地图实例时记录初始视野，后续多次查询不会被覆盖
                    originalCenter = mapRef.getCenter();
                    originalZoom = mapRef.getZoom();
                    // 保存所有车辆标记（CircleMarker），用于轨迹查看时隐藏
                    mapRef.eachLayer(function(layer) {{
                        if (layer instanceof L.CircleMarker) {{
                            savedMarkers.push(layer);
                        }}
                    }});
                }}
                callback();
            }} else {{
                setTimeout(function() {{ waitForMap(callback); }}, 100);
            }}
        }}

        // 速度输入框：可以直接输入任意数字，跟单车动画轨迹的"动画速度倍数"输入框一样，
        // 不再限制成几个固定档位。边输入边生效，下一帧动画就会用上新数值，不用重新点查询。
        document.addEventListener('input', function(e) {{
            if (e.target && e.target.id === 'traj-speed-input') {{
                var v = parseFloat(e.target.value);
                if (!isNaN(v) && v > 0) {{
                    window.trajectoryAnimSpeed = v;
                }}
            }}
        }});

        // 返回按钮：恢复全部车辆标记、地图视野，并清掉当前画的后续轨迹/动画
        document.addEventListener('click', function(e) {{
            if (e.target && e.target.id === 'traj-back-btn') {{
                waitForMap(function() {{
                    clearLookupLayers();
                    // 恢复所有被隐藏的车辆标记
                    savedMarkers.forEach(function(m) {{ m.addTo(mapRef); }});
                    document.getElementById('traj-speed-panel').style.display = 'none';
                    if (originalCenter) {{
                        mapRef.setView(originalCenter, originalZoom);
                    }}
                }});
            }}
        }});

        function clearLookupLayers() {{
            lookupLayers.forEach(function(layer) {{ mapRef.removeLayer(layer); }});
            lookupLayers = [];
            if (lookupMarker) {{ mapRef.removeLayer(lookupMarker); lookupMarker = null; }}
            if (lookupTimer) {{ clearTimeout(lookupTimer); lookupTimer = null; }}
            lookupIndex = 0;
            lookupCurrentStatus = null;
            lookupCurrentCoords = [];
        }}

        function startLookupSegment(status, coord) {{
            lookupCurrentStatus = status;
            lookupCurrentCoords = [coord];
            var outlineColor = status === 1 ? '#ffffff' : '#000000';
            lookupOuterLine = L.polyline([], {{color: outlineColor, weight: 7, opacity: 0.9}}).addTo(mapRef);
            lookupInnerLine = L.polyline([], {{color: '#d90429', weight: 4, opacity: 0.95}}).addTo(mapRef);
            lookupLayers.push(lookupOuterLine, lookupInnerLine);
            updateLookupSegment();
        }}

        function updateLookupSegment() {{
            lookupOuterLine.setLatLngs(lookupCurrentCoords);
            lookupInnerLine.setLatLngs(lookupCurrentCoords);
        }}

        function animateLookupStep(points, vehicleId) {{
            if (lookupIndex >= points.length) {{
                var last = points[points.length - 1];
                var endMarker = L.circleMarker([last.lat, last.lng], {{
                    radius: 6, color: '#fb5607', fillColor: '#fb5607', fillOpacity: 1
                }}).addTo(mapRef).bindPopup('后续轨迹终点（车辆 ' + vehicleId + '）');
                lookupLayers.push(endMarker);
                return;
            }}

            var point = points[lookupIndex];
            var nextPoint = points[lookupIndex + 1];
            var coord = [point.lat, point.lng];

            if (!lookupMarker) {{
                lookupMarker = L.marker(coord, {{
                    icon: L.icon({{
                        iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-red.png',
                        iconSize: [25, 41], iconAnchor: [12, 41]
                    }})
                }}).addTo(mapRef);
                var startMarker = L.circleMarker(coord, {{
                    radius: 6, color: '#2b9348', fillColor: '#2b9348', fillOpacity: 1
                }}).addTo(mapRef).bindPopup('后续轨迹起点（当前时刻）');
                lookupLayers.push(startMarker);
            }} else {{
                lookupMarker.setLatLng(coord);
            }}
            lookupMarker.bindPopup(
                '车辆ID: ' + vehicleId + '<br>速度: ' + point.speed + ' km/h<br>状态: ' +
                (point.status === 1 ? '载客' : '空载') + '<br>时间: ' + point.time
            );

            if (lookupCurrentStatus === null || point.status !== lookupCurrentStatus) {{
                startLookupSegment(point.status, coord);
            }} else {{
                lookupCurrentCoords.push(coord);
                updateLookupSegment();
            }}

            if (lookupIndex % 5 === 0) {{
                mapRef.panTo(coord, {{animate: true, duration: 0.4}});
            }}

            var speedFactor = window.trajectoryAnimSpeed || 200;
            var delay = 1000 / speedFactor;
            if (nextPoint) {{
                var timeDiff = new Date(nextPoint.time) - new Date(point.time);
                delay = Math.max(Math.min(timeDiff / speedFactor, 500), 16);
            }}

            lookupIndex++;
            lookupTimer = setTimeout(function() {{ animateLookupStep(points, vehicleId); }}, delay);
        }}

        window.queryVehicleTrajectoryAfter = function(vehicleId, afterTimeStr) {{
            if (!window.confirm('是否查看车辆 ' + vehicleId + ' 在此刻之后的轨迹？')) {{
                return;
            }}
            fetch('/api/vehicle_trajectory?vehicle_id=' + encodeURIComponent(vehicleId)
                  + '&start_time=' + encodeURIComponent(afterTimeStr))
                .then(function(resp) {{ return resp.json(); }})
                .then(function(data) {{
                    if (data.error) {{
                        window.alert('查询失败：' + data.error);
                        return;
                    }}
                    if (!data.points || data.points.length === 0) {{
                        window.alert('未查到该车后续轨迹数据');
                        return;
                    }}
                    waitForMap(function() {{
                        clearLookupLayers();
                        // 隐藏全部车辆标记，只保留当前查看的车辆
                        savedMarkers.forEach(function(m) {{ mapRef.removeLayer(m); }});
                        document.getElementById('traj-speed-panel').style.display = 'block';

                        var coords = data.points.map(function(p) {{ return [p.lat, p.lng]; }});
                        mapRef.fitBounds(L.latLngBounds(coords), {{padding: [30, 30]}});

                        animateLookupStep(data.points, vehicleId);
                    }});
                }})
                .catch(function(err) {{
                    window.alert('请求出错：' + err);
                }});
        }};
    }})();
    </script>
    """


def plot_minute_snapshot(time_str, vehicle_ids=None, max_vehicles=500,
                         id_min=None, id_max=None, status_filter=None,
                         enable_trajectory_lookup=True):
    """
    绘制某一时刻的车辆位置分布

    Args:
        time_str: 时间字符串
        vehicle_ids: 指定车辆ID列表（None表示所有车辆）
        max_vehicles: 最大显示车辆数
        id_min: 车辆ID下限（含），None表示不限
        id_max: 车辆ID上限（含），None表示不限
        status_filter: None=全部, 1=仅载客, 0=仅空载
        enable_trajectory_lookup: 是否开启点击查看后续轨迹功能

    Returns:
        folium.Map: 地图对象
    """
    df = load_minute_snapshot(time_str)

    # 筛选指定 ID 列表
    if vehicle_ids:
        df = df[df['id'].isin(vehicle_ids)]

    # 筛选 ID 范围
    if id_min is not None:
        df = df[df['id'] >= int(id_min)]
    if id_max is not None:
        df = df[df['id'] <= int(id_max)]

    # 筛选载客状态
    if status_filter is not None:
        df = df[df['status'] == int(status_filter)]

    # 限制显示数量
    if len(df) > max_vehicles:
        df = df.sample(n=max_vehicles, random_state=42)

    filter_desc = []
    if id_min is not None: filter_desc.append(f"ID≥{id_min}")
    if id_max is not None: filter_desc.append(f"ID≤{id_max}")
    if status_filter == 1: filter_desc.append("载客")
    elif status_filter == 0: filter_desc.append("空载")
    filter_str = f"  [{', '.join(filter_desc)}]" if filter_desc else ""
    m = create_base_map(title=f"车辆位置快照 ({time_str}){filter_str}  共 {len(df)} 辆")
    map_js_name = m.get_name()

    # 添加车辆位置标记
    for _, row in df.iterrows():
        color = 'red' if row['status'] == 1 else 'blue'
        status_text = "载客" if row['status'] == 1 else "空载"
        vehicle_id = int(row['id'])
        time_str_iso = pd.to_datetime(row['time']).strftime('%Y-%m-%d %H:%M:%S')

        if enable_trajectory_lookup:
            # popup里加一个按钮，点击后触发"弹窗确认+当前页面画后续轨迹"的JS逻辑
            popup_html = f"""
            <div style="font-size:13px;line-height:1.7;">
              车辆ID: {vehicle_id}<br>
              状态: {status_text}<br>
              速度: {row['speed']}km/h<br>
              <button onclick="window.queryVehicleTrajectoryAfter({vehicle_id}, '{time_str_iso}')"
                      style="margin-top:6px;padding:4px 12px;border:0;border-radius:6px;
                             background:#0f6cbd;color:#fff;cursor:pointer;">
                查看后续轨迹
              </button>
            </div>
            """
            popup = folium.Popup(popup_html, max_width=220)
        else:
            popup = f"车辆ID: {vehicle_id}<br>状态: {status_text}<br>速度: {row['speed']}km/h"

        folium.CircleMarker(
            location=[row['lati'], row['long']],
            radius=3,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7,
            popup=popup
        ).add_to(m)

    if enable_trajectory_lookup:
        m.get_root().html.add_child(folium.Element(_build_trajectory_lookup_js(map_js_name)))

    return m


# --------------------------- 4. OD点标注 ---------------------------
def plot_od_points(start_time=None, end_time=None, max_points=200):
    """
    绘制上车点和下车点分布

    Args:
        start_time: 开始时间
        end_time: 结束时间
        max_points: 最大显示点数

    Returns:
        folium.Map: 地图对象
    """
    df = pd.read_csv(OD_CACHE_PATH)
    df['开始时间'] = pd.to_datetime(df['开始时间'])
    df['结束时间'] = pd.to_datetime(df['结束时间'])

    # 时间过滤
    if start_time:
        start_time = pd.to_datetime(start_time)
        df = df[df['开始时间'] >= start_time]

    if end_time:
        end_time = pd.to_datetime(end_time)
        df = df[df['结束时间'] <= end_time]

    # 限制显示数量
    if len(df) > max_points:
        df = df.sample(n=max_points, random_state=42)

    m = create_base_map(title=f"上下车点分布 ({len(df)}个订单)")

    # 添加上车点
    pickup_layer = folium.FeatureGroup(name='上车点')
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row['开始纬度'], row['开始经度']],
            radius=4,
            color=COLOR_PICKUP,
            fill=True,
            fillColor=COLOR_PICKUP,
            fillOpacity=0.6,
            popup=f"上车<br>时间: {row['开始时间']}<br>车辆: {int(row['车辆id'])}"
        ).add_to(pickup_layer)

    # 添加下车点
    dropoff_layer = folium.FeatureGroup(name='下车点')
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row['结束纬度'], row['结束经度']],
            radius=4,
            color=COLOR_DROPOFF,
            fill=True,
            fillColor=COLOR_DROPOFF,
            fillOpacity=0.6,
            popup=f"下车<br>时间: {row['结束时间']}<br>车辆: {int(row['车辆id'])}"
        ).add_to(dropoff_layer)

    pickup_layer.add_to(m)
    dropoff_layer.add_to(m)
    folium.LayerControl().add_to(m)

    return m


# --------------------------- 5. 单车动画轨迹 ---------------------------
def create_animated_trajectory(vehicle_ids, start_time=None, end_time=None, speed_factor=100):
    """
    创建多车动画轨迹（支持单车或多车，各车图标同步沿轨迹移动）

    Args:
        vehicle_ids: 车辆ID（int）或车辆ID列表（list of int）
        start_time: 开始时间
        end_time: 结束时间
        speed_factor: 动画速度倍数（越大越快）

    Returns:
        folium.Map: 包含动画的地图对象
    """
    import json as _json

    if isinstance(vehicle_ids, int):
        vehicle_ids = [vehicle_ids]

    if len(vehicle_ids) > 10:
        raise ValueError(f'动画模式最多支持 10 辆车，当前输入了 {len(vehicle_ids)} 辆，请减少车辆数量')

    # 加载所有车辆数据
    all_vehicles = []
    for vid in vehicle_ids:
        try:
            df = load_vehicle_trajectory(vid, start_time, end_time)
            if df.empty or len(df) < 2:
                continue
            pts = [
                {
                    'lat': float(row['lati']),
                    'lon': float(row['long']),
                    'time': row['time'].isoformat(),
                    'speed': float(row['speed']),
                    'status': int(row['status']),
                }
                for _, row in df.iterrows()
            ]
            all_vehicles.append({'id': vid, 'points': pts})
        except FileNotFoundError:
            continue

    if not all_vehicles:
        raise ValueError('所有车辆在指定时间段内数据不足')

    # 地图中心取第一辆车起点
    first_pt = all_vehicles[0]['points'][0]
    m = create_base_map(
        center=[first_pt['lat'], first_pt['lon']],
        title=f"动画轨迹（{'、'.join(str(v['id']) for v in all_vehicles)}）"
    )
    m.options['zoom'] = 14
    map_js_name = m.get_name()

    colors_js = _json.dumps(MULTI_TRAJECTORY_COLORS)
    vehicles_js = _json.dumps(all_vehicles, ensure_ascii=False)

    animation_js = f"""
<script>
(function() {{
  var COLORS = {colors_js};
  var allVehicles = {vehicles_js};
  var speedFactor = {speed_factor};
  var mapInstance = null;
  var isPlaying = true;
  var animTimer = null;

  // 每辆车的状态
  var vehicles = allVehicles.map(function(v, i) {{
    return {{
      id: v.id,
      points: v.points,
      color: COLORS[i % COLORS.length],
      index: 0,
      marker: null,
      outerLine: null,
      innerLine: null,
      allLines: [],        // 记录该车所有已绘制的线段，供重置时清除
      currentCoords: [],
      currentStatus: null
    }};
  }});

  function waitForMap(cb) {{
    if (typeof {map_js_name} !== 'undefined' && {map_js_name}) cb({map_js_name});
    else setTimeout(function() {{ waitForMap(cb); }}, 100);
  }}

  function initVehicles(map) {{
    mapInstance = map;
    vehicles.forEach(function(v) {{
      var fp = v.points[0];
      var carIcon = L.divIcon({{
        html: '<div style="width:14px;height:14px;border-radius:50%;background:' + v.color +
              ';border:2px solid #fff;box-shadow:0 0 4px rgba(0,0,0,0.5)"></div>',
        iconSize: [14, 14], iconAnchor: [7, 7], className: ''
      }});
      v.marker = L.marker([fp.lat, fp.lon], {{ icon: carIcon }}).addTo(map);
      v.marker.bindTooltip('车辆 ' + v.id, {{ permanent: false, direction: 'top' }});
    }});

    // 控制面板
    var panel = L.control({{ position: 'bottomleft' }});
    panel.onAdd = function() {{
      var div = L.DomUtil.create('div');
      div.style.cssText = 'background:rgba(15,23,42,0.88);backdrop-filter:blur(8px);border:1px solid rgba(255,255,255,0.12);border-radius:12px;padding:10px 14px;color:#e2e8f0;font-family:Microsoft YaHei,sans-serif;min-width:200px';
      div.innerHTML =
        '<div style="font-size:13px;font-weight:700;margin-bottom:8px;color:#f8fafc">▐ 动画轨迹控制</div>' +
        '<div id="anim-time" style="font-size:12px;color:#94a3b8;margin-bottom:8px">准备中…</div>' +
        '<div style="display:flex;gap:8px">' +
          '<button id="anim-play" onclick="window._animToggle()" style="flex:1;background:#0ea5e9;color:#fff;border:none;border-radius:7px;padding:6px 10px;cursor:pointer;font-size:12px">⏸ 暂停</button>' +
          '<button onclick="window._animReset()" style="background:rgba(255,255,255,0.1);color:#cbd5e1;border:none;border-radius:7px;padding:6px 10px;cursor:pointer;font-size:12px">↺ 重置</button>' +
        '</div>' +
        '<div style="margin-top:10px">' + vehicles.map(function(v) {{
          return '<div style="display:flex;align-items:center;gap:6px;margin-top:4px">' +
            '<div style="width:10px;height:10px;border-radius:50%;background:' + v.color + '"></div>' +
            '<span style="font-size:12px;color:#94a3b8">车辆 ' + v.id + '</span></div>';
        }}).join('') + '</div>';
      L.DomEvent.disableClickPropagation(div);
      return div;
    }};
    panel.addTo(map);

    startAnimation();
  }}

  function startNewSegment(v, status, coord) {{
    // 旧段保留在地图上，只新建当前段的两层线
    v.currentStatus = status;
    v.currentCoords = [coord];

    // 空载：白边 + 彩色主线虚线
    // 载客：黑边 + 彩色主线实线
    var borderColor = status === 1 ? '#111111' : '#ffffff';
    var borderW    = status === 1 ? 6.5 : 5.5;
    var innerW     = status === 1 ? 3.5 : 2.5;
    var innerOpacity = status === 1 ? 0.98 : 0.70;
    var dash       = status === 1 ? null : '7,5';

    v.outerLine = L.polyline([coord], {{
      color: borderColor, weight: borderW, opacity: 0.88
    }}).addTo(mapInstance);
    v.innerLine = L.polyline([coord], {{
      color: v.color, weight: innerW, opacity: innerOpacity, dashArray: dash
    }}).addTo(mapInstance);

    // 记录到 allLines 以便重置时精确清除
    v.allLines.push(v.outerLine, v.innerLine);
  }}

  function stepVehicle(v) {{
    if (v.index >= v.points.length) return false;
    var pt = v.points[v.index];
    var coord = [pt.lat, pt.lon];

    v.marker.setLatLng(coord);
    v.marker.setTooltipContent(
      '车辆 ' + v.id + '<br>' + (pt.status === 1 ? '载客' : '空载') +
      ' ' + pt.speed + ' km/h<br>' + pt.time.replace('T', ' ').slice(0,19)
    );

    if (v.currentStatus === null || pt.status !== v.currentStatus) {{
      startNewSegment(v, pt.status, coord);
    }} else {{
      v.currentCoords.push(coord);
      if (v.outerLine) v.outerLine.setLatLngs(v.currentCoords);
      if (v.innerLine) v.innerLine.setLatLngs(v.currentCoords);
    }}

    v.index++;
    return v.index < v.points.length;
  }}

  // ── 全局时间轴 ─────────────────────────────────────────────────────────
  // 收集所有车辆的时间戳，去重排序，建立统一时间轴
  var allTimes = [];
  vehicles.forEach(function(v) {{
    v.points.forEach(function(p) {{ allTimes.push(new Date(p.time).getTime()); }});
  }});
  allTimes = allTimes.filter(function(v,i,a){{ return a.indexOf(v)===i; }}).sort(function(a,b){{return a-b;}});
  var globalIndex = 0;

  function startAnimation() {{
    if (animTimer) clearTimeout(animTimer);

    function tick() {{
      if (!isPlaying) return;
      if (globalIndex >= allTimes.length) return;

      var globalNow = allTimes[globalIndex];

      // 各车消费所有 time <= globalNow 的数据点，保持各车时间和全局时间同步
      vehicles.forEach(function(v) {{
        while (v.index < v.points.length &&
               new Date(v.points[v.index].time).getTime() <= globalNow) {{
          stepVehicle(v);
        }}
      }});

      // 控制框统一显示全局时间
      var el = document.getElementById('anim-time');
      if (el) {{
        var d = new Date(globalNow);
        // 用本地时间显示，避免 toISOString() 转 UTC 导致时差偏移
        var pad = function(n){{ return n < 10 ? '0'+n : ''+n; }};
        el.textContent = d.getFullYear()+'-'+pad(d.getMonth()+1)+'-'+pad(d.getDate())+
          ' '+pad(d.getHours())+':'+pad(d.getMinutes())+':'+pad(d.getSeconds());
      }}

      // 地图视野跟随第一辆车
      var lead = vehicles[0];
      if (lead.index > 0 && globalIndex % 4 === 0) {{
        var lp = lead.points[Math.min(lead.index, lead.points.length - 1)];
        mapInstance.panTo([lp.lat, lp.lon], {{ animate: true, duration: 0.4 }});
      }}

      globalIndex++;

      var delay = 50;
      if (globalIndex < allTimes.length) {{
        delay = Math.min(500, Math.max(16,
          (allTimes[globalIndex] - globalNow) / speedFactor));
      }}
      animTimer = setTimeout(tick, delay);
    }}

    animTimer = setTimeout(tick, 800);
  }}

  window._animToggle = function() {{
    isPlaying = !isPlaying;
    var btn = document.getElementById('anim-play');
    if (isPlaying) {{ btn.textContent = '⏸ 暂停'; startAnimation(); }}
    else {{ btn.textContent = '▶ 播放'; if (animTimer) clearTimeout(animTimer); }}
  }};

  window._animReset = function() {{
    if (animTimer) clearTimeout(animTimer);
    // 精确移除所有已绘制的轨迹线，不影响底图和标记
    vehicles.forEach(function(v) {{
      v.allLines.forEach(function(l) {{ mapInstance.removeLayer(l); }});
      v.index = 0; v.currentStatus = null; v.currentCoords = [];
      v.allLines = []; v.outerLine = null; v.innerLine = null;
      if (v.points.length) v.marker.setLatLng([v.points[0].lat, v.points[0].lon]);
    }});
    globalIndex = 0;
    isPlaying = true;
    document.getElementById('anim-play').textContent = '⏸ 暂停';
    startAnimation();
  }};

  waitForMap(initVehicles);
}})();
</script>
"""
    m.get_root().html.add_child(folium.Element(animation_js))
    return m

# --------------------------- 6. 路网校正轨迹展示 ---------------------------
COLOR_ORIGINAL = '#f97316'  # 原始GPS轨迹：亮橙色（原灰色 #6b7280）
COLOR_CORRECTED = '#059669'
COLOR_DEBUG = '#f59e0b'
COLOR_DIRECTION_WARNING = '#dc2626'


def plot_corrected_trajectory(vehicle_ids, start_time=None, end_time=None,
                              enable_correction=True, use_undirected=False):
    """
    在地图上同时展示原始轨迹与路网校正后轨迹，支持图层切换。

    Args:
        vehicle_ids: 单个 int 或车辆 ID 列表（建议 1-3 辆）
        start_time: 开始时间
        end_time: 结束时间
        enable_correction: 是否执行路网校正
        use_undirected: 是否全程使用无向图做最短路径

    Returns:
        folium.Map
    """
    from road_correction import correct_vehicle_trajectory, load_road_network

    if isinstance(vehicle_ids, int):
        vehicle_ids = [vehicle_ids]

    all_original = []
    all_corrected = []
    all_debug_segments = []
    stats_lines = []

    if enable_correction:
        load_road_network()

    for vehicle_id in vehicle_ids:
        if enable_correction:
            result = correct_vehicle_trajectory(
                vehicle_id, start_time, end_time, use_undirected=use_undirected
            )
            orig = result["original_coords"]
            corr = result["corrected_coords"]
            debug_segments = result.get("debug_segments", [])
            s = result["stats"]
            rate = s["success_segments"] / max(s["total_segments"], 1) * 100
            degraded = s.get("degraded_segments", 0)
            stats_lines.append(
                f"车辆 {vehicle_id}: {len(result['df'])} 点, "
                f"路段成功 {s['success_segments']}/{s['total_segments']} ({rate:.0f}%), "
                f"可疑段 {len(debug_segments)}, 低置信降级片段 {degraded}"
            )
        else:
            df = load_vehicle_trajectory(vehicle_id, start_time, end_time)
            if df.empty:
                continue
            orig = df[["lati", "long"]].values.tolist()
            corr = orig
            debug_segments = []

        all_original.append((vehicle_id, orig))
        all_corrected.append((vehicle_id, corr))
        all_debug_segments.append((vehicle_id, debug_segments))

    if not all_original:
        raise ValueError("所有车辆在指定时间段内都没有可用轨迹数据")

    flat_orig = [pt for _, coords in all_original for pt in coords]
    center_lat = sum(p[0] for p in flat_orig) / len(flat_orig)
    center_lon = sum(p[1] for p in flat_orig) / len(flat_orig)

    title = "路网校正轨迹对比" if enable_correction else "原始轨迹（未启用校正）"
    m = create_base_map(
        center=[center_lat, center_lon],
        title=title,
        show_boundary=False,
        prefer_canvas=True,
    )

    original_group = folium.FeatureGroup(name="原始轨迹", show=True)
    corrected_group = folium.FeatureGroup(name="校正后轨迹", show=enable_correction)
    debug_group = folium.FeatureGroup(name="可疑/回退路段", show=enable_correction)

    for index, (vehicle_id, orig) in enumerate(all_original):
        color = MULTI_TRAJECTORY_COLORS[index % len(MULTI_TRAJECTORY_COLORS)]
        folium.PolyLine(
            orig,
            color=COLOR_ORIGINAL,
            weight=4,
            opacity=0.75,
            dash_array="8 6",
            popup=f"车辆 {vehicle_id} 原始轨迹",
        ).add_to(original_group)

        if enable_correction:
            corr = all_corrected[index][1]
            folium.PolyLine(
                corr,
                color=COLOR_CORRECTED,
                weight=5,
                opacity=0.9,
                popup=f"车辆 {vehicle_id} 校正轨迹",
            ).add_to(corrected_group)

            folium.Marker(
                orig[0],
                popup=f"车辆 {vehicle_id} 起点",
                icon=folium.Icon(color='green', icon='play'),
            ).add_to(corrected_group)

            for segment in all_debug_segments[index][1]:
                seg_type = segment.get("type", "debug")
                seg_color = COLOR_DIRECTION_WARNING if seg_type in {"direction_warning", "segment_degraded"} else COLOR_DEBUG
                seg_dash = "4 8" if seg_type in {"offroad", "jump", "route_failed", "detour", "segment_degraded"} else "2 8"
                folium.PolyLine(
                    segment.get("coords", []),
                    color=seg_color,
                    weight=6,
                    opacity=0.95,
                    dash_array=seg_dash,
                    popup=f"车辆 {vehicle_id}: {segment.get('message', '可疑路段')}",
                ).add_to(debug_group)

    original_group.add_to(m)
    if enable_correction:
        corrected_group.add_to(m)
        debug_group.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    stats_html = "<br>".join(stats_lines) if stats_lines else "未启用路网校正"
    toggle_html = f"""
    <div style="position:fixed; bottom:30px; left:50px; z-index:9999;
                background:white; border:2px solid #059669; border-radius:10px;
                padding:12px 16px; font-size:13px; max-width:360px; line-height:1.6;">
        <div style="font-weight:bold; margin-bottom:6px;">路网校正</div>
        <label style="cursor:pointer;">
            <input type="checkbox" id="toggle-corrected" {'checked' if enable_correction else ''}
                   onchange="toggleCorrectedLayer(this.checked)">
            显示校正后轨迹（绿色实线）
        </label>
        <div style="margin-top:6px; color:#92400e; font-size:12px;">
            橙色/红色虚线 = 回退段、可疑方向段、无向图兜底段
        </div>
        <div style="margin-top:8px; color:#374151; font-size:12px;">{stats_html}</div>
        <div style="margin-top:6px; color:#6b7280; font-size:11px;">
            橙色虚线 = 原始 GPS &nbsp;|&nbsp; 绿色实线 = 路网校正
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(toggle_html))

    map_js_name = m.get_name()
    toggle_js = f"""
    <script>
    var correctedLayerGroup = null;
    function findCorrectedLayer(map) {{
        map.eachLayer(function(layer) {{
            if (layer instanceof L.FeatureGroup && layer.getLayers().length > 0) {{
                var first = layer.getLayers()[0];
                if (first instanceof L.Polyline && first.options.color === '{COLOR_CORRECTED}') {{
                    correctedLayerGroup = layer;
                }}
            }}
        }});
    }}
    function toggleCorrectedLayer(show) {{
        if (!correctedLayerGroup) return;
        if (show) {{ mapInstance.addLayer(correctedLayerGroup); }}
        else {{ mapInstance.removeLayer(correctedLayerGroup); }}
    }}
    var mapInstance = null;
    function initToggle() {{
        if (typeof {map_js_name} !== 'undefined') {{
            mapInstance = {map_js_name};
            findCorrectedLayer(mapInstance);
        }} else {{
            setTimeout(initToggle, 100);
        }}
    }}
    initToggle();
    </script>
    """
    m.get_root().html.add_child(folium.Element(toggle_js))

    sample_df = pd.DataFrame(flat_orig, columns=['lati', 'long'])
    fit_map_to_points(m, sample_df, padding=(40, 40), max_zoom=15)
    return m


# --------------------------- 7. 地图选点功能 ---------------------------
def create_point_picker_map():
    """
    创建地图选点工具：点击地图获取并显示经纬度。

    Returns:
        folium.Map: 地图对象
    """
    m = create_base_map(title="地图选点工具（点击地图获取坐标）")
    map_js_name = m.get_name()

    coord_panel = """
    <div id="coord-panel" style="position:fixed; top:70px; right:20px; z-index:9999;
                background:white; border:2px solid #0f6cbd; border-radius:10px;
                padding:14px 18px; font-size:14px; min-width:260px; line-height:1.8;">
        <div style="font-weight:bold; margin-bottom:8px;">选点坐标</div>
        <div>纬度 (lat): <span id="pick-lat" style="color:#0f6cbd;">—</span></div>
        <div>经度 (lng): <span id="pick-lng" style="color:#0f6cbd;">—</span></div>
        <div style="margin-top:6px; font-size:12px; color:#637487;">点击地图任意位置更新坐标</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(coord_panel))

    click_js = f"""
    <script>
    var mapInstance = null;
    var selectedPoints = [];
    var markers = [];
    var lineLayer = null;

    function bindPointPicker() {{
        mapInstance.on('click', function(e) {{
            var lat = e.latlng.lat.toFixed(6);
            var lon = e.latlng.lng.toFixed(6);

            document.getElementById('pick-lat').textContent = lat;
            document.getElementById('pick-lng').textContent = lon;

            var marker = L.marker([lat, lon]).addTo(mapInstance);
            marker.bindPopup('纬度: ' + lat + '<br>经度: ' + lon).openPopup();

            selectedPoints.push([lat, lon]);
            markers.push(marker);
            console.log('选中点 ' + selectedPoints.length + ': lat=' + lat + ', lng=' + lon);

            if (selectedPoints.length === 2) {{
                if (lineLayer) {{ mapInstance.removeLayer(lineLayer); }}
                lineLayer = L.polyline(selectedPoints, {{color: 'red', dashArray: '5, 10'}}).addTo(mapInstance);
                alert('起点: lat=' + selectedPoints[0][0] + ', lng=' + selectedPoints[0][1] +
                      '\\n终点: lat=' + selectedPoints[1][0] + ', lng=' + selectedPoints[1][1]);

                selectedPoints = [];
                markers.forEach(function(m) {{ mapInstance.removeLayer(m); }});
                markers = [];
            }}
        }});
    }}

    function waitForMap() {{
        if (typeof {map_js_name} !== 'undefined' && {map_js_name}) {{
            mapInstance = {map_js_name};
            bindPointPicker();
        }} else {{
            setTimeout(waitForMap, 100);
        }}
    }}
    waitForMap();
    </script>
    """

    m.get_root().html.add_child(folium.Element(click_js))
    return m


# --------------------------- 8. 主函数与示例 ---------------------------
def main():
    """生成示例地图"""
    os.makedirs(MAP_OUTPUT_DIR, exist_ok=True)

    print("开始生成地图示例...")

    # 示例1: 单车轨迹（区分载客状态）
    print("1. 生成单车轨迹地图...")
    try:
        m1 = plot_vehicle_trajectory(
            vehicle_id=22223,
            start_time='2013-10-22 08:00:00',
            end_time='2013-10-22 12:00:00',
            show_status=True
        )
        m1.save(os.path.join(MAP_OUTPUT_DIR, '01_vehicle_trajectory.html'))
        print("   [OK] 已保存: maps/01_vehicle_trajectory.html")
    except Exception as e:
        print(f"   [ERROR] 失败: {e}")

    # 示例2: 某时刻车辆位置快照
    print("2. 生成车辆位置快照...")
    try:
        m2 = plot_minute_snapshot('2013-10-22 08:00', max_vehicles=500)
        m2.save(os.path.join(MAP_OUTPUT_DIR, '02_minute_snapshot.html'))
        print("   [OK] 已保存: maps/02_minute_snapshot.html")
    except Exception as e:
        print(f"   [ERROR] 失败: {e}")

    # 示例3: 上下车点分布
    print("3. 生成上下车点分布地图...")
    try:
        m3 = plot_od_points(
            start_time='2013-10-22 08:00:00',
            end_time='2013-10-22 09:00:00',
            max_points=300
        )
        m3.save(os.path.join(MAP_OUTPUT_DIR, '03_od_points.html'))
        print("   [OK] 已保存: maps/03_od_points.html")
    except Exception as e:
        print(f"   [ERROR] 失败: {e}")

    # 示例4: 动画轨迹
    print("4. 生成动画轨迹地图...")
    try:
        m4 = create_animated_trajectory(
            vehicle_id=22223,
            start_time='2013-10-22 08:00:00',
            end_time='2013-10-22 11:30:00',
            speed_factor=200
        )
        m4.save(os.path.join(MAP_OUTPUT_DIR, '04_animated_trajectory.html'))
        print("   [OK] 已保存: maps/04_animated_trajectory.html")
    except Exception as e:
        print(f"   [ERROR] 失败: {e}")

    # 示例5: 地图选点工具
    print("5. 生成地图选点工具...")
    try:
        m5 = create_point_picker_map()
        m5.save(os.path.join(MAP_OUTPUT_DIR, '05_point_picker.html'))
        print("   [OK] 已保存: maps/05_point_picker.html")
    except Exception as e:
        print(f"   [ERROR] 失败: {e}")

    # 示例6: 路网校正轨迹（1-3 辆样例车，短时间窗口）
    print("6. 生成路网校正对比地图...")
    try:
        m6 = plot_corrected_trajectory(
            vehicle_ids=[22223, 22224],
            start_time='2013-10-22 00:00:00',
            end_time='2013-10-22 00:20:00',
            enable_correction=True,
        )
        m6.save(os.path.join(MAP_OUTPUT_DIR, '06_road_corrected_trajectory.html'))
        print("   [OK] 已保存: maps/06_road_corrected_trajectory.html")
    except Exception as e:
        print(f"   [ERROR] 失败: {e}")

    print("\n[OK] 地图生成完成！请在浏览器中打开maps目录下的HTML文件查看效果。")


if __name__ == "__main__":
    main()