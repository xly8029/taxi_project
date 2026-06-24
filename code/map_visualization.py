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

MAP_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../maps")

# 深圳地图中心点和缩放级别
SHENZHEN_CENTER = [22.52847, 114.05454]
DEFAULT_ZOOM = 12

# 颜色配置
COLOR_OCCUPIED = '#FF4444'    # 载客状态：红色
COLOR_EMPTY = '#4444FF'       # 空载状态：蓝色
COLOR_PICKUP = '#00CC00'      # 上车点：绿色
COLOR_DROPOFF = '#FF8800'     # 下车点：橙色
# ===============================================================================


# --------------------------- 1. 基础地图创建 ---------------------------
def create_base_map(center=SHENZHEN_CENTER, zoom=DEFAULT_ZOOM, title="深圳出租车GPS分析"):
    """创建基础地图"""
    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles='OpenStreetMap',
        control_scale=True
    )
    
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


def plot_minute_snapshot(time_str, vehicle_ids=None, max_vehicles=100):
    """
    绘制某一时刻的车辆位置分布
    
    Args:
        time_str: 时间字符串
        vehicle_ids: 指定车辆ID列表（None表示所有车辆）
        max_vehicles: 最大显示车辆数（避免地图过于拥挤）
    
    Returns:
        folium.Map: 地图对象
    """
    df = load_minute_snapshot(time_str)
    
    # 筛选指定车辆
    if vehicle_ids:
        df = df[df['id'].isin(vehicle_ids)]
    
    # 限制显示数量
    if len(df) > max_vehicles:
        df = df.sample(n=max_vehicles, random_state=42)
    
    m = create_base_map(title=f"车辆位置快照 ({time_str})")
    
    # 添加车辆位置标记
    for _, row in df.iterrows():
        color = 'red' if row['status'] == 1 else 'blue'
        status_text = "载客" if row['status'] == 1 else "空载"
        
        folium.CircleMarker(
            location=[row['lati'], row['long']],
            radius=3,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7,
            popup=f"车辆ID: {int(row['id'])}<br>状态: {status_text}<br>速度: {row['speed']}km/h"
        ).add_to(m)
    
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
def create_animated_trajectory(vehicle_id, start_time=None, end_time=None, 
                               speed_factor=100):
    """
    创建单车动画轨迹（车辆图标沿轨迹移动）
    
    Args:
        vehicle_id: 车辆ID
        start_time: 开始时间
        end_time: 结束时间
        speed_factor: 动画速度倍数（越大越快）
    
    Returns:
        folium.Map: 包含动画的地图对象
    """
    df = load_vehicle_trajectory(vehicle_id, start_time, end_time)
    
    if df.empty or len(df) < 2:
        raise ValueError(f"车辆 {vehicle_id} 在指定时间段内数据不足")
    
    # 创建地图
    center_lat = df['lati'].mean()
    center_lon = df['long'].mean()
    m = create_base_map(
        center=[center_lat, center_lon],
        title=f"车辆 {vehicle_id} 动画轨迹"
    )
    
    # 准备轨迹数据
    points = []
    for _, row in df.iterrows():
        points.append({
            'lat': row['lati'],
            'lon': row['long'],
            'time': row['time'].isoformat(),
            'speed': float(row['speed']),
            'status': int(row['status'])
        })
    
    # 生成JavaScript动画代码
    animation_js = f"""
    <script>
    var points = {json.dumps(points)};
    var currentIndex = 0;
    var marker = null;
    var polyline = null;
    var pathCoords = [];
    
    function initAnimation() {{
        var firstPoint = points[0];
        
        // 创建车辆标记
        marker = L.marker([firstPoint.lat, firstPoint.lon], {{
            icon: L.icon({{
                iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-red.png',
                iconSize: [25, 41],
                iconAnchor: [12, 41]
            }})
        }}).addTo(map);
        
        marker.bindPopup('车辆ID: {vehicle_id}<br>速度: ' + firstPoint.speed + ' km/h<br>状态: ' + 
                        (firstPoint.status === 1 ? '载客' : '空载'));
        
        // 创建轨迹线
        polyline = L.polyline([], {{color: 'blue', weight: 3}}).addTo(map);
        
        // 开始动画
        animateVehicle();
    }}
    
    function animateVehicle() {{
        if (currentIndex >= points.length) {{
            return;
        }}
        
        var point = points[currentIndex];
        var nextPoint = points[currentIndex + 1];
        
        // 更新标记位置
        marker.setLatLng([point.lat, point.lon]);
        marker.setPopupContent('车辆ID: {vehicle_id}<br>速度: ' + point.speed + 
                              ' km/h<br>状态: ' + (point.status === 1 ? '载客' : '空载') +
                              '<br>时间: ' + point.time);
        
        // 更新轨迹线
        pathCoords.push([point.lat, point.lon]);
        polyline.setLatLngs(pathCoords);
        
        // 根据速度和时间差计算延迟
        var delay = 1000 / {speed_factor};  // 基础延迟
        if (nextPoint) {{
            var timeDiff = new Date(nextPoint.time) - new Date(point.time);
            delay = Math.min(timeDiff / {speed_factor}, 500);  // 限制最大延迟
        }}
        
        currentIndex++;
        setTimeout(animateVehicle, delay);
    }}
    
    // 等待地图加载完成后启动动画
    map.whenReady(function() {{
        setTimeout(initAnimation, 1000);
    }});
    </script>
    """
    
    m.get_root().html.add_child(folium.Element(animation_js))
    
    return m


# --------------------------- 6. 地图选点功能（为路网校正和ETA预留） ---------------------------
def create_point_picker_map():
    """
    创建带有地图选点功能的地图，用户点击地图可获取坐标
    用于后续路网校正和ETA预测的起终点选择
    
    Returns:
        folium.Map: 地图对象
    """
    m = create_base_map(title="地图选点工具（点击地图获取坐标）")
    
    # 添加点击事件获取坐标的JavaScript
    click_js = """
    <script>
    var selectedPoints = [];
    var markers = [];
    
    map.on('click', function(e) {
        var lat = e.latlng.lat.toFixed(6);
        var lon = e.latlng.lng.toFixed(6);
        
        // 添加标记
        var marker = L.marker([lat, lon]).addTo(map);
        marker.bindPopup('坐标: [' + lat + ', ' + lon + ']<br>点击数: ' + (selectedPoints.length + 1)).openPopup();
        
        selectedPoints.push([lat, lon]);
        markers.push(marker);
        
        // 在控制台输出坐标
        console.log('选中点 ' + selectedPoints.length + ': [' + lat + ', ' + lon + ']');
        
        // 如果选了两个点，绘制连线
        if (selectedPoints.length === 2) {
            L.polyline(selectedPoints, {color: 'red', dashArray: '5, 10'}).addTo(map);
            alert('起点: [' + selectedPoints[0] + ']\\n终点: [' + selectedPoints[1] + ']');
            
            // 清空准备下一次选择
            selectedPoints = [];
            markers.forEach(m => map.removeLayer(m));
            markers = [];
        }
    });
    </script>
    """
    
    m.get_root().html.add_child(folium.Element(click_js))
    
    return m


# --------------------------- 7. 主函数与示例 ---------------------------
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
            end_time='2013-10-22 10:00:00',
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
            end_time='2013-10-22 08:30:00',
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
    
    print("\n[OK] 地图生成完成！请在浏览器中打开maps目录下的HTML文件查看效果。")


if __name__ == "__main__":
    main()
