# -*- coding: utf-8 -*-
"""
交互式地图查询页面
==================
基于 Flask 提供前端表单，支持在页面输入参数后实时生成地图。
"""

import os
import uuid

import pandas as pd
from flask import Flask, request, render_template_string, send_from_directory, jsonify

from data_analysis import (
    MAP_DIR as ANALYSIS_MAP_DIR,
    TABLE_DIR as ANALYSIS_TABLE_DIR,
    build_static_vehicle_heatmap,
    build_static_pickup_heatmap,
    build_dynamic_pickup_heatmap,
    build_dynamic_vehicle_heatmap,
    build_dbscan_cluster_map,
    run_pickup_dbscan,
)
from map_visualization import (
    MAP_OUTPUT_DIR,
    plot_vehicle_trajectory,
    plot_multi_vehicle_trajectory,
    plot_minute_snapshot,
    plot_od_points,
    create_animated_trajectory,
    create_point_picker_map,
    plot_corrected_trajectory,
)


app = Flask(__name__)
os.makedirs(MAP_OUTPUT_DIR, exist_ok=True)
os.makedirs(ANALYSIS_MAP_DIR, exist_ok=True)


PAGE_TEMPLATE = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>出租车地图交互查询</title>
  <style>
    :root {
      --bg: #f4f7fb;
      --panel: #ffffff;
      --line: #d7e0ea;
      --text: #1d2a38;
      --muted: #637487;
      --accent: #0f6cbd;
      --accent-dark: #0a4f8a;
      --danger: #b42318;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Microsoft YaHei", "PingFang SC", sans-serif;
      color: var(--text);
      background: linear-gradient(135deg, #eef4fb 0%, #f8fbff 100%);
    }
    .layout {
      display: grid;
      grid-template-columns: 360px 1fr;
      min-height: 100vh;
    }
    .top-nav {
      display: flex;
      gap: 10px;
      margin-bottom: 18px;
      flex-wrap: wrap;
    }
    .top-nav a {
      text-decoration: none;
      padding: 8px 12px;
      border-radius: 999px;
      font-size: 13px;
      border: 1px solid var(--line);
      color: var(--text);
      background: #fff;
    }
    .top-nav a.active {
      background: var(--accent);
      color: #fff;
      border-color: var(--accent);
    }
    .sidebar {
      padding: 24px;
      background: rgba(255, 255, 255, 0.96);
      border-right: 1px solid var(--line);
      overflow-y: auto;
    }
    .main {
      padding: 20px;
    }
    h1 {
      font-size: 24px;
      margin: 0 0 8px;
    }
    .subtitle {
      font-size: 14px;
      color: var(--muted);
      margin: 0 0 20px;
      line-height: 1.6;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 16px;
      margin-bottom: 16px;
      box-shadow: 0 10px 30px rgba(31, 60, 96, 0.06);
    }
    .panel h2 {
      margin: 0 0 12px;
      font-size: 16px;
    }
    .field {
      margin-bottom: 12px;
    }
    label {
      display: block;
      margin-bottom: 6px;
      font-size: 13px;
      color: var(--muted);
    }
    input, select {
      width: 100%;
      padding: 10px 12px;
      border-radius: 10px;
      border: 1px solid var(--line);
      font-size: 14px;
      background: #fff;
    }
    .button-row {
      display: flex;
      gap: 10px;
      margin-top: 16px;
    }
    button {
      border: 0;
      border-radius: 10px;
      padding: 11px 16px;
      font-size: 14px;
      cursor: pointer;
    }
    .primary {
      background: var(--accent);
      color: #fff;
      flex: 1;
    }
    .primary:hover { background: var(--accent-dark); }
    .secondary {
      background: #eef4fb;
      color: var(--text);
    }
    .tips {
      font-size: 13px;
      color: var(--muted);
      line-height: 1.7;
    }
    .status {
      margin-bottom: 12px;
      padding: 12px 14px;
      border-radius: 10px;
      font-size: 14px;
    }
    .status.error {
      background: #fdecec;
      color: var(--danger);
      border: 1px solid #f7c5c0;
    }
    .status.ok {
      background: #edf7ed;
      color: #1e6b34;
      border: 1px solid #cde7d1;
    }
    .map-wrap {
      width: 100%;
      height: calc(100vh - 40px);
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 16px;
      overflow: hidden;
      box-shadow: 0 10px 30px rgba(31, 60, 96, 0.06);
    }
    iframe {
      width: 100%;
      height: 100%;
      border: 0;
    }
    .empty {
      display: flex;
      align-items: center;
      justify-content: center;
      height: 100%;
      color: var(--muted);
      font-size: 15px;
      background: radial-gradient(circle at top, #f8fbff, #eef4fb);
    }
    @media (max-width: 1100px) {
      .layout {
        grid-template-columns: 1fr;
      }
      .main {
        padding-top: 0;
      }
      .map-wrap {
        height: 72vh;
      }
    }
  </style>
  <script>
    function onModeChange() {
      const mode = document.getElementById('mode').value;
      const vehicleSection = document.getElementById('vehicle-section');
      const timeRangeSection = document.getElementById('time-range-section');
      const minuteSection = document.getElementById('minute-section');
      const maxVehiclesSection = document.getElementById('max-vehicles-section');
      const odSection = document.getElementById('od-section');
      const animationSection = document.getElementById('animation-section');
      const roadCorrectionSection = document.getElementById('road-correction-section');

      vehicleSection.style.display = ['trajectory', 'animation', 'road_correction'].includes(mode) ? 'block' : 'none';
      timeRangeSection.style.display = ['trajectory', 'od_points', 'animation', 'road_correction'].includes(mode) ? 'block' : 'none';
      minuteSection.style.display = mode === 'snapshot' ? 'block' : 'none';
      maxVehiclesSection.style.display = mode === 'snapshot' ? 'block' : 'none';
      odSection.style.display = mode === 'od_points' ? 'block' : 'none';
      animationSection.style.display = mode === 'animation' ? 'block' : 'none';
      roadCorrectionSection.style.display = mode === 'road_correction' ? 'block' : 'none';
    }

    window.addEventListener('DOMContentLoaded', onModeChange);
  </script>
</head>
<body>
  <div class="layout">
    <aside class="sidebar">
      <div class="top-nav">
        <a href="/" class="{{ 'active' if active_page == 'map' else '' }}">地图查询</a>
        <a href="/analysis" class="{{ 'active' if active_page == 'analysis' else '' }}">热力图分析</a>
      </div>
      <h1>出租车地图交互查询</h1>
      <form method="post" class="panel">
        <h2>查询参数</h2>

        <div class="field">
          <label for="mode">查询模式</label>
          <select id="mode" name="mode" onchange="onModeChange()">
            <option value="trajectory" {% if form.mode == 'trajectory' %}selected{% endif %}>车辆轨迹</option>
            <option value="snapshot" {% if form.mode == 'snapshot' %}selected{% endif %}>分钟位置快照</option>
            <option value="od_points" {% if form.mode == 'od_points' %}selected{% endif %}>上下车点分布</option>
            <option value="animation" {% if form.mode == 'animation' %}selected{% endif %}>动画轨迹</option>
            <option value="road_correction" {% if form.mode == 'road_correction' %}selected{% endif %}>路网校正轨迹</option>
          </select>
        </div>

        <div id="vehicle-section">
          <div class="field">
            <label for="vehicle_id">车辆 ID</label>
            <input id="vehicle_id" name="vehicle_id" value="{{ form.vehicle_id }}" placeholder="例如 22223，多辆车用逗号分隔 22223,22224">
          </div>
        </div>

        <div id="time-range-section">
          <div class="field">
            <label for="start_time">开始时间</label>
            <input id="start_time" name="start_time" value="{{ form.start_time }}" placeholder="2013-10-22 08:00:00">
          </div>
          <div class="field">
            <label for="end_time">结束时间</label>
            <input id="end_time" name="end_time" value="{{ form.end_time }}" placeholder="2013-10-22 10:00:00">
          </div>
        </div>

        <div id="minute-section">
          <div class="field">
            <label for="snapshot_time">快照时间</label>
            <input id="snapshot_time" name="snapshot_time" value="{{ form.snapshot_time }}" placeholder="2013-10-22 08:00">
          </div>
          <div class="field">
            <label for="snapshot_vehicle_id_min">车辆 ID 下限</label>
            <input id="snapshot_vehicle_id_min" name="snapshot_vehicle_id_min" value="{{ form.snapshot_vehicle_id_min }}" placeholder="例如 20000，可为空">
          </div>
          <div class="field">
            <label for="snapshot_vehicle_id_max">车辆 ID 上限</label>
            <input id="snapshot_vehicle_id_max" name="snapshot_vehicle_id_max" value="{{ form.snapshot_vehicle_id_max }}" placeholder="例如 30000，可为空">
          </div>
          <div class="field">
            <label for="snapshot_status_filter">载客状态</label>
            <select id="snapshot_status_filter" name="snapshot_status_filter">
              <option value="" {% if form.snapshot_status_filter == '' %}selected{% endif %}>全部</option>
              <option value="1" {% if form.snapshot_status_filter == '1' %}selected{% endif %}>仅载客</option>
              <option value="0" {% if form.snapshot_status_filter == '0' %}selected{% endif %}>仅空载</option>
            </select>
          </div>
        </div>

        <div id="max-vehicles-section">
          <div class="field">
            <label for="max_vehicles">最多显示车辆数</label>
            <input id="max_vehicles" name="max_vehicles" value="{{ form.max_vehicles }}" placeholder="500">
          </div>
        </div>

        <div id="od-section">
          <div class="field">
            <label for="max_points">最多显示订单点数</label>
            <input id="max_points" name="max_points" value="{{ form.max_points }}" placeholder="300">
          </div>
        </div>

        <div id="animation-section">
          <div class="field">
            <label for="speed_factor">动画速度倍数</label>
            <input id="speed_factor" name="speed_factor" value="{{ form.speed_factor }}" placeholder="200">
          </div>
        </div>

        <div id="road-correction-section">
          <div class="field">
            <label>
              <input type="checkbox" name="enable_correction" value="1"
                     {% if form.enable_correction %}checked{% endif %}>
              启用路网校正（取消勾选仅显示原始轨迹）
            </label>
          </div>
          <div class="field">
            <label>
              <input type="checkbox" name="use_undirected" value="1"
                     {% if form.use_undirected %}checked{% endif %}>
              使用无向图做最短路径（有向图不可达时自动回退）
            </label>
          </div>
        </div>

        <div class="button-row">
          <button class="primary" type="submit">生成地图</button>
          <button class="secondary" type="button" onclick="window.location='/'">重置</button>
        </div>
      </form>

      <div class="panel tips">
        <h2>填写说明</h2>
        <div>车辆轨迹/动画轨迹：需要填 `车辆 ID`、`开始时间`、`结束时间`。</div>
        <div>两种模式均支持多辆车，使用英文逗号分隔，例如 `22223,22224,22225`。</div>
        <div>分钟位置快照：需要填 `快照时间`，可选筛选车辆 ID 下限/上限和载客状态，点击车辆可查看指定结束时间内的后续轨迹。</div>
        <div>上下车点分布：填写时间范围，系统从 OD 缓存中抽样展示。</div>
        <div>路网校正轨迹：填 1-3 辆车 ID 和短时间窗口，建议 ≤30 分钟。支持直接在地图上点选坐标。</div>
      </div>
    </aside>

    <main class="main">
      {% if message %}
      <div class="status {{ 'error' if error else 'ok' }}">{{ message }}</div>
      {% endif %}

      <div class="map-wrap">
        {% if map_file %}
        <iframe src="/maps/{{ map_file }}"></iframe>
        {% else %}
        <div class="empty">提交左侧查询参数后，这里会显示生成好的交互式地图。</div>
        {% endif %}
      </div>
    </main>
  </div>
</body>
</html>
"""


ANALYSIS_TEMPLATE = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>热力图与统计分析</title>
  <style>
    :root {
      --bg: #f4f7fb;
      --panel: #ffffff;
      --line: #d7e0ea;
      --text: #1d2a38;
      --muted: #637487;
      --accent: #0f6cbd;
      --accent-dark: #0a4f8a;
      --danger: #b42318;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Microsoft YaHei", "PingFang SC", sans-serif;
      color: var(--text);
      background: linear-gradient(135deg, #eef4fb 0%, #f8fbff 100%);
    }
    .layout {
      display: grid;
      grid-template-columns: 380px 1fr;
      min-height: 100vh;
    }
    .sidebar {
      padding: 24px;
      background: rgba(255, 255, 255, 0.96);
      border-right: 1px solid var(--line);
      overflow-y: auto;
    }
    .main { padding: 20px; }
    .top-nav {
      display: flex;
      gap: 10px;
      margin-bottom: 18px;
      flex-wrap: wrap;
    }
    .top-nav a {
      text-decoration: none;
      padding: 8px 12px;
      border-radius: 999px;
      font-size: 13px;
      border: 1px solid var(--line);
      color: var(--text);
      background: #fff;
    }
    .top-nav a.active {
      background: var(--accent);
      color: #fff;
      border-color: var(--accent);
    }
    h1 { font-size: 24px; margin: 0 0 8px; }
    .subtitle { font-size: 14px; color: var(--muted); margin: 0 0 20px; line-height: 1.6; }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 16px;
      margin-bottom: 16px;
      box-shadow: 0 10px 30px rgba(31, 60, 96, 0.06);
    }
    .panel h2 { margin: 0 0 12px; font-size: 16px; }
    .field { margin-bottom: 12px; }
    label { display: block; margin-bottom: 6px; font-size: 13px; color: var(--muted); }
    input, select {
      width: 100%;
      padding: 10px 12px;
      border-radius: 10px;
      border: 1px solid var(--line);
      font-size: 14px;
      background: #fff;
    }
    button {
      width: 100%;
      border: 0;
      border-radius: 10px;
      padding: 11px 16px;
      font-size: 14px;
      cursor: pointer;
      background: var(--accent);
      color: #fff;
    }
    button:hover { background: var(--accent-dark); }
    .status {
      margin-bottom: 12px;
      padding: 12px 14px;
      border-radius: 10px;
      font-size: 14px;
    }
    .status.error {
      background: #fdecec;
      color: var(--danger);
      border: 1px solid #f7c5c0;
    }
    .status.ok {
      background: #edf7ed;
      color: #1e6b34;
      border: 1px solid #cde7d1;
    }
    .preview-card {
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 14px;
      overflow: hidden;
      box-shadow: 0 10px 30px rgba(31, 60, 96, 0.06);
      min-height: calc(100vh - 40px);
    }
    .preview-card h3 {
      margin: 0;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      font-size: 15px;
      background: #f9fbfe;
    }
    iframe {
      width: 100%;
      height: calc(100vh - 96px);
      border: 0;
    }
    .table-list {
      font-size: 13px;
      line-height: 1.8;
      color: var(--muted);
    }
    .table-list code {
      background: #eef4fb;
      padding: 2px 6px;
      border-radius: 6px;
    }
    @media (max-width: 1200px) {
      .layout { grid-template-columns: 1fr; }
      .preview-card { min-height: 72vh; }
      iframe { height: 68vh; }
    }
  </style>
  <script>
    function onAnalysisModeChange() {
      const mode = document.getElementById('analysis_mode').value;
      document.getElementById('time-range-analysis').style.display = ['static_pickup_heatmap', 'od_points_like', 'dbscan'].includes(mode) ? 'block' : 'none';
      document.getElementById('snapshot-analysis').style.display = mode === 'static_vehicle_heatmap' ? 'block' : 'none';
      document.getElementById('freq-analysis').style.display = ['dynamic_pickup_heatmap', 'dynamic_vehicle_heatmap'].includes(mode) ? 'block' : 'none';
    }
    window.addEventListener('DOMContentLoaded', onAnalysisModeChange);
  </script>
</head>
<body>
  <div class="layout">
    <aside class="sidebar">
      <div class="top-nav">
        <a href="/">地图查询</a>
        <a href="/analysis" class="active">热力图分析</a>
      </div>
      <h1>热力图与统计分析</h1>
      <p class="subtitle">在页面中选择热力图或聚类模式，动态生成阶段05结果，并保留统计表输出位置。</p>

      <form method="post" class="panel">
        <h2>分析参数</h2>
        <div class="field">
          <label for="analysis_mode">分析模式</label>
          <select id="analysis_mode" name="analysis_mode" onchange="onAnalysisModeChange()">
            <option value="static_vehicle_heatmap" {% if form.analysis_mode == 'static_vehicle_heatmap' %}selected{% endif %}>车辆位置静态热力图</option>
            <option value="static_pickup_heatmap" {% if form.analysis_mode == 'static_pickup_heatmap' %}selected{% endif %}>上车点静态热力图</option>
            <option value="dynamic_pickup_heatmap" {% if form.analysis_mode == 'dynamic_pickup_heatmap' %}selected{% endif %}>上车点动态热力图</option>
            <option value="dynamic_vehicle_heatmap" {% if form.analysis_mode == 'dynamic_vehicle_heatmap' %}selected{% endif %}>车辆位置动态热力图</option>
            <option value="dbscan" {% if form.analysis_mode == 'dbscan' %}selected{% endif %}>DBSCAN 聚类中心图</option>
          </select>
        </div>

        <div id="snapshot-analysis">
          <div class="field">
            <label for="snapshot_time">快照时间</label>
            <input id="snapshot_time" name="snapshot_time" value="{{ form.snapshot_time }}" placeholder="2013-10-22 08:00">
          </div>
        </div>

        <div id="time-range-analysis">
          <div class="field">
            <label for="start_time">开始时间</label>
            <input id="start_time" name="start_time" value="{{ form.start_time }}" placeholder="2013-10-22 08:00:00">
          </div>
          <div class="field">
            <label for="end_time">结束时间</label>
            <input id="end_time" name="end_time" value="{{ form.end_time }}" placeholder="2013-10-22 09:00:00">
          </div>
        </div>

        <div id="freq-analysis">
          <div class="field">
            <label for="freq">聚合粒度</label>
            <select id="freq" name="freq">
              <option value="15min" {% if form.freq == '15min' %}selected{% endif %}>15 分钟</option>
              <option value="30min" {% if form.freq == '30min' %}selected{% endif %}>30 分钟</option>
              <option value="60min" {% if form.freq == '60min' %}selected{% endif %}>60 分钟</option>
            </select>
          </div>
        </div>

        <button type="submit">生成分析结果</button>
      </form>

      <div class="panel table-list">
        <h2>统计结果文件</h2>
        <div><code>analysis/tables/hourly_order_occupied_ratio.csv</code></div>
        <div><code>analysis/tables/pickup_dbscan_clusters.csv</code></div>
        <div><code>analysis/tables/trip_distance_levels.csv</code></div>
        <div><code>analysis/tables/vehicle_daily_stats.csv</code></div>
        <div><code>analysis/tables/dynamic_pickup_heatmap_15min.json</code></div>
      </div>
    </aside>

    <main class="main">
      {% if message %}
      <div class="status {{ 'error' if error else 'ok' }}">{{ message }}</div>
      {% endif %}

      <div class="preview-card">
        <h3>地图预览</h3>
        {% if map_file %}
        <iframe src="/analysis_maps/{{ map_file }}"></iframe>
        {% else %}
        <div style="padding:24px;color:#637487;">提交左侧参数后，这里显示热力图或聚类结果。</div>
        {% endif %}
      </div>
    </main>
  </div>
</body>
</html>
"""


def _default_form():
    return {
        'mode': 'trajectory',
        'vehicle_id': '22223',
        'start_time': '2013-10-22 08:00:00',
        'end_time': '2013-10-22 10:00:00',
        'snapshot_time': '2013-10-22 08:00',
        'snapshot_vehicle_id_min': '',
        'snapshot_vehicle_id_max': '',
        'snapshot_status_filter': '',
        'max_vehicles': '500',
        'max_points': '300',
        'speed_factor': '200',
        'enable_correction': True,
        'use_undirected': False,
    }


def _default_analysis_form():
    return {
        'analysis_mode': 'static_vehicle_heatmap',
        'snapshot_time': '2013-10-22 08:00',
        'start_time': '2013-10-22 08:00:00',
        'end_time': '2013-10-22 09:00:00',
        'freq': '15min',
    }


def _build_map(form):
    mode = form['mode']

    if mode == 'trajectory':
        vehicle_ids = [int(item.strip()) for item in form['vehicle_id'].split(',') if item.strip()]
        if not vehicle_ids:
            raise ValueError('请至少输入一个车辆 ID')
        if len(vehicle_ids) == 1:
            return plot_vehicle_trajectory(vehicle_ids[0], form['start_time'], form['end_time'])
        return plot_multi_vehicle_trajectory(vehicle_ids, form['start_time'], form['end_time'])

    if mode == 'snapshot':
        max_vehicles = int(form['max_vehicles'])
        snapshot_vehicle_id_min = form.get('snapshot_vehicle_id_min', '').strip()
        snapshot_vehicle_id_max = form.get('snapshot_vehicle_id_max', '').strip()
        snapshot_status_filter = form.get('snapshot_status_filter', '').strip()
        id_min = int(snapshot_vehicle_id_min) if snapshot_vehicle_id_min else None
        id_max = int(snapshot_vehicle_id_max) if snapshot_vehicle_id_max else None
        status_filter = int(snapshot_status_filter) if snapshot_status_filter else None
        return plot_minute_snapshot(
            form['snapshot_time'],
            max_vehicles=max_vehicles,
            id_min=id_min,
            id_max=id_max,
            status_filter=status_filter,
        )

    if mode == 'od_points':
        max_points = int(form['max_points'])
        return plot_od_points(form['start_time'], form['end_time'], max_points=max_points)

    if mode == 'animation':
        vehicle_ids = [int(x.strip()) for x in form['vehicle_id'].split(',') if x.strip()]
        if not vehicle_ids:
            raise ValueError('请至少输入一个车辆 ID')
        speed_factor = int(form['speed_factor'])
        return create_animated_trajectory(vehicle_ids, form['start_time'], form['end_time'], speed_factor=speed_factor)

    if mode == 'road_correction':
        vehicle_ids = [int(item.strip()) for item in form['vehicle_id'].split(',') if item.strip()]
        if not vehicle_ids:
            raise ValueError('请至少输入一个车辆 ID')
        if len(vehicle_ids) > 3:
            raise ValueError('路网校正样例最多支持 3 辆车')
        enable_correction = form.get('enable_correction', True)
        use_undirected = form.get('use_undirected', False)
        m = plot_corrected_trajectory(
            vehicle_ids,
            form['start_time'],
            form['end_time'],
            enable_correction=enable_correction,
            use_undirected=use_undirected,
        )
        # 将选点功能注入到校正轨迹地图中
        map_js_name = m.get_name()
        import folium as _folium
        coord_panel = """
        <div id="coord-panel" style="position:fixed; bottom:20px; right:20px; z-index:9999;
                    background:rgba(255,255,255,0.92); border:2px solid #0f6cbd; border-radius:10px;
                    padding:12px 16px; font-size:13px; min-width:200px; line-height:1.7;
                    box-shadow: 0 4px 16px rgba(15, 108, 189, 0.15); pointer-events:auto;">
            <div style="font-weight:bold; margin-bottom:4px;">📍 地图选点</div>
            <div>纬度 (lat): <span id="pick-lat" style="color:#0f6cbd;">—</span></div>
            <div>经度 (lng): <span id="pick-lng" style="color:#0f6cbd;">—</span></div>
            <div style="margin-top:4px; font-size:12px; color:#637487;">点击地图任意位置更新坐标</div>
        </div>
        """
        click_js = f"""
        <script>
        (function() {{
            var currentMarker = null;
            function bindPointPicker(map) {{
                map.on('click', function(e) {{
                    var lat = e.latlng.lat.toFixed(6);
                    var lon = e.latlng.lng.toFixed(6);
                    document.getElementById('pick-lat').textContent = lat;
                    document.getElementById('pick-lng').textContent = lon;
                    if (currentMarker) {{ map.removeLayer(currentMarker); }}
                    currentMarker = L.marker([lat, lon]).addTo(map);
                    currentMarker.bindPopup('纬度: ' + lat + '<br>经度: ' + lon).openPopup();
                }});
            }}
            function waitForMap() {{
                if (typeof {map_js_name} !== 'undefined' && {map_js_name}) {{
                    bindPointPicker({map_js_name});
                }} else {{
                    setTimeout(waitForMap, 100);
                }}
            }}
            waitForMap();
        }})();
        </script>
        """
        m.get_root().html.add_child(_folium.Element(coord_panel))
        m.get_root().html.add_child(_folium.Element(click_js))
        return m

    raise ValueError('不支持的查询模式')


def _build_analysis_map(form):
    mode = form['analysis_mode']

    if mode == 'static_vehicle_heatmap':
        file_name = f"analysis_{uuid.uuid4().hex}.html"
        out_path = build_static_vehicle_heatmap(form['snapshot_time'], output_name=file_name)
        return os.path.basename(out_path), '基于分钟缓存绘制某一时刻所有车辆位置热力图。', None

    if mode == 'static_pickup_heatmap':
        file_name = f"analysis_{uuid.uuid4().hex}.html"
        out_path = build_static_pickup_heatmap(form['start_time'], form['end_time'], output_name=file_name)
        return os.path.basename(out_path), '基于 OD 上车点绘制指定时间范围的乘客需求热力图。', None

    if mode == 'dynamic_pickup_heatmap':
        file_name = f"analysis_{uuid.uuid4().hex}.html"
        out_path = build_dynamic_pickup_heatmap(form['freq'], output_name=file_name)
        return os.path.basename(out_path), '按时间片组织的上车点动态热力图，适合看需求随时间变化。', 'analysis/tables/dynamic_pickup_heatmap_15min.json'

    if mode == 'dynamic_vehicle_heatmap':
        file_name = f"analysis_{uuid.uuid4().hex}.html"
        out_path = build_dynamic_vehicle_heatmap(form['freq'], output_name=file_name)
        return os.path.basename(out_path), '按时间片组织的车辆位置动态热力图，适合看供给分布变化。', None

    if mode == 'dbscan':
        cluster_df, cluster_path = run_pickup_dbscan(output_name='pickup_dbscan_clusters.csv')
        file_name = f"analysis_{uuid.uuid4().hex}.html"
        out_path = build_dbscan_cluster_map(cluster_df, output_name=file_name)
        return os.path.basename(out_path), 'DBSCAN 聚类中心图已生成。', cluster_path.replace('\\', '/')

    raise ValueError('不支持的分析模式')


@app.route('/', methods=['GET', 'POST'])
def index():
    form = _default_form()
    map_file = None
    message = None
    error = False

    if request.method == 'POST':
        form.update({
            'mode': request.form.get('mode', form['mode']).strip(),
            'vehicle_id': request.form.get('vehicle_id', form['vehicle_id']).strip(),
            'start_time': request.form.get('start_time', form['start_time']).strip(),
            'end_time': request.form.get('end_time', form['end_time']).strip(),
            'snapshot_time': request.form.get('snapshot_time', form['snapshot_time']).strip(),
            'snapshot_vehicle_id_min': request.form.get('snapshot_vehicle_id_min', form['snapshot_vehicle_id_min']).strip(),
            'snapshot_vehicle_id_max': request.form.get('snapshot_vehicle_id_max', form['snapshot_vehicle_id_max']).strip(),
            'snapshot_status_filter': request.form.get('snapshot_status_filter', form['snapshot_status_filter']).strip(),
            'max_vehicles': request.form.get('max_vehicles', form['max_vehicles']).strip(),
            'max_points': request.form.get('max_points', form['max_points']).strip(),
            'speed_factor': request.form.get('speed_factor', form['speed_factor']).strip(),
            'enable_correction': request.form.get('enable_correction') == '1',
            'use_undirected': request.form.get('use_undirected') == '1',
        })

        try:
            map_obj = _build_map(form)
            map_file = f"query_{uuid.uuid4().hex}.html"
            map_obj.save(os.path.join(MAP_OUTPUT_DIR, map_file))
            message = '地图生成成功，可以直接在右侧交互查看。'
        except Exception as exc:
            error = True
            message = f'生成失败：{exc}'

    return render_template_string(
        PAGE_TEMPLATE,
        form=form,
        map_file=map_file,
        message=message,
        error=error,
        active_page='map',
    )


@app.route('/analysis', methods=['GET', 'POST'])
def analysis_page():
    form = _default_analysis_form()
    map_file = None
    message = None
    error = False
    detail_text = '选择左侧分析模式后，这里会展示热力图或聚类说明。'
    extra_file = None

    if request.method == 'POST':
        form.update({
            'analysis_mode': request.form.get('analysis_mode', form['analysis_mode']).strip(),
            'snapshot_time': request.form.get('snapshot_time', form['snapshot_time']).strip(),
            'start_time': request.form.get('start_time', form['start_time']).strip(),
            'end_time': request.form.get('end_time', form['end_time']).strip(),
            'freq': request.form.get('freq', form['freq']).strip(),
        })

        try:
            map_file, detail_text, extra_file = _build_analysis_map(form)
            message = '阶段05分析结果已生成。'
        except Exception as exc:
            error = True
            message = f'生成失败：{exc}'

    return render_template_string(
        ANALYSIS_TEMPLATE,
        form=form,
        map_file=map_file,
        message=message,
        error=error,
        detail_text=detail_text,
        extra_file=extra_file,
    )


@app.route('/maps/<path:filename>')
def serve_map(filename):
    return send_from_directory(MAP_OUTPUT_DIR, filename)


@app.route('/analysis_maps/<path:filename>')
def serve_analysis_map(filename):
    return send_from_directory(ANALYSIS_MAP_DIR, filename)


@app.route('/api/vehicle_trajectory')
def api_vehicle_trajectory():
    """
    API 端点：返回指定车辆从给定时间点开始的后续轨迹数据（JSON 格式）。
    供分钟快照地图中的"查看后续轨迹"功能使用。

    Query params:
        vehicle_id: 车辆 ID（整数）
        start_time: 起始时间（字符串，如 "2013-10-22 08:00:00"）
        end_time: 结束时间（可选，字符串，如 "2013-10-22 08:30:00"）
    """
    try:
        vehicle_id = int(request.args.get('vehicle_id', ''))
        start_time = request.args.get('start_time', '').strip()
        end_time = request.args.get('end_time', '').strip()
        if not start_time:
            return jsonify({'error': '缺少 start_time 参数'})
        if end_time and pd.to_datetime(end_time) < pd.to_datetime(start_time):
            return jsonify({'error': 'end_time 不能早于 start_time'})

        from map_visualization import load_vehicle_trajectory
        df = load_vehicle_trajectory(vehicle_id, start_time=start_time, end_time=end_time or None)

        if df.empty:
            return jsonify({'error': f'车辆 {vehicle_id} 在该时间点之后无数据'})

        points = [
            {
                'lat': float(row['lati']),
                'lng': float(row['long']),
                'time': str(row['time']),
                'speed': float(row['speed']),
                'status': int(row['status']),
            }
            for _, row in df.iterrows()
        ]
        return jsonify({'vehicle_id': vehicle_id, 'points': points})
    except FileNotFoundError:
        return jsonify({'error': f'车辆 {vehicle_id} 的缓存数据不存在'})
    except Exception as exc:
        return jsonify({'error': str(exc)})


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)