# -*- coding: utf-8 -*-
"""
交互式地图查询页面
==================
基于 Flask 提供前端表单，支持在页面输入参数后实时生成地图。
"""

import os
import uuid
import json
import pickle

import pandas as pd
import networkx as nx
import osmnx as ox
import geopandas as gpd
from shapely.geometry import LineString, Point, mapping
from shapely.ops import substring, unary_union
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE7_GRAPH_PATH = os.path.join(PROJECT_ROOT, 'data', 'cache', 'shenzhen_drive_stage7.pkl')
SHENZHEN_BOUNDARY_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', '深圳市.json')
STAGE7_OD_PATH = os.path.join(PROJECT_ROOT, 'data', 'cache', 'od', 'od_cache_stage7.parquet')
STAGE7_TRACK_PATH = os.path.join(PROJECT_ROOT, 'data', 'cache', 'matched_trajectory_stage7.parquet')
ROAD_CORRECTION_CACHE_DIR = os.path.join(PROJECT_ROOT, 'data', 'cache', 'road_correction')
_STAGE7_GRAPH = None
_SHENZHEN_BOUNDARY = None
_STAGE7_OD = None


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
      height: 100vh;
      overflow: hidden;
      font-family: "Microsoft YaHei", "PingFang SC", sans-serif;
      color: var(--text);
      background: linear-gradient(135deg, #eef4fb 0%, #f8fbff 100%);
    }
    .layout {
      display: grid;
      grid-template-columns: 360px 1fr;
      height: 100vh;
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
        <a href="/routes" class="{{ 'active' if active_page == 'routes' else '' }}">路线规划</a>
        <a href="/orders" class="{{ 'active' if active_page == 'orders' else '' }}">订单对比</a>
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
      height: 100vh;
      background: rgba(255, 255, 255, 0.96);
      border-right: 1px solid var(--line);
      overflow-y: auto;
    }
    .main { padding: 20px; height: 100vh; overflow: hidden; }
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
        <a href="/routes">路线规划</a>
        <a href="/orders">订单对比</a>
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


ROUTE_TEMPLATE = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>出租车路线规划</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
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
      height: 100vh;
      overflow: hidden;
      font-family: "Microsoft YaHei", "PingFang SC", sans-serif;
      color: var(--text);
      background: linear-gradient(135deg, #eef4fb 0%, #f8fbff 100%);
    }
    .layout {
      display: grid;
      grid-template-columns: 360px 1fr;
      height: 100vh;
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
      height: 100vh;
      background: rgba(255, 255, 255, 0.96);
      border-right: 1px solid var(--line);
      overflow-y: auto;
    }
    .main { padding: 20px; height: 100vh; overflow: hidden; }
    h1 { font-size: 24px; margin: 0 0 8px; }
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
    .panel h2 { margin: 0 0 12px; font-size: 16px; }
    .status {
      margin-bottom: 12px;
      padding: 12px 14px;
      border-radius: 10px;
      font-size: 14px;
      background: #edf7ed;
      color: #1e6b34;
      border: 1px solid #cde7d1;
    }
    .status.error {
      background: #fdecec;
      color: var(--danger);
      border: 1px solid #f7c5c0;
    }
    .tips { font-size: 13px; color: var(--muted); line-height: 1.8; }
    .summary-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-top: 12px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px 12px;
      background: #fafcff;
    }
    .metric .label { font-size: 12px; color: var(--muted); }
    .metric .value { font-size: 16px; margin-top: 4px; }
    #route-map {
      width: 100%;
      height: calc(100vh - 40px);
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 16px;
      overflow: hidden;
      box-shadow: 0 10px 30px rgba(31, 60, 96, 0.06);
    }
    .legend-row { display: flex; align-items: center; gap: 8px; margin-top: 8px; font-size: 13px; }
    .legend-line { width: 30px; height: 4px; border-radius: 99px; }
    .legend-dash { width: 30px; height: 0; border-top: 2px dashed #7a8a9a; }
    button {
      border: 0;
      border-radius: 10px;
      padding: 11px 16px;
      font-size: 14px;
      cursor: pointer;
      background: var(--accent);
      color: #fff;
      margin-top: 12px;
    }
    button:hover { background: var(--accent-dark); }
    @media (max-width: 1100px) {
      .layout { grid-template-columns: 1fr; }
      #route-map { height: 72vh; }
    }
  </style>
</head>
<body>
  <div class="layout">
    <aside class="sidebar">
      <div class="top-nav">
        <a href="/">地图查询</a>
        <a href="/analysis">热力图分析</a>
        <a href="/routes" class="active">路线规划</a>
        <a href="/orders">订单对比</a>
      </div>
      <h1>最短与最快路线</h1>
      <p class="subtitle">连续点击地图选择起点和终点。蓝线是最短距离路线，绿线是基准最快路线，灰色虚线用于连接点击位置与吸附到路网后的起终点。</p>

      <div class="panel">
        <h2>交互说明</h2>
        <div class="tips">
          <div>第一次点击设置起点。</div>
          <div>第二次点击设置终点，并自动计算两条路线。</div>
          <div>如果想重新选点，点击下方按钮。</div>
          <div>边界为深圳市行政区范围，仅作参考显示。</div>
        </div>
        <button type="button" onclick="resetRoute()">重新选点</button>
      </div>

      <div id="route-status" class="status">点击地图选择起点。</div>

      <div class="panel">
        <h2>结果摘要</h2>
        <div id="route-empty" class="tips">等待路线计算。</div>
        <div id="route-summary" style="display:none;">
          <div class="summary-grid">
            <div class="metric">
              <div class="label">最短距离</div>
              <div class="value" id="shortest-distance">-</div>
            </div>
            <div class="metric">
              <div class="label">最短路线成本</div>
              <div class="value" id="shortest-cost">-</div>
            </div>
            <div class="metric">
              <div class="label">最快路线距离</div>
              <div class="value" id="fastest-distance">-</div>
            </div>
            <div class="metric">
              <div class="label">最快路线成本</div>
              <div class="value" id="fastest-cost">-</div>
            </div>
            <div class="metric">
              <div class="label">起点吸附距离</div>
              <div class="value" id="origin-snap">-</div>
            </div>
            <div class="metric">
              <div class="label">终点吸附距离</div>
              <div class="value" id="dest-snap">-</div>
            </div>
          </div>
        </div>
      </div>

      <div class="panel">
        <h2>图例</h2>
        <div class="legend-row"><span class="legend-line" style="background:blue;"></span><span>最短距离路线</span></div>
        <div class="legend-row"><span class="legend-line" style="background:green;"></span><span>基准最快路线</span></div>
        <div class="legend-row"><span class="legend-dash"></span><span>点击位置到路网连接线</span></div>
        <div class="legend-row"><span class="legend-line" style="background:#123b7a;"></span><span>深圳行政边界</span></div>
      </div>
    </aside>

    <main class="main">
      <div id="route-map"></div>
    </main>
  </div>

  <script>
    const map = L.map('route-map').setView([22.58, 114.08], 11);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; OpenStreetMap'
    }).addTo(map);

    const routeState = {
      points: [],
      markers: [],
      shortest: null,
      fastest: null,
      connectors: [],
      boundary: null,
      requestSeq: 0,
      activeRequestSeq: 0,
    };

    function clearRouteLayers() {
      if (routeState.shortest) {
        map.removeLayer(routeState.shortest);
        routeState.shortest = null;
      }
      if (routeState.fastest) {
        map.removeLayer(routeState.fastest);
        routeState.fastest = null;
      }
      routeState.connectors.forEach(l => map.removeLayer(l));
      routeState.connectors = [];
    }

    function setStatus(text, isError=false) {
      const box = document.getElementById('route-status');
      box.textContent = text;
      box.className = isError ? 'status error' : 'status';
    }

    function resetRoute() {
      routeState.requestSeq += 1;
      routeState.activeRequestSeq = routeState.requestSeq;
      routeState.points = [];
      routeState.markers.forEach(m => map.removeLayer(m));
      routeState.markers = [];
      clearRouteLayers();
      document.getElementById('route-empty').style.display = 'block';
      document.getElementById('route-summary').style.display = 'none';
      setStatus('点击地图选择起点。');
    }

    function formatDistance(m) {
      return m >= 1000 ? (m / 1000).toFixed(2) + ' km' : Math.round(m) + ' m';
    }

    function formatMinutes(sec) {
      return (sec / 60).toFixed(1) + ' min';
    }

    function updateSummary(summary) {
      document.getElementById('route-empty').style.display = 'none';
      document.getElementById('route-summary').style.display = 'block';
      document.getElementById('shortest-distance').textContent = formatDistance(summary.shortest_distance_m);
      document.getElementById('shortest-cost').textContent = formatMinutes(summary.shortest_cost_s);
      document.getElementById('fastest-distance').textContent = formatDistance(summary.fastest_distance_m);
      document.getElementById('fastest-cost').textContent = formatMinutes(summary.fastest_cost_s);
      document.getElementById('origin-snap').textContent = formatDistance(summary.origin_snap_distance_m);
      document.getElementById('dest-snap').textContent = formatDistance(summary.dest_snap_distance_m);
    }

    function drawConnector(a, b) {
      const line = L.polyline([
        [a.lat, a.lon],
        [b.lat, b.lon]
      ], {
        color: '#7a8a9a',
        weight: 2,
        dashArray: '6, 6',
        opacity: 0.9
      }).addTo(map);
      routeState.connectors.push(line);
    }

    async function loadBoundary() {
      const resp = await fetch('/api/shenzhen-boundary');
      const data = await resp.json();
      if (data.error) { return; }
      routeState.boundary = L.geoJSON(data, {
        style: {
          color: '#123b7a',
          weight: 2,
          opacity: 0.9,
          fillOpacity: 0.02
        }
      }).addTo(map);
    }

    map.on('click', async function(e) {
      if (routeState.points.length === 2) {
        resetRoute();
      }

      const point = { lat: e.latlng.lat, lon: e.latlng.lng };
      routeState.points.push(point);
      const label = routeState.points.length === 1 ? '起点' : '终点';
      const marker = L.marker([point.lat, point.lon]).addTo(map).bindTooltip(label).openTooltip();
      routeState.markers.push(marker);

      if (routeState.points.length === 1) {
        setStatus('已选择起点，点击地图选择终点。');
        return;
      }

      clearRouteLayers();
      setStatus('正在计算路线...');

      try {
        routeState.requestSeq += 1;
        const currentRequestSeq = routeState.requestSeq;
        routeState.activeRequestSeq = currentRequestSeq;

        const response = await fetch('/api/routes', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            origin: routeState.points[0],
            destination: routeState.points[1]
          })
        });
        const result = await response.json();

        if (currentRequestSeq !== routeState.activeRequestSeq) {
          return;
        }

        if (!response.ok || result.error) {
          setStatus(result.error || '路线计算失败。', true);
          return;
        }

        routeState.shortest = L.geoJSON(result.shortest, {
          style: { color: 'blue', weight: 5, opacity: 0.8 }
        }).addTo(map);
        routeState.fastest = L.geoJSON(result.fastest, {
          style: { color: 'green', weight: 5, opacity: 0.8 }
        }).addTo(map);

        drawConnector(routeState.points[0], result.connectors.origin_snap);
        drawConnector(routeState.points[1], result.connectors.destination_snap);

        updateSummary(result.summary);
        const bounds = routeState.shortest.getBounds().extend(routeState.fastest.getBounds());
        map.fitBounds(bounds.pad(0.15));
        setStatus('路线计算完成。');
      } catch (err) {
        setStatus('请求失败：' + err.message, true);
      }
    });

    loadBoundary();
  </script>
</body>
</html>
"""


ORDER_COMPARE_TEMPLATE = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>历史订单三路线对比</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
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
    body { margin: 0; height: 100vh; overflow: hidden; font-family: "Microsoft YaHei", "PingFang SC", sans-serif; color: var(--text); background: linear-gradient(135deg, #eef4fb 0%, #f8fbff 100%); }
    .layout { display: grid; grid-template-columns: 390px 1fr; height: 100vh; }
    .top-nav { display: flex; gap: 10px; margin-bottom: 18px; flex-wrap: wrap; }
    .top-nav a { text-decoration: none; padding: 8px 12px; border-radius: 999px; font-size: 13px; border: 1px solid var(--line); color: var(--text); background: #fff; }
    .top-nav a.active { background: var(--accent); color: #fff; border-color: var(--accent); }
    .sidebar { padding: 24px; height: 100vh; background: rgba(255,255,255,.96); border-right: 1px solid var(--line); overflow-y: auto; }
    .main { padding: 20px; height: 100vh; overflow: hidden; }
    h1 { font-size: 24px; margin: 0 0 8px; }
    .subtitle { font-size: 14px; color: var(--muted); margin: 0 0 20px; line-height: 1.6; }
    .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 16px; margin-bottom: 16px; box-shadow: 0 10px 30px rgba(31, 60, 96, 0.06); }
    .panel h2 { margin: 0 0 12px; font-size: 16px; }
    .field { margin-bottom: 12px; }
    label { display: block; margin-bottom: 6px; font-size: 13px; color: var(--muted); }
    input, select { width: 100%; padding: 10px 12px; border-radius: 10px; border: 1px solid var(--line); font-size: 14px; background: #fff; }
    button { border: 0; border-radius: 10px; padding: 11px 16px; font-size: 14px; cursor: pointer; background: var(--accent); color: #fff; }
    button:hover { background: var(--accent-dark); }
    .tips { font-size: 13px; color: var(--muted); line-height: 1.8; }
    .status { margin-bottom: 12px; padding: 12px 14px; border-radius: 10px; font-size: 14px; background: #edf7ed; color: #1e6b34; border: 1px solid #cde7d1; }
    .status.error { background: #fdecec; color: var(--danger); border: 1px solid #f7c5c0; }
    .results { max-height: 260px; overflow-y: auto; border: 1px solid var(--line); border-radius: 10px; }
    .result-item { padding: 10px 12px; border-bottom: 1px solid var(--line); cursor: pointer; background: #fff; }
    .result-item:hover { background: #f7fbff; }
    .result-item.active { background: #e9f3ff; }
    .result-title { font-size: 13px; font-weight: 600; }
    .result-meta { margin-top: 4px; font-size: 12px; color: var(--muted); line-height: 1.6; }
    .summary-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .metric { border: 1px solid var(--line); border-radius: 10px; padding: 10px 12px; background: #fafcff; }
    .metric .label { font-size: 12px; color: var(--muted); }
    .metric .value { font-size: 16px; margin-top: 4px; }
    #compare-map { width: 100%; height: calc(100vh - 40px); background: #fff; border: 1px solid var(--line); border-radius: 16px; overflow: hidden; box-shadow: 0 10px 30px rgba(31, 60, 96, 0.06); }
    .legend-row { display:flex; align-items:center; gap:8px; margin-top:8px; font-size:13px; }
    .legend-line { width:30px; height:4px; border-radius:99px; }
    @media (max-width: 1100px) { .layout { grid-template-columns: 1fr; } #compare-map { height: 72vh; } }
  </style>
</head>
<body>
  <div class="layout">
    <aside class="sidebar">
      <div class="top-nav">
        <a href="/">地图查询</a>
        <a href="/analysis">热力图分析</a>
        <a href="/routes">路线规划</a>
        <a href="/orders" class="active">订单对比</a>
      </div>
      <h1>历史订单三路线对比</h1>
      <p class="subtitle">直接读取阶段 07 的校正 OD 缓存和校正轨迹缓存，对同一订单叠加展示历史实际路线、最短距离路线、基准最快路线。</p>

      <div class="panel">
        <h2>订单筛选</h2>
        <div class="field">
          <label for="vehicle-id">车辆 ID</label>
          <input id="vehicle-id" value="22223" placeholder="例如 22223，可为空">
        </div>
        <div class="field">
          <label for="date-filter">日期</label>
          <input id="date-filter" value="2013-10-22" placeholder="2013-10-22">
        </div>
        <div class="field">
          <label for="start-filter">开始时间下限</label>
          <input id="start-filter" value="2013-10-22 08:00:00" placeholder="2013-10-22 08:00:00，可为空">
        </div>
        <div class="field">
          <label for="end-filter">开始时间上限</label>
          <input id="end-filter" value="2013-10-22 10:00:00" placeholder="2013-10-22 10:00:00，可为空">
        </div>
        <div class="field">
          <label for="limit-filter">返回数量</label>
          <input id="limit-filter" value="20">
        </div>
        <button type="button" onclick="loadOrders()">查询订单</button>
      </div>

      <div id="orders-status" class="status">点击“查询订单”加载候选订单。</div>

      <div class="panel">
        <h2>候选订单</h2>
        <div id="orders-empty" class="tips">当前还没有查询结果。</div>
        <div id="orders-list" class="results" style="display:none;"></div>
      </div>

      <div class="panel">
        <h2>路线图例</h2>
        <div class="legend-row"><span class="legend-line" style="background:blue;"></span><span>最短距离路线</span></div>
        <div class="legend-row"><span class="legend-line" style="background:green;"></span><span>基准最快路线</span></div>
        <div class="legend-row"><span class="legend-line" style="background:red;"></span><span>历史实际路线</span></div>
        <div class="legend-row"><span class="legend-line" style="background:#123b7a;"></span><span>深圳行政边界</span></div>
      </div>

      <div class="panel">
        <h2>指标摘要</h2>
        <div id="compare-empty" class="tips">选择订单后显示三路线对比指标。</div>
        <div id="compare-summary" style="display:none;">
          <div class="summary-grid">
            <div class="metric"><div class="label">实际距离</div><div class="value" id="m-actual-distance">-</div></div>
            <div class="metric"><div class="label">实际耗时</div><div class="value" id="m-actual-duration">-</div></div>
            <div class="metric"><div class="label">最短距离</div><div class="value" id="m-shortest-distance">-</div></div>
            <div class="metric"><div class="label">最快路线距离</div><div class="value" id="m-fastest-distance">-</div></div>
            <div class="metric"><div class="label">距离差</div><div class="value" id="m-gap">-</div></div>
            <div class="metric"><div class="label">绕行比例</div><div class="value" id="m-detour">-</div></div>
            <div class="metric"><div class="label">与最短路线重合率</div><div class="value" id="m-shortest-overlap">-</div></div>
            <div class="metric"><div class="label">与最快路线重合率</div><div class="value" id="m-fastest-overlap">-</div></div>
          </div>
        </div>
      </div>
    </aside>

    <main class="main">
      <div id="compare-map"></div>
    </main>
  </div>

  <script>
    const compareMap = L.map('compare-map').setView([22.58, 114.08], 11);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { attribution: '&copy; OpenStreetMap' }).addTo(compareMap);

    const compareState = {
      boundary: null,
      orderItems: [],
      activeOrderIndex: null,
      shortestLayer: null,
      fastestLayer: null,
      actualLayer: null,
      endpointMarkers: [],
      requestSeq: 0,
    };

    function compareStatus(text, isError=false) {
      const box = document.getElementById('orders-status');
      box.textContent = text;
      box.className = isError ? 'status error' : 'status';
    }

    function fmtDistance(m) {
      if (m === null || m === undefined) return '不可用';
      return m >= 1000 ? (m / 1000).toFixed(2) + ' km' : Math.round(m) + ' m';
    }

    function fmtDuration(s) {
      if (s === null || s === undefined) return '不可用';
      return (s / 60).toFixed(1) + ' min';
    }

    function fmtPercent(v) {
      if (v === null || v === undefined) return '不可用';
      return (v * 100).toFixed(1) + '%';
    }

    function clearCompareLayers() {
      if (compareState.shortestLayer) { compareMap.removeLayer(compareState.shortestLayer); compareState.shortestLayer = null; }
      if (compareState.fastestLayer) { compareMap.removeLayer(compareState.fastestLayer); compareState.fastestLayer = null; }
      if (compareState.actualLayer) { compareMap.removeLayer(compareState.actualLayer); compareState.actualLayer = null; }
      compareState.endpointMarkers.forEach(m => compareMap.removeLayer(m));
      compareState.endpointMarkers = [];
    }

    async function loadBoundaryCompare() {
      const resp = await fetch('/api/shenzhen-boundary');
      const data = await resp.json();
      if (data.error) return;
      compareState.boundary = L.geoJSON(data, {
        style: { color: '#123b7a', weight: 2, opacity: 0.9, fillOpacity: 0.02 }
      }).addTo(compareMap);
    }

    async function loadOrders() {
      compareStatus('正在查询订单...');
      const params = new URLSearchParams();
      const vehicleId = document.getElementById('vehicle-id').value.trim();
      const dateFilter = document.getElementById('date-filter').value.trim();
      const startFilter = document.getElementById('start-filter').value.trim();
      const endFilter = document.getElementById('end-filter').value.trim();
      const limitFilter = document.getElementById('limit-filter').value.trim();
      if (vehicleId) params.set('vehicle_id', vehicleId);
      if (dateFilter) params.set('date', dateFilter);
      if (startFilter) params.set('start_time', startFilter);
      if (endFilter) params.set('end_time', endFilter);
      if (limitFilter) params.set('limit', limitFilter);

      const resp = await fetch('/api/orders?' + params.toString());
      const data = await resp.json();
      if (!resp.ok || data.error) {
        compareStatus(data.error || '订单查询失败', true);
        return;
      }

      const list = document.getElementById('orders-list');
      list.innerHTML = '';
      compareState.orderItems = data.orders || [];
      compareState.activeOrderIndex = null;

      if (!compareState.orderItems.length) {
        document.getElementById('orders-empty').style.display = 'block';
        list.style.display = 'none';
        compareStatus('没有匹配到订单。');
        return;
      }

      document.getElementById('orders-empty').style.display = 'none';
      list.style.display = 'block';

      compareState.orderItems.forEach(item => {
        const div = document.createElement('div');
        div.className = 'result-item';
        div.dataset.orderIndex = item.order_index;
        div.innerHTML =
          '<div class="result-title">订单 #' + item.order_index + ' / 车辆 ' + item.id + '</div>' +
          '<div class="result-meta">上车: ' + item.pickup_time + '<br>下车: ' + item.dropoff_time + '<br>起终点节点: ' + item.pickup_node + ' -> ' + item.dropoff_node + '</div>';
        div.addEventListener('click', () => loadOrderCompare(item.order_index));
        list.appendChild(div);
      });

      compareStatus('已加载 ' + compareState.orderItems.length + ' 条候选订单。点击任意订单查看三路线对比。');
    }

    async function loadOrderCompare(orderIndex) {
      clearCompareLayers();
      compareState.requestSeq += 1;
      const currentRequest = compareState.requestSeq;
      compareStatus('正在计算订单三路线对比...');

      document.querySelectorAll('.result-item').forEach(el => {
        el.classList.toggle('active', String(orderIndex) === el.dataset.orderIndex);
      });

      const resp = await fetch('/api/orders/' + orderIndex + '/compare');
      const data = await resp.json();

      if (currentRequest !== compareState.requestSeq) {
        return;
      }

      if (!resp.ok || data.error) {
        compareStatus(data.error || '订单对比失败', true);
        return;
      }

      const order = data.order;
      const metrics = data.metrics;

      if (data.shortest && data.shortest.available) {
        compareState.shortestLayer = L.geoJSON(data.shortest.geojson, {
          style: { color: 'blue', weight: 5, opacity: 0.75 }
        }).addTo(compareMap);
      }

      if (data.fastest && data.fastest.available) {
        compareState.fastestLayer = L.geoJSON(data.fastest.geojson, {
          style: { color: 'green', weight: 5, opacity: 0.75 }
        }).addTo(compareMap);
      }

      compareState.actualLayer = L.geoJSON(data.actual.geojson, {
        style: { color: 'red', weight: 5, opacity: 0.85 }
      }).addTo(compareMap);

      const pickupMarker = L.marker([order.pickup_matched_lat, order.pickup_matched_lon]).addTo(compareMap).bindTooltip('上车点');
      const dropoffMarker = L.marker([order.dropoff_matched_lat, order.dropoff_matched_lon]).addTo(compareMap).bindTooltip('下车点');
      compareState.endpointMarkers = [pickupMarker, dropoffMarker];

      const allLayers = [compareState.actualLayer, compareState.shortestLayer, compareState.fastestLayer].filter(Boolean);
      if (allLayers.length) {
        let bounds = allLayers[0].getBounds();
        allLayers.slice(1).forEach(layer => { bounds = bounds.extend(layer.getBounds()); });
        compareMap.fitBounds(bounds.pad(0.12));
      }

      document.getElementById('compare-empty').style.display = 'none';
      document.getElementById('compare-summary').style.display = 'block';
      document.getElementById('m-actual-distance').textContent = fmtDistance(metrics.actual_distance_m);
      document.getElementById('m-actual-duration').textContent = fmtDuration(metrics.actual_duration_s);
      document.getElementById('m-shortest-distance').textContent = fmtDistance(metrics.shortest_distance_m);
      document.getElementById('m-fastest-distance').textContent = fmtDistance(metrics.fastest_distance_m);
      document.getElementById('m-gap').textContent = fmtDistance(metrics.distance_gap_m);
      document.getElementById('m-detour').textContent = fmtPercent(metrics.detour_ratio);
      document.getElementById('m-shortest-overlap').textContent = fmtPercent(metrics.shortest_overlap);
      document.getElementById('m-fastest-overlap').textContent = fmtPercent(metrics.fastest_overlap);

      compareStatus('订单对比完成。');
    }

    loadBoundaryCompare();
  </script>
</body>
</html>
"""


def _haversine_meters(lat1, lon1, lat2, lon2):
    from math import radians, sin, cos, sqrt, atan2
    r = 6371000.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return r * 2 * atan2(sqrt(a), sqrt(1 - a))


def _load_stage7_graph():
    global _STAGE7_GRAPH
    if _STAGE7_GRAPH is None:
        if not os.path.exists(STAGE7_GRAPH_PATH):
            raise FileNotFoundError(
                f'未找到带 route_cost 的路网缓存: {STAGE7_GRAPH_PATH}，请先运行 python code/stage7_route.py speed'
            )
        with open(STAGE7_GRAPH_PATH, 'rb') as f:
            _STAGE7_GRAPH = pickle.load(f)
    return _STAGE7_GRAPH


def _load_shenzhen_boundary():
    global _SHENZHEN_BOUNDARY
    if _SHENZHEN_BOUNDARY is None:
        with open(SHENZHEN_BOUNDARY_PATH, 'r', encoding='utf-8') as f:
            _SHENZHEN_BOUNDARY = json.load(f)
    return _SHENZHEN_BOUNDARY


def _load_stage7_od():
    global _STAGE7_OD
    if _STAGE7_OD is None:
        if not os.path.exists(STAGE7_OD_PATH):
            raise FileNotFoundError(
                f'未找到阶段07校正OD缓存: {STAGE7_OD_PATH}，请先运行 python code/stage7_route.py'
            )
        od = pd.read_parquet(STAGE7_OD_PATH)
        od['pickup_time'] = pd.to_datetime(od['pickup_time'])
        od['dropoff_time'] = pd.to_datetime(od['dropoff_time'])
        _STAGE7_OD = od
    return _STAGE7_OD


def _load_stage7_track(order_id=None, columns=None):
    if not os.path.exists(STAGE7_TRACK_PATH):
        raise FileNotFoundError(
            f'未找到阶段07校正轨迹缓存: {STAGE7_TRACK_PATH}，请先生成 matched_trajectory_stage7.parquet'
        )
    cols = columns or ['id', 'time', 'status', 'matched_lon', 'matched_lat', 'edge_u', 'edge_v', 'edge_key']
    track = pd.read_parquet(STAGE7_TRACK_PATH, columns=cols)
    track['time'] = pd.to_datetime(track['time'])
    if order_id is not None:
        track = track[track['id'] == int(order_id)].copy()
    return track


def _load_road_correction_entry(vehicle_id):
    cache_path = os.path.join(ROAD_CORRECTION_CACHE_DIR, f'{int(vehicle_id)}.pkl')
    if not os.path.exists(cache_path):
        return None
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)
    entries = data.get('entries') or {}
    if not entries:
        return None
    return next(iter(entries.values()))


def _route_gdf_from_nodes(g, route_nodes, weight_name):
    rows = []
    index = []
    prev_geom = None

    for u, v in zip(route_nodes[:-1], route_nodes[1:]):
        edge_options = g.get_edge_data(u, v, default={})
        if not edge_options:
            continue

        best_key = None
        best_data = None
        best_score = None
        for key, data in edge_options.items():
            weight = float(data.get(weight_name, data.get('length', 1.0)) or 1.0)
            geom = _edge_linestring(g, u, v, key, data)
            continuity_penalty = 0.0
            if prev_geom is not None:
                prev_end = Point(prev_geom.coords[-1])
                curr_start = Point(geom.coords[0])
                continuity_penalty = prev_end.distance(curr_start) * 100000.0
            score = weight + continuity_penalty
            if best_score is None or score < best_score:
                best_key = key
                best_data = data
                best_score = score

        if best_data is None:
            continue
        row = dict(best_data)
        row['u'] = u
        row['v'] = v
        row['key'] = best_key
        row['geometry'] = _edge_linestring(g, u, v, best_key, best_data)
        rows.append(row)
        index.append((u, v, best_key))
        prev_geom = row['geometry']

    if not rows:
        return ox.routing.route_to_gdf(g, route_nodes, weight=weight_name)

    import geopandas as gpd
    gdf = gpd.GeoDataFrame(rows, geometry='geometry', crs=g.graph.get('crs'))
    gdf.index = pd.MultiIndex.from_tuples(index, names=['u', 'v', 'key'])
    return gdf


def _edge_set_from_gdf(route_gdf):
    return set(route_gdf.index.to_list())


def _actual_edges_from_points(actual_points):
    edge_columns = ['edge_u', 'edge_v', 'edge_key']
    actual_edges = actual_points.dropna(subset=edge_columns)[edge_columns].copy()
    if actual_edges.empty:
        return actual_edges.reset_index(drop=True)
    changed = actual_edges.ne(actual_edges.shift()).any(axis=1)
    return actual_edges.loc[changed].reset_index(drop=True)


def _actual_route_geometry(g, actual_edges):
    actual_features = []
    actual_distance = 0.0
    for row in actual_edges.itertuples(index=False):
        edge_data = g.get_edge_data(int(row.edge_u), int(row.edge_v), int(row.edge_key))
        if edge_data is None:
            continue
        if isinstance(edge_data, dict) and 'length' not in edge_data and int(row.edge_key) in edge_data:
            edge_data = edge_data[int(row.edge_key)]
        length = float(edge_data.get('length', 0.0))
        actual_distance += length
        geom = edge_data.get('geometry')
        if geom is None:
            geom = LineString([
                (float(g.nodes[int(row.edge_u)]['x']), float(g.nodes[int(row.edge_u)]['y'])),
                (float(g.nodes[int(row.edge_v)]['x']), float(g.nodes[int(row.edge_v)]['y'])),
            ])
        actual_features.append(_feature_from_geom(geom, {
            'kind': 'actual',
            'edge_u': int(row.edge_u),
            'edge_v': int(row.edge_v),
            'edge_key': int(row.edge_key),
            'length': length,
        }))
    return _feature_collection(actual_features), actual_distance


def _actual_matched_points_geojson(actual_points):
    points = actual_points[['matched_lat', 'matched_lon']].dropna().copy()
    if len(points) < 2:
        return _feature_collection([])

    coords = [(float(row.matched_lon), float(row.matched_lat)) for row in points.itertuples(index=False)]
    line = LineString(coords)
    return _feature_collection([
        _feature_from_geom(line, {
            'kind': 'actual_matched_points',
            'point_count': int(len(points)),
        })
    ])


def _actual_corrected_geojson_from_cache(vehicle_id, pickup_time, dropoff_time):
    entry = _load_road_correction_entry(vehicle_id)
    if entry is None:
        return None

    timed_pieces = entry.get('timed_pieces') or []
    if not timed_pieces:
        return None

    pickup_time = pd.to_datetime(pickup_time)
    dropoff_time = pd.to_datetime(dropoff_time)
    features = []
    current_segment = []

    def flush_segment():
        nonlocal current_segment
        if len(current_segment) >= 2:
            line = LineString([(float(lon), float(lat)) for lat, lon in current_segment])
            features.append(_feature_from_geom(line, {
                'kind': 'actual_corrected_segment',
                'point_count': int(len(current_segment)),
            }))
        current_segment = []

    for piece in timed_pieces:
        piece_start = pd.to_datetime(piece.get('start_time'))
        piece_end = pd.to_datetime(piece.get('end_time'))
        if piece_end < pickup_time or piece_start > dropoff_time:
            continue
        coords = piece.get('coords') or []
        if piece.get('break_before'):
            flush_segment()
        for coord in coords:
            if not current_segment or current_segment[-1] != coord:
                current_segment.append(coord)

    flush_segment()

    if not features:
        return None
    return _feature_collection(features)


def _matched_points_distance_m(actual_points):
    points = actual_points[['matched_lat', 'matched_lon']].dropna().reset_index(drop=True)
    if len(points) < 2:
        return 0.0

    total = 0.0
    prev_lat = float(points.loc[0, 'matched_lat'])
    prev_lon = float(points.loc[0, 'matched_lon'])
    for i in range(1, len(points)):
        curr_lat = float(points.loc[i, 'matched_lat'])
        curr_lon = float(points.loc[i, 'matched_lon'])
        total += _haversine_meters(prev_lat, prev_lon, curr_lat, curr_lon)
        prev_lat = curr_lat
        prev_lon = curr_lon
    return float(total)


def _geojson_length_overlap_ratio(actual_geojson, route_gdf, buffer_m=15.0):
    if actual_geojson is None or route_gdf is None or route_gdf.empty:
        return 0.0

    actual_features = actual_geojson.get('features') or []
    if not actual_features:
        return 0.0

    actual_gdf = gpd.GeoDataFrame.from_features(actual_features, crs='EPSG:4326')
    if actual_gdf.empty:
        return 0.0

    route_metric = route_gdf.to_crs(epsg=3857)
    actual_metric = actual_gdf.to_crs(epsg=3857)

    route_union = unary_union(route_metric.geometry.tolist())
    if route_union.is_empty:
        return 0.0

    actual_buffer = unary_union(actual_metric.geometry.buffer(buffer_m).tolist())
    if actual_buffer.is_empty:
        return 0.0

    overlap = route_union.intersection(actual_buffer)
    route_length = float(route_union.length)
    if route_length <= 0:
        return 0.0
    return float(max(0.0, min(1.0, overlap.length / route_length)))


def _order_candidates(limit=100, vehicle_id=None, date_str=None, start_time=None, end_time=None):
    od = _load_stage7_od().copy()
    od = od[
        od['pickup_node'].notna()
        & od['dropoff_node'].notna()
        & (od['dropoff_time'] > od['pickup_time'])
    ].copy()
    if vehicle_id:
        od = od[od['id'] == int(vehicle_id)]
    if date_str:
        od = od[od['pickup_time'].dt.strftime('%Y-%m-%d') == date_str]
    if start_time:
        od = od[od['pickup_time'] >= pd.to_datetime(start_time)]
    if end_time:
        od = od[od['pickup_time'] <= pd.to_datetime(end_time)]
    od = od.sort_values(['pickup_time', 'id']).head(limit).copy()
    od = od.reset_index().rename(columns={'index': 'order_index'})
    return od


def _compute_stage8_order_comparison(order_index):
    od = _load_stage7_od().copy()
    od = od.reset_index().rename(columns={'index': 'order_index'})
    selected = od.loc[od['order_index'] == int(order_index)]
    if selected.empty:
        raise ValueError('未找到指定订单')
    order = selected.iloc[0]

    if pd.isna(order['pickup_node']) or pd.isna(order['dropoff_node']):
        raise ValueError('该订单缺少校正起终点节点')
    if order['dropoff_time'] <= order['pickup_time']:
        raise ValueError('该订单上下车时间无效')

    g = _load_stage7_graph()
    track = _load_stage7_track(
        order_id=int(order['id']),
        columns=['id', 'time', 'status', 'matched_lon', 'matched_lat', 'edge_u', 'edge_v', 'edge_key']
    )

    actual_points = track[
        track['time'].between(order['pickup_time'], order['dropoff_time'], inclusive='both')
    ].sort_values('time').reset_index(drop=True)

    if len(actual_points) < 2:
        raise ValueError('该订单在校正轨迹缓存中没有足够的轨迹点')

    actual_edges = _actual_edges_from_points(actual_points)
    if actual_edges.empty:
        raise ValueError('该订单缺少有效道路边序列，无法恢复实际路线')

    actual_edge_geojson, actual_edge_distance = _actual_route_geometry(g, actual_edges)
    actual_display_geojson = _actual_corrected_geojson_from_cache(
        int(order['id']), order['pickup_time'], order['dropoff_time']
    )
    if actual_display_geojson is None:
        actual_display_geojson = _actual_matched_points_geojson(actual_points)
    actual_matched_distance = _matched_points_distance_m(actual_points)
    actual_distance = max(float(actual_edge_distance), float(actual_matched_distance))
    actual_duration = float((order['dropoff_time'] - order['pickup_time']).total_seconds())

    origin_node = int(order['pickup_node'])
    destination_node = int(order['dropoff_node'])

    try:
        shortest_route = nx.shortest_path(g, origin_node, destination_node, weight='length')
        shortest_gdf = _route_gdf_from_nodes(g, shortest_route, 'length')
    except (nx.NodeNotFound, nx.NetworkXNoPath):
        shortest_gdf = None

    try:
        fastest_route = nx.shortest_path(g, origin_node, destination_node, weight='route_cost')
        fastest_gdf = _route_gdf_from_nodes(g, fastest_route, 'route_cost')
    except (nx.NodeNotFound, nx.NetworkXNoPath):
        fastest_gdf = None

    shortest_distance = float(shortest_gdf['length'].sum()) if shortest_gdf is not None else None
    fastest_distance = float(fastest_gdf['length'].sum()) if fastest_gdf is not None else None
    shortest_cost = float(shortest_gdf['route_cost'].sum()) if shortest_gdf is not None else None
    fastest_cost = float(fastest_gdf['route_cost'].sum()) if fastest_gdf is not None else None

    distance_gap = None if shortest_distance in (None, 0) else float(actual_distance - shortest_distance)
    detour_ratio = None if shortest_distance in (None, 0) else float((actual_distance - shortest_distance) / shortest_distance)

    shortest_overlap = _geojson_length_overlap_ratio(actual_display_geojson, shortest_gdf) if shortest_gdf is not None else 0.0
    fastest_overlap = _geojson_length_overlap_ratio(actual_display_geojson, fastest_gdf) if fastest_gdf is not None else 0.0

    return {
        'order': {
            'order_index': int(order['order_index']),
            'id': int(order['id']),
            'pickup_time': str(order['pickup_time']),
            'dropoff_time': str(order['dropoff_time']),
            'pickup_matched_lon': float(order['pickup_matched_lon']),
            'pickup_matched_lat': float(order['pickup_matched_lat']),
            'dropoff_matched_lon': float(order['dropoff_matched_lon']),
            'dropoff_matched_lat': float(order['dropoff_matched_lat']),
            'pickup_node': int(order['pickup_node']),
            'dropoff_node': int(order['dropoff_node']),
        },
        'actual': {
            'geojson': actual_display_geojson,
            'edge_geojson': actual_edge_geojson,
            'distance_m': float(actual_distance),
            'edge_distance_m': float(actual_edge_distance),
            'matched_distance_m': float(actual_matched_distance),
            'duration_s': actual_duration,
            'point_count': int(len(actual_points)),
            'edge_count': int(len(actual_edges)),
        },
        'shortest': {
            'geojson': json.loads(shortest_gdf.to_json()) if shortest_gdf is not None else None,
            'distance_m': shortest_distance,
            'cost_s': shortest_cost,
            'edge_count': int(len(shortest_gdf)) if shortest_gdf is not None else None,
            'available': shortest_gdf is not None,
        },
        'fastest': {
            'geojson': json.loads(fastest_gdf.to_json()) if fastest_gdf is not None else None,
            'distance_m': fastest_distance,
            'cost_s': fastest_cost,
            'edge_count': int(len(fastest_gdf)) if fastest_gdf is not None else None,
            'available': fastest_gdf is not None,
        },
        'metrics': {
            'actual_distance_m': float(actual_distance),
            'actual_edge_distance_m': float(actual_edge_distance),
            'actual_matched_distance_m': float(actual_matched_distance),
            'shortest_distance_m': shortest_distance,
            'fastest_distance_m': fastest_distance,
            'actual_duration_s': actual_duration,
            'distance_gap_m': distance_gap,
            'detour_ratio': detour_ratio,
            'shortest_overlap': shortest_overlap,
            'fastest_overlap': fastest_overlap,
        },
    }


def _edge_linestring(g, u, v, key, data):
    geom = data.get('geometry')
    if geom is not None:
        return geom
    return LineString([
        (float(g.nodes[u]['x']), float(g.nodes[u]['y'])),
        (float(g.nodes[v]['x']), float(g.nodes[v]['y'])),
    ])


def _safe_speed_kph(data):
    speed = data.get('baseline_speed_kph')
    if speed is None or speed <= 0:
        speed = 30.0
    return max(5.0, min(float(speed), 90.0))


def _segment_cost_seconds(length_m, speed_kph):
    return float(length_m) / max(float(speed_kph), 1.0) * 3.6


def _coords_latlon(geom):
    return [[float(lat), float(lon)] for lon, lat in geom.coords]


def _feature_from_geom(geom, properties=None):
    return {
        'type': 'Feature',
        'geometry': mapping(geom),
        'properties': properties or {},
    }


def _feature_collection(features):
    return {'type': 'FeatureCollection', 'features': features}


def _snap_point_to_edge(g, lon, lat):
    u, v, key = ox.distance.nearest_edges(g, lon, lat)
    data = g.edges[u, v, key]
    geom = _edge_linestring(g, u, v, key, data)
    point = Point(float(lon), float(lat))
    progress = geom.project(point, normalized=True)
    snap_point = geom.interpolate(progress, normalized=True)
    total_len = max(float(geom.length), 1e-9)
    forward_ratio = max(0.0, min(1.0, progress))
    backward_ratio = 1.0 - forward_ratio
    speed = _safe_speed_kph(data)

    candidates = []

    forward_geom = substring(geom, progress, 1.0, normalized=True)
    if forward_geom and not forward_geom.is_empty:
        candidates.append({
            'node': int(v),
            'entry_cost_s': _segment_cost_seconds(float(total_len * forward_ratio), speed),
            'entry_length_m': float(total_len * forward_ratio),
            'entry_geom': forward_geom,
        })

    reverse_data = g.get_edge_data(v, u, default={})
    reverse_key = next(iter(reverse_data.keys()), None) if reverse_data else None
    if reverse_key is not None:
        reverse_geom = reverse_data[reverse_key].get('geometry')
        if reverse_geom is None:
            reverse_geom = LineString([
                (float(g.nodes[v]['x']), float(g.nodes[v]['y'])),
                (float(g.nodes[u]['x']), float(g.nodes[u]['y'])),
            ])
        reverse_progress = reverse_geom.project(snap_point, normalized=True)
        backward_geom = substring(reverse_geom, reverse_progress, 1.0, normalized=True)
        reverse_speed = _safe_speed_kph(reverse_data[reverse_key])
        if backward_geom and not backward_geom.is_empty:
            candidates.append({
                'node': int(u),
                'entry_cost_s': _segment_cost_seconds(float(total_len * backward_ratio), reverse_speed),
                'entry_length_m': float(total_len * backward_ratio),
                'entry_geom': backward_geom,
            })

    return {
        'edge': (int(u), int(v), int(key)),
        'snap_lon': float(snap_point.x),
        'snap_lat': float(snap_point.y),
        'click_lon': float(lon),
        'click_lat': float(lat),
        'candidates': candidates,
    }


def _destination_candidates(g, lon, lat):
    snapped = _snap_point_to_edge(g, lon, lat)
    u, v, key = snapped['edge']
    data = g.edges[u, v, key]
    geom = _edge_linestring(g, u, v, key, data)
    snap_point = Point(snapped['snap_lon'], snapped['snap_lat'])
    progress = geom.project(snap_point, normalized=True)
    total_len = max(float(geom.length), 1e-9)
    speed = _safe_speed_kph(data)

    candidates = []

    forward_arrival = substring(geom, 0.0, progress, normalized=True)
    if forward_arrival and not forward_arrival.is_empty:
        candidates.append({
            'node': int(u),
            'exit_cost_s': _segment_cost_seconds(float(total_len * progress), speed),
            'exit_length_m': float(total_len * progress),
            'exit_geom': forward_arrival,
        })

    reverse_data = g.get_edge_data(v, u, default={})
    reverse_key = next(iter(reverse_data.keys()), None) if reverse_data else None
    if reverse_key is not None:
        reverse_geom = reverse_data[reverse_key].get('geometry')
        if reverse_geom is None:
            reverse_geom = LineString([
                (float(g.nodes[v]['x']), float(g.nodes[v]['y'])),
                (float(g.nodes[u]['x']), float(g.nodes[u]['y'])),
            ])
        reverse_progress = reverse_geom.project(snap_point, normalized=True)
        reverse_total_len = max(float(reverse_geom.length), 1e-9)
        reverse_speed = _safe_speed_kph(reverse_data[reverse_key])
        reverse_arrival = substring(reverse_geom, 0.0, reverse_progress, normalized=True)
        if reverse_arrival and not reverse_arrival.is_empty:
            candidates.append({
                'node': int(v),
                'exit_cost_s': _segment_cost_seconds(float(reverse_total_len * reverse_progress), reverse_speed),
                'exit_length_m': float(reverse_total_len * reverse_progress),
                'exit_geom': reverse_arrival,
            })

    snapped['candidates'] = candidates
    return snapped


def _direct_same_edge_solution(origin_snap, dest_snap, weight_name):
    if origin_snap['edge'] != dest_snap['edge']:
        return None
    if not origin_snap['candidates'] or not dest_snap['candidates']:
        return None

    g = _load_stage7_graph()
    u, v, key = origin_snap['edge']
    data = g.edges[u, v, key]
    geom = _edge_linestring(g, u, v, key, data)

    origin_point = Point(origin_snap['snap_lon'], origin_snap['snap_lat'])
    dest_point = Point(dest_snap['snap_lon'], dest_snap['snap_lat'])
    origin_progress = geom.project(origin_point, normalized=True)
    dest_progress = geom.project(dest_point, normalized=True)
    total_len = max(float(geom.length), 1e-9)
    speed = _safe_speed_kph(data)

    solutions = []

    if dest_progress >= origin_progress:
        seg = substring(geom, origin_progress, dest_progress, normalized=True)
        if seg and not seg.is_empty:
            seg_len = float(total_len * (dest_progress - origin_progress))
            solutions.append({
                'route_nodes': [],
                'middle_features': [_feature_from_geom(seg, {'kind': 'same_edge'})],
                'distance_m': seg_len,
                'cost_s': _segment_cost_seconds(seg_len, speed),
                'edge_count': 1,
            })

    reverse_data = g.get_edge_data(v, u, default={})
    reverse_key = next(iter(reverse_data.keys()), None) if reverse_data else None
    if reverse_key is not None:
        reverse_geom = reverse_data[reverse_key].get('geometry')
        if reverse_geom is None:
            reverse_geom = LineString([
                (float(g.nodes[v]['x']), float(g.nodes[v]['y'])),
                (float(g.nodes[u]['x']), float(g.nodes[u]['y'])),
            ])
        reverse_speed = _safe_speed_kph(reverse_data[reverse_key])
        reverse_origin_progress = reverse_geom.project(origin_point, normalized=True)
        reverse_dest_progress = reverse_geom.project(dest_point, normalized=True)
        reverse_total_len = max(float(reverse_geom.length), 1e-9)
        if reverse_dest_progress >= reverse_origin_progress:
            seg = substring(reverse_geom, reverse_origin_progress, reverse_dest_progress, normalized=True)
            if seg and not seg.is_empty:
                seg_len = float(reverse_total_len * (reverse_dest_progress - reverse_origin_progress))
                solutions.append({
                    'route_nodes': [],
                    'middle_features': [_feature_from_geom(seg, {'kind': 'same_edge_reverse'})],
                    'distance_m': seg_len,
                    'cost_s': _segment_cost_seconds(seg_len, reverse_speed),
                    'edge_count': 1,
                })

    if not solutions:
        return None

    if weight_name == 'length':
        return min(solutions, key=lambda item: item['distance_m'])
    return min(solutions, key=lambda item: item['cost_s'])


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


def _compute_stage7_routes(origin_lon, origin_lat, destination_lon, destination_lat):
    g = _load_stage7_graph()

    origin_snap = _snap_point_to_edge(g, origin_lon, origin_lat)
    dest_snap = _destination_candidates(g, destination_lon, destination_lat)

    if not origin_snap['candidates']:
        raise ValueError('起点附近没有可接入的道路方向')
    if not dest_snap['candidates']:
        raise ValueError('终点附近没有可接入的道路方向')

    def solve(weight_name):
        best = None

        direct = _direct_same_edge_solution(origin_snap, dest_snap, weight_name)
        if direct is not None:
            best = direct

        for o in origin_snap['candidates']:
            for d in dest_snap['candidates']:
                try:
                    route_nodes = nx.shortest_path(g, o['node'], d['node'], weight=weight_name)
                except (nx.NodeNotFound, nx.NetworkXNoPath):
                    continue

                middle_gdf = _route_gdf_from_nodes(g, route_nodes, weight_name)
                middle_distance = float(middle_gdf['length'].sum()) if 'length' in middle_gdf else 0.0
                middle_cost = float(middle_gdf['route_cost'].sum()) if 'route_cost' in middle_gdf else 0.0

                total_distance = o['entry_length_m'] + middle_distance + d['exit_length_m']
                total_cost = o['entry_cost_s'] + middle_cost + d['exit_cost_s']

                candidate = {
                    'route_nodes': route_nodes,
                    'middle_features': json.loads(middle_gdf.to_json())['features'],
                    'distance_m': total_distance,
                    'cost_s': total_cost,
                    'edge_count': max(len(route_nodes) - 1, 0),
                    'origin_entry_geom': o['entry_geom'],
                    'dest_exit_geom': d['exit_geom'],
                }

                key = candidate['distance_m'] if weight_name == 'length' else candidate['cost_s']
                if best is None:
                    best = candidate
                else:
                    best_key = best['distance_m'] if weight_name == 'length' else best['cost_s']
                    if key < best_key:
                        best = candidate

        if best is None:
            raise ValueError('起终点之间无可达路径')

        features = []
        if best.get('origin_entry_geom') is not None:
            features.append(_feature_from_geom(best['origin_entry_geom'], {'kind': 'origin_partial'}))
        features.extend(best['middle_features'])
        if best.get('dest_exit_geom') is not None:
            features.append(_feature_from_geom(best['dest_exit_geom'], {'kind': 'destination_partial'}))

        return {
            'geojson': _feature_collection(features),
            'distance_m': float(best['distance_m']),
            'cost_s': float(best['cost_s']),
            'edge_count': int(best['edge_count']),
        }

    shortest = solve('length')
    fastest = solve('route_cost')

    return {
        'shortest': shortest,
        'fastest': fastest,
        'summary': {
            'shortest_distance_m': shortest['distance_m'],
            'shortest_cost_s': shortest['cost_s'],
            'fastest_distance_m': fastest['distance_m'],
            'fastest_cost_s': fastest['cost_s'],
            'origin_snap_distance_m': float(_haversine_meters(origin_lat, origin_lon, origin_snap['snap_lat'], origin_snap['snap_lon'])),
            'dest_snap_distance_m': float(_haversine_meters(destination_lat, destination_lon, dest_snap['snap_lat'], dest_snap['snap_lon'])),
        },
        'connectors': {
            'origin_snap': {'lat': origin_snap['snap_lat'], 'lon': origin_snap['snap_lon']},
            'destination_snap': {'lat': dest_snap['snap_lat'], 'lon': dest_snap['snap_lon']},
        },
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
                    var coordPanel = document.getElementById('coord-panel');
                    if (coordPanel && coordPanel.style.display === 'none') {{
                        return;
                    }}
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


@app.route('/routes')
def route_page():
    return render_template_string(ROUTE_TEMPLATE, active_page='routes')


@app.route('/orders')
def order_compare_page():
    return render_template_string(ORDER_COMPARE_TEMPLATE, active_page='orders')


@app.route('/maps/<path:filename>')
def serve_map(filename):
    return send_from_directory(MAP_OUTPUT_DIR, filename)


@app.route('/analysis_maps/<path:filename>')
def serve_analysis_map(filename):
    return send_from_directory(ANALYSIS_MAP_DIR, filename)


@app.route('/api/shenzhen-boundary')
def api_shenzhen_boundary():
    try:
        return jsonify(_load_shenzhen_boundary())
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/api/routes', methods=['POST'])
def api_routes():
    try:
        payload = request.get_json(silent=True) or {}
        origin = payload.get('origin') or {}
        destination = payload.get('destination') or {}

        origin_lon = float(origin['lon'])
        origin_lat = float(origin['lat'])
        destination_lon = float(destination['lon'])
        destination_lat = float(destination['lat'])

        if not (22.35 <= origin_lat <= 22.95 and 113.70 <= origin_lon <= 114.75):
            raise ValueError('起点超出深圳路网范围')
        if not (22.35 <= destination_lat <= 22.95 and 113.70 <= destination_lon <= 114.75):
            raise ValueError('终点超出深圳路网范围')

        result = _compute_stage7_routes(
            origin_lon, origin_lat,
            destination_lon, destination_lat,
        )

        return jsonify({
            'shortest': result['shortest']['geojson'],
            'fastest': result['fastest']['geojson'],
            'summary': result['summary'],
            'connectors': result['connectors'],
        })
    except KeyError:
        return jsonify({'error': '请求缺少 origin/destination 坐标字段'}), 400
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/api/orders')
def api_orders():
    try:
        vehicle_id = request.args.get('vehicle_id', '').strip() or None
        date_str = request.args.get('date', '').strip() or None
        start_time = request.args.get('start_time', '').strip() or None
        end_time = request.args.get('end_time', '').strip() or None
        limit = int(request.args.get('limit', '20'))
        limit = max(1, min(limit, 100))

        result_df = _order_candidates(
            limit=limit,
            vehicle_id=vehicle_id,
            date_str=date_str,
            start_time=start_time,
            end_time=end_time,
        )

        records = []
        for row in result_df.itertuples(index=False):
            records.append({
                'order_index': int(row.order_index),
                'id': int(row.id),
                'pickup_time': str(row.pickup_time),
                'dropoff_time': str(row.dropoff_time),
                'pickup_node': int(row.pickup_node),
                'dropoff_node': int(row.dropoff_node),
            })

        return jsonify({'orders': records})
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/api/orders/<int:order_index>/compare')
def api_order_compare(order_index):
    try:
        result = _compute_stage8_order_comparison(order_index)
        return jsonify(result)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


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
