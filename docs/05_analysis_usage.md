# 阶段05 热力图与统计分析 - 使用说明

## 已完成内容

- 车辆位置静态热力图
- 上车点静态热力图
- 上车点动态热力图（15分钟粒度）
- 车辆位置动态热力图（60分钟粒度）
- DBSCAN 上车点聚类结果导出
- 每小时订单数量、载客车辆数量、载客率统计
- 订单时长箱线图
- 短途 / 中途 / 长途订单数量与占比统计
- 车辆全天载客率、总里程、载客里程、空载里程统计

## 运行方式

```bash
python code/data_analysis.py
```

## 输出位置

### 地图文件
- `analysis/maps/01_static_vehicle_heatmap.html`
- `analysis/maps/02_static_pickup_heatmap.html`
- `analysis/maps/03_dynamic_pickup_heatmap_15min.html`
- `analysis/maps/04_dynamic_vehicle_heatmap_60min.html`

### 图表文件
- `analysis/figures/01_hourly_order_occupied_ratio.png`
- `analysis/figures/02_order_duration_boxplot.png`

### 表格文件
- `analysis/tables/hourly_order_occupied_ratio.csv`
- `analysis/tables/pickup_dbscan_clusters.csv`
- `analysis/tables/dynamic_pickup_heatmap_15min.json`
- `analysis/tables/trip_distance_levels.csv`
- `analysis/tables/vehicle_daily_stats.csv`

## 数据来源说明

- 车辆位置热力图：来自 `data/cache/minute/`
- 上车点热力图 / DBSCAN / 订单统计：来自 `data/cache/od/od_cache.csv`
- 车辆全天统计：来自 `data/cache/vehicle/`

## 统计口径

### 动态热力图
- 上车点动态热力图：15分钟聚合
- 车辆位置动态热力图：60分钟聚合
- 动态数据结构：时间片列表，每片为若干 `[lat, lon, weight]`

### DBSCAN 聚类
- `eps = 0.005`
- `min_samples = 8`
- 输出字段：`lat`, `lng`, `count`, `time`, `cluster_id`

### 距离分层
- 短途：小于 4 km
- 中途：4 km 到 8 km
- 长途：大于 8 km

### 车辆全天统计
- `总里程_km`
- `载客里程_km`
- `空载里程_km`
- `全天载客率`

## 给老师展示时可以这样讲

### 1. 两类热力图的区别
- 车辆位置热力图反映某时刻车辆空间分布
- 上车点热力图反映乘客需求空间分布
- 两者含义不同，不能混着解释

### 2. 动态热力图的意义
- 上车点动态热力图更适合看需求时段变化
- 车辆位置动态热力图更适合看供给时段变化

### 3. 订单统计
- `hourly_order_occupied_ratio.csv` 可解释每小时需求与载客强度
- `01_hourly_order_occupied_ratio.png` 可直接展示高峰与低谷

### 4. 路程分析
- `trip_distance_levels.csv` 用于说明短中长途订单结构
- 项目要求阈值已经按 `<4km / 4-8km / >8km` 实现

### 5. 车辆运营分析
- `vehicle_daily_stats.csv` 用于说明单车载客率和营运效率
- 可从中找高效率车辆和低效率车辆做样例

## 当前实现说明

- 地图使用 `folium` 生成 HTML，可直接浏览器打开
- 动态热力图使用 `HeatMapWithTime`
- 聚类结果输出的是“聚类中心 + 权重”，不是原始每一个点

## 后续可以继续增强

- 增加 1分钟 / 30分钟 动态热力图版本
- 给 DBSCAN 参数做可调入口
- 把阶段05也接入前端交互页面
- 增加道路平均速度分时统计图
