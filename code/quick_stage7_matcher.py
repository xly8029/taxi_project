# -*- coding: utf-8 -*-
"""
阶段 7 快速地图匹配 - 简化版（最近节点吸附）
==============================================
目标：快速生成全量校正轨迹，20-30 分钟完成 11,120 辆车

策略：
    1. 使用 OSMnx nearest_nodes 直接吸附（跳过 HMM）
    2. 从连续节点推断道路边
    3. 多进程并行 + 分批保存
    4. Parquet + 数据类型优化

输出字段：
    - id, time, status, speed
    - matched_lon, matched_lat, matched_node
    - edge_u, edge_v, edge_key
"""

import os
import sys
import pickle
import multiprocessing as mp
from pathlib import Path

import pandas as pd
import numpy as np
import osmnx as ox
from tqdm import tqdm

# ========================= 配置 =========================
PROJECT_ROOT = Path(__file__).parent.parent
ROAD_NETWORK_PKL = PROJECT_ROOT / "data" / "raw" / "shenzhen_drive.pkl"
VEHICLE_CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "vehicle"
OUTPUT_CACHE = PROJECT_ROOT / "data" / "cache" / "matched_trajectory_stage7.parquet"
TEMP_DIR = PROJECT_ROOT / "data" / "cache" / "_stage7_temp"

# 全局路网（在每个子进程中加载一次）
_G = None
_edge_lookup = None


def init_worker():
    """子进程初始化：加载路网"""
    global _G, _edge_lookup
    if _G is None:
        print(f"[Worker {os.getpid()}] Loading road network...")
        with open(ROAD_NETWORK_PKL, "rb") as f:
            _G = pickle.load(f)
        
        # 构建节点对 -> 道路边的映射
        _edge_lookup = {}
        for u, v, key in _G.edges(keys=True):
            _edge_lookup[(u, v)] = (u, v, key)
        
        print(f"[Worker {os.getpid()}] Road network loaded: {_G.number_of_nodes()} nodes, {_G.number_of_edges()} edges")


def snap_trajectory_fast(vehicle_id):
    """
    快速匹配单个车辆轨迹（最近节点吸附）
    
    返回: DataFrame with [id, time, status, speed,
                          matched_lon, matched_lat, matched_node,
                          edge_u, edge_v, edge_key]
    """
    try:
        # 1. 读取原始轨迹
        vehicle_csv = VEHICLE_CACHE_DIR / f"{vehicle_id}.csv"
        if not vehicle_csv.exists():
            return None
        
        df = pd.read_csv(vehicle_csv)
        df.columns = [c.strip() for c in df.columns]
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").reset_index(drop=True)
        
        # 检查必要字段
        required = {"id", "time", "long", "lati", "status", "speed"}
        if not required.issubset(df.columns):
            return None
        
        if len(df) < 2:
            return None
        
        # 2. 使用 nearest_nodes 批量吸附（快速！）
        global _G
        
        lons = df["long"].values
        lats = df["lati"].values
        
        # OSMnx 支持批量查询
        matched_nodes = ox.distance.nearest_nodes(_G, lons, lats)
        
        # 3. 获取匹配节点的坐标
        matched_lons = []
        matched_lats = []
        
        for node in matched_nodes:
            node_data = _G.nodes[node]
            matched_lons.append(node_data['x'])
            matched_lats.append(node_data['y'])
        
        # 4. 构建结果 DataFrame
        result = pd.DataFrame({
            "id": df["id"],
            "time": df["time"],
            "status": df["status"],
            "speed": df["speed"],
            "matched_lon": matched_lons,
            "matched_lat": matched_lats,
            "matched_node": matched_nodes,
        })
        
        # 5. 推断道路边（基于连续节点）
        global _edge_lookup
        
        result["prev_node"] = result["matched_node"].shift(1)
        
        edge_u_list = []
        edge_v_list = []
        edge_key_list = []
        
        for _, row in result.iterrows():
            prev = row["prev_node"]
            curr = row["matched_node"]
            
            if pd.isna(prev) or int(prev) == int(curr):
                edge_u_list.append(None)
                edge_v_list.append(None)
                edge_key_list.append(None)
            else:
                edge = _edge_lookup.get((int(prev), int(curr)))
                if edge:
                    edge_u_list.append(edge[0])
                    edge_v_list.append(edge[1])
                    edge_key_list.append(edge[2])
                else:
                    edge_u_list.append(None)
                    edge_v_list.append(None)
                    edge_key_list.append(None)
        
        result["edge_u"] = edge_u_list
        result["edge_v"] = edge_v_list
        result["edge_key"] = edge_key_list
        result.drop(columns=["prev_node"], inplace=True)
        
        # 6. 优化数据类型
        result["id"] = result["id"].astype("int32")
        result["status"] = result["status"].astype("int8")
        result["speed"] = result["speed"].astype("float32")
        result["matched_lon"] = result["matched_lon"].astype("float32")
        result["matched_lat"] = result["matched_lat"].astype("float32")
        result["matched_node"] = result["matched_node"].astype("int64")  # 节点 ID 可能很大
        result["edge_u"] = pd.to_numeric(result["edge_u"], errors="coerce").astype("Int64")
        result["edge_v"] = pd.to_numeric(result["edge_v"], errors="coerce").astype("Int64")
        result["edge_key"] = pd.to_numeric(result["edge_key"], errors="coerce").astype("Int8")
        
        return result
        
    except Exception as e:
        print(f"\n[WARN] Vehicle {vehicle_id} failed: {e}")
        return None


def save_batch(batch_results, batch_num):
    """保存一批结果到临时文件"""
    valid = [r for r in batch_results if r is not None and not r.empty]
    if not valid:
        return
    
    df_batch = pd.concat(valid, ignore_index=True)
    
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    temp_file = TEMP_DIR / f"batch_{batch_num:04d}.parquet"
    df_batch.to_parquet(
        temp_file,
        engine="pyarrow",
        compression="zstd",
        index=False,
    )


def build_stage7_cache_fast(max_workers=8, batch_size=200, limit=None):
    """
    快速并行构建阶段 7 缓存（简化版）
    
    Args:
        max_workers: 并行进程数
        batch_size: 每批保存的车辆数
        limit: 限制处理车辆数（测试用）
    """
    # 获取所有车辆 ID
    vehicle_ids = sorted([
        int(f.stem)
        for f in VEHICLE_CACHE_DIR.glob("*.csv")
    ])
    
    if limit:
        vehicle_ids = vehicle_ids[:limit]
    
    total_vehicles = len(vehicle_ids)
    
    print(f"\n{'='*60}")
    print(f"阶段 7 快速地图匹配 - 简化版")
    print(f"{'='*60}")
    print(f"车辆总数: {total_vehicles}")
    print(f"并行进程: {max_workers}")
    print(f"批量大小: {batch_size}")
    print(f"匹配方法: 最近节点吸附 (nearest_nodes)")
    print(f"{'='*60}\n")
    
    # 清理旧的临时文件
    if TEMP_DIR.exists():
        for f in TEMP_DIR.glob("*.parquet"):
            f.unlink()
    
    # 初始化进程池
    with mp.Pool(processes=max_workers, initializer=init_worker) as pool:
        batch_results = []
        batch_num = 0
        
        for result in tqdm(
            pool.imap_unordered(
                snap_trajectory_fast,
                vehicle_ids,
                chunksize=10,
            ),
            total=total_vehicles,
            desc="处理进度",
            unit="车",
            ncols=80,
        ):
            batch_results.append(result)
            
            # 达到批量大小，保存临时文件
            if len(batch_results) >= batch_size:
                save_batch(batch_results, batch_num)
                batch_num += 1
                batch_results = []
        
        # 保存最后一批
        if batch_results:
            save_batch(batch_results, batch_num)
    
    # 合并所有临时文件
    print("\n正在合并临时文件...")
    temp_files = sorted(TEMP_DIR.glob("*.parquet"))
    
    if not temp_files:
        print("[错误] 没有生成任何有效数据")
        return
    
    dfs = []
    for f in tqdm(temp_files, desc="读取文件", ncols=80):
        dfs.append(pd.read_parquet(f))
    
    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all.sort_values(["id", "time"]).reset_index(drop=True)
    
    # 保存最终文件
    print("正在保存最终缓存...")
    OUTPUT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_parquet(
        OUTPUT_CACHE,
        engine="pyarrow",
        compression="zstd",
        index=False,
    )
    
    # 清理临时文件
    for f in temp_files:
        f.unlink()
    if TEMP_DIR.exists():
        try:
            TEMP_DIR.rmdir()
        except OSError:
            pass
    
    # 统计信息
    size_mb = OUTPUT_CACHE.stat().st_size / 1024 / 1024
    
    print(f"\n{'='*60}")
    print(f"生成完成！")
    print(f"{'='*60}")
    print(f"缓存文件: {OUTPUT_CACHE}")
    print(f"总记录数: {len(df_all):,}")
    print(f"车辆数量: {df_all['id'].nunique()}")
    print(f"时间范围: {df_all['time'].min()} ~ {df_all['time'].max()}")
    print(f"文件大小: {size_mb:.2f} MB")
    
    # 数据质量检查
    node_ratio = df_all["matched_node"].notna().mean() * 100
    edge_ratio = df_all["edge_u"].notna().mean() * 100
    
    print(f"\n数据质量:")
    print(f"  匹配节点覆盖率: {node_ratio:.1f}%")
    print(f"  有效道路边比例: {edge_ratio:.1f}%")
    print(f"  速度范围: {df_all['speed'].min():.1f} ~ {df_all['speed'].max():.1f} km/h")
    
    print(f"\n数据预览（前 10 条）:")
    print(df_all.head(10).to_string(index=False))
    print(f"{'='*60}\n")


def test_read_cache():
    """测试读取生成的缓存"""
    if not OUTPUT_CACHE.exists():
        print(f"[错误] 缓存文件不存在: {OUTPUT_CACHE}")
        return
    
    print("\n正在读取缓存...")
    df = pd.read_parquet(OUTPUT_CACHE)
    
    print(f"\n成功读取 {len(df):,} 条记录")
    print(f"车辆数: {df['id'].nunique()}")
    print(f"时间范围: {df['time'].min()} ~ {df['time'].max()}")
    
    print(f"\n前 5 条记录:")
    print(df.head())
    
    # 测试按道路边聚合速度（阶段 7 核心功能）
    print(f"\n{'='*60}")
    print("测试道路速度统计（阶段 7 核心功能）")
    print(f"{'='*60}")
    
    valid = df[
        df["edge_u"].notna()
        & df["edge_v"].notna()
        & df["speed"].between(1, 120)
    ]
    
    print(f"有效样本: {len(valid):,} / {len(df):,} ({len(valid)/len(df)*100:.1f}%)")
    
    speed_stats = (
        valid.groupby(["edge_u", "edge_v", "edge_key"])
        .agg(
            avg_speed=("speed", "mean"),
            sample_count=("speed", "size"),
            vehicle_count=("id", "nunique"),
        )
        .reset_index()
        .sort_values("sample_count", ascending=False)
    )
    
    print(f"统计道路边数量: {len(speed_stats):,}")
    print(f"\n样本最多的前 10 条道路:")
    print(speed_stats.head(10).to_string(index=False))
    
    print(f"\n道路边样本数分布:")
    print(speed_stats["sample_count"].describe())
    print(f"{'='*60}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("\n*** 测试模式：前 50 辆车 ***")
        build_stage7_cache_fast(max_workers=4, batch_size=25, limit=50)
    elif len(sys.argv) > 1 and sys.argv[1] == "read":
        test_read_cache()
    else:
        print("\n*** 全量处理模式 ***")
        build_stage7_cache_fast(max_workers=8, batch_size=200)
    
    print("\n完成！")
