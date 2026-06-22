# -*- coding: utf-8 -*-
"""
出租车GPS数据分析及可视化模块
================================
依据《数据简单处理.docx》"三、数据分析及可视化"需求12-13。
输入：data_cleaning.py 生成的 OD 订单表 taxi_od.csv
      （列名：车辆id, 开始时间, 开始经度, 开始纬度, 结束时间, 结束经度, 结束纬度）

包含：
    需求12：统计各小时的订单数分布，绘制折线图+柱状图
    需求13：统计各时段订单的时长分布，绘制箱型图
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ========================= 配置区 =========================
OD_DATA_PATH = "../data/processed/taxi_od.csv"
FIG_DIR = "../docs/figures"   # 图表输出目录

# 解决中文显示问题，按你系统实际安装的字体改（Windows用"SimHei"，Mac用"PingFang SC"等）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False
# ===========================================================


def load_order_data(path: str) -> pd.DataFrame:
    """读取OD订单表，并把开始/结束时间转换为datetime类型（从csv重新读取后默认是字符串）"""
    df_order = pd.read_csv(path)
    df_order['开始时间'] = pd.to_datetime(df_order['开始时间'])
    df_order['结束时间'] = pd.to_datetime(df_order['结束时间'])
    return df_order


def analyze_hourly_count(df_order: pd.DataFrame, save_path: str = None) -> pd.DataFrame:
    """
    需求12：统计各小时的订单数分布
    """
    df_order = df_order.copy()
    df_order['小时'] = df_order['开始时间'].dt.hour

    df_hourcnt = df_order.groupby('小时')['车辆id'].count()
    df_hourcnt = df_hourcnt.rename('数量').reset_index()

    fig = plt.figure(1, (8, 4), dpi=200)
    ax = plt.subplot(111)
    plt.plot(df_hourcnt['小时'], df_hourcnt['数量'], 'k-')
    plt.plot(df_hourcnt['小时'], df_hourcnt['数量'], 'k.')
    plt.bar(df_hourcnt['小时'], df_hourcnt['数量'])
    plt.ylabel('数量')
    plt.xlabel('小时')
    plt.xticks(range(24), range(24))
    plt.title('出行小时数量统计')
    plt.ylim(0, df_hourcnt['数量'].max() * 1.15)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        print(f"已保存：{save_path}")
    plt.show()
    plt.close(fig)

    return df_order  # 返回带"小时"列的df_order，供需求13复用


def analyze_duration_distribution(df_order: pd.DataFrame, save_path: str = None) -> pd.DataFrame:
    """
    需求13：统计各时段订单的时长分布（箱型图）
    要求 df_order 已包含"小时"列（即先跑过 analyze_hourly_count）
    """
    df_order = df_order.copy()
    if '小时' not in df_order.columns:
        df_order['小时'] = df_order['开始时间'].dt.hour

    # 开始结束时间作差，转化为秒单位（订单时长）
    df_order['订单时长'] = (df_order['结束时间'] - df_order['开始时间']).dt.seconds

    fig = plt.figure(1, (6, 4), dpi=150)
    ax = plt.subplot(111)
    plt.sca(ax)
    sns.boxplot(x="小时", y=df_order["订单时长"] / 60, data=df_order, ax=ax)
    plt.ylabel('订单时长(分钟)')
    plt.xlabel('订单时段')
    plt.ylim(0, 60)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        print(f"已保存：{save_path}")
    plt.show()
    plt.close(fig)

    return df_order


def main():
    df_order = load_order_data(OD_DATA_PATH)
    print(f"订单数：{df_order.shape[0]}")

    df_order = analyze_hourly_count(
        df_order, save_path=os.path.join(FIG_DIR, "01_订单小时分布.png")
    )
    df_order = analyze_duration_distribution(
        df_order, save_path=os.path.join(FIG_DIR, "02_订单时长分布箱型图.png")
    )

    print("数据分析与可视化完成。")
    print("观察要点（可写入项目日志）：")
    print("- 各时段是否都有订单；晚高峰(20-22点)是否明显偏多，凌晨3-5点是否明显偏少。")
    print("- 早晚高峰(8-9点、17-18点)的订单时长中位数是否高于其他时段（堵车导致）。")


if __name__ == "__main__":
    main()