import pandas as pd
import numpy as np


def process_dwts_data(input_file, output_file):
    """
    处理DWTS数据（宽格式）：
    1. 识别真正的淘汰周数
    2. 绝对周转化为相对进度
    3. 绝对排名转化为百分位（同season内）
    """

    # 读取数据
    print("Reading data...")
    df = pd.read_csv(input_file)

    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()[:10]}...")  # 显示前10列

    # ========== 1. 识别真正的淘汰周数 ==========
    print("\nIdentifying elimination weeks...")

    def find_elimination_week_from_scores(row):
        """
        从每周的分数列中找到真正的淘汰周
        逻辑：找到首次所有judge_score为0的周，淘汰周是它的前一周
        """
        elimination_week = None

        # 检查week 1-11
        for week in range(1, 12):
            # 查找该周的所有judge score列（非norm）
            # 格式: week1_judge1_score, week1_judge2_score等
            week_score_cols = [
                col
                for col in df.columns
                if f"week{week}_judge" in col and "_score" in col and "norm" not in col
            ]

            if len(week_score_cols) == 0:
                continue

            # 获取该周的所有分数
            scores = row[week_score_cols].values

            # 过滤掉N/A
            valid_scores = [s for s in scores if pd.notna(s)]

            if len(valid_scores) == 0:
                # 该周没有有效分数，说明比赛还没到这周
                break

            # 检查是否所有有效分数都是0
            if all(s == 0 for s in valid_scores):
                # 找到首次全0的周，淘汰周是前一周
                elimination_week = week - 1 if week > 1 else 1
                break

        # 如果没找到全0的周，说明该选手参赛到最后
        if elimination_week is None:
            # 找最后一个有非0分数的周
            for week in range(11, 0, -1):
                week_score_cols = [
                    col
                    for col in df.columns
                    if f"week{week}_judge" in col
                    and "_score" in col
                    and "norm" not in col
                ]

                if len(week_score_cols) > 0:
                    scores = row[week_score_cols].values
                    valid_scores = [s for s in scores if pd.notna(s)]

                    if len(valid_scores) > 0 and any(s > 0 for s in valid_scores):
                        elimination_week = week
                        break

        # 如果还是None，默认为1
        return elimination_week if elimination_week is not None else 1

    # 应用到每一行
    print("Processing each contestant...")
    df["elimination_week"] = df.apply(find_elimination_week_from_scores, axis=1)

    # ========== 2. 计算每季的统计数据 ==========
    print("Computing season statistics...")

    # 每季的最大周数（该季所有选手中最晚淘汰的周数）
    season_max_weeks = df.groupby("season")["elimination_week"].max()

    # 每季的选手数量
    season_n_contestants = df.groupby("season")["celebrity_name"].nunique()

    # 合并到主数据
    df["max_weeks_in_season"] = df["season"].map(season_max_weeks)
    df["n_contestants_in_season"] = df["season"].map(season_n_contestants)

    # ========== 3. 转化为相对进度 ==========
    print("Converting to relative progress...")

    # 相对进度 = 淘汰周数 / 该季最大周数
    df["relative_progress"] = df["elimination_week"] / df["max_weeks_in_season"]
    df["relative_progress"] = df["relative_progress"].round(4)

    # ========== 4. 转化placement为百分位 ==========
    print("Converting placement to percentile...")

    # 百分位 = (n_contestants - placement + 1) / n_contestants
    # 第1名 → 接近1.0
    # 最后一名 → 接近0
    df["placement_percentile"] = (
        df["n_contestants_in_season"] - df["placement"] + 1
    ) / df["n_contestants_in_season"]
    df["placement_percentile"] = df["placement_percentile"].round(4)

    # ========== 5. 清理和输出 ==========
    print("Cleaning up...")

    # 删除临时列
    df = df.drop(
        columns=["max_weeks_in_season", "n_contestants_in_season"], errors="ignore"
    )

    # 保存
    print(f"Saving to {output_file}...")
    df.to_csv(output_file, index=False)

    # ========== 6. 输出统计信息 ==========
    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE!")
    print("=" * 70)

    print(f"\nOutput saved to: {output_file}")
    print(f"Total rows: {len(df)}")

    print(f"\n📊 New columns added:")
    print(f"  1. elimination_week     - 真正的淘汰周数")
    print(f"  2. relative_progress    - 相对进度 (0-1)")
    print(f"  3. placement_percentile - 排名百分位 (0-1)")

    print(f"\n📈 Statistics:")
    print(
        f"  Elimination week range: {df['elimination_week'].min():.0f} - {df['elimination_week'].max():.0f}"
    )
    print(
        f"  Relative progress range: {df['relative_progress'].min():.3f} - {df['relative_progress'].max():.3f}"
    )
    print(
        f"  Placement percentile range: {df['placement_percentile'].min():.3f} - {df['placement_percentile'].max():.3f}"
    )

    # ========== 7. 验证示例 ==========
    print("\n" + "=" * 70)
    print("SAMPLE DATA (First 10 rows):")
    print("=" * 70)

    sample_cols = [
        "celebrity_name",
        "season",
        "placement",
        "results",
        "elimination_week",
        "relative_progress",
        "placement_percentile",
    ]

    # 确保所有列都存在
    existing_cols = [col for col in sample_cols if col in df.columns]

    print(df[existing_cols].head(10).to_string(index=False))

    # ========== 8. 按Season验证 ==========
    print("\n" + "=" * 70)
    print("VERIFICATION BY SEASON:")
    print("=" * 70)

    # 显示每季的统计
    for season in sorted(df["season"].unique())[:3]:  # 只显示前3季作为例子
        season_data = df[df["season"] == season][existing_cols].sort_values("placement")
        print(f"\nSeason {season}:")
        print(season_data.to_string(index=False))

    # ========== 9. 逻辑验证 ==========
    print("\n" + "=" * 70)
    print("LOGIC VERIFICATION:")
    print("=" * 70)
    print("✓ elimination_week: 首次评分为0的前一周")
    print("✓ relative_progress = elimination_week / max_weeks_in_season")
    print("✓ placement_percentile = (n - placement + 1) / n")
    print("  → 1st place (placement=1) → percentile ≈ 1.0 (highest)")
    print("  → Last place (placement=n) → percentile ≈ 1/n (lowest)")

    # 检查是否有异常值
    print("\n🔍 Checking for anomalies...")

    # 检查是否有relative_progress > 1
    if (df["relative_progress"] > 1.0).any():
        print("⚠️  WARNING: Some relative_progress > 1.0")
        print(df[df["relative_progress"] > 1.0][existing_cols])

    # 检查是否有placement_percentile > 1
    if (df["placement_percentile"] > 1.0).any():
        print("⚠️  WARNING: Some placement_percentile > 1.0")
        print(df[df["placement_percentile"] > 1.0][existing_cols])

    # 检查是否有0值
    if (df["elimination_week"] == 0).any():
        print("⚠️  WARNING: Some elimination_week = 0")
        print(df[df["elimination_week"] == 0][existing_cols])

    if not (
        (df["relative_progress"] > 1.0).any()
        or (df["placement_percentile"] > 1.0).any()
        or (df["elimination_week"] == 0).any()
    ):
        print("✓ No anomalies detected!")

    return df


# ========== 主程序 ==========
if __name__ == "__main__":
    # 设置文件路径
    input_file = "dwts_data.csv"  # 输入文件
    output_file = "dwts_data_processed.csv"  # 输出文件

    try:
        # 处理数据
        df_processed = process_dwts_data(input_file, output_file)

        print("\n✅ SUCCESS! Data processing completed.")
        print(f"📁 Output file: {output_file}")

    except FileNotFoundError:
        print(f"❌ ERROR: Input file '{input_file}' not found!")
        print("Please make sure the file exists in the current directory.")

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback

        traceback.print_exc()
