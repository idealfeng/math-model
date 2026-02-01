import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path


def create_judge_score_aggregated_features(input_file, output_file):
    """
    基于标准化后的judge score创建聚合统计特征
    """

    print("Reading data...")
    df = pd.read_csv(input_file)

    print(f"Data shape: {df.shape}")
    print("Creating aggregated judge score features...")

    # Preselect columns for each week (avoid rescanning df.columns per row).
    # Prefer normalized columns (*_score_norm) if present; otherwise fall back to raw (*_score).
    week_cols_by_week = {}
    using_norm_cols = False
    for week in range(1, 12):
        norm_cols = [
            col
            for col in df.columns
            if f"week{week}_judge" in col and col.endswith("_score_norm")
        ]
        raw_cols = [
            col
            for col in df.columns
            if f"week{week}_judge" in col
            and col.endswith("_score")
            and not col.endswith("_score_norm")
        ]

        if len(norm_cols) > 0:
            week_cols_by_week[week] = norm_cols
            using_norm_cols = True
        elif len(raw_cols) > 0:
            week_cols_by_week[week] = raw_cols
        else:
            week_cols_by_week[week] = []

    if using_norm_cols:
        print("Using normalized score columns (*_score_norm).")
    else:
        print("No *_score_norm columns found; falling back to raw *_score columns.")

    features_list = []

    for idx, row in df.iterrows():
        # 进度显示
        if idx % 50 == 0:
            print(f"Processing row {idx}/{len(df)}...")

        # ========== 收集数据 ==========
        all_scores = []  # 所有有效分数
        week_avg_scores = []  # 每周平均分
        week_variances = []  # 每周评委分歧（方差）

        for week in range(1, 12):
            # 找该周的所有标准化judge score列
            week_cols = week_cols_by_week.get(week, [])
            if len(week_cols) == 0:
                continue

            # 获取该周的分数
            scores = row[week_cols].values

            # 过滤掉NA和0（0表示已淘汰）
            valid_scores = [s for s in scores if pd.notna(s) and s != 0]

            if len(valid_scores) == 0:
                break  # 该周没有有效分数，说明已淘汰或比赛未进行到此周

            # 收集所有分数
            all_scores.extend(valid_scores)

            # 该周平均分
            week_avg = np.mean(valid_scores)
            week_avg_scores.append(week_avg)

            # 该周评委分歧（方差）
            if len(valid_scores) > 1:
                week_var = np.var(valid_scores, ddof=1)  # 样本方差
                week_variances.append(week_var)

        # ========== 计算聚合特征 ==========

        if len(all_scores) > 0:
            features = {
                # 1. 整体水平特征
                "avg_judge_score": np.mean(all_scores),
                "std_judge_score": (
                    np.std(all_scores, ddof=1) if len(all_scores) > 1 else 0
                ),
                "min_judge_score": np.min(all_scores),
                "max_judge_score": np.max(all_scores),
                "score_range": np.max(all_scores) - np.min(all_scores),
                # 2. 趋势特征（基于每周平均分）
                "score_trend": np.nan,
                "score_improvement": np.nan,
                # 3. 一致性特征（评委分歧）
                "avg_judge_disagreement": (
                    np.mean(week_variances) if len(week_variances) > 0 else 0
                ),
                "max_judge_disagreement": (
                    np.max(week_variances) if len(week_variances) > 0 else 0
                ),
                # 4. 稳定性特征（跨周波动）
                "performance_consistency": np.nan,
                # 5. 辅助信息
                "n_weeks_with_scores": len(week_avg_scores),
                "total_judge_scores": len(all_scores),
            }

            # 计算趋势（线性回归斜率）
            if len(week_avg_scores) >= 2:
                weeks = np.arange(len(week_avg_scores))
                # 使用scipy.stats.linregress更稳健
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    weeks, week_avg_scores
                )
                features["score_trend"] = slope
                features["score_trend_r2"] = r_value**2  # R²值

                # 改善幅度（最后一周 - 第一周）
                features["score_improvement"] = week_avg_scores[-1] - week_avg_scores[0]
            else:
                features["score_trend"] = 0
                features["score_trend_r2"] = 0
                features["score_improvement"] = 0

            # 计算表现一致性（跨周标准差）
            if len(week_avg_scores) >= 2:
                features["performance_consistency"] = np.std(week_avg_scores, ddof=1)
            else:
                features["performance_consistency"] = 0

        else:
            # 没有有效数据（不应该发生，但以防万一）
            features = {
                "avg_judge_score": 0,
                "std_judge_score": 0,
                "min_judge_score": 0,
                "max_judge_score": 0,
                "score_range": 0,
                "score_trend": 0,
                "score_trend_r2": 0,
                "score_improvement": 0,
                "avg_judge_disagreement": 0,
                "max_judge_disagreement": 0,
                "performance_consistency": 0,
                "n_weeks_with_scores": 0,
                "total_judge_scores": 0,
            }

        features_list.append(features)

    # 转为DataFrame
    features_df = pd.DataFrame(features_list)

    # 合并到原数据
    df_final = pd.concat([df, features_df], axis=1)

    # 保存
    output_path = Path(output_file)
    print(f"\nSaving to {output_path}...")
    try:
        df_final.to_csv(output_path, index=False)
        saved_path = output_path
    except PermissionError:
        saved_path = output_path.with_name(
            f"{output_path.stem}_normalized{output_path.suffix}"
        )
        df_final.to_csv(saved_path, index=False)
        print(
            f"WARNING: Could not overwrite '{output_path}' (file may be open). "
            f"Saved to '{saved_path}' instead."
        )

    # ========== 输出统计信息 ==========
    print("\n" + "=" * 70)
    print("JUDGE SCORE FEATURES CREATED")
    print("=" * 70)

    print("\nNew features added (13 total):")
    print(f"  1. avg_judge_score          - 所有周平均分")
    print(f"  2. std_judge_score          - 所有周标准差")
    print(f"  3. min_judge_score          - 最低分")
    print(f"  4. max_judge_score          - 最高分")
    print(f"  5. score_range              - 分数范围 (max - min)")
    print(f"  6. score_trend              - 趋势斜率（线性回归）")
    print("  7. score_trend_r2           - 趋势拟合度 (R^2)")
    print(f"  8. score_improvement        - 改善幅度 (最后 - 最初)")
    print(f"  9. avg_judge_disagreement   - 平均评委分歧（方差）")
    print(f" 10. max_judge_disagreement   - 最大评委分歧")
    print(f" 11. performance_consistency  - 表现稳定性（跨周标准差）")
    print(f" 12. n_weeks_with_scores      - 有分数的周数")
    print(f" 13. total_judge_scores       - 总评分次数")

    # 统计信息
    print("\nFeature statistics:")
    print(features_df.describe().round(3))

    # 显示样例
    print("\n" + "=" * 70)
    print("SAMPLE DATA (First 10 rows):")
    print("=" * 70)

    sample_cols = [
        "celebrity_name",
        "season",
        "placement",
        "avg_judge_score",
        "score_trend",
        "score_improvement",
        "avg_judge_disagreement",
        "performance_consistency",
        "n_weeks_with_scores",
    ]

    existing_cols = [c for c in sample_cols if c in df_final.columns]
    print(df_final[existing_cols].head(10).to_string(index=False))

    # ========== 验证和洞察 ==========
    print("\n" + "=" * 70)
    print("INSIGHTS:")
    print("=" * 70)

    # 按placement分组看特征
    if "placement" in df_final.columns:
        print("\nAverage features by placement (Top 3):")
        top3_analysis = (
            df_final[df_final["placement"] <= 3]
            .groupby("placement")
            .agg(
                {
                    "avg_judge_score": "mean",
                    "score_trend": "mean",
                    "avg_judge_disagreement": "mean",
                    "performance_consistency": "mean",
                }
            )
            .round(3)
        )
        print(top3_analysis)

        print("\nCorrelation with placement (negative = better placement):")
        correlations = (
            df_final[
                [
                    "placement",
                    "avg_judge_score",
                    "score_trend",
                    "avg_judge_disagreement",
                    "performance_consistency",
                ]
            ]
            .corr()["placement"]
            .sort_values()
        )
        print(correlations.round(3))

    # 检查异常值
    print("\n" + "=" * 70)
    print("QUALITY CHECK:")
    print("=" * 70)

    # 检查是否有NA
    na_counts = features_df.isna().sum()
    if na_counts.sum() > 0:
        print("WARNING: Features with NA values:")
        print(na_counts[na_counts > 0])
    else:
        print("No NA values in features")

    # 检查极端值
    print("\nExtreme values check:")
    for col in ["avg_judge_score", "score_trend", "avg_judge_disagreement"]:
        if col in features_df.columns:
            q1 = features_df[col].quantile(0.01)
            q99 = features_df[col].quantile(0.99)
            print(f"  {col}: [{q1:.3f}, {q99:.3f}]")

    print(f"\nSuccess! Output saved to: {saved_path}")
    print(f"Total features in output: {len(df_final.columns)}")

    return df_final


# ========== 主程序 ==========
if __name__ == "__main__":
    input_file = "dwts_data_normalized.csv"  # 标准化后的输入文件（包含 *_score_norm）
    output_file = "dwts_data_with_all_features.csv"  # 输出文件

    try:
        df_final = create_judge_score_aggregated_features(input_file, output_file)

        print("\n" + "=" * 70)
        print("ALL FEATURES CREATED SUCCESSFULLY!")
        print("=" * 70)

        print("\nYou can now use these features for modeling:")
        print("  - avg_judge_score (整体水平)")
        print("  - score_trend (进步趋势)")
        print("  - avg_judge_disagreement (评委分歧)")
        print("  - performance_consistency (稳定性)")

    except FileNotFoundError:
        print(f"ERROR: Input file '{input_file}' not found!")
        print("Please check the file name and path.")

    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback

        traceback.print_exc()
