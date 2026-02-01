import pandas as pd
import numpy as np


def standardize_scores_by_season_week(data):
    """
    对每个season的每个week内的所有选手评委分数进行标准化

    核心思想:
    - Season 1, Week 1的所有选手 → 标准化为一组
    - Season 1, Week 2的所有选手 → 标准化为另一组
    - 依此类推

    这样可以消除不同季度评委打分风格的差异
    """

    data_norm = data.copy()

    # ========== Step 1: 识别所有week和judge列 ==========
    # 找出所有类似 "week1_judge1_score" 的列
    score_columns = [col for col in data.columns if "judge" in col and "score" in col]

    # 提取week信息
    # week1_judge1_score → week=1
    # week2_judge1_score → week=2
    weeks = set()
    for col in score_columns:
        week_num = int(col.split("_")[0].replace("week", ""))
        weeks.add(week_num)

    max_week = max(weeks)
    print(f"Found {max_week} weeks in the data")

    # ========== Step 2: 对每个season × week进行标准化 ==========
    for season in data["season"].unique():
        print(f"\nProcessing Season {season}...")

        # 该季的数据
        season_mask = data["season"] == season

        for week in range(1, max_week + 1):
            # 找出该周的所有judge列
            week_judge_cols = [col for col in score_columns if f"week{week}_" in col]

            if len(week_judge_cols) == 0:
                continue

            # 该季该周有数据的选手 (排除score=0的已淘汰选手)
            week_data = data.loc[season_mask, week_judge_cols]

            # 将所有judge分数收集到一起
            all_scores = []
            valid_mask = season_mask.copy()

            for col in week_judge_cols:
                scores = data.loc[season_mask, col]
                # 排除0分(已淘汰)和N/A
                valid_scores = scores[(scores > 0) & (scores.notna())]
                all_scores.extend(valid_scores.values)

                # 更新valid_mask: 该周该列有有效分数的选手
                valid_mask = valid_mask & (data[col] > 0) & (data[col].notna())

            # 如果该周有有效分数
            if len(all_scores) > 0:
                # 计算该季该周所有分数的均值和标准差
                week_mean = np.mean(all_scores)
                week_std = np.std(all_scores)

                # 避免除以0
                if week_std < 1e-6:
                    week_std = 1.0

                print(
                    f"  Week {week}: mean={week_mean:.2f}, std={week_std:.2f}, n_scores={len(all_scores)}"
                )

                # 标准化该周的所有judge列
                for col in week_judge_cols:
                    # 创建标准化后的列名
                    norm_col = col.replace("_score", "_score_norm")

                    # 标准化: (x - mean) / std
                    data_norm.loc[season_mask, norm_col] = data.loc[
                        season_mask, col
                    ].apply(
                        lambda x: (
                            (x - week_mean) / week_std if (pd.notna(x) and x > 0) else 0
                        )
                    )

    return data_norm


# ========== 使用示例 ==========

# 从CSV文件中读取原始数据
dwts_data = pd.read_csv("dwts_data.csv")

# 调用标准化函数，得到标准化后的数据
df_normalized = standardize_scores_by_season_week(dwts_data)

# ========== 输出到新文件并覆盖相应的列 ==========

# 创建一个新的副本，保持原始列
dwts_data_copy = dwts_data.copy()

# 找到所有标准化的列（weekX_judgeY_score_norm）
norm_columns = [col for col in df_normalized.columns if "_score_norm" in col]

# 将标准化数据覆盖到副本中的相应位置
for col in norm_columns:
    dwts_data_copy[col] = df_normalized[col]

# 保存新的数据文件
dwts_data_copy.to_csv("dwts_data_normalized.csv", index=False)

print("Standardized data has been saved to 'dwts_data_normalized.csv'.")
