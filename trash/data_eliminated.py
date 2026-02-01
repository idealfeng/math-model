import pandas as pd
import numpy as np


def identify_special_weeks_FIXED(df):
    """
    修正版：识别无淘汰周和多淘汰周
    关键修正：决赛选手不算"被淘汰"
    """

    print("Identifying special weeks (FIXED version)...")

    # 创建字典存储每个season每周的淘汰情况
    week_info = {}

    # 遍历每个season
    for season in sorted(df["season"].unique()):
        week_info[season] = {}
        season_data = df[df["season"] == season]

        # 获取该季的最大周数（决赛周）
        max_week = season_data["elimination_week"].max()

        # 遍历每周
        for week in range(1, int(max_week) + 1):
            # 🔧 关键修正：只统计真正被淘汰的选手
            # 规则：elimination_week == week 且 week < max_week
            #      或者：placement > 3 (保守)

            if week == max_week:
                # 决赛周：不算淘汰
                eliminated_this_week = pd.DataFrame()  # 空
            else:
                # 非决赛周：正常统计
                eliminated_this_week = season_data[
                    season_data["elimination_week"] == week
                ]

            n_eliminated = len(eliminated_this_week)

            week_info[season][week] = {
                "n_eliminated": n_eliminated,
                "zero_eliminate": 1 if n_eliminated == 0 else 0,
                "multi_eliminate": 1 if n_eliminated > 1 else 0,
                "eliminated_names": (
                    eliminated_this_week["celebrity_name"].tolist()
                    if "celebrity_name" in eliminated_this_week.columns
                    else []
                ),
                "is_finals": 1 if week == max_week else 0,  # 新增：标记决赛周
            }

    return week_info


def add_special_week_features_FIXED(df, week_info):
    """
    修正版：添加特殊周特征
    """

    print("Adding special week features (FIXED version)...")

    # 方法1: 标记选手被淘汰的那周是否特殊
    def get_elimination_week_type(row):
        season = row["season"]
        elim_week = int(row["elimination_week"])

        # 🔧 关键修正：判断是否为决赛选手
        season_data = df[df["season"] == season]
        max_week = season_data["elimination_week"].max()

        is_finalist = elim_week == max_week

        if is_finalist:
            # 决赛选手：不算被淘汰
            return pd.Series(
                {
                    "eliminated_in_multi_week": 0,
                    "n_eliminated_same_week": 0,
                    "is_finalist": 1,
                }
            )
        elif season in week_info and elim_week in week_info[season]:
            # 非决赛选手：正常统计
            info = week_info[season][elim_week]
            return pd.Series(
                {
                    "eliminated_in_multi_week": info["multi_eliminate"],
                    "n_eliminated_same_week": info["n_eliminated"],
                    "is_finalist": 0,
                }
            )
        else:
            return pd.Series(
                {
                    "eliminated_in_multi_week": 0,
                    "n_eliminated_same_week": 1,
                    "is_finalist": 0,
                }
            )

    df[["eliminated_in_multi_week", "n_eliminated_same_week", "is_finalist"]] = (
        df.apply(get_elimination_week_type, axis=1)
    )

    # 方法2: 统计选手参赛期间遇到的特殊周次数
    def count_special_weeks_experienced(row):
        season = row["season"]
        elim_week = int(row["elimination_week"])

        # 🔧 关键修正：决赛周不统计
        season_data = df[df["season"] == season]
        max_week = season_data["elimination_week"].max()

        n_zero_weeks = 0
        n_multi_weeks = 0

        if season in week_info:
            # 统计从week1到elimination_week期间的特殊周
            # 但排除决赛周
            for week in range(1, elim_week):  # 改为 elim_week 而不是 elim_week + 1
                if week in week_info[season] and week < max_week:
                    n_zero_weeks += week_info[season][week]["zero_eliminate"]
                    n_multi_weeks += week_info[season][week]["multi_eliminate"]

        return pd.Series(
            {
                "n_zero_eliminate_weeks_experienced": n_zero_weeks,
                "n_multi_eliminate_weeks_experienced": n_multi_weeks,
            }
        )

    df[
        ["n_zero_eliminate_weeks_experienced", "n_multi_eliminate_weeks_experienced"]
    ] = df.apply(count_special_weeks_experienced, axis=1)

    return df


def create_week_level_table_FIXED(df, week_info):
    """
    修正版：创建周级别表
    """

    print("Creating week-level reference table (FIXED version)...")

    week_records = []

    for season in sorted(week_info.keys()):
        for week, info in sorted(week_info[season].items()):
            week_records.append(
                {
                    "season": season,
                    "week": week,
                    "n_eliminated": info["n_eliminated"],
                    "zero_eliminate": info["zero_eliminate"],
                    "multi_eliminate": info["multi_eliminate"],
                    "is_finals": info["is_finals"],
                    "eliminated_names": (
                        ", ".join(info["eliminated_names"])
                        if info["eliminated_names"]
                        else "None"
                    ),
                }
            )

    week_df = pd.DataFrame(week_records)
    return week_df


def process_special_weeks_FIXED(input_file, output_file, week_table_file=None):
    """
    完整流程：识别并标记特殊周（修正版）
    """

    # 读取数据
    print("Reading data...")
    df = pd.read_csv(input_file)

    # 检查必要列
    if "elimination_week" not in df.columns:
        raise ValueError("请先运行淘汰周识别代码！需要'elimination_week'列。")

    # 识别特殊周（修正版）
    week_info = identify_special_weeks_FIXED(df)

    # 添加特殊周特征（修正版）
    df = add_special_week_features_FIXED(df, week_info)

    # 创建周级别表（可选）
    if week_table_file:
        week_df = create_week_level_table_FIXED(df, week_info)
        week_df.to_csv(week_table_file, index=False)
        print(f"Week-level table saved to: {week_table_file}")

    # 保存
    df.to_csv(output_file, index=False)

    # ========== 统计信息 ==========
    print("\n" + "=" * 70)
    print("SPECIAL WEEKS ANALYSIS (FIXED)")
    print("=" * 70)

    # 按season统计（显示前5季）
    for season in sorted(week_info.keys())[:5]:
        print(f"\nSeason {season}:")

        # 获取决赛周
        finals_week = [
            w for w, info in week_info[season].items() if info["is_finals"] == 1
        ]

        # 无淘汰周（排除决赛）
        zero_weeks = [
            w
            for w, info in week_info[season].items()
            if info["zero_eliminate"] == 1 and info["is_finals"] == 0
        ]

        # 多淘汰周（排除决赛）
        multi_weeks = [
            w
            for w, info in week_info[season].items()
            if info["multi_eliminate"] == 1 and info["is_finals"] == 0
        ]

        if finals_week:
            print(f"  决赛周: Week {finals_week[0]}")

        if zero_weeks:
            print(f"  无淘汰周: Week {zero_weeks}")
        else:
            print(f"  无淘汰周: None")

        if multi_weeks:
            print(f"  多淘汰周: Week {multi_weeks}")
            for w in multi_weeks:
                names = week_info[season][w]["eliminated_names"]
                print(f"    Week {w}: {len(names)} eliminated - {names}")
        else:
            print(f"  多淘汰周: None")

    # 整体统计
    print("\n" + "=" * 70)
    print("OVERALL STATISTICS")
    print("=" * 70)

    # 统计（排除决赛周）
    total_zero_weeks = sum(
        sum(
            info["zero_eliminate"]
            for week, info in season_info.items()
            if info["is_finals"] == 0
        )
        for season_info in week_info.values()
    )

    total_multi_weeks = sum(
        sum(
            info["multi_eliminate"]
            for week, info in season_info.items()
            if info["is_finals"] == 0
        )
        for season_info in week_info.values()
    )

    total_finals = len(week_info)  # 每季一个决赛

    print(f"Total seasons: {len(week_info)}")
    print(f"Total zero-elimination weeks (excluding finals): {total_zero_weeks}")
    print(f"Total multi-elimination weeks (excluding finals): {total_multi_weeks}")
    print(f"Total finals weeks: {total_finals}")

    # 决赛选手统计
    n_finalists = (df["is_finalist"] == 1).sum()
    print(f"\nTotal finalists (completed all weeks): {n_finalists}")
    print(f"Average finalists per season: {n_finalists / len(week_info):.1f}")

    # 选手级别统计
    print("\n" + "=" * 70)
    print("CONTESTANT-LEVEL FEATURES")
    print("=" * 70)

    print(f"\nNew columns added:")
    print(f"  1. eliminated_in_multi_week              - 是否在多淘汰周被淘汰 (0/1)")
    print(f"  2. n_eliminated_same_week                - 同周被淘汰人数")
    print("  3. is_finalist                           - 是否为决赛选手 (0/1)")
    print(f"  4. n_zero_eliminate_weeks_experienced    - 参赛期间经历的无淘汰周数")
    print(f"  5. n_multi_eliminate_weeks_experienced   - 参赛期间经历的多淘汰周数")

    # 显示样例
    print("\nSample data (First 10 rows):")
    sample_cols = [
        "celebrity_name",
        "season",
        "placement",
        "elimination_week",
        "is_finalist",
        "eliminated_in_multi_week",
        "n_eliminated_same_week",
    ]

    existing_cols = [c for c in sample_cols if c in df.columns]
    print(df[existing_cols].head(10).to_string(index=False))

    # 验证：检查决赛选手
    print("\n" + "=" * 70)
    print("VERIFICATION - Finalists Check")
    print("=" * 70)

    finalists = df[df["is_finalist"] == 1][
        [
            "celebrity_name",
            "season",
            "placement",
            "elimination_week",
            "eliminated_in_multi_week",
            "n_eliminated_same_week",
        ]
    ].head(10)

    print("\nSample finalists (should have eliminated_in_multi_week=0):")
    print(finalists.to_string(index=False))

    # 检查是否还有决赛被标记为多淘汰的bug
    bug_check = df[(df["is_finalist"] == 1) & (df["eliminated_in_multi_week"] == 1)]
    if len(bug_check) > 0:
        print("\nWARNING: Found finalists marked as multi-elimination!")
        print(bug_check[existing_cols])
    else:
        print("\nNo bugs detected - finalists correctly marked!")

    print(f"\nOutput saved to: {output_file}")

    return df, week_info


# ========== 主程序 ==========
if __name__ == "__main__":
    input_file = "dwts_data_processed_percentage.csv"
    output_file = "dwts_data_with_special_weeks_fixed.csv"
    week_table_file = "week_elimination_info_fixed.csv"

    try:
        df, week_info = process_special_weeks_FIXED(
            input_file, output_file, week_table_file
        )

        print("\n" + "=" * 70)
        print("SUCCESS! Bug fixed - finals no longer marked as multi-elimination")
        print("=" * 70)

    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback

        traceback.print_exc()
