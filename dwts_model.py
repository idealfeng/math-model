import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# ==================== 数据加载和预处理 ====================


class DWTSDataPreprocessor:
    """DWTS数据预处理器 - 修复版"""

    def __init__(self, data_dir="./data/"):
        self.data_dir = data_dir

    def load_all_data(self):
        """加载所有数据文件"""
        self.main_df = pd.read_csv(f"{self.data_dir}2026_MCM_Problem_C_Data.csv")

        # 修复：读取Excel后清理列名
        self.zero_elim = pd.read_excel(f"{self.data_dir}zero_eliminate.xlsx")
        self.zero_elim.columns = self.zero_elim.columns.str.strip()  # 去除空格

        self.multi_elim = pd.read_excel(f"{self.data_dir}multi_eliminate.xlsx")
        self.multi_elim.columns = self.multi_elim.columns.str.strip()

        self.topscore = pd.read_excel(f"{self.data_dir}topscore.xlsx")
        self.sharpness = pd.read_excel(f"{self.data_dir}sharpness.xlsx")
        self.growth = pd.read_excel(f"{self.data_dir}growth.xlsx")
        self.personal = pd.read_excel(f"{self.data_dir}personal.xlsx")
        self.relative_rank = pd.read_excel(f"{self.data_dir}relativeRank.xlsx")

        return self

    def extract_judge_scores(self):
        """提取评委分数（周平均）"""
        judge_cols = [
            col
            for col in self.main_df.columns
            if "judge" in col.lower() and "score" in col.lower()
        ]

        weeks_data = {}
        for col in judge_cols:
            parts = col.split("_")
            week = parts[0]

            if week not in weeks_data:
                weeks_data[week] = []
            weeks_data[week].append(col)

        for week, cols in weeks_data.items():
            self.main_df[f"{week}_avg_judge_score"] = self.main_df[cols].mean(axis=1)

        return self

    def extract_elimination_info(self):
        """提取淘汰信息"""

        def get_elimination_week(result):
            if pd.isna(result):
                return None
            result_str = str(result)
            if "Eliminated Week" in result_str:
                try:
                    return int(result_str.split("Week")[-1].strip())
                except:
                    return None
            elif any(
                x in result_str
                for x in [
                    "1st Place",
                    "2nd Place",
                    "3rd Place",
                    "4th Place",
                    "5th Place",
                ]
            ):
                return 999
            return None

        self.main_df["elimination_week"] = self.main_df["results"].apply(
            get_elimination_week
        )
        return self

    def build_weekly_dataset(self):
        """构建周级别的数据集 - 优化版"""
        weekly_data = []
        state_memory = {}  # 记录每个选手的state历史

        for season in sorted(self.main_df["season"].unique()):
            season_df = self.main_df[self.main_df["season"] == season].copy()
            season_num_contestants = int(len(season_df))

            # 获取该季度的周数
            week_cols = [
                col
                for col in season_df.columns
                if "week" in col and "avg_judge_score" in col
            ]
            max_week = int(len(week_cols)) if len(week_cols) > 0 else 1

            # 重置state记忆
            state_memory = {}

            for week in range(1, max_week + 1):
                week_str = f"week{week}"
                week_pct = float(week) / float(max_week) if max_week > 0 else 0.0

                # 该周所有选手（包括即将被淘汰的）
                active_contestants = season_df[
                    (season_df["elimination_week"].isna())
                    | (season_df["elimination_week"] >= week)
                ].copy()

                if len(active_contestants) == 0:
                    continue

                # 该周被淘汰的选手
                eliminated_this_week = season_df[season_df["elimination_week"] == week][
                    "celebrity_name"
                ].tolist()

                for idx, row in active_contestants.iterrows():
                    contestant_name = row["celebrity_name"]
                    partner_name = row["ballroom_partner"]
                    contestant_key = f"{contestant_name}_{partner_name}_{season}"

                    # 获取评委分数
                    judge_score = row.get(f"{week_str}_avg_judge_score", np.nan)
                    if pd.isna(judge_score):
                        continue

                    # 静态特征
                    personal_info = self.personal[
                        (self.personal["celebrity_name"] == contestant_name)
                        & (self.personal["ballroom_partner"] == partner_name)
                        & (self.personal["season"] == season)
                    ]

                    if len(personal_info) == 0:
                        continue
                    personal_info = personal_info.iloc[0]

                    # 动态特征 - State (成长潜力)
                    alpha = 0.7
                    try:
                        growth_idx = (
                            self.growth[
                                (self.growth.index == idx)
                                | (abs(self.growth.index - idx) < 0.5)
                            ].index[0]
                            if len(self.growth) > idx
                            else idx
                        )

                        current_score_expected = self.growth.loc[
                            growth_idx, f"{week_str}_judge_score_expected"
                        ]

                        if pd.isna(current_score_expected):
                            current_score_expected = 0.0
                    except:
                        current_score_expected = 0.0

                    if week == 1:
                        state = float(current_score_expected)
                    else:
                        prev_state = state_memory.get(contestant_key, 0.0)
                        state = (
                            alpha * float(current_score_expected)
                            + (1 - alpha) * prev_state
                        )

                    state_memory[contestant_key] = state

                    # MaxScore
                    try:
                        max_score_row = self.topscore[
                            (self.topscore["celebrity_name"] == contestant_name)
                            & (self.topscore["ballroom_partner"] == partner_name)
                            & (self.topscore["season"] == season)
                        ]
                        max_score = (
                            max_score_row.iloc[0][f"{week_str}_max_score"]
                            if len(max_score_row) > 0
                            else 0
                        )
                        if pd.isna(max_score):
                            max_score = 0.0
                    except:
                        max_score = 0.0

                    # Sharpness
                    try:
                        sharpness_row = self.sharpness[
                            (self.sharpness["celebrity_name"] == contestant_name)
                            & (self.sharpness["ballroom_partner"] == partner_name)
                            & (self.sharpness["season"] == season)
                        ]
                        sharpness = (
                            sharpness_row.iloc[0][week_str]
                            if len(sharpness_row) > 0
                            else 0
                        )
                        if pd.isna(sharpness):
                            sharpness = 0.0
                    except:
                        sharpness = 0.0

                    # PrevRank
                    try:
                        rank_row = self.relative_rank[
                            (self.relative_rank["celebrity_name"] == contestant_name)
                            & (self.relative_rank["ballroom_partner"] == partner_name)
                            & (self.relative_rank["season"] == season)
                        ]
                        prev_rank = (
                            rank_row.iloc[0][f"{week_str}_relative_rank"]
                            if len(rank_row) > 0
                            else 0.5
                        )
                        if pd.isna(prev_rank):
                            prev_rank = 0.5
                    except:
                        prev_rank = 0.5

                    # 意外变量
                    is_all_star = 1 if season == 15 else 0

                    try:
                        zero_elim_row = self.zero_elim[
                            self.zero_elim["zero_eliminate"] == season
                        ]
                        no_elim = (
                            zero_elim_row.iloc[0][week_str]
                            if len(zero_elim_row) > 0
                            else 0
                        )
                        if pd.isna(no_elim):
                            no_elim = 0
                    except:
                        no_elim = 0

                    try:
                        multi_elim_row = self.multi_elim[
                            self.multi_elim["multi_eliminate"] == season
                        ]
                        multi_elim_flag = (
                            multi_elim_row.iloc[0][week_str]
                            if len(multi_elim_row) > 0
                            else 0
                        )
                        if pd.isna(multi_elim_flag):
                            multi_elim_flag = 0
                    except:
                        multi_elim_flag = 0

                    # 是否被淘汰
                    is_eliminated = 1 if contestant_name in eliminated_this_week else 0

                    sample = {
                        "contestant_id": contestant_key,
                        "season": season,
                        "week": week,
                        "num_contestants": len(active_contestants),
                        "week_pct": float(week_pct),
                        "contestants_pct": float(len(active_contestants))
                        / float(season_num_contestants)
                        if season_num_contestants > 0
                        else 0.0,
                        # 静态特征
                        "age": float(personal_info["age_A"]),
                        "fame": float(personal_info["Frame_A"]),
                        "gender": float(personal_info["sex_A"]),
                        "industry": float(personal_info["profession_A_influence_0_1"]),
                        "experience": float(personal_info["times_A"]),
                        "is_hetero": float(1 - personal_info["is_sexSame"]),
                        # 动态特征
                        "state": float(state),
                        "max_score": float(max_score),
                        "sharpness": float(sharpness),
                        "prev_rank": float(prev_rank),
                        # 评委分数
                        "judge_score": float(judge_score),
                        # 意外变量
                        "is_all_star": float(is_all_star),
                        "no_elim": float(no_elim),
                        "multi_elim": float(multi_elim_flag),
                        # 目标
                        "is_eliminated": float(is_eliminated),
                    }

                    weekly_data.append(sample)

        self.weekly_df = pd.DataFrame(weekly_data)

        # 清理无效数据
        self.weekly_df = self.weekly_df.dropna(subset=["judge_score"])
        self.weekly_df = self.weekly_df[self.weekly_df["judge_score"] > 0]

        return self

    def prepare_training_data(self):
        """准备训练数据"""
        self.weekly_df["season_week"] = (
            self.weekly_df["season"].astype(str)
            + "_"
            + self.weekly_df["week"].astype(str)
        )

        # 按季度标准化人气
        for season in self.weekly_df["season"].unique():
            mask = self.weekly_df["season"] == season
            fame_mean = self.weekly_df.loc[mask, "fame"].mean()
            fame_std = self.weekly_df.loc[mask, "fame"].std()

            if fame_std > 0:
                self.weekly_df.loc[mask, "fame_normalized"] = (
                    self.weekly_df.loc[mask, "fame"] - fame_mean
                ) / fame_std
            else:
                self.weekly_df.loc[mask, "fame_normalized"] = 0.0

        # 标准化其他连续特征
        for col in ["age", "state", "max_score"]:
            mean_val = self.weekly_df[col].mean()
            std_val = self.weekly_df[col].std()
            if std_val > 0:
                self.weekly_df[f"{col}_normalized"] = (
                    self.weekly_df[col] - mean_val
                ) / std_val
            else:
                self.weekly_df[f"{col}_normalized"] = 0.0

        # Fill remaining NaNs to keep torch tensors finite
        self.weekly_df = self.weekly_df.fillna(0)

        # Integer id for contestant embedding (stable within this dataset)
        self.weekly_df["contestant_idx"] = (
            pd.factorize(self.weekly_df["contestant_id"])[0].astype(int)
        )

        return self


# ==================== 改进的PyTorch模型 ====================


class DWTSModel(nn.Module):
    """DWTS MNL model used to estimate fan vote probabilities."""

    def __init__(
        self,
        n_static_features: int,
        n_dynamic_features: int,
        n_context_features: int,
        tau: float = 0.5,
    ):
        super().__init__()
        self.tau = tau

        # Linear utility parameters
        self.beta = nn.Parameter(torch.randn(n_static_features))
        self.gamma = nn.Parameter(torch.randn(n_dynamic_features))
        self.delta = nn.Parameter(torch.randn(n_context_features))

    def compute_utility(self, x_static, x_dynamic, x_context):
        eta = (
            torch.matmul(x_static, self.beta)
            + torch.matmul(x_dynamic, self.gamma)
            + torch.matmul(x_context, self.delta)
        )
        return eta

    def compute_fan_prob(self, eta):
        # Stable softmax
        eta = eta - eta.max()
        exp_eta = torch.exp(eta)
        return exp_eta / (exp_eta.sum() + 1e-12)

    def soft_rank_optimized(self, P):
        """
        Vectorized soft-rank:
          R_i = 1 + Σ_{j≠i} σ((P_j - P_i) / τ)
        """
        diff_matrix = P.unsqueeze(0) - P.unsqueeze(1)  # [n,n] = P_j - P_i
        sigmoid_matrix = torch.sigmoid(diff_matrix / self.tau)
        # exclude diagonal (σ(0)=0.5)
        return 1.0 + sigmoid_matrix.sum(dim=1) - torch.diag(sigmoid_matrix)

    def forward(self, x_static, x_dynamic, x_context, judge_scores, method="rank"):
        eta = self.compute_utility(x_static, x_dynamic, x_context)
        eta = torch.clamp(eta, min=-10, max=10)
        P_fan = self.compute_fan_prob(eta)

        n = int(judge_scores.shape[0])

        if method == "rank":
            # Ranks start at 1, smaller is better
            judge_rank = (
                torch.argsort(torch.argsort(judge_scores, descending=True)).float() + 1
            )
            fan_rank = self.soft_rank_optimized(P_fan)
            combined_rank = judge_rank + fan_rank  # smaller is better

            # Convert to "higher is better" score for the loss
            max_possible_rank = 2 * n
            combined_score = max_possible_rank - combined_rank
            return combined_score, P_fan

        # percentage
        judge_rank = torch.argsort(torch.argsort(judge_scores, descending=True))
        judge_pct = (n - judge_rank.float() - 1) / max(n - 1, 1)
        combined_score = judge_pct + P_fan
        return combined_score, P_fan


class DWTSLoss(nn.Module):
    """似然损失 - 数值稳定版"""

    def __init__(self):
        super(DWTSLoss, self).__init__()

    def forward(self, combined_scores, eliminated_mask, no_elim_flag):
        """
        成对比较损失
        combined_scores: 分数越高越好
        """
        if no_elim_flag == 1:
            # No-elimination week provides no supervision; keep graph connectivity.
            return combined_scores.sum() * 0.0

        eliminated_idx = torch.where(eliminated_mask == 1)[0]
        survived_idx = torch.where(eliminated_mask == 0)[0]

        if len(eliminated_idx) == 0 or len(survived_idx) == 0:
            return combined_scores.sum() * 0.0

        # Vectorized pairwise loss: sum_{j in survived, i in eliminated} log(sigmoid(score_j - score_i))
        survived_scores = combined_scores[survived_idx]  # [S]
        eliminated_scores = combined_scores[eliminated_idx]  # [E]
        diff = survived_scores.unsqueeze(1) - eliminated_scores.unsqueeze(0)  # [S, E]
        log_likelihood = torch.nn.functional.logsigmoid(diff).sum()
        return -log_likelihood


class DWTSFocalPairwiseLoss(nn.Module):
    """
    Focal-style pairwise loss for elimination comparisons.

    For each survived-vs-eliminated pair, we want p = sigmoid(score_surv - score_elim) -> 1.
    Focal weighting down-weights easy pairs:
        loss = - Σ (1 - p)^gamma * log(p)
    """

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be >= 0")
        self.gamma = float(gamma)

    def forward(self, combined_scores, eliminated_mask, no_elim_flag):
        if no_elim_flag == 1:
            return combined_scores.sum() * 0.0

        eliminated_idx = torch.where(eliminated_mask == 1)[0]
        survived_idx = torch.where(eliminated_mask == 0)[0]

        if len(eliminated_idx) == 0 or len(survived_idx) == 0:
            return combined_scores.sum() * 0.0

        survived_scores = combined_scores[survived_idx]  # [S]
        eliminated_scores = combined_scores[eliminated_idx]  # [E]
        diff = survived_scores.unsqueeze(1) - eliminated_scores.unsqueeze(0)  # [S, E]

        # p = sigmoid(diff), log(p) = logsigmoid(diff)
        logp = torch.nn.functional.logsigmoid(diff)
        if self.gamma == 0.0:
            return -logp.sum()

        p = torch.sigmoid(diff)
        w = torch.pow(1.0 - p.detach(), self.gamma)
        return -(w * logp).sum()


class UnifiedDWTSModel(nn.Module):
    """统一 Fan Vote 模型：参数唯一，组合规则按 season 自动选择。"""

    def __init__(
        self,
        n_static,
        n_dynamic,
        n_context,
        tau=0.5,
        hidden=32,
        dropout=0.1,
        num_contestants=None,
        emb_dim=8,
        learn_mixing_weights: bool = False,
        fan_utility: str = "mlp",  # "mlp" or "linear"
    ):
        super().__init__()
        self.tau = tau
        self.learn_mixing_weights = bool(learn_mixing_weights)
        self.fan_utility = str(fan_utility).lower().strip()

        # Fan-utility network (allows nonlinear interactions).
        # We also allow current performance (judge score) to influence fan vote as a proxy for "performance effect".
        if num_contestants is None:
            raise ValueError("num_contestants is required for the unified model.")

        base_dim = n_static + n_dynamic + n_context + 1  # + judge_score_z
        if self.fan_utility == "mlp":
            self.contestant_emb = nn.Embedding(num_contestants, emb_dim)
            in_dim = base_dim + emb_dim  # + emb
            self.fan_net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )
            self.fan_linear = None
            self.contestant_bias = None
        elif self.fan_utility == "linear":
            # Linear eta_{i,w,s} = theta_i + w^T x_{i,w,s}
            # theta_i is implemented as a learned per-contestant scalar bias (random effect).
            self.contestant_emb = None
            self.fan_net = None
            self.fan_linear = nn.Linear(base_dim, 1, bias=False)
            self.contestant_bias = nn.Embedding(num_contestants, 1)
        else:
            raise ValueError("fan_utility must be 'mlp' or 'linear'.")

        # Mixing weights between judge component and fan component.
        # IMPORTANT:
        # - For "official-rule" reproduction/counterfactuals, keep weights fixed (1,1).
        # - Learnable weights can improve in-sample fit but entangle fan-vote estimates with reweighting.
        # Use softplus to keep weights positive and avoid sign flips.
        if self.learn_mixing_weights:
            self._w_j_rank = nn.Parameter(torch.tensor(0.0))
            self._w_f_rank = nn.Parameter(torch.tensor(0.0))
            self._w_j_pct = nn.Parameter(torch.tensor(0.0))
            self._w_f_pct = nn.Parameter(torch.tensor(0.0))
        else:
            self._w_j_rank = None
            self._w_f_rank = None
            self._w_j_pct = None
            self._w_f_pct = None

    @staticmethod
    def _softmax_stable(eta):
        eta = eta - eta.max()
        exp_eta = torch.exp(eta)
        return exp_eta / (exp_eta.sum() + 1e-12)

    def compute_fan_prob(self, x_static, x_dynamic, x_context, judge_scores, contestant_idx):
        # Standardize judge scores within the week to a stable scale (performance effect).
        mean = judge_scores.mean()
        std = judge_scores.std(unbiased=False)
        if std < 1e-6:
            judge_z = judge_scores * 0.0
        else:
            judge_z = (judge_scores - mean) / std

        x_base = torch.cat([x_static, x_dynamic, x_context, judge_z.unsqueeze(1)], dim=1)

        if self.fan_utility == "mlp":
            emb = self.contestant_emb(contestant_idx)  # [n, emb_dim]
            x = torch.cat([x_base, emb], dim=1)  # [n, in_dim]
            eta = self.fan_net(x).squeeze(1)  # [n]
        else:
            eta = self.fan_linear(x_base).squeeze(1)  # [n]
            eta = eta + self.contestant_bias(contestant_idx).squeeze(1)

        eta = torch.clamp(eta, -10, 10)
        return self._softmax_stable(eta)

    def soft_rank(self, P):
        diff = P.unsqueeze(0) - P.unsqueeze(1)  # P_j - P_i
        sigmoid = torch.sigmoid(diff / self.tau)
        return 1.0 + sigmoid.sum(dim=1) - torch.diag(sigmoid)

    def combine_scores(self, P_fan, judge_scores, method):
        n = int(judge_scores.shape[0])

        if method == "rank":
            j_rank = torch.argsort(torch.argsort(judge_scores, descending=True)).float() + 1
            f_rank = self.soft_rank(P_fan)
            if self.learn_mixing_weights:
                w_j = torch.nn.functional.softplus(self._w_j_rank) + 1e-6
                w_f = torch.nn.functional.softplus(self._w_f_rank) + 1e-6
            else:
                w_j = 1.0
                w_f = 1.0
            combined = w_j * j_rank + w_f * f_rank
            # 转成高分=好
            return (2 * n) - combined

        # percentage
        # DWTS percent rule: judge percent is score share (not rank percentile).
        denom = judge_scores.sum() + 1e-12
        j_pct = judge_scores / denom
        if self.learn_mixing_weights:
            w_j = torch.nn.functional.softplus(self._w_j_pct) + 1e-6
            w_f = torch.nn.functional.softplus(self._w_f_pct) + 1e-6
        else:
            w_j = 1.0
            w_f = 1.0
        return w_j * j_pct + w_f * P_fan

    def get_mixing_weights(self) -> dict:
        """Return effective mixing weights for reporting/debugging."""
        if not self.learn_mixing_weights:
            return {"mode": "fixed", "rank": (1.0, 1.0), "percentage": (1.0, 1.0)}

        w_j_rank = float(torch.nn.functional.softplus(self._w_j_rank).detach().cpu().item()) + 1e-6
        w_f_rank = float(torch.nn.functional.softplus(self._w_f_rank).detach().cpu().item()) + 1e-6
        w_j_pct = float(torch.nn.functional.softplus(self._w_j_pct).detach().cpu().item()) + 1e-6
        w_f_pct = float(torch.nn.functional.softplus(self._w_f_pct).detach().cpu().item()) + 1e-6
        return {"mode": "learned", "rank": (w_j_rank, w_f_rank), "percentage": (w_j_pct, w_f_pct)}

    @staticmethod
    def choose_method_by_season(season: int) -> str:
        # 历史规则：Season 1-2 和 >=28 用 rank，否则用 percentage
        return "rank" if (season in [1, 2] or season >= 28) else "percentage"

    def forward(self, x_static, x_dynamic, x_context, judge_scores, contestant_idx, season: int):
        P_fan = self.compute_fan_prob(
            x_static, x_dynamic, x_context, judge_scores, contestant_idx
        )
        method = self.choose_method_by_season(int(season))
        combined_score = self.combine_scores(P_fan, judge_scores, method)
        return combined_score, P_fan, method


def train_dwts_model(weekly_df, num_epochs=1000, lr=0.001, method="rank", weight_decay=0.01):
    """修复的训练函数"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    static_features = [
        "age_normalized",
        "fame_normalized",
        "gender",
        "industry",
        "experience",
        "is_hetero",
    ]
    dynamic_features = [
        "state_normalized",
        "max_score_normalized",
        "sharpness",
        "prev_rank",
    ]
    context_features = ["is_all_star", "no_elim", "multi_elim", "week_pct", "contestants_pct"]

    # 打印特征统计
    print("\n=== Feature Statistics ===")
    for col in static_features + dynamic_features + context_features:
        if col in weekly_df.columns:
            mean_val = weekly_df[col].mean()
            std_val = weekly_df[col].std()
            nonzero_pct = (weekly_df[col] != 0).mean() * 100
            print(
                f"{col:25s} mean={mean_val:7.3f}, std={std_val:7.3f}, nonzero={nonzero_pct:5.1f}%"
            )

    model = DWTSModel(
        n_static_features=len(static_features),
        n_dynamic_features=len(dynamic_features),
        n_context_features=len(context_features),
        tau=0.5,
    ).to(device)

    criterion = DWTSLoss()

    # 修复：分组优化（context特征不正则化）
    optimizer = optim.Adam(
        [
            {"params": [model.beta, model.gamma], "weight_decay": weight_decay},
            {"params": [model.delta], "weight_decay": 0.0},
        ],
        lr=lr,
    )

    # NOTE: Some torch versions don't accept the 'verbose' kwarg; keep args minimal for compatibility.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=50
    )

    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_weeks = 0

        for season_week, group in weekly_df.groupby("season_week"):
            if len(group) < 2:
                continue

            try:
                x_static = torch.tensor(
                    group[static_features].values, dtype=torch.float32
                ).to(device)
                x_dynamic = torch.tensor(
                    group[dynamic_features].values, dtype=torch.float32
                ).to(device)
                x_context = torch.tensor(
                    group[context_features].values, dtype=torch.float32
                ).to(device)
                judge_scores = torch.tensor(
                    group["judge_score"].values, dtype=torch.float32
                ).to(device)
                eliminated_mask = torch.tensor(
                    group["is_eliminated"].values, dtype=torch.float32
                ).to(device)
                no_elim = group["no_elim"].values[0]

                combined_scores, P_fan = model(
                    x_static, x_dynamic, x_context, judge_scores, method=method
                )
                loss = criterion(combined_scores, eliminated_mask, no_elim)

                if torch.isnan(loss):
                    continue

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                epoch_loss += loss.item()
                num_weeks += 1

            except Exception as e:
                continue

        avg_loss = epoch_loss / num_weeks if num_weeks > 0 else 0
        losses.append(avg_loss)
        scheduler.step(avg_loss)

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    return model, losses


def train_unified_model(
    weekly_df,
    num_epochs=600,
    lr=0.003,
    weight_decay=0.001,
    hidden=64,
    dropout=0.15,
    learn_mixing_weights: bool = False,
    fan_utility: str = "mlp",
    focal_gamma: float | None = None,
    skip_unsupervised_weeks: bool = True,
    feature_noise_std: float | None = None,
):
    """训练统一模型（只有一套参数，按 season 自动选组合方式）。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    static_features = [
        "age_normalized",
        "fame_normalized",
        "gender",
        "industry",
        "experience",
        "is_hetero",
    ]
    dynamic_features = [
        "state_normalized",
        "max_score_normalized",
        "sharpness",
        "prev_rank",
    ]
    context_features = ["is_all_star", "no_elim", "multi_elim", "week_pct", "contestants_pct"]

    num_contestants = int(weekly_df["contestant_idx"].max()) + 1
    model = UnifiedDWTSModel(
        n_static=len(static_features),
        n_dynamic=len(dynamic_features),
        n_context=len(context_features),
        tau=0.5,
        hidden=hidden,
        dropout=dropout,
        num_contestants=num_contestants,
        emb_dim=8,
        learn_mixing_weights=learn_mixing_weights,
        fan_utility=fan_utility,
    ).to(device)

    if focal_gamma is None:
        criterion = DWTSLoss()
    else:
        criterion = DWTSFocalPairwiseLoss(gamma=float(focal_gamma))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=50
    )

    # Cache per-week tensors once (much faster than regrouping every epoch)
    batches = []
    for season_week, group in weekly_df.groupby("season_week"):
        if len(group) < 2:
            continue

        season = int(group["season"].values[0])
        no_elim = float(group["no_elim"].values[0])

        x_static = torch.tensor(group[static_features].values, dtype=torch.float32).to(
            device
        )
        x_dynamic = torch.tensor(group[dynamic_features].values, dtype=torch.float32).to(
            device
        )
        x_context = torch.tensor(group[context_features].values, dtype=torch.float32).to(
            device
        )
        judge_scores = torch.tensor(group["judge_score"].values, dtype=torch.float32).to(
            device
        )
        eliminated_mask = torch.tensor(group["is_eliminated"].values, dtype=torch.float32).to(
            device
        )
        contestant_idx = torch.tensor(group["contestant_idx"].values, dtype=torch.long).to(
            device
        )

        batches.append(
            (
                season,
                no_elim,
                x_static,
                x_dynamic,
                x_context,
                judge_scores,
                contestant_idx,
                eliminated_mask,
            )
        )

    if skip_unsupervised_weeks:
        filtered = []
        for b in batches:
            season, no_elim, x_static, x_dynamic, x_context, judge_scores, contestant_idx, eliminated_mask = b
            if no_elim == 1.0:
                continue
            k = float(eliminated_mask.sum().detach().cpu().item())
            if k <= 0.0 or k >= float(eliminated_mask.shape[0]):
                continue
            filtered.append(b)
        batches = filtered

    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_weeks = 0

        for (
            season,
            no_elim,
            x_static,
            x_dynamic,
            x_context,
            judge_scores,
            contestant_idx,
            eliminated_mask,
        ) in batches:
            if feature_noise_std is not None and float(feature_noise_std) > 0.0:
                s = float(feature_noise_std)
                x_static_in = x_static + s * torch.randn_like(x_static)
                x_dynamic_in = x_dynamic + s * torch.randn_like(x_dynamic)
                x_context_in = x_context + s * torch.randn_like(x_context)
            else:
                x_static_in = x_static
                x_dynamic_in = x_dynamic
                x_context_in = x_context

            combined_scores, _, _ = model(
                x_static_in,
                x_dynamic_in,
                x_context_in,
                judge_scores,
                contestant_idx,
                season,
            )
            loss = criterion(combined_scores, eliminated_mask, no_elim)

            if torch.isnan(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += float(loss.item())
            num_weeks += 1

        avg_loss = epoch_loss / num_weeks if num_weeks > 0 else 0.0
        losses.append(avg_loss)
        scheduler.step(avg_loss)

        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    return model, losses


def train_unified_ensemble(
    weekly_df: pd.DataFrame,
    ensemble_size: int = 5,
    seeds: list[int] | None = None,
    num_epochs: int = 80,
    hidden: int = 64,
    dropout: float = 0.15,
    lr: float = 0.003,
    weight_decay: float = 0.001,
    fan_utility: str = "mlp",
    focal_gamma: float | None = None,
    feature_noise_std: float | None = None,
):
    """Train multiple unified models with different random seeds (bagging by init)."""
    if ensemble_size <= 0:
        raise ValueError("ensemble_size must be >= 1")

    if seeds is None:
        seeds = list(range(ensemble_size))
    if len(seeds) < ensemble_size:
        raise ValueError("seeds length must be >= ensemble_size")

    models = []
    for i in range(ensemble_size):
        seed = int(seeds[i])
        torch.manual_seed(seed)
        np.random.seed(seed)
        model, _ = train_unified_model(
            weekly_df,
            num_epochs=num_epochs,
            lr=lr,
            weight_decay=weight_decay,
            hidden=hidden,
            dropout=dropout,
            learn_mixing_weights=False,
            fan_utility=fan_utility,
            focal_gamma=focal_gamma,
            skip_unsupervised_weeks=True,
            feature_noise_std=feature_noise_std,
        )
        models.append(model)
    return models


def evaluate_predictions_unified_ensemble(weekly_df: pd.DataFrame, models: list[nn.Module]):
    """
    Evaluate an ensemble by averaging per-model combined scores (higher=better) within each week.
    Predict eliminated contestants as the k lowest average combined-scores.
    """
    if len(models) == 0:
        raise ValueError("models must be non-empty")

    device = next(models[0].parameters()).device
    for m in models[1:]:
        if next(m.parameters()).device != device:
            raise ValueError("All ensemble models must be on the same device.")

    for m in models:
        m.eval()

    static_features = [
        "age_normalized",
        "fame_normalized",
        "gender",
        "industry",
        "experience",
        "is_hetero",
    ]
    dynamic_features = [
        "state_normalized",
        "max_score_normalized",
        "sharpness",
        "prev_rank",
    ]
    context_features = ["is_all_star", "no_elim", "multi_elim", "week_pct", "contestants_pct"]

    total = 0
    hit_sum = 0.0
    exact_match = 0

    weeks_rank = 0
    weeks_pct = 0
    hit_sum_rank = 0.0
    hit_sum_pct = 0.0

    with torch.no_grad():
        for season_week, group in weekly_df.groupby("season_week"):
            if len(group) < 2:
                continue
            if float(group["no_elim"].values[0]) == 1.0:
                continue

            actual_elim_idx = group[group["is_eliminated"] == 1].index
            if len(actual_elim_idx) == 0:
                continue
            k = int(len(actual_elim_idx))

            season = int(group["season"].values[0])
            method = UnifiedDWTSModel.choose_method_by_season(season)
            if method == "rank":
                weeks_rank += 1
            else:
                weeks_pct += 1

            x_static = torch.tensor(group[static_features].values, dtype=torch.float32).to(device)
            x_dynamic = torch.tensor(group[dynamic_features].values, dtype=torch.float32).to(device)
            x_context = torch.tensor(group[context_features].values, dtype=torch.float32).to(device)
            judge_scores = torch.tensor(group["judge_score"].values, dtype=torch.float32).to(device)
            contestant_idx = torch.tensor(group["contestant_idx"].values, dtype=torch.long).to(device)

            scores = []
            for m in models:
                combined_scores, _, _ = m(x_static, x_dynamic, x_context, judge_scores, contestant_idx, season)
                scores.append(combined_scores)
            avg_scores = torch.stack(scores, dim=0).mean(dim=0)

            pred_positions = torch.argsort(avg_scores)[:k].tolist()
            predicted_elim_idx = set(group.index[pred_positions].tolist())
            actual_set = set(actual_elim_idx.tolist())
            hits = len(predicted_elim_idx & actual_set)
            hit_rate = hits / k if k > 0 else 0.0
            is_exact = 1 if hits == k else 0

            total += 1
            hit_sum += hit_rate
            exact_match += is_exact
            if method == "rank":
                hit_sum_rank += hit_rate
            else:
                hit_sum_pct += hit_rate

    overall_hit = hit_sum / total if total > 0 else 0.0
    overall_exact = exact_match / total if total > 0 else 0.0
    rank_hit = hit_sum_rank / weeks_rank if weeks_rank > 0 else 0.0
    pct_hit = hit_sum_pct / weeks_pct if weeks_pct > 0 else 0.0

    print("\n" + "=" * 60)
    print("ENSEMBLE EVALUATION RESULTS")
    print("=" * 60)
    print(f"Ensemble size: {len(models)}")
    print(f"HitRate@|E_w| (avg combined score): {overall_hit:.2%} (weeks={total})")
    print(f"ExactMatch@|E_w|: {overall_exact:.2%} (weeks={total})")
    print(f"  - Rank weeks HitRate: {rank_hit:.2%} (weeks={weeks_rank})")
    print(f"  - Percentage weeks HitRate: {pct_hit:.2%} (weeks={weeks_pct})")

    return {"hit_rate": overall_hit, "exact_match": overall_exact, "weeks": total}


def evaluate_predictions_unified_dual_ensemble(
    weekly_df: pd.DataFrame,
    models_rank: list[nn.Module],
    models_percentage: list[nn.Module],
):
    """
    Evaluate using two ensembles:
      - models_rank for seasons using rank combination
      - models_percentage for seasons using percentage combination

    This is useful when different loss settings (e.g., focal gamma) work better for different rules.
    """
    if len(models_rank) == 0 or len(models_percentage) == 0:
        raise ValueError("Both models_rank and models_percentage must be non-empty.")

    device = next(models_rank[0].parameters()).device
    for m in models_rank[1:]:
        if next(m.parameters()).device != device:
            raise ValueError("All models_rank must be on the same device.")
    for m in models_percentage:
        if next(m.parameters()).device != device:
            raise ValueError("All models_percentage must be on the same device as models_rank.")

    for m in models_rank:
        m.eval()
    for m in models_percentage:
        m.eval()

    static_features = [
        "age_normalized",
        "fame_normalized",
        "gender",
        "industry",
        "experience",
        "is_hetero",
    ]
    dynamic_features = [
        "state_normalized",
        "max_score_normalized",
        "sharpness",
        "prev_rank",
    ]
    context_features = ["is_all_star", "no_elim", "multi_elim", "week_pct", "contestants_pct"]

    total = 0
    hit_sum = 0.0
    exact_match = 0

    weeks_rank = 0
    weeks_pct = 0
    hit_sum_rank = 0.0
    hit_sum_pct = 0.0

    with torch.no_grad():
        for season_week, group in weekly_df.groupby("season_week"):
            if len(group) < 2:
                continue
            if float(group["no_elim"].values[0]) == 1.0:
                continue

            actual_elim_idx = group[group["is_eliminated"] == 1].index
            if len(actual_elim_idx) == 0:
                continue
            k = int(len(actual_elim_idx))

            season = int(group["season"].values[0])
            method = UnifiedDWTSModel.choose_method_by_season(season)
            models = models_rank if method == "rank" else models_percentage
            if method == "rank":
                weeks_rank += 1
            else:
                weeks_pct += 1

            x_static = torch.tensor(group[static_features].values, dtype=torch.float32).to(device)
            x_dynamic = torch.tensor(group[dynamic_features].values, dtype=torch.float32).to(device)
            x_context = torch.tensor(group[context_features].values, dtype=torch.float32).to(device)
            judge_scores = torch.tensor(group["judge_score"].values, dtype=torch.float32).to(device)
            contestant_idx = torch.tensor(group["contestant_idx"].values, dtype=torch.long).to(device)

            scores = []
            for m in models:
                combined_scores, _, _ = m(
                    x_static, x_dynamic, x_context, judge_scores, contestant_idx, season
                )
                scores.append(combined_scores)
            avg_scores = torch.stack(scores, dim=0).mean(dim=0)

            pred_positions = torch.argsort(avg_scores)[:k].tolist()
            predicted_elim_idx = set(group.index[pred_positions].tolist())
            actual_set = set(actual_elim_idx.tolist())
            hits = len(predicted_elim_idx & actual_set)
            hit_rate = hits / k if k > 0 else 0.0
            is_exact = 1 if hits == k else 0

            total += 1
            hit_sum += hit_rate
            exact_match += is_exact
            if method == "rank":
                hit_sum_rank += hit_rate
            else:
                hit_sum_pct += hit_rate

    overall_hit = hit_sum / total if total > 0 else 0.0
    overall_exact = exact_match / total if total > 0 else 0.0
    rank_hit = hit_sum_rank / weeks_rank if weeks_rank > 0 else 0.0
    pct_hit = hit_sum_pct / weeks_pct if weeks_pct > 0 else 0.0

    print("\n" + "=" * 60)
    print("DUAL-ENSEMBLE EVALUATION RESULTS")
    print("=" * 60)
    print(f"Rank ensemble size: {len(models_rank)}")
    print(f"Percentage ensemble size: {len(models_percentage)}")
    print(f"HitRate@|E_w|: {overall_hit:.2%} (weeks={total})")
    print(f"ExactMatch@|E_w|: {overall_exact:.2%} (weeks={total})")
    print(f"  - Rank weeks HitRate: {rank_hit:.2%} (weeks={weeks_rank})")
    print(f"  - Percentage weeks HitRate: {pct_hit:.2%} (weeks={weeks_pct})")

    return {"hit_rate": overall_hit, "exact_match": overall_exact, "weeks": total}


def predict_fan_votes_unified_ensemble(weekly_df: pd.DataFrame, models: list[nn.Module]):
    """
    Produce per-row ensemble fan-vote probability mean/std and combined-score mean/std.
    Note: combined_score is the rule-consistent score used for elimination (higher=better).
    """
    if len(models) == 0:
        raise ValueError("models must be non-empty")

    device = next(models[0].parameters()).device
    for m in models[1:]:
        if next(m.parameters()).device != device:
            raise ValueError("All ensemble models must be on the same device.")

    for m in models:
        m.eval()

    static_features = [
        "age_normalized",
        "fame_normalized",
        "gender",
        "industry",
        "experience",
        "is_hetero",
    ]
    dynamic_features = [
        "state_normalized",
        "max_score_normalized",
        "sharpness",
        "prev_rank",
    ]
    context_features = ["is_all_star", "no_elim", "multi_elim", "week_pct", "contestants_pct"]

    out = weekly_df.copy()
    out["predicted_fan_vote_mean"] = np.nan
    out["predicted_fan_vote_std"] = np.nan
    out["combined_score_mean"] = np.nan
    out["combined_score_std"] = np.nan
    out["method_used"] = "unknown"

    with torch.no_grad():
        for season_week, group in out.groupby("season_week"):
            if len(group) < 2:
                continue

            season = int(group["season"].values[0])
            method = UnifiedDWTSModel.choose_method_by_season(season)

            x_static = torch.tensor(group[static_features].values, dtype=torch.float32).to(device)
            x_dynamic = torch.tensor(group[dynamic_features].values, dtype=torch.float32).to(device)
            x_context = torch.tensor(group[context_features].values, dtype=torch.float32).to(device)
            judge_scores = torch.tensor(group["judge_score"].values, dtype=torch.float32).to(device)
            contestant_idx = torch.tensor(group["contestant_idx"].values, dtype=torch.long).to(device)

            p_list = []
            s_list = []
            for m in models:
                combined_scores, p_fan, _ = m(
                    x_static, x_dynamic, x_context, judge_scores, contestant_idx, season
                )
                p_list.append(p_fan)
                s_list.append(combined_scores)

            P = torch.stack(p_list, dim=0)  # [M, n]
            S = torch.stack(s_list, dim=0)  # [M, n]

            p_mean = P.mean(dim=0).cpu().numpy()
            p_std = P.std(dim=0, unbiased=False).cpu().numpy()
            s_mean = S.mean(dim=0).cpu().numpy()
            s_std = S.std(dim=0, unbiased=False).cpu().numpy()

            out.loc[group.index, "predicted_fan_vote_mean"] = p_mean
            out.loc[group.index, "predicted_fan_vote_std"] = p_std
            out.loc[group.index, "combined_score_mean"] = s_mean
            out.loc[group.index, "combined_score_std"] = s_std
            out.loc[group.index, "method_used"] = method

    return out


def benchmark_linear_vs_mlp(
    weekly_df: pd.DataFrame,
    num_epochs: int = 150,
    lr_mlp: float = 0.003,
    lr_linear: float = 0.01,
    hidden: int = 64,
    dropout: float = 0.15,
    weight_decay: float = 0.001,
    seed: int = 0,
):
    """Train/evaluate linear vs MLP fan-utility models under fixed DWTS equal-weight combining."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("\n" + "=" * 60)
    print("BENCHMARK: LINEAR vs MLP (fixed equal-weight combining)")
    print("=" * 60)

    print("\n[1/2] Training LINEAR utility model...")
    model_lin, _ = train_unified_model(
        weekly_df,
        num_epochs=num_epochs,
        lr=lr_linear,
        hidden=hidden,
        dropout=dropout,
        weight_decay=weight_decay,
        learn_mixing_weights=False,
        fan_utility="linear",
    )
    acc_lin = evaluate_predictions_unified(weekly_df, model_lin)

    print("\n[2/2] Training MLP utility model...")
    model_mlp, _ = train_unified_model(
        weekly_df,
        num_epochs=num_epochs,
        lr=lr_mlp,
        hidden=hidden,
        dropout=dropout,
        weight_decay=weight_decay,
        learn_mixing_weights=False,
        fan_utility="mlp",
    )
    acc_mlp = evaluate_predictions_unified(weekly_df, model_mlp)

    print("\n=== SUMMARY (same data, same evaluation) ===")
    print(f"Linear  HitRate@|E_w|: {float(acc_lin['hit_rate']):.2%}, ExactMatch: {float(acc_lin['exact_match']):.2%}, weeks={int(acc_lin['weeks'])}")
    print(f"MLP     HitRate@|E_w|: {float(acc_mlp['hit_rate']):.2%}, ExactMatch: {float(acc_mlp['exact_match']):.2%}, weeks={int(acc_mlp['weeks'])}")

    return {"linear": acc_lin, "mlp": acc_mlp}


def predict_fan_votes(model, weekly_df, method="rank"):
    """预测观众投票"""
    device = next(model.parameters()).device
    model.eval()

    static_features = [
        "age_normalized",
        "fame_normalized",
        "gender",
        "industry",
        "experience",
        "is_hetero",
    ]
    dynamic_features = [
        "state_normalized",
        "max_score_normalized",
        "sharpness",
        "prev_rank",
    ]
    context_features = ["is_all_star", "no_elim", "multi_elim", "week_pct", "contestants_pct"]

    predictions = []

    with torch.no_grad():
        for season_week, group in weekly_df.groupby("season_week"):
            try:
                x_static = torch.tensor(
                    group[static_features].values, dtype=torch.float32
                ).to(device)
                x_dynamic = torch.tensor(
                    group[dynamic_features].values, dtype=torch.float32
                ).to(device)
                x_context = torch.tensor(
                    group[context_features].values, dtype=torch.float32
                ).to(device)
                judge_scores = torch.tensor(
                    group["judge_score"].values, dtype=torch.float32
                ).to(device)

                combined_scores, P_fan = model(
                    x_static, x_dynamic, x_context, judge_scores, method=method
                )

                predictions.extend(P_fan.cpu().numpy())
            except:
                predictions.extend([np.nan] * len(group))

    weekly_df["predicted_fan_vote"] = predictions
    return weekly_df


def predict_fan_votes_unified(model, weekly_df):
    """用统一模型预测 fan vote，同时记录每个 season 使用的组合方法。"""
    device = next(model.parameters()).device
    model.eval()

    static_features = [
        "age_normalized",
        "fame_normalized",
        "gender",
        "industry",
        "experience",
        "is_hetero",
    ]
    dynamic_features = [
        "state_normalized",
        "max_score_normalized",
        "sharpness",
        "prev_rank",
    ]
    context_features = ["is_all_star", "no_elim", "multi_elim", "week_pct", "contestants_pct"]

    predictions = []
    methods = []

    with torch.no_grad():
        for season_week, group in weekly_df.groupby("season_week"):
            try:
                season = int(group["season"].values[0])
                x_static = torch.tensor(
                    group[static_features].values, dtype=torch.float32
                ).to(device)
                x_dynamic = torch.tensor(
                    group[dynamic_features].values, dtype=torch.float32
                ).to(device)
                x_context = torch.tensor(
                    group[context_features].values, dtype=torch.float32
                ).to(device)
                judge_scores = torch.tensor(
                    group["judge_score"].values, dtype=torch.float32
                ).to(device)
                contestant_idx = torch.tensor(
                    group["contestant_idx"].values, dtype=torch.long
                ).to(device)

                _, P_fan, method_used = model(
                    x_static, x_dynamic, x_context, judge_scores, contestant_idx, season
                )
                predictions.extend(P_fan.cpu().numpy().tolist())
                methods.extend([method_used] * len(group))
            except Exception:
                predictions.extend([np.nan] * len(group))
                methods.extend(["unknown"] * len(group))

    out = weekly_df.copy()
    out["predicted_fan_vote"] = predictions
    out["method_used"] = methods
    return out


def counterfactual_analysis(model, weekly_df):
    """
    在同一套 fan vote 概率下，对比 rank vs percentage 两种组合方式的预测差异。
    返回每个 season_week 是否一致（预测淘汰者相同）。
    """
    device = next(model.parameters()).device
    model.eval()

    static_features = [
        "age_normalized",
        "fame_normalized",
        "gender",
        "industry",
        "experience",
        "is_hetero",
    ]
    dynamic_features = [
        "state_normalized",
        "max_score_normalized",
        "sharpness",
        "prev_rank",
    ]
    context_features = ["is_all_star", "no_elim", "multi_elim", "week_pct", "contestants_pct"]

    records = []

    with torch.no_grad():
        for season_week, group in weekly_df.groupby("season_week"):
            if len(group) < 2:
                continue

            try:
                season = int(group["season"].values[0])
                x_static = torch.tensor(
                    group[static_features].values, dtype=torch.float32
                ).to(device)
                x_dynamic = torch.tensor(
                    group[dynamic_features].values, dtype=torch.float32
                ).to(device)
                x_context = torch.tensor(
                    group[context_features].values, dtype=torch.float32
                ).to(device)
                judge_scores = torch.tensor(
                    group["judge_score"].values, dtype=torch.float32
                ).to(device)
                contestant_idx = torch.tensor(
                    group["contestant_idx"].values, dtype=torch.long
                ).to(device)

                P_fan = model.compute_fan_prob(
                    x_static, x_dynamic, x_context, judge_scores, contestant_idx
                )
                score_rank = model.combine_scores(P_fan, judge_scores, "rank")
                score_pct = model.combine_scores(P_fan, judge_scores, "percentage")

                # 分数低 => 更可能被淘汰（因为高分=好）
                elim_rank_idx = int(torch.argmin(score_rank).item())
                elim_pct_idx = int(torch.argmin(score_pct).item())
                contestant_ids = group["contestant_id"].tolist()
                elim_rank = contestant_ids[elim_rank_idx]
                elim_pct = contestant_ids[elim_pct_idx]

                chosen_method = UnifiedDWTSModel.choose_method_by_season(season)
                chosen_elim = elim_rank if chosen_method == "rank" else elim_pct
                other_elim = elim_pct if chosen_method == "rank" else elim_rank

                records.append(
                    {
                        "season_week": season_week,
                        "season": season,
                        "chosen_method": chosen_method,
                        "elim_rank": elim_rank,
                        "elim_percentage": elim_pct,
                        "methods_agree": elim_rank == elim_pct,
                        "chosen_vs_other_agree": chosen_elim == other_elim,
                    }
                )
            except Exception:
                continue

    return pd.DataFrame(records)


def run_diagnostics(model, weekly_df_with_pred, model_accuracy=None):
    print("\n" + "=" * 60)
    print("DIAGNOSTIC ANALYSIS")
    print("=" * 60)

    def judge_only_metrics(df):
        """Judge-only baseline using lowest judge scores as eliminated."""
        total = 0
        hit_sum = 0.0
        exact = 0

        for _, g in df.groupby("season_week"):
            if len(g) < 2:
                continue

            # If this is tagged as no-elimination, skip for fairness.
            if "no_elim" in g.columns and float(g["no_elim"].values[0]) == 1.0:
                continue

            actual_elim_idx = g[g["is_eliminated"] == 1].index
            if len(actual_elim_idx) == 0:
                continue
            k = int(len(actual_elim_idx))

            pred_positions = g["judge_score"].sort_values(ascending=True).head(k).index
            pred_set = set(pred_positions.tolist())
            actual_set = set(actual_elim_idx.tolist())
            hits = len(pred_set & actual_set)
            hit_rate = hits / k if k > 0 else 0.0

            total += 1
            hit_sum += hit_rate
            exact += 1 if hits == k else 0

        return {
            "weeks": total,
            "hit_rate": (hit_sum / total if total > 0 else 0.0),
            "exact_match": (exact / total if total > 0 else 0.0),
        }

    judge_base = judge_only_metrics(weekly_df_with_pred)
    print(f"\n[1] Judge-only HitRate@|E_w|: {judge_base['hit_rate']:.2%} (weeks={judge_base['weeks']})")
    print(f"    Judge-only ExactMatch@|E_w|: {judge_base['exact_match']:.2%} (weeks={judge_base['weeks']})")
    if model_accuracy is not None:
        if isinstance(model_accuracy, dict) and "hit_rate" in model_accuracy:
            model_hit = float(model_accuracy["hit_rate"])
            print(f"    Model HitRate@|E_w|: {model_hit:.2%}")
            print(f"    (Note) Baseline uses judge_score only; model uses judge+fan via DWTS rule.")
        else:
            print(f"    Model accuracy: {model_accuracy:.2%}")
            # If user still passes a legacy scalar accuracy, avoid claiming a fan-vote contribution here.

    print("\n[2] Fan vote variance check:")
    eligible = weekly_df_with_pred.groupby("season_week").filter(lambda x: len(x) >= 5)
    if len(eligible) == 0:
        print("  Not enough weeks with >=5 contestants.")
    else:
        sample_weeks = eligible.groupby("season_week").head(1).sample(
            n=min(3, eligible["season_week"].nunique()), random_state=0
        )
        for _, row in sample_weeks.iterrows():
            sw = row["season_week"]
            g = weekly_df_with_pred[weekly_df_with_pred["season_week"] == sw]
            mean_v = g["predicted_fan_vote"].mean()
            std_v = g["predicted_fan_vote"].std()
            cv = std_v / mean_v if mean_v and mean_v != 0 else float("nan")
            print(
                f"  {sw}: CV={cv:.3f}, range=[{g['predicted_fan_vote'].min():.4f}, {g['predicted_fan_vote'].max():.4f}]"
            )

    print("\n[3] Implicit weight analysis:")
    for sw in ["12_2", "5_9", "27_10"]:
        g = weekly_df_with_pred[weekly_df_with_pred["season_week"] == sw]
        if len(g) == 0:
            continue
        judge_corr = g["judge_score"].corr(g["is_eliminated"])
        fan_corr = g["predicted_fan_vote"].corr(g["is_eliminated"])
        print(f"  {sw}: Judge corr={judge_corr:.3f}, Fan corr={fan_corr:.3f}")

    print("\n[4] Context feature actual impact:")
    if hasattr(model, "delta"):
        print(f"  Delta coefficients: {model.delta.detach().cpu().numpy()}")
    elif hasattr(model, "get_mixing_weights"):
        w = model.get_mixing_weights()
        print(f"  Mixing weights mode: {w['mode']}")
        print(f"  Rank weights (judge, fan): {w['rank']}")
        print(f"  Percentage weights (judge, fan): {w['percentage']}")
    else:
        print("  (No context coefficients exposed by this model class.)")
    all_star_elim = weekly_df_with_pred[weekly_df_with_pred["is_all_star"] == 1][
        "is_eliminated"
    ].mean()
    regular_elim = weekly_df_with_pred[weekly_df_with_pred["is_all_star"] == 0][
        "is_eliminated"
    ].mean()
    print(f"  All-star weeks eliminated: {all_star_elim:.2%}")
    print(f"  Regular weeks eliminated: {regular_elim:.2%}")


def validate_feature_effects(model_rank, model_pct):
    """Validate sign consistency against simple priors."""
    feature_labels = ["age", "fame", "gender", "industry", "experience", "is_hetero"]
    expected_signs = {
        "age": "-",  # younger tends to perform better
        "fame": "+",  # popularity tends to increase fan votes
        "gender": "?",  # no strong prior
        "industry": "?",  # no strong prior
        "experience": "+",  # more experience tends to help
        "is_hetero": "+",  # may increase audience appeal (heuristic)
    }

    beta_rank = model_rank.beta.detach().cpu().numpy()
    beta_pct = model_pct.beta.detach().cpu().numpy()

    def _sign(x):
        return "+" if x > 0 else "-"

    print("\n=== Feature Effect Validation ===")
    print(f"{'Feature':<15} | {'Expected':<8} | {'Rank':<8} | {'Percentage':<10} | Status")
    print("-" * 70)

    for i, feat in enumerate(feature_labels):
        expected = expected_signs.get(feat, "?")
        sign_rank = _sign(beta_rank[i])
        sign_pct = _sign(beta_pct[i])

        if expected == "?":
            status = "OK (no prior)"
        elif expected == sign_rank == sign_pct:
            status = "OK"
        elif expected in [sign_rank, sign_pct]:
            status = "MIXED"
        else:
            status = "REVERSED"

        print(
            f"{feat:<15} | {expected:<8} | {sign_rank:<8} | {sign_pct:<10} | {status}"
        )


def evaluate_predictions(weekly_df_pred):
    """Evaluate whether the lowest predicted fan vote matches actual elimination."""
    metrics = {}

    for season_week, group in weekly_df_pred.groupby("season_week"):
        if len(group) < 2:
            continue

        actual_elim_idx = group[group["is_eliminated"] == 1].index
        if len(actual_elim_idx) == 0:
            continue

        predicted_elim_idx = group["predicted_fan_vote"].idxmin()
        correct = predicted_elim_idx in actual_elim_idx.tolist()
        metrics[season_week] = {"correct": correct, "num_eliminated": len(actual_elim_idx)}

    if len(metrics) == 0:
        print("\n=== Prediction Accuracy ===")
        print("Weeks evaluated: 0 (no elimination labels found)")
        return 0.0

    accuracy = sum(m["correct"] for m in metrics.values()) / len(metrics)

    single = [m for m in metrics.values() if m["num_eliminated"] == 1]
    if len(single) > 0:
        single_acc = sum(m["correct"] for m in single) / len(single)
    else:
        single_acc = float("nan")

    print("\n=== Prediction Accuracy ===")
    print(f"Weeks evaluated: {len(metrics)}")
    print(f"Accuracy (predicting eliminated by min fan vote): {accuracy:.2%}")
    if not np.isnan(single_acc):
        print(f"Accuracy (single elimination only): {single_acc:.2%}")
    else:
        print("Accuracy (single elimination only): N/A")

    return accuracy


def evaluate_predictions_unified(weekly_df, model):
    """
    Correct evaluation for the unified model:
    - Use the same combined-score rule as training (rank/percentage chosen by season).
    - Predict eliminated contestant as the one with the lowest combined_score.
    """
    device = next(model.parameters()).device
    model.eval()

    static_features = [
        "age_normalized",
        "fame_normalized",
        "gender",
        "industry",
        "experience",
        "is_hetero",
    ]
    dynamic_features = [
        "state_normalized",
        "max_score_normalized",
        "sharpness",
        "prev_rank",
    ]
    context_features = ["is_all_star", "no_elim", "multi_elim", "week_pct", "contestants_pct"]

    total = 0
    hit_sum = 0.0
    exact_match = 0
    weeks_rank = 0
    weeks_pct = 0
    hit_sum_rank = 0.0
    hit_sum_pct = 0.0

    with torch.no_grad():
        for season_week, group in weekly_df.groupby("season_week"):
            if len(group) < 2:
                continue

            # Weeks with no elimination provide no labeled elimination signal; skip for fair evaluation.
            if "no_elim" in group.columns and float(group["no_elim"].values[0]) == 1.0:
                continue

            actual_elim_idx = group[group["is_eliminated"] == 1].index
            if len(actual_elim_idx) == 0:
                continue
            k = int(len(actual_elim_idx))

            season = int(group["season"].values[0])
            method = UnifiedDWTSModel.choose_method_by_season(season)
            if method == "rank":
                weeks_rank += 1
            else:
                weeks_pct += 1

            x_static = torch.tensor(group[static_features].values, dtype=torch.float32).to(
                device
            )
            x_dynamic = torch.tensor(group[dynamic_features].values, dtype=torch.float32).to(
                device
            )
            x_context = torch.tensor(group[context_features].values, dtype=torch.float32).to(
                device
            )
            judge_scores = torch.tensor(group["judge_score"].values, dtype=torch.float32).to(
                device
            )

            contestant_idx = torch.tensor(
                group["contestant_idx"].values, dtype=torch.long
            ).to(device)

            combined_scores, _, _ = model(
                x_static, x_dynamic, x_context, judge_scores, contestant_idx, season
            )

            # Predict the lowest-|E_w| contestants as eliminated
            pred_positions = torch.argsort(combined_scores)[:k].tolist()
            predicted_elim_idx = set(group.index[pred_positions].tolist())
            actual_set = set(actual_elim_idx.tolist())
            hits = len(predicted_elim_idx & actual_set)
            hit_rate = hits / k if k > 0 else 0.0
            is_exact = 1 if hits == k else 0

            total += 1
            hit_sum += hit_rate
            exact_match += is_exact
            if method == "rank":
                hit_sum_rank += hit_rate
            else:
                hit_sum_pct += hit_rate

    overall_hit = hit_sum / total if total > 0 else 0.0
    overall_exact = exact_match / total if total > 0 else 0.0
    rank_hit = hit_sum_rank / weeks_rank if weeks_rank > 0 else 0.0
    pct_hit = hit_sum_pct / weeks_pct if weeks_pct > 0 else 0.0

    print("\n" + "=" * 60)
    print("CORRECT EVALUATION RESULTS")
    print("=" * 60)
    print(f"HitRate@|E_w| (combined score): {overall_hit:.2%} (weeks={total})")
    print(f"ExactMatch@|E_w|: {overall_exact:.2%} (weeks={total})")
    print(f"  - Rank weeks HitRate: {rank_hit:.2%} (weeks={weeks_rank})")
    print(f"  - Percentage weeks HitRate: {pct_hit:.2%} (weeks={weeks_pct})")

    return {"hit_rate": overall_hit, "exact_match": overall_exact, "weeks": total}


def _build_train_contestant_index(train_df: pd.DataFrame) -> dict:
    """Map contestant_id to contiguous indices; reserve 0 for unseen contestants at test time."""
    ids = train_df["contestant_id"].astype(str).unique().tolist()
    return {cid: i + 1 for i, cid in enumerate(ids)}


def _apply_contestant_index(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    out = df.copy()
    out["contestant_idx"] = out["contestant_id"].map(mapping).fillna(0).astype(int)
    return out


def validate_generalization_time_split(
    weekly_df: pd.DataFrame,
    train_max_season: int = 25,
    num_epochs: int = 80,
):
    """
    Time-based generalization check:
      Train on seasons <= train_max_season, test on seasons > train_max_season.
    Important: contestant embeddings are fit on train contestants only; unseen test contestants map to idx=0.
    """
    train_df = weekly_df[weekly_df["season"] <= train_max_season].copy()
    test_df = weekly_df[weekly_df["season"] > train_max_season].copy()

    if len(train_df) == 0 or len(test_df) == 0:
        print("\n=== GENERALIZATION CHECK (TIME SPLIT) ===")
        print("Not enough data for the requested split.")
        return None

    mapping = _build_train_contestant_index(train_df)
    train_df = _apply_contestant_index(train_df, mapping)
    test_df = _apply_contestant_index(test_df, mapping)

    model, _ = train_unified_model(train_df, num_epochs=num_epochs)
    print("\n" + "=" * 60)
    print("GENERALIZATION CHECK (TIME SPLIT)")
    print("=" * 60)
    print(f"Train seasons: <= {train_max_season} (samples={len(train_df)})")
    print(f"Test seasons:  > {train_max_season} (samples={len(test_df)})")

    metrics = evaluate_predictions_unified(test_df, model)
    return metrics


def compare_elimination_methods(weekly_df, model):
    """
    Counterfactual comparison:
    - Use the unified fan-vote probabilities P_fan.
    - Compare rank vs percentage combination rules in terms of predicted eliminated contestant.
    - Also record correctness against actual elimination (when labeled).
    """
    device = next(model.parameters()).device
    model.eval()

    static_features = [
        "age_normalized",
        "fame_normalized",
        "gender",
        "industry",
        "experience",
        "is_hetero",
    ]
    dynamic_features = [
        "state_normalized",
        "max_score_normalized",
        "sharpness",
        "prev_rank",
    ]
    context_features = ["is_all_star", "no_elim", "multi_elim", "week_pct", "contestants_pct"]

    rows = []
    with torch.no_grad():
        for season_week, group in weekly_df.groupby("season_week"):
            if len(group) < 2:
                continue

            season = int(group["season"].values[0])
            actual_elim = group[group["is_eliminated"] == 1]["contestant_id"].tolist()
            if len(actual_elim) == 0:
                continue

            x_static = torch.tensor(group[static_features].values, dtype=torch.float32).to(
                device
            )
            x_dynamic = torch.tensor(group[dynamic_features].values, dtype=torch.float32).to(
                device
            )
            x_context = torch.tensor(group[context_features].values, dtype=torch.float32).to(
                device
            )
            judge_scores = torch.tensor(group["judge_score"].values, dtype=torch.float32).to(
                device
            )

            contestant_idx = torch.tensor(
                group["contestant_idx"].values, dtype=torch.long
            ).to(device)

            P_fan = model.compute_fan_prob(
                x_static, x_dynamic, x_context, judge_scores, contestant_idx
            )
            score_rank = model.combine_scores(P_fan, judge_scores, "rank")
            score_pct = model.combine_scores(P_fan, judge_scores, "percentage")

            pred_rank = group.iloc[int(torch.argmin(score_rank).item())]["contestant_id"]
            pred_pct = group.iloc[int(torch.argmin(score_pct).item())]["contestant_id"]

            actual_method = UnifiedDWTSModel.choose_method_by_season(season)

            rows.append(
                {
                    "season_week": season_week,
                    "season": season,
                    "actual_method": actual_method,
                    "actual_elim": actual_elim[0] if len(actual_elim) == 1 else str(actual_elim),
                    "pred_rank": pred_rank,
                    "pred_pct": pred_pct,
                    "methods_agree": pred_rank == pred_pct,
                    "rank_correct": pred_rank in actual_elim,
                    "pct_correct": pred_pct in actual_elim,
                }
            )

    return pd.DataFrame(rows)


def analyze_fame_effect(weekly_df):
    """Compare elimination rates between high-fame and low-fame groups."""
    if "fame_normalized" not in weekly_df.columns:
        print("\n=== Fame Effect Analysis ===")
        print("Missing column: fame_normalized")
        return

    high_fame = weekly_df[weekly_df["fame_normalized"] > 0.5]
    low_fame = weekly_df[weekly_df["fame_normalized"] < -0.5]

    high_elim_rate = high_fame["is_eliminated"].mean() if len(high_fame) > 0 else float("nan")
    low_elim_rate = low_fame["is_eliminated"].mean() if len(low_fame) > 0 else float("nan")

    print("\n=== Fame Effect Analysis ===")
    print(f"High fame (>0.5 std) samples: {len(high_fame)}")
    print(f"Low fame (<-0.5 std) samples: {len(low_fame)}")
    print(f"High fame elimination rate: {high_elim_rate:.2%}" if not np.isnan(high_elim_rate) else "High fame elimination rate: N/A")
    print(f"Low fame elimination rate: {low_elim_rate:.2%}" if not np.isnan(low_elim_rate) else "Low fame elimination rate: N/A")
    if not (np.isnan(high_elim_rate) or np.isnan(low_elim_rate)):
        diff_pp = (high_elim_rate - low_elim_rate) * 100
        print(f"Difference (high - low): {diff_pp:.1f} percentage points")


# ==================== 主程序 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("DWTS Fan Vote Estimation - Fixed Version")
    print("=" * 60)

    # 1. 数据预处理
    print("\n[1/3] Loading and preprocessing data...")
    preprocessor = DWTSDataPreprocessor(data_dir="./data/")
    preprocessor.load_all_data()
    preprocessor.extract_judge_scores()
    preprocessor.extract_elimination_info()
    preprocessor.build_weekly_dataset()
    preprocessor.prepare_training_data()

    weekly_df = preprocessor.weekly_df
    print(f"Total samples: {len(weekly_df)}")
    print(f"Total season-weeks: {weekly_df['season_week'].nunique()}")
    print(f"Seasons: {weekly_df['season'].min()} - {weekly_df['season'].max()}")

    # 2. 训练统一模型
    print("\n[2/3] Training unified fan vote model...")
    # Keep mixing weights fixed by default to match the show's rule (judge component + fan component).
    model, losses = train_unified_model(weekly_df, learn_mixing_weights=False)

    # 3. 预测与保存
    print("\n[3/3] Predicting and saving results...")
    weekly_df_unified = predict_fan_votes_unified(model, weekly_df)
    weekly_df_unified.to_csv("predictions_unified.csv", index=False)

    # 6. 输出结果
    print("\n" + "=" * 60)
    print("UNIFIED FAN VOTE MODEL")
    print("=" * 60)

    print(f"Final Loss: {losses[-1]:.4f}")
    if hasattr(model, "get_mixing_weights"):
        w = model.get_mixing_weights()
        print(f"Mixing weights mode: {w['mode']}")
        print(f"  Rank weights (judge, fan): {w['rank']}")
        print(f"  Percentage weights (judge, fan): {w['percentage']}")
    if hasattr(model, "fan_utility"):
        print(f"Fan utility type: {model.fan_utility}")
    if getattr(model, "fan_net", None) is not None:
        print(f"Fan utility model: {model.fan_net.__class__.__name__} (MLP)")
    elif getattr(model, "fan_linear", None) is not None:
        print("Fan utility model: Linear")

    print("\nDone! Check predictions_unified.csv")

    # ==================== Post-run validation ====================
    # (1) The old evaluation uses min fan-vote; keep it for reference if needed.
    # evaluate_predictions(weekly_df_unified)
    # (2) Correct evaluation uses the combined score (same rule as training).
    model_acc = evaluate_predictions_unified(weekly_df, model)
    analyze_fame_effect(weekly_df)

    print("\n=== Context Feature Coverage ===")
    print(f"is_all_star=1: {(weekly_df['is_all_star'] == 1).sum()} samples")
    print(f"no_elim=1: {(weekly_df['no_elim'] == 1).sum()} samples")
    print(f"multi_elim=1: {(weekly_df['multi_elim'] == 1).sum()} samples")

    print("\n=== Fame Distribution by Season ===")
    for s in [1, 10, 20, 30]:
        season_data = weekly_df[weekly_df["season"] == s]
        if len(season_data) == 0:
            continue
        print(
            f"Season {s}: fame mean={season_data['fame'].mean():.2f}, "
            f"fame_normalized mean={season_data['fame_normalized'].mean():.2f}"
        )

    print("\n=== Season 12 Week 2 Full Results ===")
    week_12_2 = weekly_df_unified[weekly_df_unified["season_week"] == "12_2"].copy()
    if len(week_12_2) == 0:
        print("No records found for season_week=12_2")
    else:
        cols = ["contestant_id", "judge_score", "predicted_fan_vote", "is_eliminated"]
        existing = [c for c in cols if c in week_12_2.columns]
        print(week_12_2[existing].sort_values("predicted_fan_vote", ascending=False).to_string(index=False))

    # 统一模型下的反事实：rank vs percentage 组合方式差异
    print("\n=== Method Comparison ===")
    cf = counterfactual_analysis(model, weekly_df)
    if len(cf) == 0:
        print("No season-weeks available for comparison.")
    else:
        print(f"Weeks where methods agree: {int(cf['methods_agree'].sum())}/{len(cf)}")
        print(f"Agreement rate: {cf['methods_agree'].mean():.2%}")

    run_diagnostics(model, weekly_df_unified, model_accuracy=model_acc)

    # Save detailed method comparison (correct)
    comparison_results = compare_elimination_methods(weekly_df, model)
    comparison_results.to_csv("method_comparison_results.csv", index=False)
    print("\nDetailed comparison saved to method_comparison_results.csv")

    # Optional: one-shot dual-ensemble benchmark (best-effort in-sample improvement)
    # Comment out if you don't want the extra training time.
    print("\n=== Dual-ensemble (focal gamma rule-specific) ===")
    models_rank = train_unified_ensemble(
        weekly_df,
        ensemble_size=3,
        num_epochs=60,
        lr=0.003,
        hidden=64,
        dropout=0.15,
        fan_utility="mlp",
        focal_gamma=1.0,
    )
    models_pct = train_unified_ensemble(
        weekly_df,
        ensemble_size=3,
        num_epochs=60,
        lr=0.003,
        hidden=64,
        dropout=0.15,
        fan_utility="mlp",
        focal_gamma=2.0,
    )
    _ = evaluate_predictions_unified_dual_ensemble(weekly_df, models_rank, models_pct)
