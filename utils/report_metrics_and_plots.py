import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_write_csv(df: pd.DataFrame, path: Path) -> Path:
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        alt = path.with_name(f"{path.stem}_new{path.suffix}")
        df.to_csv(alt, index=False)
        return alt


def _safe_write_text(text: str, path: Path, encoding: str = "utf-8") -> Path:
    try:
        path.write_text(text, encoding=encoding)
        return path
    except PermissionError:
        alt = path.with_name(f"{path.stem}_new{path.suffix}")
        alt.write_text(text, encoding=encoding)
        return alt


def _choose_prob_col(df: pd.DataFrame) -> str:
    for c in ["predicted_fan_vote_mean", "predicted_fan_vote"]:
        if c in df.columns:
            return c
    raise ValueError("Missing probability column: expected predicted_fan_vote or predicted_fan_vote_mean")


def _choose_score_col(df: pd.DataFrame) -> str | None:
    for c in ["combined_score_mean", "combined_score"]:
        if c in df.columns:
            return c
    return None


def _soft_rank_numpy(P: np.ndarray, tau: float = 0.5) -> np.ndarray:
    P = np.asarray(P, dtype=float)
    diff = P.reshape(1, -1) - P.reshape(-1, 1)  # [n,n] = P_j - P_i
    sig = 1.0 / (1.0 + np.exp(-(diff / float(tau))))
    return 1.0 + sig.sum(axis=1) - np.diag(sig)


def _compute_combined_score_fallback(df: pd.DataFrame, prob_col: str, tau: float = 0.5) -> pd.Series:
    """
    Fallback if combined_score is missing:
      - Rank rule: (2n) - (judge_rank + soft_fan_rank(P_fan))
      - Percentage rule: judge_share + P_fan
    """
    if "season_week" not in df.columns:
        raise ValueError("Missing season_week in predictions dataframe.")
    if "judge_score" not in df.columns:
        raise ValueError("Missing judge_score in predictions dataframe.")
    if "season" not in df.columns:
        raise ValueError("Missing season in predictions dataframe.")

    def choose_method(season: int) -> str:
        return "rank" if (season in [1, 2] or season >= 28) else "percentage"

    out = pd.Series(index=df.index, dtype=float)
    for sw, g in df.groupby("season_week"):
        if len(g) < 2:
            continue
        season = int(g["season"].values[0])
        method = choose_method(season)
        p = g[prob_col].to_numpy(dtype=float)
        j = g["judge_score"].to_numpy(dtype=float)
        n = int(len(g))

        if method == "percentage":
            denom = float(j.sum()) + 1e-12
            j_pct = j / denom
            out.loc[g.index] = j_pct + p
            continue

        # rank
        j_rank = (-j).argsort().argsort().astype(float) + 1.0  # 1..n, smaller is better
        f_rank = _soft_rank_numpy(p, tau=tau)
        combined = (2.0 * n) - (j_rank + f_rank)  # higher is better
        out.loc[g.index] = combined

    return out


def shannon_entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    p = p[np.isfinite(p)]
    if len(p) == 0:
        return float("nan")
    s = float(p.sum())
    if s <= 0:
        return float("nan")
    p = p / s
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


def add_weekly_entropy_columns(
    df: pd.DataFrame,
    prob_col: str,
    group_col: str = "season_week",
    out_entropy_col: str = "entropy",
    out_entropy_norm_col: str = "entropy_norm",
) -> pd.DataFrame:
    out = df.copy()
    out[out_entropy_col] = np.nan
    out[out_entropy_norm_col] = np.nan

    for sw, g in out.groupby(group_col):
        n = int(len(g))
        if n == 0:
            continue
        H = shannon_entropy(g[prob_col].to_numpy(dtype=float))
        if not np.isfinite(H) or n <= 1:
            Hn = float("nan")
        else:
            Hn = float(H) / float(np.log(n))
        out.loc[g.index, out_entropy_col] = float(H)
        out.loc[g.index, out_entropy_norm_col] = float(Hn)
    return out


def build_week_level_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Collapse contestant-level rows to one row per season_week (taking the first non-null value).
    """
    if "season_week" not in df.columns:
        raise ValueError("Missing season_week.")
    use = [c for c in ["season_week", "season", "week", "no_elim"] + cols if c in df.columns]
    g = df[use].copy()
    # entropy is week-level already; still safe to "first".
    return g.groupby("season_week", as_index=False).first()


def kendall_tau_rank_score(rank_values: np.ndarray, score_values: np.ndarray) -> float:
    """
    tau = (2 / (n(n-1))) * sum_{i<j} sign(rank_i - rank_j) * sign(score_i - score_j)
    Ties contribute 0.
    """
    r = np.asarray(rank_values, dtype=float)
    s = np.asarray(score_values, dtype=float)
    n = int(len(r))
    if n < 2:
        return float("nan")

    def sgn(x: float) -> int:
        if x > 0:
            return 1
        if x < 0:
            return -1
        return 0

    pair_sum = 0.0
    for i in range(n - 1):
        for j in range(i + 1, n):
            pair_sum += sgn(r[i] - r[j]) * sgn(s[i] - s[j])
    return (2.0 / (n * (n - 1))) * float(pair_sum)


def build_weekly_kendall_tau_table(
    df: pd.DataFrame,
    score_col: str,
    actual_rank_col: str = "true_elim_week",
    group_col: str = "season_week",
    no_elim_col: str = "no_elim",
    skip_no_elim: bool = True,
) -> pd.DataFrame:
    if actual_rank_col not in df.columns:
        raise ValueError(f"Missing {actual_rank_col} for Kendall tau.")
    rows = []
    for sw, g in df.groupby(group_col):
        if len(g) < 2:
            continue
        if skip_no_elim and no_elim_col in g.columns and float(g[no_elim_col].values[0]) == 1.0:
            continue
        r = g[actual_rank_col].to_numpy(dtype=float)
        s = g[score_col].to_numpy(dtype=float)
        if not (np.isfinite(r).all() and np.isfinite(s).all()):
            continue
        tau = kendall_tau_rank_score(r, s)
        season = int(g["season"].values[0]) if "season" in g.columns else None
        week = int(g["week"].values[0]) if "week" in g.columns else None
        rows.append({"season_week": sw, "season": season, "week": week, "tau": float(tau), "n": int(len(g))})
    return pd.DataFrame(rows)


def build_weekly_accuracy_table(
    df: pd.DataFrame,
    score_col: str,
    group_col: str = "season_week",
    eliminated_col: str = "is_eliminated",
    no_elim_col: str = "no_elim",
    skip_no_elim: bool = True,
) -> pd.DataFrame:
    rows = []
    for sw, g in df.groupby(group_col):
        if len(g) < 2:
            continue
        if skip_no_elim and no_elim_col in g.columns and float(g[no_elim_col].values[0]) == 1.0:
            continue
        actual_idx = g[g[eliminated_col] == 1].index
        if len(actual_idx) == 0:
            continue
        k = int(len(actual_idx))
        scores = g[score_col]
        if scores.isna().any():
            continue
        pred_idx = g.sort_values(score_col, ascending=True).head(k).index
        hits = len(set(pred_idx.tolist()) & set(actual_idx.tolist()))
        season = int(g["season"].values[0]) if "season" in g.columns else None
        week = int(g["week"].values[0]) if "week" in g.columns else None
        rows.append(
            {
                "season_week": sw,
                "season": season,
                "week": week,
                "n": int(len(g)),
                "k": k,
                "hits": hits,
                "hit_rate": hits / k if k > 0 else float("nan"),
                "exact_match": 1 if hits == k else 0,
            }
        )
    return pd.DataFrame(rows)


def plot_entropy_heatmap(
    week_table: pd.DataFrame,
    outpath: Path,
    value_col: str = "entropy_norm",
    title: str = "Uncertainty Heatmap (Normalized Entropy)",
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib

    if not {"season", "week", value_col}.issubset(set(week_table.columns)):
        raise ValueError(f"Need columns season, week, {value_col} for heatmap.")

    seasons = sorted(week_table["season"].dropna().astype(int).unique().tolist())
    weeks = sorted(week_table["week"].dropna().astype(int).unique().tolist())
    if len(seasons) == 0 or len(weeks) == 0:
        raise ValueError("No seasons/weeks available for heatmap.")

    grid = np.full((len(seasons), len(weeks)), np.nan, dtype=float)
    s_to_i = {s: i for i, s in enumerate(seasons)}
    w_to_j = {w: j for j, w in enumerate(weeks)}

    for _, row in week_table.iterrows():
        s = int(row["season"])
        w = int(row["week"])
        v = float(row[value_col]) if pd.notna(row[value_col]) else np.nan
        if s in s_to_i and w in w_to_j:
            grid[s_to_i[s], w_to_j[w]] = v

    masked = np.ma.masked_invalid(grid)

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "font.size": 11,
        }
    )
    fig, ax = plt.subplots(figsize=(12, 7))

    def _truncate_cmap(base, lo: float = 0.15, hi: float = 0.85, n: int = 256):
        base = matplotlib.colormaps.get_cmap(base) if isinstance(base, str) else base
        colors = base(np.linspace(lo, hi, n))
        return LinearSegmentedColormap.from_list(f"{base.name}_trunc", colors)

    # Warm, soft palette with reduced contrast:
    # - Truncate to avoid very dark reds.
    # - Keep a linear norm so colorbar tick distances are uniform.
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    warm_cmap = _truncate_cmap("YlOrRd", lo=0.05, hi=0.75)
    im = ax.imshow(masked, aspect="auto", interpolation="nearest", cmap=warm_cmap, norm=norm, alpha=0.92)

    ax.set_title(title)
    ax.set_xlabel("Week")
    ax.set_ylabel("Season")

    def _tick_idx(n: int, max_labels: int) -> list[int]:
        if n <= 0:
            return []
        if n <= max_labels:
            return list(range(n))
        step = int(np.ceil((n - 1) / float(max_labels - 1)))
        idx = list(range(0, n, step))
        if idx[-1] != n - 1:
            idx.append(n - 1)
        return idx

    # ticks (thin to avoid clutter)
    xticks = list(range(len(weeks)))
    xlabels = [str(w) for w in weeks]
    sel_x = _tick_idx(len(xticks), max_labels=(len(xticks) if len(xticks) <= 20 else 16))
    ax.set_xticks(sel_x)
    ax.set_xticklabels([xlabels[i] for i in sel_x])

    # Seasons: show 1,4,7,... every 3 seasons
    yticks = list(range(len(seasons)))
    ylabels = [str(s) for s in seasons]
    sel_y = [i for i, s in enumerate(seasons) if (int(s) - 1) % 3 == 0]
    if len(sel_y) == 0:
        sel_y = _tick_idx(len(yticks), max_labels=(len(yticks) if len(yticks) <= 40 else 12))
    ax.set_yticks(sel_y)
    ax.set_yticklabels([ylabels[i] for i in sel_y])
    ax.tick_params(axis="y", labelsize=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("H_norm (0=confident, 1=uncertain)")
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_tau_heatmap(
    tau_table: pd.DataFrame,
    outpath: Path,
    value_col: str = "tau",
    title: str = "Ranking Consistency Heatmap (Kendall's Tau)",
) -> None:
    import matplotlib.pyplot as plt

    if not {"season", "week", value_col}.issubset(set(tau_table.columns)):
        raise ValueError(f"Need columns season, week, {value_col} for tau heatmap.")

    seasons = sorted(tau_table["season"].dropna().astype(int).unique().tolist())
    weeks = sorted(tau_table["week"].dropna().astype(int).unique().tolist())
    if len(seasons) == 0 or len(weeks) == 0:
        raise ValueError("No seasons/weeks available for tau heatmap.")

    grid = np.full((len(seasons), len(weeks)), np.nan, dtype=float)
    s_to_i = {s: i for i, s in enumerate(seasons)}
    w_to_j = {w: j for j, w in enumerate(weeks)}

    for _, row in tau_table.iterrows():
        s = int(row["season"])
        w = int(row["week"])
        v = float(row[value_col]) if pd.notna(row[value_col]) else np.nan
        if s in s_to_i and w in w_to_j:
            grid[s_to_i[s], w_to_j[w]] = v

    masked = np.ma.masked_invalid(grid)

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "font.size": 11,
        }
    )
    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(
        masked,
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
    )

    ax.set_title(title)
    ax.set_xlabel("Week")
    ax.set_ylabel("Season")

    def _tick_idx(n: int, max_labels: int) -> list[int]:
        if n <= 0:
            return []
        if n <= max_labels:
            return list(range(n))
        step = int(np.ceil((n - 1) / float(max_labels - 1)))
        idx = list(range(0, n, step))
        if idx[-1] != n - 1:
            idx.append(n - 1)
        return idx

    xticks = list(range(len(weeks)))
    xlabels = [str(w) for w in weeks]
    sel_x = _tick_idx(len(xticks), max_labels=(len(xticks) if len(xticks) <= 20 else 16))
    ax.set_xticks(sel_x)
    ax.set_xticklabels([xlabels[i] for i in sel_x])

    yticks = list(range(len(seasons)))
    ylabels = [str(s) for s in seasons]
    # Seasons: show 1,4,7,... every 3 seasons
    sel_y = [i for i, s in enumerate(seasons) if (int(s) - 1) % 3 == 0]
    if len(sel_y) == 0:
        sel_y = _tick_idx(len(yticks), max_labels=(len(yticks) if len(yticks) <= 40 else 12))
    ax.set_yticks(sel_y)
    ax.set_yticklabels([ylabels[i] for i in sel_y])
    ax.tick_params(axis="y", labelsize=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Kendall's Tau (higher = more consistent)")

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def _tau_method_by_season(season: int) -> str:
    return "rank" if (season in [1, 2] or season >= 28) else "percentage"


def summarize_kendall_tau(tau_table: pd.DataFrame) -> pd.DataFrame:
    """
    Core summary stats for Kendall's Tau:
      mean/median/std/min/max/IQR, positive/negative rates.
    Also split by DWTS method (rank vs percentage) inferred by season.
    """
    df = tau_table.copy()
    df = df.dropna(subset=["tau", "season", "week"])
    if len(df) == 0:
        return pd.DataFrame()

    df["method"] = df["season"].astype(int).map(_tau_method_by_season)

    def _summ(g: pd.DataFrame, label: str) -> dict:
        t = g["tau"].to_numpy(dtype=float)
        return {
            "group": label,
            "weeks": int(len(t)),
            "tau_mean": float(np.mean(t)),
            "tau_median": float(np.median(t)),
            "tau_std": float(np.std(t, ddof=0)),
            "tau_min": float(np.min(t)),
            "tau_max": float(np.max(t)),
            "tau_q25": float(np.quantile(t, 0.25)),
            "tau_q75": float(np.quantile(t, 0.75)),
            "tau_mean_abs": float(np.mean(np.abs(t))),
            "pos_rate": float(np.mean(t > 0)),
            "neg_rate": float(np.mean(t < 0)),
        }

    rows = [_summ(df, "overall")]
    for m in ["rank", "percentage"]:
        gm = df[df["method"] == m]
        if len(gm) > 0:
            rows.append(_summ(gm, m))

    return pd.DataFrame(rows)

def plot_line_with_band(
    x: np.ndarray,
    y_mean: np.ndarray,
    y_se: np.ndarray,
    outpath: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "font.size": 11,
        }
    )
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(x, y_mean, linewidth=2)
    ax.fill_between(x, y_mean - y_se, y_mean + y_se, alpha=0.2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance_linear(
    weights: np.ndarray,
    feature_names: list[str],
    outpath: Path,
    title: str = "Feature Importance (Linear Utility Weights)",
) -> None:
    import matplotlib.pyplot as plt

    w = np.asarray(weights, dtype=float).reshape(-1)
    names = feature_names[: len(w)]

    order = np.argsort(np.abs(w))[::-1]
    w_sorted = w[order]
    names_sorted = [names[i] for i in order]
    colors = ["#1f77b4" if v >= 0 else "#d62728" for v in w_sorted]

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "font.size": 11,
        }
    )
    fig, ax = plt.subplots(figsize=(10, 6.5))
    y = np.arange(len(w_sorted))
    ax.barh(y, w_sorted, color=colors, alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(names_sorted)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=1, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("Weight (sign shows direction on fan utility)")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


@dataclass
class ReportOutputs:
    outdir: Path
    weekly_metrics_csv: Path
    kendall_tau_csv: Path | None
    entropy_week_csv: Path
    entropy_heatmap_png: Path
    entropy_by_week_png: Path
    hitrate_by_week_png: Path
    feature_importance_png: Path | None


def main() -> None:
    parser = argparse.ArgumentParser(description="DWTS paper metrics + plots (entropy heatmap, feature importance).")
    parser.add_argument("--pred-csv", type=str, default="predictions_unified.csv", help="Predictions CSV path.")
    parser.add_argument("--outdir", type=str, default="reports", help="Output directory.")
    parser.add_argument("--bootstrap-B", type=int, default=0, help="If >0, run bootstrap with B replicates (slow).")
    parser.add_argument("--bootstrap-epochs", type=int, default=60, help="Epochs per bootstrap replicate (if enabled).")
    parser.add_argument(
        "--bootstrap-fan-utility",
        choices=["linear", "mlp"],
        default="linear",
        help="Fan utility used for bootstrap uncertainty (linear is much faster/more interpretable).",
    )
    parser.add_argument(
        "--feature-importance",
        choices=["none", "linear"],
        default="linear",
        help="Feature importance plot mode. 'linear' trains a linear fan-utility model for interpretability.",
    )
    parser.add_argument("--linear-epochs", type=int, default=150, help="Epochs for linear importance training.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    args = parser.parse_args()

    outdir = _ensure_dir(Path(args.outdir))

    pred_path = Path(args.pred_csv)
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {pred_path}")

    df = pd.read_csv(pred_path)
    prob_col = _choose_prob_col(df)
    score_col = _choose_score_col(df)
    if score_col is None:
        df["combined_score"] = _compute_combined_score_fallback(df, prob_col=prob_col)
        score_col = "combined_score"

    # Entropy / uncertainty columns
    df = add_weekly_entropy_columns(df, prob_col=prob_col)
    week_entropy = build_week_level_table(df, cols=["entropy", "entropy_norm"])

    entropy_week_csv = _safe_write_csv(week_entropy, outdir / "entropy_by_season_week.csv")

    # Weekly accuracy table + by-week plot
    weekly_metrics = build_weekly_accuracy_table(df, score_col=score_col)
    weekly_metrics_csv = _safe_write_csv(weekly_metrics, outdir / "weekly_metrics.csv")

    by_week = weekly_metrics.dropna(subset=["week"]).groupby("week")["hit_rate"].agg(["mean", "count", "std"]).reset_index()
    by_week["se"] = by_week["std"] / np.sqrt(np.maximum(by_week["count"].to_numpy(dtype=float), 1.0))
    plot_line_with_band(
        x=by_week["week"].to_numpy(dtype=float),
        y_mean=by_week["mean"].to_numpy(dtype=float),
        y_se=by_week["se"].to_numpy(dtype=float),
        outpath=outdir / "hitrate_by_week.png",
        title="HitRate@|E_w| by Week Number",
        xlabel="Week",
        ylabel="HitRate@|E_w|",
    )

    # Entropy heatmap + by-week plot
    plot_entropy_heatmap(
        week_table=week_entropy.dropna(subset=["entropy_norm"]),
        outpath=outdir / "entropy_heatmap.png",
        value_col="entropy_norm",
        title="Uncertainty Heatmap (Normalized Shannon Entropy)",
    )
    ent_by_week = week_entropy.dropna(subset=["week", "entropy_norm"]).groupby("week")["entropy_norm"].agg(["mean", "count", "std"]).reset_index()
    ent_by_week["se"] = ent_by_week["std"] / np.sqrt(np.maximum(ent_by_week["count"].to_numpy(dtype=float), 1.0))
    plot_line_with_band(
        x=ent_by_week["week"].to_numpy(dtype=float),
        y_mean=ent_by_week["mean"].to_numpy(dtype=float),
        y_se=ent_by_week["se"].to_numpy(dtype=float),
        outpath=outdir / "entropy_by_week.png",
        title="Mean Uncertainty by Week (Normalized Entropy)",
        xlabel="Week",
        ylabel="Mean H_norm",
    )

    # Kendall tau table (if possible)
    kendall_tau_csv = None
    kendall_tau_csv_path = outdir / "kendall_tau_by_week.csv"
    tau_table = None
    if kendall_tau_csv_path.exists():
        # If the CSV is open/locked by another program (e.g., Excel), avoid overwriting.
        tau_table = pd.read_csv(kendall_tau_csv_path)
        kendall_tau_csv = kendall_tau_csv_path
    elif "true_elim_week" in df.columns:
        tau_table = build_weekly_kendall_tau_table(df, score_col=score_col)
        kendall_tau_csv = _safe_write_csv(tau_table, kendall_tau_csv_path)

    if tau_table is not None and len(tau_table) > 0:
        # Core stats + heatmap for tau (paper-friendly)
        tau_summary = summarize_kendall_tau(tau_table)
        if len(tau_summary) > 0:
            _safe_write_csv(tau_summary, outdir / "kendall_tau_summary.csv")
            _safe_write_text(
                tau_summary.to_string(index=False) + "\n",
                outdir / "kendall_tau_summary.txt",
                encoding="utf-8",
            )
        if "season" in tau_table.columns and "week" in tau_table.columns and "tau" in tau_table.columns:
            plot_tau_heatmap(
                tau_table=tau_table.dropna(subset=["season", "week", "tau"]),
                outpath=outdir / "kendall_tau_heatmap.png",
                value_col="tau",
                title="Ranking Consistency Heatmap (Kendall's Tau)",
            )

    # Feature importance (linear, interpretability-focused)
    feature_importance_png = None
    if args.feature_importance == "linear":
        from dwts_model import DWTSDataPreprocessor, train_unified_model

        torch_seed = int(args.seed)
        np.random.seed(torch_seed)
        try:
            import torch

            torch.manual_seed(torch_seed)
        except Exception:
            pass

        pre = DWTSDataPreprocessor(data_dir="./data/")
        pre.load_all_data().extract_judge_scores().extract_elimination_info().build_weekly_dataset().prepare_training_data()
        weekly_df = pre.weekly_df

        model, _ = train_unified_model(
            weekly_df,
            num_epochs=int(args.linear_epochs),
            lr=0.01,
            fan_utility="linear",
            focal_gamma=None,
            skip_unsupervised_weeks=True,
        )

        w = model.fan_linear.weight.detach().cpu().numpy().reshape(-1)
        static_features = ["age_normalized", "fame_normalized", "gender", "industry", "experience", "is_hetero"]
        dynamic_features = ["state_normalized", "max_score_normalized", "sharpness", "prev_rank"]
        context_features = ["is_all_star", "no_elim", "multi_elim", "week_pct", "contestants_pct"]
        feature_names = static_features + dynamic_features + context_features + ["judge_score_z"]

        feature_importance_png = outdir / "feature_importance_linear.png"
        plot_feature_importance_linear(
            weights=w,
            feature_names=feature_names,
            outpath=feature_importance_png,
            title="Feature Importance (Linear Fan-Utility Weights)",
        )

    # Bootstrap uncertainty (optional; slow)
    if int(args.bootstrap_B) > 0:
        from dwts_model import DWTSDataPreprocessor, bootstrap_unified_predictions
        from dwts_model import anova_oneway

        pre = DWTSDataPreprocessor(data_dir="./data/")
        pre.load_all_data().extract_judge_scores().extract_elimination_info().build_weekly_dataset().prepare_training_data()
        weekly_df = pre.weekly_df

        boot = bootstrap_unified_predictions(
            weekly_df,
            B=int(args.bootstrap_B),
            num_epochs=int(args.bootstrap_epochs),
            fan_utility=str(args.bootstrap_fan_utility),
            seed=int(args.seed),
        )

        boot_pred = boot["predictions"]
        _safe_write_csv(boot_pred, outdir / "bootstrap_predictions.csv")
        if boot.get("linear_params", None) is not None:
            _safe_write_csv(boot["linear_params"], outdir / "bootstrap_linear_params.csv")

        # Week-level boundary confidence table (if available)
        week_boundary_path = None
        if "week_elim_boundary_certainty" in boot_pred.columns:
            week_boundary = (
                boot_pred[
                    [
                        "season_week",
                        "season",
                        "week",
                        "week_elim_k",
                        "week_cutoff_mean",
                        "week_cutoff_std",
                        "week_cutoff_ci_low",
                        "week_cutoff_ci_high",
                        "week_elim_entropy_norm",
                        "week_elim_boundary_certainty",
                    ]
                ]
                .drop_duplicates("season_week")
                .sort_values(["season", "week"])
            )
            week_boundary_path = _safe_write_csv(week_boundary, outdir / "bootstrap_boundary_by_season_week.csv")

        # Contestant-level uncertainty summary (CI width + CV + elim probability)
        contestant_uncertainty_path = None
        if {"contestant_id", "season_week", "predicted_fan_vote_cv", "predicted_fan_vote_ci_low", "predicted_fan_vote_ci_high"}.issubset(
            set(boot_pred.columns)
        ):
            tmp = boot_pred[
                [
                    "contestant_id",
                    "season_week",
                    "predicted_fan_vote_cv",
                    "predicted_fan_vote_ci_low",
                    "predicted_fan_vote_ci_high",
                    "elim_prob" if "elim_prob" in boot_pred.columns else None,
                ]
            ].copy()
            tmp = tmp[[c for c in tmp.columns if c is not None]]
            tmp["ci_width"] = tmp["predicted_fan_vote_ci_high"] - tmp["predicted_fan_vote_ci_low"]
            tmp = tmp.drop_duplicates(["contestant_id", "season_week"])
            agg_cols = {
                "predicted_fan_vote_cv": "mean",
                "ci_width": "mean",
            }
            if "elim_prob" in tmp.columns:
                agg_cols["elim_prob"] = "mean"
            cu = tmp.groupby("contestant_id").agg(agg_cols)
            cu["T_i"] = tmp.groupby("contestant_id")["season_week"].nunique()
            cu = cu.reset_index().rename(
                columns={
                    "predicted_fan_vote_cv": "mean_CV_P",
                    "ci_width": "mean_CI_width_P",
                    "elim_prob": "mean_elim_prob",
                }
            )
            contestant_uncertainty_path = _safe_write_csv(cu, outdir / "contestant_uncertainty.csv")

        # ANOVA: does uncertainty vary by week number?
        # Week-level uncertainty is entropy_norm(P_w); run ANOVA across week groups.
        if {"week", "entropy_norm"}.issubset(set(week_entropy.columns)):
            wdf = week_entropy.dropna(subset=["week", "entropy_norm"])
            if len(wdf) > 0:
                res = anova_oneway(wdf["entropy_norm"].to_numpy(), wdf["week"].astype(int).to_numpy())
                _safe_write_text(
                    "ANOVA: entropy_norm ~ week\n" + str(res) + "\n",
                    outdir / "anova_entropy_by_week.txt",
                    encoding="utf-8",
                )

        # ANOVA for decision-boundary certainty across week number (bootstrap only).
        if week_boundary_path is not None and week_boundary_path.exists():
            bdf = pd.read_csv(week_boundary_path).dropna(subset=["week", "week_elim_boundary_certainty"])
            if len(bdf) > 0:
                res2 = anova_oneway(
                    bdf["week_elim_boundary_certainty"].to_numpy(),
                    bdf["week"].astype(int).to_numpy(),
                )
                _safe_write_text(
                    "ANOVA: week_elim_boundary_certainty ~ week\n" + str(res2) + "\n",
                    outdir / "anova_boundary_certainty_by_week.txt",
                    encoding="utf-8",
                )

    outputs = ReportOutputs(
        outdir=outdir,
        weekly_metrics_csv=weekly_metrics_csv,
        kendall_tau_csv=kendall_tau_csv,
        entropy_week_csv=entropy_week_csv,
        entropy_heatmap_png=outdir / "entropy_heatmap.png",
        entropy_by_week_png=outdir / "entropy_by_week.png",
        hitrate_by_week_png=outdir / "hitrate_by_week.png",
        feature_importance_png=feature_importance_png,
    )

    print("\n" + "=" * 60)
    print("REPORT OUTPUTS")
    print("=" * 60)
    print(f"Output dir: {outputs.outdir}")
    print(f"- Weekly metrics CSV: {outputs.weekly_metrics_csv}")
    if outputs.kendall_tau_csv is not None:
        print(f"- Kendall tau CSV: {outputs.kendall_tau_csv}")
    else:
        print("- Kendall tau CSV: (skipped; missing true_elim_week)")
    print(f"- Entropy CSV: {outputs.entropy_week_csv}")
    print(f"- Entropy heatmap: {outputs.entropy_heatmap_png}")
    print(f"- Entropy by week: {outputs.entropy_by_week_png}")
    print(f"- HitRate by week: {outputs.hitrate_by_week_png}")
    if outputs.feature_importance_png is not None:
        print(f"- Feature importance: {outputs.feature_importance_png}")
    if int(args.bootstrap_B) > 0:
        print(f"- Bootstrap predictions: {outdir / 'bootstrap_predictions.csv'}")
        if (outdir / 'bootstrap_linear_params.csv').exists():
            print(f"- Bootstrap params: {outdir / 'bootstrap_linear_params.csv'}")


if __name__ == "__main__":
    main()
