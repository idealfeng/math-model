import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Paths:
    outdir: Path
    edr_summary_txt: Path
    edr_by_season_csv: Path
    spearman_by_season_week_csv: Path
    spearman_heatmap_png: Path
    fanfavor_by_contestant_week_csv: Path
    fanfavor_elim_by_week_csv: Path
    fanfavor_elim_hist_png: Path
    judge_bottom2_scenarios_csv: Path


def _safe_write_csv(df: pd.DataFrame, path: Path) -> Path:
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        alt = path.with_name(path.stem + "_new" + path.suffix)
        df.to_csv(alt, index=False)
        return alt


def _safe_write_text(text: str, path: Path) -> Path:
    try:
        path.write_text(text, encoding="utf-8")
        return path
    except PermissionError:
        alt = path.with_name(path.stem + "_new" + path.suffix)
        alt.write_text(text, encoding="utf-8")
        return alt


def _pct_rank(values: pd.Series, higher_better: bool = True) -> pd.Series:
    """
    Percentile rank in [0,1], higher=better.
    Uses deterministic tie-breaking by index order via 'first' ranking.
    """
    n = int(len(values))
    if n <= 1:
        return pd.Series([0.0] * n, index=values.index, dtype=float)
    rank = values.rank(ascending=not higher_better, method="first")  # 1..n
    pct = (n - rank) / float(n - 1)
    return pct.astype(float)


def _rank_pos(values: pd.Series, higher_better: bool = True) -> pd.Series:
    """Deterministic 1..n rank positions (1=best)."""
    rank = values.rank(ascending=not higher_better, method="first")
    return rank.astype(int)


def _spearman_rho_from_ranks(r1: np.ndarray, r2: np.ndarray) -> float:
    """Spearman rho as Pearson correlation of rank vectors (handles ties if present)."""
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    if len(r1) < 3:
        return float("nan")
    if np.all(r1 == r1[0]) or np.all(r2 == r2[0]):
        return float("nan")
    c = np.corrcoef(r1, r2)
    return float(c[0, 1])


def _heatmap_grid(
    table: pd.DataFrame, value_col: str, season_col: str = "season", week_col: str = "week"
) -> tuple[list[int], list[int], np.ma.MaskedArray]:
    seasons_raw = table[season_col].dropna().astype(int)
    weeks_raw = table[week_col].dropna().astype(int)
    seasons = sorted(seasons_raw.unique().tolist())
    weeks = sorted(weeks_raw.unique().tolist())
    grid = np.full((len(seasons), len(weeks)), np.nan, dtype=float)
    s_to_i = {s: i for i, s in enumerate(seasons)}
    w_to_j = {w: j for j, w in enumerate(weeks)}
    for _, row in table.iterrows():
        if pd.isna(row.get(season_col)) or pd.isna(row.get(week_col)) or pd.isna(row.get(value_col)):
            continue
        s = int(row[season_col])
        w = int(row[week_col])
        v = float(row[value_col])
        if s in s_to_i and w in w_to_j:
            grid[s_to_i[s], w_to_j[w]] = v
    return seasons, weeks, np.ma.masked_invalid(grid)


def plot_spearman_heatmap(
    spearman_by_week: pd.DataFrame,
    outpath: Path,
    value_col: str = "spearman_rho",
    title: str = "Rank-vs-Percentage Ranking Similarity (Spearman rho)",
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib

    seasons, weeks, masked = _heatmap_grid(spearman_by_week, value_col=value_col)

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

    def _truncate_cmap(base, lo: float = 0.10, hi: float = 0.90, n: int = 256):
        base = matplotlib.colormaps.get_cmap(base) if isinstance(base, str) else base
        colors = base(np.linspace(lo, hi, n))
        return LinearSegmentedColormap.from_list(f"{base.name}_trunc", colors)

    # rho in [-1,1], but in practice mostly [0,1]. Keep a diverging map anyway.
    norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)
    cmap = _truncate_cmap("RdBu_r", lo=0.12, hi=0.88)
    cmap.set_bad(color=(1.0, 1.0, 1.0, 0.0))
    im = ax.imshow(masked, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm, alpha=0.92)

    ax.set_title(title)
    ax.set_xlabel("Week")
    ax.set_ylabel("Season")

    ax.set_xticks(list(range(len(weeks))))
    ax.set_xticklabels([str(w) for w in weeks])

    sel_y = [i for i, s in enumerate(seasons) if (int(s) - 1) % 3 == 0]
    ax.set_yticks(sel_y)
    ax.set_yticklabels([str(seasons[i]) for i in sel_y])
    ax.tick_params(axis="y", labelsize=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Spearman rho")
    cbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])

    fig.tight_layout()
    try:
        fig.savefig(outpath, bbox_inches="tight")
    except PermissionError:
        fig.savefig(outpath.with_name(outpath.stem + "_new" + outpath.suffix), bbox_inches="tight")
    plt.close(fig)


def plot_fanfavor_elim_hist(
    by_week: pd.DataFrame,
    outpath: Path,
    col_rank: str = "fanfavor_elim_rank",
    col_pct: str = "fanfavor_elim_percentage",
    title: str = "Fan-Favor Index of Predicted Eliminations (Rank vs Percentage)",
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib

    df = by_week.dropna(subset=[col_rank, col_pct]).copy()
    if len(df) == 0:
        raise ValueError("No rows to plot fanFavor elimination histogram.")

    r = df[col_rank].astype(float).to_numpy()
    p = df[col_pct].astype(float).to_numpy()

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "font.size": 11,
        }
    )
    fig, ax = plt.subplots(figsize=(10.5, 5.2))

    cmap = matplotlib.colormaps.get_cmap("YlOrRd")
    c1 = cmap(0.35)
    c2 = cmap(0.65)

    bins = 28
    lo = float(np.nanquantile(np.concatenate([r, p]), 0.01))
    hi = float(np.nanquantile(np.concatenate([r, p]), 0.99))
    ax.hist(r, bins=bins, range=(lo, hi), alpha=0.55, color=c1, label="Rank method")
    ax.hist(p, bins=bins, range=(lo, hi), alpha=0.45, color=c2, label="Percentage method")

    ax.axvline(0.0, color="#8a3b12", linewidth=1.0, alpha=0.45)
    ax.set_title(title)
    ax.set_xlabel("fanFavor (PctFan - PctJudge) for eliminated contestants")
    ax.set_ylabel("Number of weeks")
    ax.legend(frameon=False)

    fig.tight_layout()
    try:
        fig.savefig(outpath, bbox_inches="tight")
    except PermissionError:
        fig.savefig(outpath.with_name(outpath.stem + "_new" + outpath.suffix), bbox_inches="tight")
    plt.close(fig)


def compute_week_outputs(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Given predictions_unified.csv (must include judge_score, predicted_fan_vote, is_eliminated),
    compute rank vs percentage outcomes for every season_week.
    """
    required = {"season_week", "season", "week", "contestant_id", "judge_score", "predicted_fan_vote"}
    if not required.issubset(set(df.columns)):
        missing = sorted(required - set(df.columns))
        raise ValueError(f"Missing required columns: {missing}")

    eliminated_col = "is_eliminated"
    no_elim_col = "no_elim"
    has_labels = eliminated_col in df.columns

    week_rows = []
    spearman_rows = []
    fanfavor_rows = []
    judge_scenario_rows = []

    for sw, g0 in df.groupby("season_week"):
        g = g0.copy().sort_values("contestant_id").reset_index(drop=True)
        season = int(g["season"].values[0])
        week = int(g["week"].values[0])
        n = int(len(g))
        if n < 2:
            continue

        judge = g["judge_score"].astype(float)
        fan = g["predicted_fan_vote"].astype(float)

        # Fan/Judge percentile ranks for fanFavor
        pct_judge_rank = _pct_rank(judge, higher_better=True)
        pct_fan_rank = _pct_rank(fan, higher_better=True)
        fanfavor = (pct_fan_rank - pct_judge_rank).astype(float)
        g["pct_judge_rank"] = pct_judge_rank
        g["pct_fan_rank"] = pct_fan_rank
        g["fanFavor"] = fanfavor
        fanfavor_rows.append(g[["season_week", "season", "week", "contestant_id", "fanFavor", "pct_fan_rank", "pct_judge_rank"]])

        # Combine rules
        # Percentage method (DWTS): judge share + fan share
        judge_pct = judge / float(judge.sum() + 1e-12)
        score_pct = judge_pct + fan

        # Rank method: sum ranks (1=best), eliminate by largest combined_rank (worst); equivalently lowest combined_score.
        judge_rank = _rank_pos(judge, higher_better=True).astype(float)
        fan_rank = _rank_pos(fan, higher_better=True).astype(float)
        combined_rank = judge_rank + fan_rank
        score_rank = (2.0 * n) - combined_rank  # higher is better

        # Spearman correlation between the two induced rankings (best=1).
        rank_pos_rank = _rank_pos(score_rank, higher_better=True).to_numpy()
        rank_pos_pct = _rank_pos(score_pct, higher_better=True).to_numpy()
        rho = _spearman_rho_from_ranks(rank_pos_rank, rank_pos_pct)
        spearman_rows.append({"season_week": sw, "season": season, "week": week, "n": n, "spearman_rho": rho})

        # Elimination set comparisons require labels.
        k = int(g[eliminated_col].sum()) if has_labels else 0
        no_elim = float(g[no_elim_col].values[0]) if no_elim_col in g.columns else 0.0
        valid_elim_week = (k > 0) and (no_elim != 1.0)

        pred_rank_set = None
        pred_pct_set = None
        actual_set = None

        if valid_elim_week:
            actual_set = set(g.loc[g[eliminated_col] == 1, "contestant_id"].tolist())
            pred_rank = g.assign(score_rank=score_rank).sort_values("score_rank", ascending=True).head(k)["contestant_id"].tolist()
            pred_pct = g.assign(score_pct=score_pct).sort_values("score_pct", ascending=True).head(k)["contestant_id"].tolist()
            pred_rank_set = set(pred_rank)
            pred_pct_set = set(pred_pct)
            edr = 1 if pred_rank_set != pred_pct_set else 0

            # fanFavor of predicted eliminated sets (average, so multi-elim works).
            fanfavor_elim_rank = float(g.set_index("contestant_id").loc[list(pred_rank_set), "fanFavor"].mean()) if len(pred_rank_set) else float("nan")
            fanfavor_elim_pct = float(g.set_index("contestant_id").loc[list(pred_pct_set), "fanFavor"].mean()) if len(pred_pct_set) else float("nan")

            # Which method looks more fan-favoring? Lower eliminated fanFavor => protects fan-favored contestants more.
            week_rows.append(
                {
                    "season_week": sw,
                    "season": season,
                    "week": week,
                    "n": n,
                    "k": k,
                    "edr": edr,
                    "rank_elim": sorted(pred_rank_set),
                    "pct_elim": sorted(pred_pct_set),
                    "fanfavor_elim_rank": fanfavor_elim_rank,
                    "fanfavor_elim_percentage": fanfavor_elim_pct,
                    "rank_hits": len(pred_rank_set & actual_set),
                    "pct_hits": len(pred_pct_set & actual_set),
                }
            )

            # Hypothetical "bottom two + judge chooses one" scenario, only meaningful for single-elimination weeks.
            if k == 1 and n >= 3:
                # Bottom two under each method
                bottom2_rank = (
                    g.assign(score_rank=score_rank).sort_values("score_rank", ascending=True).head(2).reset_index(drop=True)
                )
                bottom2_pct = (
                    g.assign(score_pct=score_pct).sort_values("score_pct", ascending=True).head(2).reset_index(drop=True)
                )

                def _judge_choose(bottom2: pd.DataFrame, mode: str) -> str:
                    if mode == "lowest_judge":
                        return str(bottom2.sort_values("judge_score", ascending=True).iloc[0]["contestant_id"])
                    if mode == "fan_safe":
                        return str(bottom2.sort_values("predicted_fan_vote", ascending=True).iloc[0]["contestant_id"])
                    if mode == "fan_drama":
                        return str(bottom2.sort_values("predicted_fan_vote", ascending=False).iloc[0]["contestant_id"])
                    raise ValueError("unknown mode")

                judge_rank_lowest = _judge_choose(bottom2_rank, "lowest_judge")
                judge_rank_safe = _judge_choose(bottom2_rank, "fan_safe")
                judge_rank_drama = _judge_choose(bottom2_rank, "fan_drama")

                judge_pct_lowest = _judge_choose(bottom2_pct, "lowest_judge")
                judge_pct_safe = _judge_choose(bottom2_pct, "fan_safe")
                judge_pct_drama = _judge_choose(bottom2_pct, "fan_drama")

                judge_scenario_rows.append(
                    {
                        "season_week": sw,
                        "season": season,
                        "week": week,
                        "n": n,
                        "actual_elim": next(iter(actual_set)) if actual_set else "",
                        "rank_bottom2": bottom2_rank["contestant_id"].tolist(),
                        "rank_elim_original": next(iter(pred_rank_set)) if pred_rank_set else "",
                        "rank_elim_judge_lowest": judge_rank_lowest,
                        "rank_elim_judge_fan_safe": judge_rank_safe,
                        "rank_elim_judge_fan_drama": judge_rank_drama,
                        "pct_bottom2": bottom2_pct["contestant_id"].tolist(),
                        "pct_elim_original": next(iter(pred_pct_set)) if pred_pct_set else "",
                        "pct_elim_judge_lowest": judge_pct_lowest,
                        "pct_elim_judge_fan_safe": judge_pct_safe,
                        "pct_elim_judge_fan_drama": judge_pct_drama,
                        "rank_judge_lowest_correct": int(judge_rank_lowest in actual_set) if actual_set else 0,
                        "pct_judge_lowest_correct": int(judge_pct_lowest in actual_set) if actual_set else 0,
                    }
                )

    out = {}
    out["week_comparison"] = pd.DataFrame(week_rows)
    out["spearman_by_week"] = pd.DataFrame(spearman_rows)
    out["fanfavor_by_contestant_week"] = pd.concat(fanfavor_rows, ignore_index=True) if fanfavor_rows else pd.DataFrame()
    out["judge_bottom2_scenarios"] = pd.DataFrame(judge_scenario_rows)
    return out


def summarize_edr(
    week_comparison: pd.DataFrame,
    final_n: int = 4,
) -> tuple[pd.DataFrame, str]:
    if len(week_comparison) == 0:
        return pd.DataFrame(), "No labeled elimination weeks available for EDR.\n"

    df = week_comparison.copy()
    df["is_final_stage"] = df["n"].astype(int) <= int(final_n)

    overall = float(df["edr"].mean())
    overall_n = int(len(df))
    final_df = df[df["is_final_stage"]]
    final = float(final_df["edr"].mean()) if len(final_df) else float("nan")

    by_season = (
        df.groupby("season")
        .agg(weeks=("season_week", "count"), edr=("edr", "mean"), edr_final=("is_final_stage", lambda x: float("nan")))
        .reset_index()
    )
    # Fill final-stage EDR per season
    finals = df[df["is_final_stage"]].groupby("season")["edr"].mean()
    by_season["edr_final"] = by_season["season"].map(finals).astype(float)

    txt = []
    txt.append("Elimination Discrepancy Rate (EDR)\n")
    txt.append(f"EDR_result (overall): {overall:.2%} (weeks={overall_n})\n")
    if np.isfinite(final):
        txt.append(f"EDR_result (final stage, n<= {int(final_n)}): {final:.2%} (weeks={int(len(final_df))})\n")
    txt.append("\nInterpretation: EDR is the fraction of elimination weeks where the predicted eliminated set differs between the rank and percentage combining rules.\n")
    return by_season, "".join(txt)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare DWTS rank vs percentage combining rules using predicted fan-vote shares."
    )
    ap.add_argument("--pred-csv", default="predictions_unified.csv", help="CSV with predicted_fan_vote and judge_score.")
    ap.add_argument("--outdir", default="reports", help="Output directory.")
    ap.add_argument("--final-n", type=int, default=4, help="Define 'final stage' as weeks with n<=final_n.")
    args = ap.parse_args()

    pred_path = Path(args.pred_csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    paths = Paths(
        outdir=outdir,
        edr_summary_txt=outdir / "method_edr_summary.txt",
        edr_by_season_csv=outdir / "method_edr_by_season.csv",
        spearman_by_season_week_csv=outdir / "spearman_by_season_week.csv",
        spearman_heatmap_png=outdir / "spearman_heatmap.png",
        fanfavor_by_contestant_week_csv=outdir / "fanfavor_by_contestant_week.csv",
        fanfavor_elim_by_week_csv=outdir / "fanfavor_elim_by_week.csv",
        fanfavor_elim_hist_png=outdir / "fanfavor_elim_hist.png",
        judge_bottom2_scenarios_csv=outdir / "judge_bottom2_scenarios.csv",
    )

    df = pd.read_csv(pred_path)

    outs = compute_week_outputs(df)
    week_cmp = outs["week_comparison"]
    spearman = outs["spearman_by_week"]
    fanfavor = outs["fanfavor_by_contestant_week"]
    judge_scen = outs["judge_bottom2_scenarios"]

    # EDR summary
    by_season, edr_txt = summarize_edr(week_cmp, final_n=int(args.final_n))
    _safe_write_text(edr_txt, paths.edr_summary_txt)
    if len(by_season):
        _safe_write_csv(by_season, paths.edr_by_season_csv)

    # Spearman heatmap + CSV
    _safe_write_csv(spearman.sort_values(["season", "week"]), paths.spearman_by_season_week_csv)
    try:
        plot_spearman_heatmap(spearman, paths.spearman_heatmap_png)
    except Exception:
        pass

    # fanFavor outputs
    if len(fanfavor):
        _safe_write_csv(fanfavor, paths.fanfavor_by_contestant_week_csv)
    if len(week_cmp):
        _safe_write_csv(week_cmp, paths.fanfavor_elim_by_week_csv)
        try:
            plot_fanfavor_elim_hist(week_cmp, paths.fanfavor_elim_hist_png)
        except Exception:
            pass

    # Hypothetical bottom-two scenarios
    if len(judge_scen):
        _safe_write_csv(judge_scen, paths.judge_bottom2_scenarios_csv)

    print("\n" + "=" * 60)
    print("METHOD COMPARISON OUTPUTS")
    print("=" * 60)
    print(f"Outdir: {paths.outdir}")
    print(f"- EDR summary: {paths.edr_summary_txt}")
    if paths.edr_by_season_csv.exists():
        print(f"- EDR by season: {paths.edr_by_season_csv}")
    print(f"- Spearman by week: {paths.spearman_by_season_week_csv}")
    if paths.spearman_heatmap_png.exists():
        print(f"- Spearman heatmap: {paths.spearman_heatmap_png}")
    if paths.fanfavor_elim_by_week_csv.exists():
        print(f"- fanFavor elim-by-week: {paths.fanfavor_elim_by_week_csv}")
    if paths.fanfavor_elim_hist_png.exists():
        print(f"- fanFavor hist: {paths.fanfavor_elim_hist_png}")
    if paths.judge_bottom2_scenarios_csv.exists():
        print(f"- Bottom-two judge scenarios: {paths.judge_bottom2_scenarios_csv}")


if __name__ == "__main__":
    main()

