import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _safe_write_csv(df: pd.DataFrame, path: Path) -> Path:
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        alt = path.with_name(path.stem + "_new" + path.suffix)
        df.to_csv(alt, index=False)
        return alt


def _pct_rank(values: pd.Series, higher_better: bool = True) -> pd.Series:
    n = int(len(values))
    if n <= 1:
        return pd.Series([0.0] * n, index=values.index, dtype=float)
    rank = values.rank(ascending=not higher_better, method="first")  # 1..n
    return ((n - rank) / float(n - 1)).astype(float)


def compute_fanfavor_by_contestant_week(pred: pd.DataFrame) -> pd.DataFrame:
    required = {"season_week", "season", "week", "contestant_id", "judge_score", "predicted_fan_vote"}
    missing = sorted(required - set(pred.columns))
    if missing:
        raise ValueError(f"Missing columns in predictions: {missing}")

    rows = []
    for sw, g0 in pred.groupby("season_week"):
        g = g0.copy().sort_values("contestant_id")
        if len(g) < 2:
            continue
        pct_j = _pct_rank(g["judge_score"].astype(float), higher_better=True)
        pct_f = _pct_rank(g["predicted_fan_vote"].astype(float), higher_better=True)
        out = pd.DataFrame(
            {
                "season_week": sw,
                "season": g["season"].astype(int).values,
                "week": g["week"].astype(int).values,
                "contestant_id": g["contestant_id"].astype(str).values,
                "pct_fan_rank": pct_f.values,
                "pct_judge_rank": pct_j.values,
            }
        )
        out["fanFavor"] = out["pct_fan_rank"] - out["pct_judge_rank"]
        rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def extract_top_bottom(
    fanfavor_df: pd.DataFrame,
    q: float = 0.10,
    min_n_per_week: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = fanfavor_df.copy()
    df = df.dropna(subset=["fanFavor", "season", "week", "season_week", "contestant_id"])
    df = df[df.groupby("season_week")["contestant_id"].transform("count") >= int(min_n_per_week)]
    if len(df) == 0:
        return pd.DataFrame(), pd.DataFrame(), {"q": q, "low": np.nan, "high": np.nan, "rows": 0}

    low = float(df["fanFavor"].quantile(q))
    high = float(df["fanFavor"].quantile(1.0 - q))
    top = df[df["fanFavor"] >= high].sort_values("fanFavor", ascending=False)
    bottom = df[df["fanFavor"] <= low].sort_values("fanFavor", ascending=True)
    meta = {"q": q, "low": low, "high": high, "rows": int(len(df))}
    return top, bottom, meta


def build_bottom2_summary(
    outdir: Path,
    fanfavor_df: pd.DataFrame,
    q: float = 0.10,
) -> None:
    scen_path = outdir / "judge_bottom2_scenarios.csv"
    if not scen_path.exists():
        return

    scen = pd.read_csv(scen_path)
    if len(scen) == 0:
        return

    ff = fanfavor_df[["season_week", "contestant_id", "fanFavor"]].copy()
    ff["season_week"] = ff["season_week"].astype(str)
    ff["contestant_id"] = ff["contestant_id"].astype(str)
    ff_map = ff.set_index(["season_week", "contestant_id"])["fanFavor"]

    def _ff(sw: str, cid: str) -> float:
        try:
            return float(ff_map.loc[(str(sw), str(cid))])
        except Exception:
            return float("nan")

    rows = []
    for _, r in scen.iterrows():
        sw = str(r["season_week"])
        rows.append(
            {
                "season_week": sw,
                "season": int(r["season"]),
                "week": int(r["week"]),
                "rank_elim_original": str(r.get("rank_elim_original", "")),
                "rank_elim_judge_lowest": str(r.get("rank_elim_judge_lowest", "")),
                "rank_elim_judge_fan_safe": str(r.get("rank_elim_judge_fan_safe", "")),
                "rank_elim_judge_fan_drama": str(r.get("rank_elim_judge_fan_drama", "")),
                "pct_elim_original": str(r.get("pct_elim_original", "")),
                "pct_elim_judge_lowest": str(r.get("pct_elim_judge_lowest", "")),
                "pct_elim_judge_fan_safe": str(r.get("pct_elim_judge_fan_safe", "")),
                "pct_elim_judge_fan_drama": str(r.get("pct_elim_judge_fan_drama", "")),
                "actual_elim": str(r.get("actual_elim", "")),
            }
        )

    df = pd.DataFrame(rows)
    # Attach fanFavor for the eliminated contestant under each scenario.
    for c in [
        "rank_elim_original",
        "rank_elim_judge_lowest",
        "rank_elim_judge_fan_safe",
        "rank_elim_judge_fan_drama",
        "pct_elim_original",
        "pct_elim_judge_lowest",
        "pct_elim_judge_fan_safe",
        "pct_elim_judge_fan_drama",
    ]:
        df[c + "_fanFavor"] = [ _ff(sw, cid) for sw, cid in zip(df["season_week"], df[c]) ]

    # Define "controversial elimination" weeks as those whose eliminated contestant has high |fanFavor|.
    elim_ff = df["pct_elim_original_fanFavor"].abs()
    if elim_ff.notna().any():
        thr = float(elim_ff.quantile(1.0 - q))
        df["is_controversial_week"] = elim_ff >= thr
    else:
        df["is_controversial_week"] = False

    # Overall deltas vs original elimination (how often judge choice changes the eliminated contestant).
    def _change_rate(a: pd.Series, b: pd.Series) -> float:
        m = (a.astype(str) != b.astype(str)) & a.notna() & b.notna()
        return float(m.mean()) if len(m) else float("nan")

    summary = []
    for prefix in ["rank", "pct"]:
        orig = df[f"{prefix}_elim_original"]
        for mode in ["judge_lowest", "judge_fan_safe", "judge_fan_drama"]:
            alt = df[f"{prefix}_elim_{mode}"]
            summary.append(
                {
                    "method": prefix,
                    "scenario": mode,
                    "weeks": int(len(df)),
                    "change_rate_vs_original": _change_rate(orig, alt),
                    "accuracy_vs_actual": float((alt.astype(str) == df["actual_elim"].astype(str)).mean()),
                }
            )

    summary_df = pd.DataFrame(summary)
    _safe_write_csv(summary_df, outdir / "judge_bottom2_summary.csv")

    # Export controversial subset examples (top |fanFavor| eliminated under pct original).
    ex = df[df["is_controversial_week"]].copy()
    ex = ex.sort_values("pct_elim_original_fanFavor", key=lambda s: s.abs(), ascending=False)
    _safe_write_csv(ex, outdir / "judge_bottom2_controversial_weeks.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract top/bottom fanFavor cases and summarize bottom-2 judge scenarios.")
    ap.add_argument("--outdir", default="reports", help="Output directory (contains fanfavor/judge scenario CSVs).")
    ap.add_argument("--pred-csv", default="predictions_unified.csv", help="Predictions CSV to compute fanFavor if needed.")
    ap.add_argument("--q", type=float, default=0.10, help="Quantile for top/bottom selection, e.g. 0.10 for 10%%.")
    ap.add_argument("--min-weeks", type=int, default=2, help="Minimum contestants per week to include.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fanfavor_path = outdir / "fanfavor_by_contestant_week.csv"
    if fanfavor_path.exists():
        ff = pd.read_csv(fanfavor_path)
    else:
        pred = pd.read_csv(Path(args.pred_csv))
        ff = compute_fanfavor_by_contestant_week(pred)
        _safe_write_csv(ff, fanfavor_path)

    top, bottom, meta = extract_top_bottom(ff, q=float(args.q), min_n_per_week=int(args.min_weeks))

    top_path = _safe_write_csv(
        top[["season", "week", "season_week", "contestant_id", "fanFavor", "pct_fan_rank", "pct_judge_rank"]],
        outdir / "fanfavor_top10pct_contestant_weeks.csv",
    )
    bottom_path = _safe_write_csv(
        bottom[["season", "week", "season_week", "contestant_id", "fanFavor", "pct_fan_rank", "pct_judge_rank"]],
        outdir / "fanfavor_bottom10pct_contestant_weeks.csv",
    )

    build_bottom2_summary(outdir=outdir, fanfavor_df=ff, q=float(args.q))

    print("\n" + "=" * 60)
    print("CONTROVERSY OUTPUTS")
    print("=" * 60)
    print(f"Top {int(meta['q']*100)}% threshold: fanFavor >= {meta['high']:.3f} (rows={len(top)})")
    print(f"Bottom {int(meta['q']*100)}% threshold: fanFavor <= {meta['low']:.3f} (rows={len(bottom)})")
    print(f"- {top_path}")
    print(f"- {bottom_path}")
    if (outdir / "judge_bottom2_summary.csv").exists():
        print(f"- {outdir / 'judge_bottom2_summary.csv'}")
    if (outdir / "judge_bottom2_controversial_weeks.csv").exists():
        print(f"- {outdir / 'judge_bottom2_controversial_weeks.csv'}")


if __name__ == "__main__":
    main()
