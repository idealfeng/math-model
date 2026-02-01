import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _extract_pro_dancer(contestant_id: str) -> str:
    parts = str(contestant_id).rsplit("_", 2)
    if len(parts) != 3:
        return "UNKNOWN"
    _, pro, _ = parts
    return pro


def _zscore_within_group(x: pd.Series) -> pd.Series:
    mu = float(x.mean())
    sd = float(x.std(ddof=0))
    if not np.isfinite(sd) or sd < 1e-9:
        return x * 0.0
    return (x - mu) / sd


def _pct_rank_within_group(x: pd.Series, high_is_good: bool = True) -> pd.Series:
    n = len(x)
    if n <= 1:
        return x * 0.0
    asc = not high_is_good
    r = x.rank(method="average", ascending=asc)
    return (r - 1.0) / float(n - 1)


def _add_targets_and_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["pro_dancer"] = out["contestant_id"].map(_extract_pro_dancer)

    out["judge_score_z"] = out.groupby("season_week", sort=False)["judge_score"].transform(
        _zscore_within_group
    )

    out["pct_fan_rank"] = out.groupby("season_week", sort=False)["predicted_fan_vote"].transform(
        lambda s: _pct_rank_within_group(s, high_is_good=True)
    )
    out["pct_judge_rank"] = out.groupby("season_week", sort=False)["judge_score"].transform(
        lambda s: _pct_rank_within_group(s, high_is_good=True)
    )
    out["fanFavor"] = out["pct_fan_rank"] - out["pct_judge_rank"]

    out = out.sort_values(["season", "contestant_id", "week"]).reset_index(drop=True)
    out["fan_vote_prev"] = out.groupby(["season", "contestant_id"])["predicted_fan_vote"].shift(1)
    out["fan_vote_delta"] = out["predicted_fan_vote"] - out["fan_vote_prev"]
    out["judge_z_prev"] = out.groupby(["season", "contestant_id"])["judge_score_z"].shift(1)
    out["judge_z_delta"] = out["judge_score_z"] - out["judge_z_prev"]

    for c in ["fan_vote_prev", "fan_vote_delta", "judge_z_prev", "judge_z_delta"]:
        out[c] = out[c].fillna(0.0)

    return out


def _load_predictions(predictions_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(predictions_csv)
    required = [
        "contestant_id",
        "season",
        "week",
        "season_week",
        "judge_score",
        "predicted_fan_vote",
        "num_contestants",
        "week_pct",
        "contestants_pct",
        "age_normalized",
        "fame_normalized",
        "gender",
        "industry",
        "experience",
        "is_hetero",
        "state_normalized",
        "max_score_normalized",
        "sharpness",
        "prev_rank",
        "is_all_star",
        "no_elim",
        "multi_elim",
    ]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns in {predictions_csv}: {missing}")
    return df


@dataclass
class ModelBundle:
    name: str
    target: str
    feature_cols: list[str]
    categorical_cols: list[str]
    model: object
    explainer: object


def _train_lgbm(
    df: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    target_col: str,
    train_max_season: int,
    seed: int,
):
    import lightgbm as lgb

    train_df = df[df["season"] <= train_max_season].copy()
    test_df = df[df["season"] > train_max_season].copy()
    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError("Not enough data for time split; adjust train_max_season.")

    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].astype(float).values
    X_test = test_df[feature_cols].copy()
    y_test = test_df[target_col].astype(float).values

    for c in categorical_cols:
        if c in X_train.columns:
            X_train[c] = X_train[c].astype("category")
            X_test[c] = X_test[c].astype("category")

    seasons = np.sort(train_df["season"].unique())
    split_season = int(seasons[int(0.8 * (len(seasons) - 1))]) if len(seasons) >= 5 else int(
        seasons[-2]
    )
    tr_mask = train_df["season"] <= split_season
    va_mask = ~tr_mask

    model = lgb.LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=int(seed),
        verbose=-1,
        n_jobs=-1,
    )
    model.fit(
        X_train.loc[tr_mask],
        y_train[tr_mask.values],
        eval_set=[(X_train.loc[va_mask], y_train[va_mask.values])],
        eval_metric="l2",
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        categorical_feature=[c for c in categorical_cols if c in feature_cols],
    )

    # quick sanity metrics
    pred = model.predict(X_test)
    rmse = float(math.sqrt(np.mean((pred - y_test) ** 2)))
    r2 = float(1.0 - (np.sum((pred - y_test) ** 2) / (np.sum((y_test - y_test.mean()) ** 2) + 1e-12)))
    return model, {"rmse": rmse, "r2": r2}


def _set_plot_style():
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def _savefig(path: Path) -> None:
    import matplotlib.pyplot as plt

    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def _shap_mean_abs(shap_values: np.ndarray) -> np.ndarray:
    sv = np.asarray(shap_values, dtype=float)
    if sv.ndim != 2:
        raise ValueError("shap_values must be 2D for mean-abs importance.")
    return np.mean(np.abs(sv), axis=0)


def _pick_top_numeric_features(
    shap_values: np.ndarray, feature_cols: list[str], X: pd.DataFrame, k: int = 3
) -> list[str]:
    imp = _shap_mean_abs(shap_values)
    order = np.argsort(imp)[::-1]
    picked = []
    for idx in order:
        f = feature_cols[int(idx)]
        if f not in X.columns:
            continue
        if str(X[f].dtype) == "category":
            continue
        picked.append(f)
        if len(picked) >= k:
            break
    return picked


def _plot_shap_beeswarm(
    bundle: ModelBundle, X: pd.DataFrame, out_path: Path, max_display: int = 18
) -> None:
    import matplotlib.pyplot as plt
    import shap

    shap_values = bundle.explainer.shap_values(X)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False, max_display=int(max_display))
    plt.title(f"SHAP Summary (beeswarm) — {bundle.name}", pad=10)
    _savefig(out_path)


def _plot_shap_dependence(
    bundle: ModelBundle,
    X: pd.DataFrame,
    out_path: Path,
    feature: str,
    interaction: str | None = "auto",
) -> None:
    import matplotlib.pyplot as plt
    import shap

    shap_values = bundle.explainer.shap_values(X)
    plt.figure(figsize=(8.5, 5.5))
    shap.dependence_plot(
        feature,
        shap_values,
        X,
        interaction_index=interaction,
        show=False,
    )
    plt.title(f"SHAP Dependence — {bundle.name}: {feature}", pad=10)
    _savefig(out_path)


def _plot_shap_waterfall(
    bundle: ModelBundle,
    X_row: pd.DataFrame,
    out_path: Path,
    title: str,
    max_display: int = 14,
) -> None:
    import matplotlib.pyplot as plt
    import shap

    sv = bundle.explainer.shap_values(X_row)
    if isinstance(sv, list):
        sv = sv[0]
    sv = np.asarray(sv, dtype=float).reshape(-1)
    base = bundle.explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        base = float(np.asarray(base).reshape(-1)[0])

    exp = shap.Explanation(
        values=sv,
        base_values=base,
        data=X_row.iloc[0].values,
        feature_names=list(X_row.columns),
    )
    plt.figure(figsize=(9.5, 5.5))
    shap.plots.waterfall(exp, show=False, max_display=int(max_display))
    plt.title(title, pad=12)
    _savefig(out_path)


def _load_case_rows(
    top_csv: Path, bottom_csv: Path, n_each: int = 1
) -> tuple[list[dict], list[dict]]:
    top = pd.read_csv(top_csv)
    bottom = pd.read_csv(bottom_csv)

    # Keep deterministic: use head(n_each)
    top_rows = top.head(int(n_each)).to_dict(orient="records")
    bottom_rows = bottom.head(int(n_each)).to_dict(orient="records")
    return top_rows, bottom_rows


def run(
    predictions_csv: str,
    reports_dir: str,
    train_max_season: int,
    seed: int,
    max_samples: int,
    cases_each_side: int,
):
    _set_plot_style()
    out_dir = Path(reports_dir)
    _safe_mkdir(out_dir)

    df = _load_predictions(Path(predictions_csv))
    df = _add_targets_and_features(df)

    feature_cols = [
        "age_normalized",
        "fame_normalized",
        "gender",
        "industry",
        "experience",
        "is_hetero",
        "state_normalized",
        "max_score_normalized",
        "sharpness",
        "prev_rank",
        "is_all_star",
        "no_elim",
        "multi_elim",
        "week",
        "week_pct",
        "contestants_pct",
        "num_contestants",
        "pro_dancer",
        "season",
        "fan_vote_prev",
        "fan_vote_delta",
        "judge_z_prev",
        "judge_z_delta",
    ]
    categorical_cols = ["pro_dancer"]

    # Prepare X matrix once
    X_all = df[feature_cols].copy()
    for c in categorical_cols:
        X_all[c] = X_all[c].astype("category")

    # Sample for global plots (fast + stable)
    X_plot = X_all.sample(n=min(int(max_samples), len(X_all)), random_state=seed)

    import shap

    bundles: list[ModelBundle] = []
    for name, target, extra in [
        ("judge", "judge_score_z", []),
        ("fan", "predicted_fan_vote", ["judge_score_z"]),
        ("divergence", "fanFavor", []),
    ]:
        feats = list(feature_cols)
        if extra:
            for e in extra:
                if e not in feats and e in df.columns:
                    feats.append(e)
        X_all_t = df[feats].copy()
        for c in categorical_cols:
            if c in X_all_t.columns:
                X_all_t[c] = X_all_t[c].astype("category")
        model, _ = _train_lgbm(
            df, feature_cols=feats, categorical_cols=categorical_cols, target_col=target,
            train_max_season=train_max_season, seed=seed
        )
        explainer = shap.TreeExplainer(model)
        bundles.append(
            ModelBundle(
                name=name,
                target=target,
                feature_cols=feats,
                categorical_cols=categorical_cols,
                model=model,
                explainer=explainer,
            )
        )

        # Beeswarm
        Xp = X_all_t.loc[X_plot.index].copy()
        for c in categorical_cols:
            if c in Xp.columns:
                Xp[c] = Xp[c].astype("category")
        _plot_shap_beeswarm(bundle=bundles[-1], X=Xp, out_path=out_dir / f"shap_beeswarm_{name}.png")

        # Dependence plots for top numeric features (2)
        sv = explainer.shap_values(Xp)
        if isinstance(sv, list):
            sv = sv[0]
        top_feats = _pick_top_numeric_features(sv, bundles[-1].feature_cols, Xp, k=2)
        for f in top_feats:
            _plot_shap_dependence(
                bundle=bundles[-1],
                X=Xp,
                out_path=out_dir / f"shap_dependence_{name}_{f}.png",
                feature=f,
                interaction="auto",
            )

    # One explicit interaction plot for divergence: judge_z_delta × fan_vote_prev
    div_bundle = next(b for b in bundles if b.name == "divergence")
    X_div = df[div_bundle.feature_cols].copy()
    for c in categorical_cols:
        if c in X_div.columns:
            X_div[c] = X_div[c].astype("category")
    X_div_plot = X_div.loc[X_plot.index].copy()
    _plot_shap_dependence(
        div_bundle,
        X_div_plot,
        out_dir / "shap_interaction_divergence_judge_z_delta_x_fan_vote_prev.png",
        feature="judge_z_delta",
        interaction="fan_vote_prev",
    )

    # Case studies: pick a few extreme fanFavor rows
    top_csv = out_dir / "fanfavor_top10pct_contestant_weeks.csv"
    bottom_csv = out_dir / "fanfavor_bottom10pct_contestant_weeks.csv"
    if top_csv.exists() and bottom_csv.exists():
        top_rows, bottom_rows = _load_case_rows(top_csv, bottom_csv, n_each=cases_each_side)
    else:
        top_rows, bottom_rows = [], []

    chosen = []

    def _case_plot(case: dict, label: str, idx: int):
        contestant_id = str(case.get("contestant_id", ""))
        season_week = str(case.get("season_week", ""))
        season = int(case.get("season", -1))
        week = int(case.get("week", -1))
        ff = float(case.get("fanFavor", float("nan")))
        chosen.append(
            {
                "label": label,
                "rank": idx,
                "season_week": season_week,
                "season": season,
                "week": week,
                "contestant_id": contestant_id,
                "fanFavor": ff,
            }
        )

        row_mask = (df["season_week"].astype(str) == season_week) & (
            df["contestant_id"].astype(str) == contestant_id
        )
        if row_mask.sum() != 1:
            return

        # Divergence waterfall
        X_row_div = df.loc[row_mask, div_bundle.feature_cols].copy()
        for c in categorical_cols:
            if c in X_row_div.columns:
                X_row_div[c] = X_row_div[c].astype("category")
        _plot_shap_waterfall(
            div_bundle,
            X_row_div,
            out_dir / f"shap_case_{label}{idx}_divergence_waterfall.png",
            title=f"Case {label}{idx} — divergence (fanFavor) | {season_week}",
        )

        # Fan waterfall
        fan_bundle = next(b for b in bundles if b.name == "fan")
        X_row_fan = df.loc[row_mask, fan_bundle.feature_cols].copy()
        for c in categorical_cols:
            if c in X_row_fan.columns:
                X_row_fan[c] = X_row_fan[c].astype("category")
        _plot_shap_waterfall(
            fan_bundle,
            X_row_fan,
            out_dir / f"shap_case_{label}{idx}_fan_waterfall.png",
            title=f"Case {label}{idx} — fan vote | {season_week}",
        )

    for i, r in enumerate(top_rows, start=1):
        _case_plot(r, "TOP", i)
    for i, r in enumerate(bottom_rows, start=1):
        _case_plot(r, "BOT", i)

    if chosen:
        pd.DataFrame(chosen).to_csv(out_dir / "shap_case_selection.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate SHAP figures (beeswarm, dependence, cases).")
    ap.add_argument("--predictions-csv", default="predictions_unified.csv")
    ap.add_argument("--reports-dir", default="reports")
    ap.add_argument("--train-max-season", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-samples", type=int, default=1500)
    ap.add_argument("--cases-each-side", type=int, default=2)
    args = ap.parse_args()

    run(
        predictions_csv=args.predictions_csv,
        reports_dir=args.reports_dir,
        train_max_season=args.train_max_season,
        seed=args.seed,
        max_samples=args.max_samples,
        cases_each_side=args.cases_each_side,
    )


if __name__ == "__main__":
    main()

