import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions file: {path}")
    df = pd.read_csv(path)
    required = {
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
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"predictions file missing columns: {missing}")
    return df


def _extract_pro_dancer(contestant_id: str) -> str:
    """
    contestant_id format from this project:
      <celebrity_name>_<ballroom_partner>_<season>
    We use a right-split to be robust to underscores elsewhere.
    """
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
    """
    Percentile rank in [0,1] within a group.
    - 1.0 means best, 0.0 means worst.
    """
    n = len(x)
    if n <= 1:
        return x * 0.0
    asc = not high_is_good
    r = x.rank(method="average", ascending=asc)
    return (r - 1.0) / float(n - 1)


def _add_groupwise_targets_and_features(df: pd.DataFrame, add_momentum: bool) -> pd.DataFrame:
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

    if add_momentum:
        out = out.sort_values(["season", "contestant_id", "week"]).reset_index(drop=True)
        out["fan_vote_prev"] = out.groupby(["season", "contestant_id"])["predicted_fan_vote"].shift(
            1
        )
        out["fan_vote_delta"] = out["predicted_fan_vote"] - out["fan_vote_prev"]
        out["judge_z_prev"] = out.groupby(["season", "contestant_id"])["judge_score_z"].shift(1)
        out["judge_z_delta"] = out["judge_score_z"] - out["judge_z_prev"]

        # Keep tensors/trees finite
        for c in ["fan_vote_prev", "fan_vote_delta", "judge_z_prev", "judge_z_delta"]:
            out[c] = out[c].fillna(0.0)

    return out


def _maybe_import_lightgbm():
    try:
        import lightgbm as lgb  # type: ignore

        return lgb
    except Exception as e:
        raise RuntimeError(
            "lightgbm is required for this script. Install it in your environment."
        ) from e


def _maybe_import_shap():
    try:
        import shap  # type: ignore

        return shap
    except Exception:
        return None


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": mae, "rmse": rmse, "r2": r2}


@dataclass
class TrainResult:
    target: str
    metrics: dict
    n_train: int
    n_test: int
    feature_names: list[str]
    model: object


def _train_holdout_by_season(
    df: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    target_col: str,
    train_max_season: int,
    seed: int,
    n_estimators: int,
    learning_rate: float,
) -> TrainResult:
    lgb = _maybe_import_lightgbm()

    train_df = df[df["season"] <= train_max_season].copy()
    test_df = df[df["season"] > train_max_season].copy()
    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError(
            f"Not enough data for holdout split with train_max_season={train_max_season} "
            f"(train={len(train_df)}, test={len(test_df)})."
        )

    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].astype(float).values
    X_test = test_df[feature_cols].copy()
    y_test = test_df[target_col].astype(float).values

    for c in categorical_cols:
        if c in X_train.columns:
            X_train[c] = X_train[c].astype("category")
            X_test[c] = X_test[c].astype("category")

    # Internal validation split (by season) for early stopping
    seasons = np.sort(train_df["season"].unique())
    split_season = int(seasons[int(0.8 * (len(seasons) - 1))]) if len(seasons) >= 5 else int(
        seasons[-2]
    )
    tr_mask = train_df["season"] <= split_season
    va_mask = ~tr_mask
    X_tr = X_train.loc[tr_mask].copy()
    y_tr = y_train[tr_mask.values]
    X_va = X_train.loc[va_mask].copy()
    y_va = y_train[va_mask.values]

    model = lgb.LGBMRegressor(
        n_estimators=int(n_estimators),
        learning_rate=float(learning_rate),
        num_leaves=63,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=int(seed),
        verbose=-1,
        n_jobs=-1,
    )

    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="l2",
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        categorical_feature=[c for c in categorical_cols if c in feature_cols],
    )

    y_pred = model.predict(X_test)
    m = _metrics(y_test, y_pred)

    return TrainResult(
        target=target_col,
        metrics=m,
        n_train=int(len(train_df)),
        n_test=int(len(test_df)),
        feature_names=list(feature_cols),
        model=model,
    )


def _train_group_kfold(
    df: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    target_col: str,
    seed: int,
    n_splits: int,
    n_estimators: int,
    learning_rate: float,
) -> TrainResult:
    lgb = _maybe_import_lightgbm()
    from sklearn.model_selection import GroupKFold

    X_all = df[feature_cols].copy()
    y_all = df[target_col].astype(float).values
    groups = df["season"].values

    for c in categorical_cols:
        if c in X_all.columns:
            X_all[c] = X_all[c].astype("category")

    gkf = GroupKFold(n_splits=int(n_splits))
    fold_metrics = []

    best_model = None
    best_rmse = float("inf")

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X_all, y_all, groups), start=1):
        X_tr = X_all.iloc[tr_idx].copy()
        y_tr = y_all[tr_idx]
        X_te = X_all.iloc[te_idx].copy()
        y_te = y_all[te_idx]

        # Early stopping uses a small random split inside training fold.
        rng = np.random.default_rng(seed + fold)
        perm = rng.permutation(len(X_tr))
        split = int(0.85 * len(perm))
        tr2_idx = perm[:split]
        va_idx = perm[split:]

        model = lgb.LGBMRegressor(
            n_estimators=int(n_estimators),
            learning_rate=float(learning_rate),
            num_leaves=63,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            random_state=int(seed + fold),
            verbose=-1,
            n_jobs=-1,
        )
        model.fit(
            X_tr.iloc[tr2_idx],
            y_tr[tr2_idx],
            eval_set=[(X_tr.iloc[va_idx], y_tr[va_idx])],
            eval_metric="l2",
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
            categorical_feature=[c for c in categorical_cols if c in feature_cols],
        )

        pred = model.predict(X_te)
        m = _metrics(y_te, pred)
        fold_metrics.append(m)

        if m["rmse"] < best_rmse:
            best_rmse = m["rmse"]
            best_model = model

    mean_metrics = {
        k: float(np.mean([fm[k] for fm in fold_metrics])) for k in fold_metrics[0].keys()
    }

    return TrainResult(
        target=target_col,
        metrics={"cv_mean": mean_metrics, "cv_folds": fold_metrics},
        n_train=int(len(df)),
        n_test=0,
        feature_names=list(feature_cols),
        model=best_model,
    )


def _plot_feature_importance(model, feature_names: list[str], out_png: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    # Prefer gain importance if possible
    try:
        imp = model.booster_.feature_importance(importance_type="gain")
        names = model.booster_.feature_name()
    except Exception:
        imp = getattr(model, "feature_importances_", None)
        names = feature_names
        if imp is None:
            return

    imp = np.asarray(imp, dtype=float)
    names = list(names)
    order = np.argsort(imp)[::-1]
    top_k = min(20, len(order))
    order = order[:top_k]

    plot_imp = imp[order]
    plot_names = [names[i] for i in order]

    # Warm, paper-friendly palette
    fig = plt.figure(figsize=(10, 6), dpi=200)
    ax = plt.gca()
    ax.barh(range(len(plot_names))[::-1], plot_imp, color="#E07A5F", alpha=0.85)
    ax.set_yticks(range(len(plot_names))[::-1])
    ax.set_yticklabels(plot_names, fontsize=10)
    ax.set_xlabel("Importance (gain)", fontsize=11)
    ax.set_title(title, fontsize=14, pad=10)
    ax.grid(axis="x", alpha=0.15)
    plt.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def _try_plot_shap_summary(
    model,
    X: pd.DataFrame,
    out_png: Path,
    title: str,
    max_display: int = 20,
) -> bool:
    shap = _maybe_import_shap()
    if shap is None:
        return False

    import matplotlib.pyplot as plt

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        plt.figure(figsize=(10, 6), dpi=200)
        shap.summary_plot(
            shap_values,
            X,
            show=False,
            max_display=int(max_display),
            plot_type="bar",
            color="#E07A5F",
        )
        plt.title(title, fontsize=14, pad=10)
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()
        return True
    except Exception:
        return False


def _pro_dancer_uplift(model, X: pd.DataFrame, pro_col: str) -> pd.DataFrame:
    """
    Counterfactual-style uplift:
      For each pro dancer p, set pro_col=p for all rows and re-predict,
      then report delta vs baseline average prediction.
    """
    if pro_col not in X.columns:
        return pd.DataFrame()

    # Keep categorical encoding consistent with what LightGBM saw at fit time.
    base_is_cat = str(X[pro_col].dtype) == "category"
    base_categories = None
    if base_is_cat:
        base_categories = list(X[pro_col].cat.categories)

    baseline_pred = model.predict(X)
    baseline_mean = float(np.mean(baseline_pred))

    pros = pd.Series(X[pro_col].astype(str).unique()).sort_values().tolist()
    rows = []
    X_tmp = X.copy()
    if base_is_cat and base_categories is not None:
        X_tmp[pro_col] = X_tmp[pro_col].cat.set_categories(base_categories)
    for p in pros:
        if base_is_cat and base_categories is not None:
            X_tmp[pro_col] = pd.Categorical([p] * len(X_tmp), categories=base_categories)
        else:
            X_tmp[pro_col] = p
        pred = model.predict(X_tmp)
        mean_pred = float(np.mean(pred))
        rows.append(
            {
                "pro_dancer": p,
                "mean_pred": mean_pred,
                "delta_vs_baseline": mean_pred - baseline_mean,
            }
        )
    return pd.DataFrame(rows).sort_values("delta_vs_baseline", ascending=False)


def run(
    predictions_csv: str,
    out_dir: str,
    train_max_season: int,
    seed: int,
    add_momentum: bool,
    n_estimators: int,
    learning_rate: float,
    kfold_splits: int,
    include_judge_in_fan: bool,
) -> None:
    out_path = Path(out_dir)
    _safe_mkdir(out_path)

    df = _read_predictions(Path(predictions_csv))
    df = _add_groupwise_targets_and_features(df, add_momentum=bool(add_momentum))

    base_features = [
        # static
        "age_normalized",
        "fame_normalized",
        "gender",
        "industry",
        "experience",
        "is_hetero",
        # dynamic
        "state_normalized",
        "max_score_normalized",
        "sharpness",
        "prev_rank",
        # context
        "is_all_star",
        "no_elim",
        "multi_elim",
        # stage
        "week",
        "week_pct",
        "contestants_pct",
        "num_contestants",
        # identity
        "pro_dancer",
        "season",
    ]
    if add_momentum:
        base_features += ["fan_vote_prev", "fan_vote_delta", "judge_z_prev", "judge_z_delta"]

    categorical = ["pro_dancer"]

    # Targets
    targets = [
        ("judge", "judge_score_z", list(base_features)),
        (
            "fan",
            "predicted_fan_vote",
            list(base_features + (["judge_score_z"] if include_judge_in_fan else [])),
        ),
        ("divergence", "fanFavor", list(base_features)),
    ]

    results = {}

    for tag, target, feats in targets:
        feats = [c for c in feats if c in df.columns]
        target_series = df[target].astype(float)
        model_df = df.loc[np.isfinite(target_series), :].copy()

        # Drop any remaining NaNs in features
        Xc = model_df[feats].copy()
        for c in feats:
            if c in categorical:
                Xc[c] = Xc[c].astype("category")
        model_df = model_df.loc[Xc.dropna().index].copy()

        try:
            tr = _train_holdout_by_season(
                model_df,
                feature_cols=feats,
                categorical_cols=categorical,
                target_col=target,
                train_max_season=int(train_max_season),
                seed=int(seed),
                n_estimators=int(n_estimators),
                learning_rate=float(learning_rate),
            )
            eval_mode = f"holdout_season>{train_max_season}"
        except ValueError:
            tr = _train_group_kfold(
                model_df,
                feature_cols=feats,
                categorical_cols=categorical,
                target_col=target,
                seed=int(seed),
                n_splits=int(kfold_splits),
                n_estimators=int(n_estimators),
                learning_rate=float(learning_rate),
            )
            eval_mode = f"group_kfold(season, k={kfold_splits})"

        results[tag] = {"target": target, "eval_mode": eval_mode, "metrics": tr.metrics}

        # Plots
        imp_png = out_path / f"lgbm_{tag}_feature_importance.png"
        _plot_feature_importance(
            tr.model,
            tr.feature_names,
            imp_png,
            title=f"LightGBM Feature Importance (gain) — {tag}",
        )

        # SHAP (optional)
        shap_png = out_path / f"lgbm_{tag}_shap_summary.png"
        X_plot = model_df[tr.feature_names].copy()
        for c in categorical:
            if c in X_plot.columns:
                X_plot[c] = X_plot[c].astype("category")
        _try_plot_shap_summary(
            tr.model,
            X_plot.sample(n=min(1500, len(X_plot)), random_state=seed),
            shap_png,
            title=f"SHAP Summary (bar) — {tag}",
        )

        # Pro-dancer uplift table (optional but useful for narrative)
        uplift = _pro_dancer_uplift(tr.model, X_plot, "pro_dancer")
        if len(uplift) > 0:
            uplift.to_csv(out_path / f"lgbm_{tag}_pro_dancer_uplift.csv", index=False)

    # Save summary
    with (out_path / "lgbm_impact_summary.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Also save a short human-readable txt
    lines = []
    lines.append("LightGBM Impact Models Summary")
    lines.append("=" * 40)
    lines.append(f"predictions_csv: {predictions_csv}")
    lines.append(f"add_momentum: {bool(add_momentum)}")
    lines.append(f"include_judge_in_fan: {bool(include_judge_in_fan)}")
    lines.append("")
    for tag, info in results.items():
        lines.append(f"[{tag}] target={info['target']} eval={info['eval_mode']}")
        lines.append(json.dumps(info["metrics"], ensure_ascii=False))
        lines.append("")
    lines.append(
        "Note: SHAP plots are only generated if the 'shap' package is installed."
    )
    with (out_path / "lgbm_impact_summary.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Train 3 LightGBM regressors to quantify impacts on judge scores, "
            "fan vote estimates, and judge–fan divergence (fanFavor)."
        )
    )
    ap.add_argument(
        "--predictions-csv",
        default="predictions_unified.csv",
        help="Input CSV with features + predicted_fan_vote (default: predictions_unified.csv)",
    )
    ap.add_argument(
        "--out-dir",
        default="reports",
        help="Output directory for plots/tables (default: reports)",
    )
    ap.add_argument(
        "--train-max-season",
        type=int,
        default=25,
        help="Time split: train seasons <= this; test > this. If not possible, use season GroupKFold.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--add-momentum",
        action="store_true",
        help="Add lag/delta features (previous week fan/judge) per contestant.",
    )
    ap.add_argument("--n-estimators", type=int, default=1200)
    ap.add_argument("--learning-rate", type=float, default=0.03)
    ap.add_argument("--kfold-splits", type=int, default=5)
    ap.add_argument(
        "--include-judge-in-fan",
        action="store_true",
        help="Include judge_score_z as a covariate in the fan-vote model (captures 'performance effect').",
    )
    args = ap.parse_args()

    run(
        predictions_csv=args.predictions_csv,
        out_dir=args.out_dir,
        train_max_season=args.train_max_season,
        seed=args.seed,
        add_momentum=args.add_momentum,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        kfold_splits=args.kfold_splits,
        include_judge_in_fan=args.include_judge_in_fan,
    )


if __name__ == "__main__":
    main()
