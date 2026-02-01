# main.py

import pandas as pd
from src.data_preprocessing import DWTSDataLoader
from src.feature_engineering import FeatureEngineer
from src.model_training import DWTSTrainer
from src.uncertainty_estimation import BootstrapUncertainty
from src.evaluation import ModelEvaluator


def main():
    print("=" * 50)
    print("DWTS Fan Vote Estimation Pipeline")
    print("=" * 50)

    # Step 1: 数据加载
    print("\n[1/5] 加载数据...")
    loader = DWTSDataLoader(data_dir="./data")
    df_long = loader.merge_all_data()
    df_long.to_csv("./data/processed_data.csv", index=False)
    print(f"   完成！共{len(df_long)}条记录")

    # Step 2: 特征工程
    print("\n[2/5] 特征工程...")
    engineer = FeatureEngineer(df_long)
    data = engineer.prepare_for_modeling()
    print(f"   完成！特征矩阵形状: {data['X'].shape}")

    # Step 3: 模型训练
    print("\n[3/5] 训练模型...")
    trainer = DWTSTrainer(data, method="percentage", device="cpu")
    losses = trainer.train(epochs=1000, lr=0.01)
    predictions = trainer.predict_fan_votes()
    predictions.to_csv("./results/fan_vote_estimates.csv", index=False)
    print("   完成！")

    # Step 4: 不确定性估计（可选，耗时）
    print("\n[4/5] Bootstrap不确定性估计...")
    boot = BootstrapUncertainty(data, DWTSTrainer, n_bootstrap=100)
    boot_results = boot.run_bootstrap(method="percentage")
    ci_df, _, _ = boot.compute_confidence_intervals(boot_results)
    ci_df.to_csv("./results/uncertainty_estimates.csv", index=False)
    print("   完成！")

    # Step 5: 评估
    print("\n[5/5] 模型评估...")
    evaluator = ModelEvaluator(data, predictions)
    metrics = evaluator.evaluate_elimination_accuracy()
    tau = evaluator.compute_kendall_tau()

    print(f"\n   平均F1 Score: {metrics['f1'].mean():.3f}")
    print(f"   平均Kendall's τ: {tau['kendall_tau'].mean():.3f}")

    print("\n" + "=" * 50)
    print("Pipeline完成！结果已保存到 ./results/")
    print("=" * 50)


if __name__ == "__main__":
    main()
