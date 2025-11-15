import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 1. 数据加载与增强预处理 =====================
def load_and_preprocess_data(file_path):
    """加载数据并进行增强预处理"""
    # 读取数据
    df = pd.read_excel(file_path, sheet_name="Sheet1")

    core_cols = [
        "编号", "母亲年龄", "婚姻状况", "教育程度", "妊娠时间（周数）", "分娩方式",
        "CBTS", "EPDS", "HADS", "整晚睡眠时间（时：分：秒）", "睡醒次数", "入睡方式"
    ]

    df_clean = df[df["编号"].apply(lambda x: str(x).isdigit())][core_cols].copy()
    df_clean["编号"] = df_clean["编号"].astype(int)

    train_df = df_clean[df_clean["编号"] <= 390].reset_index(drop=True)
    predict_df = df_clean[df_clean["编号"] > 390].reset_index(drop=True)

    def time_to_hours(time_str):
        try:
            parts = str(time_str).split(":")
            hours = int(parts[0])
            minutes = int(parts[1]) if len(parts) > 1 else 0
            return round(hours + minutes / 60, 2)
        except:
            return np.nan

    train_df["整晚睡眠时间_小时"] = train_df["整晚睡眠时间（时：分：秒）"].apply(time_to_hours)
    predict_df["整晚睡眠时间_小时"] = predict_df["整晚睡眠时间（时：分：秒）"].apply(time_to_hours)

    # 缺失值处理：分类变量用众数，连续变量用中位数
    sleep_cols = ["整晚睡眠时间_小时", "睡醒次数", "入睡方式"]
    for col in sleep_cols:
        if train_df[col].dtype in [np.int64, np.float64]:
            # 连续变量用中位数填充（更抗异常值）
            train_median = train_df[col].median()
            train_df[col] = train_df[col].fillna(train_median)
            predict_df[col] = predict_df[col].fillna(train_median)
        else:
            # 分类变量用众数填充
            train_mode = train_df[col].mode()[0]
            train_df[col] = train_df[col].fillna(train_mode)
            predict_df[col] = predict_df[col].fillna(train_mode)

    # 输出数据基本信息
    print("=== 数据基本信息 ===")
    print(f"训练集样本数: {len(train_df)}, 预测集样本数: {len(predict_df)}")
    print(f"训练集中缺失值已处理，关键指标无缺失")

    return train_df, predict_df


# ===================== 2. 综合睡眠质量评价（熵权法） =====================
def evaluate_sleep_quality(train_df):
    """基于熵权法计算综合睡眠质量并分级"""
    # 提取睡眠指标
    x1 = train_df["整晚睡眠时间_小时"].values.reshape(-1, 1)
    x2 = train_df["睡醒次数"].values.reshape(-1, 1)
    x3 = train_df["入睡方式"].values.reshape(-1, 1)
    n = len(train_df)

    # 指标正向化函数
    def normalize_minmax(x, is_min_type=True):
        x_max = np.max(x)
        x_min = np.min(x)
        if x_max == x_min:
            return np.ones_like(x)
        if is_min_type:
            return (x_max - x) / (x_max - x_min)  # 极小型转极大型
        else:
            return (x - x_min) / (x_max - x_min)  # 极大型标准化

    # 正向化处理
    x1_pos = normalize_minmax(x1, is_min_type=False)  # 睡眠时间越长越好
    x2_pos = normalize_minmax(x2, is_min_type=True)  # 睡醒次数越少越好
    x3_pos = x3 / 5  # 入睡方式1-5级归一化

    # 合并指标矩阵
    x_pos = np.hstack([x1_pos, x2_pos, x3_pos])

    # 熵权法计算权重
    X = np.zeros_like(x_pos)
    for j in range(3):
        col_sum = np.sum(x_pos[:, j])
        X[:, j] = x_pos[:, j] / col_sum if col_sum != 0 else 0

    k = 1 / np.log(n) if n > 1 else 0
    E = np.zeros(3)
    for j in range(3):
        X_j = X[:, j] + 1e-10  # 避免log(0)
        E[j] = -k * np.sum(X_j * np.log(X_j))
    w = (1 - E) / np.sum(1 - E) if np.sum(1 - E) != 0 else np.ones(3) / 3
    # 计算综合得分
    S = np.dot(X, w)
    # 四分位数分级
    Q1, Q2, Q3 = np.percentile(S, [25, 50, 75])

    def score_to_grade(s):
        if s <= Q1:
            return 0  # 差
        elif s <= Q2:
            return 1  # 中
        elif s <= Q3:
            return 2  # 良
        else:
            return 3  # 优

    # 添加等级标签 - 修复错误处
    train_df["综合睡眠得分"] = S
    train_df["综合睡眠等级（编码）"] = [score_to_grade(s) for s in S]
    grade_map = {0: "差", 1: "中", 2: "良", 3: "优"}
    # 直接使用编码列映射生成名称列，而不是尝试映射一个不存在的列
    train_df["综合睡眠等级（名称）"] = train_df["综合睡眠等级（编码）"].map(grade_map)

    # 输出等级分布
    print("\n=== 综合睡眠质量等级分布 ===")
    grade_dist = train_df["综合睡眠等级（名称）"].value_counts(normalize=True).sort_index()
    for grade, ratio in grade_dist.items():
        print(f"{grade}: {ratio:.2%}")

    return train_df, grade_map, (Q1, Q2, Q3)


# ===================== 3. 网格搜索优化模型 =====================
def optimize_model(train_df):
    """使用网格搜索优化随机森林模型"""
    # 准备特征和标签
    X_cols = ["母亲年龄", "婚姻状况", "教育程度", "妊娠时间（周数）", "分娩方式", "CBTS", "EPDS", "HADS"]
    X_train = train_df[X_cols].values
    y_train = train_df["综合睡眠等级（编码）"].values

    # 处理特征缺失值（用中位数填充，更稳健）
    for j in range(X_train.shape[1]):
        col_median = np.nanmedian(X_train[:, j])
        X_train[np.isnan(X_train[:, j]), j] = col_median

    # 优化的参数网格（更聚焦合理范围）
    param_grid = {
        'n_estimators': [100, 150, 200],
        'max_depth': [8, 10, 12],
        'min_samples_split': [4, 6],
        'min_samples_leaf': [2, 3]
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 网格搜索（使用加权F1适应可能的类别不平衡）
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(
            random_state=42,
            class_weight="balanced",  # 自动平衡类别权重
            n_jobs=-1
        ),
        param_grid=param_grid,
        cv=cv,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=0
    )

    print("\n=== 开始模型参数优化 ===")
    grid_search.fit(X_train, y_train)

    print(f"最优参数组合: {grid_search.best_params_}")
    print(f"最优参数下的交叉验证F1分数: {grid_search.best_score_:.4f}")
    y_pred_train = grid_search.predict(X_train)
    print("\n=== 训练集上的模型表现 ===")
    print(classification_report(y_train, y_pred_train, target_names=["差", "中", "良", "优"]))

    return grid_search.best_estimator_, X_cols


# ===================== 4. 预测与结果可视化 =====================
def predict_and_visualize(best_model, X_cols, train_df, predict_df, grade_map):
    """预测并可视化结果"""
    # 处理预测集特征
    X_predict = predict_df[X_cols].values
    for j in range(X_predict.shape[1]):
        col_median = np.nanmedian(train_df[X_cols[j]].values)
        X_predict[np.isnan(X_predict[:, j]), j] = col_median

    # 预测
    y_predict = best_model.predict(X_predict)
    predict_df["综合睡眠等级（编码）"] = y_predict
    predict_df["综合睡眠等级（名称）"] = predict_df["综合睡眠等级（编码）"].map(grade_map)

    # 输出预测结果
    print("\n=== 391-410号样本睡眠质量预测结果 ===")
    result_cols = ["编号", "母亲年龄", "CBTS", "EPDS", "HADS", "综合睡眠等级（名称）"]
    print(predict_df[result_cols].to_string(index=False))

    # 可视化预测结果分布
    plt.figure(figsize=(10, 6))
    predict_dist = predict_df["综合睡眠等级（名称）"].value_counts().reindex(["优", "良", "中", "差"])
    predict_dist.plot(kind="bar", color=["#4CAF50", "#2196F3", "#FFC107", "#F44336"])
    plt.title("预测样本睡眠质量等级分布")
    plt.xlabel("睡眠质量等级")
    plt.ylabel("样本数量")
    plt.xticks(rotation=0)
    for i, v in enumerate(predict_dist):
        plt.text(i, v + 0.1, str(v), ha='center')
    plt.tight_layout()
    plt.savefig("预测结果分布.png")

    # 保存结果到Excel
    with pd.ExcelWriter("婴儿睡眠质量预测结果.xlsx", engine="openpyxl") as writer:
        train_result = train_df[["编号", "母亲年龄", "CBTS", "EPDS", "HADS",
                                 "综合睡眠得分", "综合睡眠等级（名称）"]]
        train_result.to_excel(writer, sheet_name="训练集结果", index=False)

        predict_result = predict_df[["编号", "母亲年龄", "CBTS", "EPDS", "HADS",
                                     "综合睡眠等级（名称）"]]
        predict_result.to_excel(writer, sheet_name="预测结果", index=False)
    print("完整结果已保存到'婴儿睡眠质量预测结果.xlsx'")

# ===================== 主函数 =====================
def main():
    # 配置文件路径
    file_path = "训练题4健康问题附件.xlsx"
    train_df, predict_df = load_and_preprocess_data(file_path)
    train_df, grade_map, _ = evaluate_sleep_quality(train_df)
    best_model, X_cols = optimize_model(train_df)
    predict_and_visualize(best_model, X_cols, train_df, predict_df, grade_map)


if __name__ == "__main__":
    main()
