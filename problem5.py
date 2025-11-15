import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ===================== 1. 数据预处理模块 =====================
def preprocess_data(df):
    """数据预处理：清洗、编码和特征工程"""
    df = df[df['编号'].notna() & df['编号'].apply(lambda x: str(x).isdigit())].copy()
    df['编号'] = df['编号'].astype(int)

    initial_count = len(df)
    df = df[df['婴儿行为特征'].notna()].copy()
    deleted_count = initial_count - len(df)
    if deleted_count > 0:
        print(f"已删除婴儿行为特征缺失记录 {deleted_count} 条，剩余 {len(df)} 条")

    def time_to_hours(time_str):
        try:
            parts = str(time_str).split(":")
            hours = int(parts[0])
            minutes = int(parts[1]) if len(parts) > 1 else 0
            return round(hours + minutes / 60, 2)
        except:
            return np.nan

    df['睡眠时长_小时'] = df['整晚睡眠时间（时：分：秒）'].apply(time_to_hours)

    # 填充其他缺失值
    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                fill_val = df[col].median()
                df[col].fillna(fill_val, inplace=True)
            else:
                fill_val = df[col].mode()[0]
                df[col].fillna(fill_val, inplace=True)
    # 分类特征编码
    code_mappings = {
        '婚姻状况': {1: 1, 2: 2},
        '教育程度': {1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
        '分娩方式': {1: 1, 2: 2},
        '婴儿性别': {1: 1, 2: 2},
        '入睡方式': {1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
        '婴儿行为特征': {'安静型': 0, '中等型': 1, '矛盾型': 2}
    }
    for col, mapping in code_mappings.items():
        if col in df.columns:
            df[f'{col}_编码'] = df[col].map(mapping)

    return df
# ===================== 2. 问题三模型（婴儿行为特征预测） =====================
def train_behavior_model(df):
    """训练婴儿行为特征分类模型"""
    # 特征与标签
    feature_cols = [
        '母亲年龄', '婚姻状况_编码', '教育程度_编码', '妊娠时间（周数）',
        '分娩方式_编码', 'CBTS', 'EPDS', 'HADS', '婴儿性别_编码',
        '婴儿年龄（月）', '睡醒次数', '入睡方式_编码'
    ]
    X = df[feature_cols]
    y = df['婴儿行为特征_编码']  # 0=安静型, 1=中等型, 2=矛盾型

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # 训练随机森林分类器
    model = RandomForestClassifier(
        n_estimators=100, max_depth=8, random_state=42
    )
    model.fit(X_train, y_train)
    # 评估模型
    y_pred = model.predict(X_test)
    print(f"行为特征模型准确率: {accuracy_score(y_test, y_pred):.4f}")

    return model, feature_cols


def get_problem3_solution(df_238, behavior_model, feature_cols):
    """获取问题三的最优方案（使行为特征达标的最小心理指标下降）"""
    initial = {
        'CBTS': df_238['CBTS'],
        'EPDS': df_238['EPDS'],
        'HADS': df_238['HADS']
    }

    # 固定非心理特征
    df_features = df_238[feature_cols].to_frame().T  # 转换为DataFrame
    cols_to_drop = [col for col in ['CBTS', 'EPDS', 'HADS'] if col in feature_cols]
    fixed_features = df_features.drop(cols_to_drop, axis=1).iloc[0].to_dict()

    # 搜索最小下降方案（优先不下降）
    best_reduction = {k: float('inf') for k in initial.keys()}
    best_scores = None
    # 先检查不下降的情况
    test_scores = {'CBTS': initial['CBTS'], 'EPDS': initial['EPDS'], 'HADS': initial['HADS']}
    features = fixed_features.copy()
    features.update(test_scores)
    pred = behavior_model.predict(pd.DataFrame([features])[feature_cols])[0]
    if pred in [0, 1]:  # 安静型或中等型
        return test_scores  # 不下降即可满足，直接返回

    # 若不下降不满足，再搜索需要下降的方案
    for h in range(initial['HADS'], -1, -1):
        for e in range(initial['EPDS'], -1, -1):
            for c in range(initial['CBTS'], -1, -1):
                test_scores = {'CBTS': c, 'EPDS': e, 'HADS': h}
                # 跳过已经检查过的初始状态
                if c == initial['CBTS'] and e == initial['EPDS'] and h == initial['HADS']:
                    continue

                features = fixed_features.copy()
                features.update(test_scores)

                # 预测行为特征
                pred = behavior_model.predict(pd.DataFrame([features])[feature_cols])[0]
                if pred in [0, 1]:  # 安静型或中等型
                    reduction = {
                        'CBTS': initial['CBTS'] - c,
                        'EPDS': initial['EPDS'] - e,
                        'HADS': initial['HADS'] - h
                    }
                    total_red = sum(reduction.values())
                    if total_red < sum(best_reduction.values()):
                        best_reduction = reduction
                        best_scores = test_scores
                        return best_scores  # 找到第一个可行解即为最优（从高到低搜索）

    return best_scores  # 问题三的最优心理指标


# ===================== 3. 问题四模型（睡眠质量评价） =====================
def calculate_sleep_weights(df):
    """用熵权法计算睡眠指标权重"""
    # 提取睡眠指标并正向化
    sleep_df = df[['睡眠时长_小时', '睡醒次数', '入睡方式']].copy()
    # 正向化（时长越大越好，次数和方式越小越好）
    def normalize_max(col):
        return (col - col.min()) / (col.max() - col.min() + 1e-10)
    def normalize_min(col):
        return (col.max() - col) / (col.max() - col.min() + 1e-10)

    sleep_df['时长_正向'] = normalize_max(sleep_df['睡眠时长_小时'])
    sleep_df['次数_正向'] = normalize_min(sleep_df['睡醒次数'])
    sleep_df['方式_正向'] = normalize_min(sleep_df['入睡方式'])

    # 计算熵权
    p = sleep_df[['时长_正向', '次数_正向', '方式_正向']].values
    p = p / p.sum(axis=0)
    n = len(sleep_df)
    entropy = -np.sum(p * np.log(p + 1e-10), axis=0) / np.log(n)
    weights = (1 - entropy) / np.sum(1 - entropy)

    # 计算优级阈值（Q3）
    sleep_df['综合得分'] = np.dot(p, weights)
    q3 = np.percentile(sleep_df['综合得分'], 75)

    print(f"睡眠指标权重: {weights.round(4)}, 优级阈值(Q3): {q3:.4f}")
    return weights, q3

def train_sleep_models(df):
    """训练睡眠指标预测模型"""
    # 明确包含心理指标
    feature_cols = [
        '母亲年龄', '婚姻状况_编码', '教育程度_编码', '妊娠时间（周数）',
        '分娩方式_编码', 'CBTS', 'EPDS', 'HADS'
    ]
    target_cols = ['睡眠时长_小时', '睡醒次数', '入睡方式']
    models = {}
    for target in target_cols:
        X = df[feature_cols]
        y = df[target]
        model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
        model.fit(X, y)
        models[target] = model

    return models, feature_cols


def evaluate_sleep_quality(scores, df, sleep_models, sleep_feats, weights, q3):
    """评估给定心理指标下的睡眠质量是否达标"""
    # 将Series转换为DataFrame再操作
    df_features = df[sleep_feats].iloc[0].to_frame().T  # 转换为DataFrame
    cols_to_drop = [col for col in ['CBTS', 'EPDS', 'HADS'] if col in sleep_feats]
    fixed_feats = df_features.drop(cols_to_drop, axis=1).iloc[0].to_dict()
    # 合并固定特征和当前心理指标得分
    features = fixed_feats.copy()
    features.update(scores)

    # 预测睡眠指标
    input_df = pd.DataFrame([features])[sleep_feats]
    pred_duration = sleep_models['睡眠时长_小时'].predict(input_df)[0]
    pred_times = sleep_models['睡醒次数'].predict(input_df)[0]
    pred_method = sleep_models['入睡方式'].predict(input_df)[0]

    # 计算综合得分
    max_vals = df[['睡眠时长_小时', '睡醒次数', '入睡方式']].max()
    min_vals = df[['睡眠时长_小时', '睡醒次数', '入睡方式']].min()
    dur_norm = (pred_duration - min_vals['睡眠时长_小时']) / (
                max_vals['睡眠时长_小时'] - min_vals['睡眠时长_小时'] + 1e-10)
    times_norm = (max_vals['睡醒次数'] - pred_times) / (max_vals['睡醒次数'] - min_vals['睡醒次数'] + 1e-10)
    method_norm = (max_vals['入睡方式'] - pred_method) / (max_vals['入睡方式'] - min_vals['入睡方式'] + 1e-10)

    score = dur_norm * weights[0] + times_norm * weights[1] + method_norm * weights[2]
    return score > q3  # 是否达优


# ===================== 4. 问题五核心逻辑 =====================
def calculate_cost(initial, target):
    """计算治疗成本（指标不下降则不花钱）"""
    # 成本参数：只对下降的部分收费
    params = {
        'CBTS': 2612/3,  # 每下降1分的费用
        'EPDS': 695,  # 每下降1分的费用
        'HADS': 2440  # 每下降1分的费用
    }

    cost = {}
    total = 0
    for metric in ['CBTS', 'EPDS', 'HADS']:
        reduction = max(0, initial[metric] - target[metric])
        cost[metric] = round(reduction * params[metric], 2)
        total += cost[metric]

    return {'总费用': round(total, 2), '分项费用': cost}


def optimize_problem5(df, df_238, behavior_model, behavior_feats,
                      sleep_models, sleep_feats, sleep_weights, sleep_q3):
    """问题五优化：先检验问题三方案，再进行启发式搜索"""
    initial_scores = {
        'CBTS': df_238['CBTS'],
        'EPDS': df_238['EPDS'],
        'HADS': df_238['HADS']
    }
    print(
        f"\n初始心理指标: CBTS={initial_scores['CBTS']}, EPDS={initial_scores['EPDS']}, HADS={initial_scores['HADS']}")

    # 步骤1: 先检查原始指标是否已满足两个约束
    print("检查初始指标是否满足所有约束...")
    # 检查行为特征约束 - 修复axis问题
    df_features = df_238[behavior_feats].to_frame().T  # 转换为DataFrame
    cols_to_drop = [col for col in ['CBTS', 'EPDS', 'HADS'] if col in behavior_feats]
    features_df = df_features.drop(cols_to_drop, axis=1)
    features = features_df.iloc[0].to_dict()
    features.update(initial_scores)
    behavior_pred = behavior_model.predict(pd.DataFrame([features])[behavior_feats])[0]
    behavior_ok = behavior_pred in [0, 1]

    # 检查睡眠约束
    sleep_ok = evaluate_sleep_quality(
        initial_scores, df, sleep_models, sleep_feats, sleep_weights, sleep_q3
    )
    if behavior_ok and sleep_ok:
        print("初始指标已满足所有约束，无需任何治疗，费用为0")
        return {
            '总费用': 0,
            '目标得分': initial_scores,
            '分项费用': {'CBTS': 0, 'EPDS': 0, 'HADS': 0},
            '指标下降': {'CBTS': 0, 'EPDS': 0, 'HADS': 0}
        }, initial_scores

    # 步骤2: 获取问题三最优方案并检验是否满足睡眠约束
    problem3_scores = get_problem3_solution(df_238, behavior_model, behavior_feats)
    if problem3_scores is None:
        print("未找到满足行为特征约束的方案")
        return None, initial_scores

    print(f"问题三最优方案: {problem3_scores}")
    sleep_ok = evaluate_sleep_quality(
        problem3_scores, df, sleep_models, sleep_feats, sleep_weights, sleep_q3
    )

    if sleep_ok:
        print("问题三方案同时满足睡眠质量约束，作为最优解")
        cost = calculate_cost(initial_scores, problem3_scores)
        return {
            '总费用': cost['总费用'],
            '目标得分': problem3_scores,
            '分项费用': cost['分项费用'],
            '指标下降': {k: initial_scores[k] - problem3_scores[k] for k in initial_scores}
        }, initial_scores

    # 步骤3: 问题三方案不满足睡眠约束，进行启发式搜索
    print("问题三方案不满足睡眠约束，开始优化搜索...")
    best_cost = float('inf')
    best_solution = None
    start_h = problem3_scores['HADS']
    start_e = problem3_scores['EPDS']
    start_c = problem3_scores['CBTS']

    # 搜索范围：在问题三基础上继续降低指标
    for h in range(start_h, max(-1, start_h - 4), -1):
        for e in range(start_e, max(-1, start_e - 4), -1):
            for c in range(start_c, max(-1, start_c - 4), -1):
                current_scores = {'CBTS': c, 'EPDS': e, 'HADS': h}

                # 检查行为特征约束
                df_features = df_238[behavior_feats].to_frame().T  # 转换为DataFrame
                cols_to_drop = [col for col in ['CBTS', 'EPDS', 'HADS'] if col in behavior_feats]
                features_df = df_features.drop(cols_to_drop, axis=1)
                features = features_df.iloc[0].to_dict()
                features.update(current_scores)

                behavior_pred = behavior_model.predict(pd.DataFrame([features])[behavior_feats])[0]
                if behavior_pred not in [0, 1]:
                    continue

                # 检查睡眠约束
                if not evaluate_sleep_quality(
                        current_scores, df, sleep_models, sleep_feats, sleep_weights, sleep_q3
                ):
                    continue

                # 计算成本
                cost = calculate_cost(initial_scores, current_scores)
                if cost['总费用'] < best_cost:
                    best_cost = cost['总费用']
                    best_solution = {
                        '总费用': cost['总费用'],
                        '目标得分': current_scores,
                        '分项费用': cost['分项费用'],
                        '指标下降': {k: initial_scores[k] - current_scores[k] for k in initial_scores}
                    }

    return best_solution, initial_scores


# ===================== 5. 结果输出 =====================
def print_result(solution, initial):
    """输出最终结果"""
    if not solution:
        print("\n未找到满足所有约束的最优方案")
        return

    print("\n" + "=" * 60)
    print("问题五：238号婴儿最优治疗策略结果")
    print("=" * 60)
    print(f"1. 最小总治疗费用: {solution['总费用']} 元")

    print("\n2. 各心理指标下降情况:")
    for metric in ['CBTS', 'EPDS', 'HADS']:
        if solution['指标下降'][metric] == 0:
            print(f"   - {metric}: 无需下降 (保持初始值: {initial[metric]})")
        else:
            print(f"   - {metric}: 下降 {solution['指标下降'][metric]} 分 "
                  f"(初始: {initial[metric]} → 目标: {solution['目标得分'][metric]})")

    print("\n3. 各指标治疗费用:")
    for metric in ['CBTS', 'EPDS', 'HADS']:
        print(f"   - {metric}: {solution['分项费用'][metric]} 元")
    print("=" * 60)


# ===================== 主函数 =====================
if __name__ == "__main__":
    EXCEL_PATH = "训练题4健康问题附件.xlsx"
    # 1. 数据加载与预处理
    print("1. 加载并预处理数据...")
    raw_df = pd.read_excel(EXCEL_PATH, sheet_name='Sheet1')
    df = preprocess_data(raw_df)

    # 2. 获取238号婴儿数据
    if 238 not in df['编号'].values:
        raise ValueError("数据中未找到238号婴儿记录")
    df_238 = df[df['编号'] == 238].iloc[0]
    # 3. 训练问题三模型
    print("\n2. 训练婴儿行为特征模型...")
    behavior_model, behavior_feats = train_behavior_model(df)
    # 4. 训练问题四模型
    print("\n3. 计算睡眠质量评价指标...")
    sleep_weights, sleep_q3 = calculate_sleep_weights(df)
    print("4. 训练睡眠指标预测模型...")
    sleep_models, sleep_feats = train_sleep_models(df)
    # 5. 执行问题五优化
    print("\n5. 执行问题五优化求解...")
    solution, initial_scores = optimize_problem5(
        df, df_238, behavior_model, behavior_feats,
        sleep_models, sleep_feats, sleep_weights, sleep_q3
    )

    # 6. 输出结果
    print_result(solution, initial_scores)
