import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from itertools import product
from tqdm import tqdm


# --------------------------
# 1. 数据加载与预处理（修复缺失值问题）
# --------------------------
def load_attachment_data(excel_path):
    """加载附件数据，处理分类特征与缺失值"""
    # 读取Excel数据
    df = pd.read_excel(excel_path, sheet_name='Sheet1')

    # 数据清洗：移除空行与无效数据
    df = df[df['编号'].notna()].copy()
    df['编号'] = df['编号'].astype(int)

    # 关键修复：处理缺失值
    print(f"原始数据形状: {df.shape}")

    # 检查并处理目标变量缺失值（婴儿行为特征）
    if df['婴儿行为特征'].isna().any():
        missing_count = df['婴儿行为特征'].isna().sum()
        print(f"发现{missing_count}条婴儿行为特征缺失数据，已移除")
        df = df[df['婴儿行为特征'].notna()].copy()

    # 检查并处理特征列缺失值
    feature_cols_initial = [
        '母亲年龄', '婚姻状况', '教育程度', '妊娠时间（周数）',
        '分娩方式', 'CBTS', 'EPDS', 'HADS', '婴儿性别',
        '婴儿年龄（月）', '睡醒次数', '入睡方式'
    ]

    # 对数值型特征填充中位数，对分类特征填充众数
    for col in feature_cols_initial:
        if df[col].isna().any():
            missing_count = df[col].isna().sum()
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].median(), inplace=True)
                print(f"特征'{col}'有{missing_count}个缺失值，已用中位数填充")
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
                print(f"特征'{col}'有{missing_count}个缺失值，已用众数填充")

    print(f"清洗后数据形状: {df.shape}")

    # 特征映射
    mappings = {
        '婚姻状况': {1: '未婚', 2: '已婚'},
        '教育程度': {1: '小学', 2: '初中', 3: '高中', 4: '大学', 5: '研究生'},
        '分娩方式': {1: '自然分娩', 2: '剖宫产'},
        '婴儿性别': {1: '男性', 2: '女性'},
        '入睡方式': {
            1: '哄睡法', 2: '抚触法', 3: '安抚奶嘴法',
            4: '环境营造法', 5: '定时法'
        },
        '婴儿行为特征': {'安静型': 0, '中等型': 1, '矛盾型': 2}
    }

    # 应用映射
    df['婚姻状况_编码'] = df['婚姻状况'].map({1: 1, 2: 2})
    df['教育程度_编码'] = df['教育程度'].map({1: 1, 2: 2, 3: 3, 4: 4, 5: 5})
    df['分娩方式_编码'] = df['分娩方式'].map({1: 1, 2: 2})
    df['婴儿性别_编码'] = df['婴儿性别'].map({1: 1, 2: 2})
    df['入睡方式_编码'] = df['入睡方式'].map({1: 1, 2: 2, 3: 3, 4: 4, 5: 5})
    df['婴儿行为特征_编码'] = df['婴儿行为特征'].map(mappings['婴儿行为特征'])

    # 提取编号238的完整数据
    if 238 not in df['编号'].values:
        raise ValueError("数据中未找到编号238的记录，请检查数据文件")

    df_238 = df[df['编号'] == 238].iloc[0]
    initial_scores = {
        'CBTS': df_238['CBTS'],
        'EPDS': df_238['EPDS'],
        'HADS': df_238['HADS']
    }

    # 定义模型输入特征
    feature_cols = [
        '母亲年龄', '婚姻状况_编码', '教育程度_编码', '妊娠时间（周数）',
        '分娩方式_编码', 'CBTS', 'EPDS', 'HADS', '婴儿性别_编码',
        '婴儿年龄（月）', '睡醒次数', '入睡方式_编码'
    ]

    return df, df_238, initial_scores, feature_cols, mappings


# --------------------------
# 2. 成本函数构建
# --------------------------
def build_cost_functions():
    """构建CBTS/EPDS/HADS的线性成本函数"""
    cost_params = {
        'CBTS': {'a': 2612 / 3, 'b': 200},
        'EPDS': {'a': 695, 'b': 500},
        'HADS': {'a': 2440, 'b': 300}
    }
    return cost_params


def calculate_total_cost(initial_S, target_S, cost_params):
    """计算总治疗费用"""
    cbts_cost = (cost_params['CBTS']['a'] * initial_S['CBTS'] + cost_params['CBTS']['b']) - \
                (cost_params['CBTS']['a'] * target_S['CBTS'] + cost_params['CBTS']['b'])
    epds_cost = (cost_params['EPDS']['a'] * initial_S['EPDS'] + cost_params['EPDS']['b']) - \
                (cost_params['EPDS']['a'] * target_S['EPDS'] + cost_params['EPDS']['b'])
    hads_cost = (cost_params['HADS']['a'] * initial_S['HADS'] + cost_params['HADS']['b']) - \
                (cost_params['HADS']['a'] * target_S['HADS'] + cost_params['HADS']['b'])

    total_cost = max(0, cbts_cost + epds_cost + hads_cost)
    return total_cost, {'CBTS': cbts_cost, 'EPDS': epds_cost, 'HADS': hads_cost}


# --------------------------
# 3. 行为特征预测模型
# --------------------------
def train_behavior_model(df, feature_cols):
    """训练婴儿行为特征分类模型"""
    # 模型输入输出
    X = df[feature_cols].copy()
    y = df['婴儿行为特征_编码'].copy()

    # 检查目标变量是否还有缺失值
    if y.isna().any():
        raise ValueError("目标变量仍存在缺失值")

    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) > 1 else None
    )
    # 训练随机森林模型
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42
    )
    model.fit(X_train, y_train)
    # 模型评估
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"模型训练完成 | 测试集准确率：{accuracy:.3f}")
    # 反向映射
    label_map = {0: '安静型', 1: '中等型', 2: '矛盾型'}
    return model, label_map


def find_min_cost(initial_scores, df_238, model, label_map, cost_params, feature_cols, target_behavior):
    """网格搜索最小治疗成本"""
    # 提取固定特征
    fixed_features = df_238[feature_cols].copy()
    fixed_features = fixed_features.drop(['CBTS', 'EPDS', 'HADS']).to_dict()

    # 生成所有可能的干预后得分组合
    cbts_range = range(0, int(initial_scores['CBTS']) + 1)
    epds_range = range(0, int(initial_scores['EPDS']) + 1)
    hads_range = range(0, int(initial_scores['HADS']) + 1)

    # 初始化最小成本与最优得分
    min_total_cost = float('inf')
    best_target_scores = None
    best_individual_costs = None

    # 遍历所有得分组合
    total_comb = len(cbts_range) * len(epds_range) * len(hads_range)
    print(f"\n=== 搜索目标：矛盾型→{target_behavior} | 总组合数：{total_comb} ===")

    for cbts, epds, hads in tqdm(product(cbts_range, epds_range, hads_range), total=total_comb):
        # 构建当前干预后的特征向量
        current_features = fixed_features.copy()
        current_features.update({'CBTS': cbts, 'EPDS': epds, 'HADS': hads})
        current_features_df = pd.DataFrame([current_features])[feature_cols]

        # 模型预测当前行为特征
        pred_label = model.predict(current_features_df)[0]
        pred_behavior = label_map[pred_label]

        # 若预测结果符合目标，计算成本
        if pred_behavior == target_behavior:
            target_scores = {'CBTS': cbts, 'EPDS': epds, 'HADS': hads}
            total_cost, individual_costs = calculate_total_cost(initial_scores, target_scores, cost_params)

            # 更新最小成本
            if total_cost < min_total_cost:
                min_total_cost = total_cost
                best_target_scores = target_scores
                best_individual_costs = individual_costs

    return min_total_cost, best_target_scores, best_individual_costs

def print_result(initial_scores, min_cost, best_scores, individual_costs, target_behavior):
    """输出单个目标的结果"""
    if best_scores is None:
        print(f"\n未找到将婴儿行为特征从矛盾型转为{target_behavior}的可行方案")
        return

    # 计算各指标下降分数
    reductions = {
        'CBTS': initial_scores['CBTS'] - best_scores['CBTS'],
        'EPDS': initial_scores['EPDS'] - best_scores['EPDS'],
        'HADS': initial_scores['HADS'] - best_scores['HADS']
    }

    # 输出
    print(f"\n================ 矛盾型→{target_behavior} 结果 ================")
    print(f"1. 最小总治疗费用：{min_cost:.2f} 元")
    print("\n2. 各心理指标下降分数：")
    for metric, reduce_val in reductions.items():
        print(f"   - {metric}：{reduce_val:.0f} 分（初始{initial_scores[metric]:.0f}→目标{best_scores[metric]:.0f}）")
    print("\n3. 各指标单独治疗费用：")
    for metric, cost_val in individual_costs.items():
        print(f"   - {metric}：{cost_val:.2f} 元")
    print("=======================================================")


def main(excel_path):
    try:
        # 加载数据
        df, df_238, initial_scores, feature_cols, _ = load_attachment_data(excel_path)
        print(
            f"编号238初始心理得分：CBTS={initial_scores['CBTS']:.0f}, EPDS={initial_scores['EPDS']:.0f}, HADS={initial_scores['HADS']:.0f}")
        # 构建成本函数
        cost_params = build_cost_functions()
        # 训练模型
        model, label_map = train_behavior_model(df, feature_cols)
        # 求解"矛盾型→中等型"的最小成本
        target1 = '中等型'
        cost1, scores1, indiv_cost1 = find_min_cost(
            initial_scores, df_238, model, label_map, cost_params, feature_cols, target1
        )
        print_result(initial_scores, cost1, scores1, indiv_cost1, target1)

        # 求解"矛盾型→安静型"的最小成本
        target2 = '安静型'
        cost2, scores2, indiv_cost2 = find_min_cost(
            initial_scores, df_238, model, label_map, cost_params, feature_cols, target2
        )
        print_result(initial_scores, cost2, scores2, indiv_cost2, target2)

    except Exception as e:
        print(f"运行出错: {str(e)}")

if __name__ == "__main__":
    EXCEL_PATH = "训练题4健康问题附件.xlsx"
    main(EXCEL_PATH)