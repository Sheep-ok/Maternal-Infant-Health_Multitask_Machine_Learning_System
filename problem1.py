import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ---------------------- 数据加载与预处理 ----------------------
df = pd.read_excel('训练题4健康问题附件.xlsx')

# 1. 母亲核心指标：年龄、婚姻状况、妊娠时间、分娩方式
target_cols = {
    # 新增：母亲人口学/生理指标
    '母亲年龄': 'Mom_Age',              # 母亲年龄（岁）
    '婚姻状况': 'Mom_Marriage',        # 婚姻状况（1=未婚，2=已婚，3=其他）
    '妊娠时间（周数）': 'Pregnancy_Weeks',# 妊娠时间（周）
    '分娩方式': 'Delivery_Mode',       # 分娩方式（1=自然分娩，2=剖宫产）
    # 原有：母亲心理指标
    'CBTS': 'CBTS',                    # 母亲创伤后应激得分
    'EPDS': 'EPDS',                    # 母亲产后抑郁得分
    'HADS': 'HADS',                    # 母亲焦虑抑郁得分
    # 原有：婴儿指标
    '婴儿行为特征': 'Baby_Behavior',    # 婴儿行为特征
    '婴儿性别': 'Baby_Gender',         # 婴儿性别（1=男，2=女）
    '婴儿年龄（月）': 'Baby_Age',       # 婴儿年龄（月）
    '整晚睡眠时间（时：分：秒）': 'Sleep_Hours',# 整晚睡眠时间（小时）
    '睡醒次数': 'Wake_Count'           # 睡醒次数
}

# 筛选并rename列，保留原始390个样本框架
df = df[list(target_cols.keys())].rename(columns=target_cols).iloc[:390].copy()

# 2. 数据类型与缺失值处理
# 处理睡眠时间（转换为小时）
def convert_sleep(time_str):
    try:
        h, m, s = map(int, str(time_str).split(':'))
        if 0 <= h < 24 and 0 <= m < 60 and 0 <= s < 60:
            return round(h + m/60 + s/3600, 2)
        return np.nan
    except:
        return np.nan

df['Sleep_Hours'] = df['Sleep_Hours'].apply(convert_sleep)

# 3. 筛选有效样本（排除异常类别/缺失值）
valid_conditions = [
    df['Baby_Behavior'].isin(['安静型', '中等型', '矛盾型']),  # 婴儿行为特征有效
    df['Baby_Gender'].isin([1, 2]),                           # 婴儿性别有效
    df['Mom_Marriage'].isin([1, 2, 3]),                       # 婚姻状况有效（1=未婚，2=已婚，3=其他）
    df['Delivery_Mode'].isin([1, 2]),                         # 分娩方式有效（1=自然，2=剖宫产）
    df['Sleep_Hours'].notna(),                                # 睡眠时间非空
    df['Pregnancy_Weeks'].between(26, 43)                     # 妊娠周数合理（26周<足月<43周）
]
df = df[np.all(valid_conditions, axis=0)].copy()

# 4. 填充缺失值（按变量类型区分）
# 数值型变量（用中位数填充，避免极端值影响）
numeric_vars = ['Mom_Age', 'Pregnancy_Weeks', 'CBTS', 'EPDS', 'HADS', 'Baby_Age', 'Sleep_Hours', 'Wake_Count']
# 分类变量（用众数填充，符合实际类别分布）
categorical_vars = ['Mom_Marriage', 'Delivery_Mode', 'Baby_Gender', 'Baby_Behavior']

for var in df.columns:
    if df[var].isna().sum() > 0:
        if var in numeric_vars:
            df[var].fillna(df[var].median(), inplace=True)
        elif var in categorical_vars:
            df[var].fillna(df[var].mode()[0], inplace=True)

# 5. 数据概览（新增母亲4类指标的分布）
print("=== 数据预处理结果（含母亲人口学/生理指标） ===")
print(f"有效样本数：{len(df)}")
print("\n变量缺失值情况：")
print(df.isna().sum())

print("\n=== 母亲核心指标分布 ===")
# 母亲年龄统计
print(f"母亲年龄：均值={df['Mom_Age'].mean():.1f}岁，范围=[{df['Mom_Age'].min()}-{df['Mom_Age'].max()}]岁")
# 婚姻状况分布（映射为中文）
marriage_map = {1:'未婚', 2:'已婚', 3:'其他'}
print(f"婚姻状况：\n{df['Mom_Marriage'].map(marriage_map).value_counts()}")
# 妊娠时间统计
print(f"妊娠时间：均值={df['Pregnancy_Weeks'].mean():.1f}周，范围=[{df['Pregnancy_Weeks'].min()}-{df['Pregnancy_Weeks'].max()}]周")
# 分娩方式分布（映射为中文）
delivery_map = {1:'自然分娩', 2:'剖宫产'}
print(f"分娩方式：\n{df['Delivery_Mode'].map(delivery_map).value_counts()}")

print("\n=== 婴儿睡眠指标统计 ===")
print(f"整晚睡眠时间：均值={df['Sleep_Hours'].mean():.2f}小时，范围=[{df['Sleep_Hours'].min()}-{df['Sleep_Hours'].max()}]小时")
print(f"睡醒次数：均值={df['Wake_Count'].mean():.2f}次，范围=[{df['Wake_Count'].min()}-{df['Wake_Count'].max()}]次")


# ---------------------- 新增：母亲4类指标与婴儿睡眠时长的相关性分析 ----------------------
print("\n=== 母亲4类核心指标与婴儿睡眠时长的相关性 ===")
# 选择母亲4类指标（数值型/有序分类）与睡眠时长计算Pearson相关系数
mom_sleep_vars = ['Mom_Age', 'Pregnancy_Weeks', 'Mom_Marriage', 'Delivery_Mode']
corr_results = []

for var in mom_sleep_vars:
    # 计算相关系数与显著性（排除极端值）
    data_pair = df[[var, 'Sleep_Hours']].dropna()
    corr, p_value = stats.pearsonr(data_pair[var], data_pair['Sleep_Hours'])
    corr_results.append({
        '指标': var,
        '指标名称': {
            'Mom_Age':'母亲年龄',
            'Pregnancy_Weeks':'妊娠时间',
            'Mom_Marriage':'婚姻状况',
            'Delivery_Mode':'分娩方式'
        }[var],
        '相关系数r': round(corr, 3),
        'P值': round(p_value, 4),
        '相关性': '显著' if p_value < 0.05 else '不显著',
        '关联方向': '正相关' if corr > 0 else '负相关' if corr < 0 else '无相关'
    })

# 打印相关性结果
corr_df = pd.DataFrame(corr_results)
print(corr_df[['指标名称', '相关系数r', 'P值', '相关性', '关联方向']].to_string(index=False))

# 绘制母亲4类指标与睡眠时长的相关性热图（整合原有心理指标）
print("\n=== 母亲全维度指标与婴儿睡眠时长相关性热图 ===")
all_mom_vars = ['Mom_Age', 'Pregnancy_Weeks', 'Mom_Marriage', 'Delivery_Mode', 'CBTS', 'EPDS', 'HADS']
all_mom_names = ['母亲年龄', '妊娠时间', '婚姻状况', '分娩方式', 'CBTS', 'EPDS', 'HADS']
corr_matrix_mom_sleep = df[all_mom_vars + ['Sleep_Hours']].corr().round(3)
# 重命名列索引为中文（便于解读）
corr_matrix_mom_sleep.columns = all_mom_names + ['婴儿睡眠时长']
corr_matrix_mom_sleep.index = all_mom_names + ['婴儿睡眠时长']

# 绘制热图
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix_mom_sleep))  # 隐藏上三角（避免重复）
sns.heatmap(
    corr_matrix_mom_sleep,
    annot=True,
    cmap='RdBu_r',
    vmin=-1, vmax=1,
    fmt='.3f',
    mask=mask,
    cbar_kws={'label': 'Pearson相关系数'}
)
plt.title('母亲全维度指标与婴儿睡眠时长相关性热图', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('母亲全维度指标-婴儿睡眠相关性热图.png', dpi=300)
plt.close()


# ---------------------- 新增：母亲4类指标与睡眠时长的分组统计与可视化 ----------------------
print("\n=== 母亲分类指标与婴儿睡眠时长的分组统计 ===")
# 1. 婚姻状况分组（已婚vs未婚/其他）
print("1. 婚姻状况分组（婴儿睡眠时长）：")
marriage_sleep = df.groupby('Mom_Marriage')['Sleep_Hours'].agg(['mean', 'std', 'count']).round(2)
marriage_sleep.index = marriage_sleep.index.map(marriage_map)
print(marriage_sleep)

# 2. 分娩方式分组（自然分娩vs剖宫产）
print("\n2. 分娩方式分组（婴儿睡眠时长）：")
delivery_sleep = df.groupby('Delivery_Mode')['Sleep_Hours'].agg(['mean', 'std', 'count']).round(2)
delivery_sleep.index = delivery_sleep.index.map(delivery_map)
print(delivery_sleep)

# 3. 母亲年龄分组（按中位数分：≤30岁vs>30岁）
print("\n3. 母亲年龄分组（婴儿睡眠时长）：")
df['Mom_Age_Group'] = pd.cut(df['Mom_Age'], bins=[0, 30, 50], labels=['≤30岁', '>30岁'])
age_sleep = df.groupby('Mom_Age_Group')['Sleep_Hours'].agg(['mean', 'std', 'count']).round(2)
print(age_sleep)

# 4. 妊娠时间分组（足月vs非足月：≥37周为足月）
print("\n4. 妊娠时间分组（婴儿睡眠时长）：")
df['Pregnancy_Group'] = pd.cut(df['Pregnancy_Weeks'], bins=[0, 37, 45], labels=['非足月(<37周)', '足月(≥37周)'])
pregnancy_sleep = df.groupby('Pregnancy_Group')['Sleep_Hours'].agg(['mean', 'std', 'count']).round(2)
print(pregnancy_sleep)

# 可视化：母亲4类指标与婴儿睡眠时长的箱线图（2x2子图）
plt.figure(figsize=(16, 12))
# 子图1：母亲年龄分组
plt.subplot(2, 2, 1)
sns.boxplot(x='Mom_Age_Group', y='Sleep_Hours', data=df, palette='Set2')
plt.title('母亲年龄分组与婴儿睡眠时长', fontsize=12)
plt.xlabel('母亲年龄')
plt.ylabel('婴儿睡眠时长（小时）')
plt.grid(alpha=0.3, linestyle='--')

# 子图2：妊娠时间分组
plt.subplot(2, 2, 2)
sns.boxplot(x='Pregnancy_Group', y='Sleep_Hours', data=df, palette='Set2')
plt.title('妊娠时间分组与婴儿睡眠时长', fontsize=12)
plt.xlabel('妊娠时间')
plt.ylabel('婴儿睡眠时长（小时）')
plt.grid(alpha=0.3, linestyle='--')

# 子图3：婚姻状况分组
plt.subplot(2, 2, 3)
sns.boxplot(x=df['Mom_Marriage'].map(marriage_map), y='Sleep_Hours', data=df, palette='Set2')
plt.title('母亲婚姻状况与婴儿睡眠时长', fontsize=12)
plt.xlabel('婚姻状况')
plt.ylabel('婴儿睡眠时长（小时）')
plt.grid(alpha=0.3, linestyle='--')

# 子图4：分娩方式分组
plt.subplot(2, 2, 4)
sns.boxplot(x=df['Delivery_Mode'].map(delivery_map), y='Sleep_Hours', data=df, palette='Set2')
plt.title('分娩方式与婴儿睡眠时长', fontsize=12)
plt.xlabel('分娩方式')
plt.ylabel('婴儿睡眠时长（小时）')
plt.grid(alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('母亲4类指标-婴儿睡眠时长箱线图.png', dpi=300)
plt.close()


# ---------------------- 新增：母亲全维度指标对睡眠时长的回归模型（含4类指标） ----------------------
print("\n=== 母亲全维度指标对婴儿睡眠时长的回归分析 ===")
# 1. 构建自变量（母亲4类指标+3类心理指标，排除共线性）
X_full = df[['Mom_Age', 'Pregnancy_Weeks', 'Mom_Marriage', 'Delivery_Mode', 'CBTS', 'EPDS', 'HADS']]
y_sleep = df['Sleep_Hours']

# 2. 多重共线性检验（VIF）
def calculate_vif(X_data):
    X_with_const = sm.add_constant(X_data)
    vif_df = pd.DataFrame()
    vif_df['指标'] = all_mom_names  # 对应中文名称
    vif_df['VIF'] = [variance_inflation_factor(X_with_const.values, i+1) for i in range(X_data.shape[1])]  # 跳过常数项
    return vif_df

vif_full = calculate_vif(X_full)
print("多重共线性检验（VIF<10为无严重共线性）：")
print(vif_full.round(3))

# 3. 多元线性回归（分训练集/测试集）
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full, y_sleep, test_size=0.3, random_state=42
)

# 构建回归模型（添加常数项）
model_full = sm.OLS(y_train_full, sm.add_constant(X_train_full)).fit()
print("\n回归模型结果（显著变量P<0.05）：")
# 提取显著变量并展示
summary_df = pd.DataFrame({
    '指标': ['常数项'] + all_mom_names,
    '系数': model_full.params.round(3),
    '标准误': model_full.bse.round(3),
    'P值': model_full.pvalues.round(4),
    '显著性': ['***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns' for p in model_full.pvalues]
})
print(summary_df[summary_df['P值'] < 0.05].to_string(index=False))  # 只显示显著变量

# 模型评估
y_pred_full = model_full.predict(sm.add_constant(X_test_full))
r2_full = r2_score(y_test_full, y_pred_full)
rmse_full = np.sqrt(mean_squared_error(y_test_full, y_pred_full))
print(f"\n模型评估：R²={r2_full:.3f}，RMSE={rmse_full:.3f}")
print(f"模型解释力：母亲全维度指标共解释婴儿睡眠时长变异的{r2_full*100:.1f}%")
