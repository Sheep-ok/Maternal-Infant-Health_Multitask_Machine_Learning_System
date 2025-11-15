import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

file_path = "训练题4健康问题附件.xlsx"
df = pd.read_excel(file_path)


df.columns = ['编号', '母亲年龄', '婚姻状况', '教育程度', '妊娠时间（周数）', '分娩方式',
              'CBTS', 'EPDS', 'HADS', '婴儿行为特征', '婴儿性别', '婴儿年龄（月）',
              '整晚睡眠时间（时：分：秒）', '睡醒次数', '入睡方式', 'Unnamed: 15',
              '补充说明（数值含义）_数值', '补充说明（数值含义）_婚姻状况',
              '补充说明（数值含义）_教育程度', '补充说明（数值含义）_分娩方式',
              '补充说明（数值含义）_婴儿性别', '补充说明（数值含义）_入睡方式']

df = df[1:].reset_index(drop=True)

# 查看婴儿行为特征的分布情况
print("婴儿行为特征分布：")
print(df['婴儿行为特征'].value_counts())

# 可视化类别分布
plt.figure(figsize=(8, 5))
sns.countplot(x='婴儿行为特征', data=df)
plt.title('婴儿行为特征类别分布')
plt.tight_layout()
plt.show()

X = df[['母亲年龄', '婚姻状况', '教育程度', '妊娠时间（周数）', '分娩方式', 'CBTS', 'EPDS', 'HADS']].copy()
y = df['婴儿行为特征'].copy()

for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')

for col in X.columns:
    X[col] = X[col].fillna(X[col].mean())

# 删除目标变量中的缺失值
non_missing_indices = y.dropna().index
X = X.loc[non_missing_indices]
y = y.loc[non_missing_indices]

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("\nSMOTE处理后的类别分布：")
print(pd.Series(y_resampled).value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)


param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced', 'balanced_subsample']
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    scoring='f1_macro'
)

grid_search.fit(X_train, y_train)
print(f"\n最佳参数: {grid_search.best_params_}")

best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'\n模型准确率: {accuracy:.2f}')
print('分类报告:')
print(classification_report(y_test, y_pred, zero_division=0))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=best_rf.classes_,
            yticklabels=best_rf.classes_)
plt.xlabel('预测类别')
plt.ylabel('实际类别')
plt.title('混淆矩阵')
plt.tight_layout()
plt.show()
# 计算并输出混淆矩阵数值
cm = confusion_matrix(y_test, y_pred)
print("\n混淆矩阵数值（行=实际类别，列=预测类别）：")
print("行/列顺序：", best_rf.classes_)  # 显示行列对应的类别
print(cm)

# 也可以格式化输出为更易读的形式
print("\n混淆矩阵详细解读：")
class_names = best_rf.classes_
for i, actual_class in enumerate(class_names):
    for j, pred_class in enumerate(class_names):
        print(f"实际为{actual_class}且预测为{pred_class}的样本数：{cm[i, j]}")

# 特征重要性分析
feature_importance = pd.DataFrame({
    '特征': X.columns,
    '重要性': best_rf.feature_importances_
}).sort_values(by='重要性', ascending=False)

print("\n特征重要性排序：")
print(feature_importance)

plt.figure(figsize=(10, 6))
sns.barplot(x='重要性', y='特征', data=feature_importance)
plt.title('特征重要性')
plt.tight_layout()
plt.show()

# 提取编号391-410的数据
missing_data = df.loc[389:408, ['母亲年龄', '婚姻状况', '教育程度', '妊娠时间（周数）',
                                '分娩方式', 'CBTS', 'EPDS', 'HADS']].copy()

# 处理预测数据集中的缺失值
for col in missing_data.columns:
    missing_data[col] = pd.to_numeric(missing_data[col], errors='coerce')
    missing_data[col] = missing_data[col].fillna(X[col].mean())

# 预测缺失的婴儿行为特征，同时获取预测概率
predicted_missing = best_rf.predict(missing_data)
predicted_proba = best_rf.predict_proba(missing_data)

# 创建结果数据框
result = pd.DataFrame({
    '编号': df.loc[389:408, '编号'].values,
    '预测婴儿行为特征': predicted_missing
})

# 添加每个类别的预测概率
for i, cls in enumerate(best_rf.classes_):
    result[f'{cls}_概率'] = predicted_proba[:, i].round(4)

# 显示编号391-410的编号和预测结果
print('\n编号391-410的预测结果:')
print(result.to_string(index=False))

# 保存结果到Excel文件
result.to_excel('问题二婴儿行为特征预测结果.xlsx', index=False)
print('\n预测结果已保存到"婴儿行为特征预测结果.xlsx"文件中')
