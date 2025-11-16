# 🌍 Maternal–Infant Health Multitask Machine Learning System

# 母婴健康多任务机器学习系统

An end-to-end AI pipeline for **infant behavior prediction**, **sleep quality evaluation**, and **maternal psychological intervention cost optimization**, based on **390 real-world mother–infant samples**.
基于 **390 例真实母婴数据**，构建婴儿行为预测、睡眠质量评估与母亲心理干预成本优化的 **端到端多任务机器学习系统**。

---

## 📘 Project Overview | 项目概述

This project integrates **data preprocessing**, **predictive modeling**, **entropy-based composite scoring**, and **multi-objective optimization** to support intelligent maternal–infant healthcare decisions.
本项目结合 **数据预处理、预测模型、熵权法评分与多目标优化算法**，构建一个面向母婴健康的智能预测与决策系统。

---

# 🧩 Modules | 模块功能

## **1. Data Preprocessing & Correlation Analysis (problem1.py)**

### **数据预处理与相关性分析**

* Clean multi-source maternal–infant data

* Convert sleep duration (“hh:mm:ss”) into numeric value

* Normalize features & impute missing values

* Pearson correlation analysis

* 清洗多源母婴数据

* 将“时:分:秒”形式的睡眠时间转化为可训练数值

* 特征标准化与缺失值填补

* 母亲身体/心理特征与婴儿行为/睡眠之间的相关性分析

---

## **2. Infant Behavioral Classification (problem2.py)**

### **婴儿行为特征分类**

* Random Forest multi-classification

* Resolve class imbalance using SMOTE

* GridSearchCV hyperparameter tuning

* Feature importance interpretation

* 使用随机森林三分类模型

* 使用 SMOTE 处理类别不平衡

* 网格搜索优化模型参数

* 特征重要性可解释性

---

## **3. Treatment Cost Optimization (problem3.py)**

### **心理干预治疗费用最优化**

* Construct linear cost functions based on CBTS / EPDS / HADS

* Compute minimum treatment cost for improving behavior

* Personalized intervention strategy generation

* 基于 CBTS/EPDS/HADS 构建线性费用函数

* 计算行为改善所需的最小治疗费用

* 输出个性化干预策略方案

---

## **4. Sleep Quality Scoring & Prediction (problem4.py)**

### **睡眠质量评分与预测**

* Apply Entropy Weight Method (EWM) to integrate multi-indicator sleep metrics

* Regression-based sleep quality prediction

* Sleep grades: **Excellent / Good / Fair / Poor**

* 使用熵权法融合多指标睡眠特征

* 构建回归模型预测睡眠等级

* 将睡眠质量划分为 **优 / 良 / 中 / 差**

---

## **5. Joint Behavior–Sleep Optimization (problem5.py)**

### **行为 + 睡眠联合优化模型**

* Multi-objective optimization across behavior & sleep

* Ensure both behavior improvement & sleep enhancement

* Find overall minimum psychological intervention cost

* 构建行为与睡眠的多目标优化模型

* 同时提升婴儿行为特征与睡眠质量

* 输出最小化心理干预费用的优化策略

---

# 🔧 Key Techniques | 核心技术

* Random Forest（分类/回归）
* SMOTE 类别不平衡处理
* Entropy Weight Method（熵权法）
* GridSearchCV 超参搜索
* Multi-objective optimization 多目标优化
* Feature importance analysis 特征可解释性分析
* Heatmaps / Confusion Matrix / Regression Plots 等可视化

---

# 📁 Project Structure | 项目结构

```
problem1.py   → Data preprocessing & correlation analysis  
problem2.py   → Infant behavior classification  
problem3.py   → Treatment cost optimization  
problem4.py   → Sleep scoring & prediction  
problem5.py   → Joint behavior–sleep optimization
```

```
problem1.py   → 数据预处理与相关性分析  
problem2.py   → 婴儿行为特征分类模型  
problem3.py   → 心理干预治疗成本优化  
problem4.py   → 睡眠综合评分与预测  
problem5.py   → 行为 + 睡眠联合优化
```

---

# 📊 Dataset Description | 数据集说明

| Category                      | Description                           | 中文说明                 |
| ----------------------------- | ------------------------------------- | -------------------- |
| Maternal physical indicators  | Age, pregnancy weeks, delivery method | 母亲年龄、孕周、分娩方式         |
| Maternal psychological scales | CBTS, EPDS, HADS                      | 三大心理量表               |
| Infant behavior               | Quiet / Moderate / Ambivalent         | 婴儿行为三分类              |
| Infant sleep                  | Duration / Awakenings / Sleep method  | 睡眠时长 / 夜间醒转次数 / 入睡方式 |
| Missing labels                | Cases 391–410                         | 391–410 号婴儿行为与睡眠标签缺失 |

---

# 🧩 Appendix | 附录说明

## 📌 **Background Information** |背景说明

The dataset includes mother–infant information for **390 babies aged 3–12 months**.
Maternal indicators include age, marital status, education level, pregnancy weeks, delivery method, and psychological health measured by:

* **CBTS** – Childbirth-Related PTSD Questionnaire
* **EPDS** – Edinburgh Postnatal Depression Scale
* **HADS** – Hospital Anxiety and Depression Scale

Baby sleep quality is evaluated via:

1. Night sleep duration
2. Number of awakenings
3. Falling-asleep method

---


数据包含 **390 名 3–12 个月婴儿及其母亲信息**，包括：

* 母亲年龄、婚姻状况、受教育程度
* 孕周、分娩方式
* 心理量表：CBTS / EPDS / HADS

婴儿睡眠质量判定指标包括：

1. 夜间睡眠时长
2. 夜间醒转次数
3. 入睡方式

# 📌 Treatment Cost Table | 治疗费用假定表

The treatment cost for improving maternal psychological indicators is assumed to be **linearly related to the severity score** of the psychological scales **CBTS / EPDS / HADS**.
Table 1 provides the reference cost at two score levels for each scale.

母亲心理健康干预的治疗费用根据题目要求，被 **假设为与心理量表得分线性相关**。
下表给出了 **CBTS / EPDS / HADS** 在两个得分点处的对应费用，用于构建线性费用函数。

---

## **📄 Table 1. Psychological Score vs Treatment Cost**

表 1. 心理量表得分与治疗费用对照表**

| Scale (量表) | Score (得分) | Cost (RMB 元) |
| ---------- | ---------- | ------------ |
| **CBTS**   | 0          | 200          |
|            | 3          | 2812         |
| **EPDS**   | 0          | 500          |
|            | 2          | 1890         |
| **HADS**   | 0          | 300          |
|            | 5          | 12500        |

---

# 📝 Explanation | 说明

* The above table serves as the basis for constructing the **linear treatment cost functions** in the optimization model.
* Given two known points for each psychological scale, we compute the slope and intercept to model:
  Cost = a × Score + b
* These functions are used to calculate the **minimum intervention cost** for improving infant behavior or sleep quality.

* 上述表格用于构建优化模型中的 **线性治疗费用函数**；
* 每个心理量表均给定两个（得分–费用）点，可据此计算斜率和截距，得到：
  费用 = a × 得分 + b
* 这一函数被用于计算改善婴儿行为或睡眠质量所需的 **最小心理干预成本**。

---

## 📌 **Core Tasks Summary | 核心任务总结**

The project aims to:

1. Build a model linking infant behavior type with maternal indicators, and predict **behavior labels for cases #391–410**.
2. For **infant #238 (Ambivalent)**, calculate the **minimum treatment cost** to achieve **Moderate / Quiet** behavior.
3. Build a **sleep-quality scoring system** and predict sleep grades for **cases #391–410**.
4. Perform **behavior + sleep joint optimization** and determine whether #238 needs treatment plan adjustment to reach **Excellent sleep**.

---

本项目旨在：

1. 建立婴儿行为类型与母亲身体/心理指标的关联模型，并预测 **391–410 号婴儿行为类型**。
2. 对 **238 号矛盾型婴儿**，计算达到 **中等型/安静型** 的最小心理干预费用。
3. 构建 **婴儿睡眠评级体系**，并预测 **391–410 号婴儿** 的睡眠等级。
4. 若要求 **238 号婴儿睡眠达到“优”**，判断治疗方案是否需要调整并求出最优策略。

---

# 🧠 Summary | 项目总结

This project unifies **prediction + evaluation + optimization** into a clinically meaningful maternal–infant AI system.
本项目实现了 **预测 + 评估 + 优化** 一体化的母婴健康智能系统，具有良好的科研与应用价值。

---
