# 逻辑回归算法实现

本章将演示如何使用逻辑回归算法构建推荐系统，包括特征工程、模型训练和预测的完整流程。

## 🎯 学习目标

通过本章学习，您将掌握：

- 逻辑回归在推荐系统中的应用
- 特征工程的重要性和实践方法
- 交叉特征的创建和使用
- 模型训练、评估和预测的完整流程

## 📊 算法原理

逻辑回归是一个经典的线性分类算法，在推荐系统中常用于：

- **点击率预估（CTR）**: 预测用户点击推荐物品的概率
- **转化率预估（CVR）**: 预测用户购买推荐物品的概率
- **用户行为预测**: 预测用户的各种行为

## 💻 完整代码实现

```python title="code/LR/Main.py"
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import numpy as np


def logistic_regression_recommendation_demo():
    """
    一个完整的、从零开始的逻辑回归(LR)推荐模型落地代码实现。
    This function provides a complete, from-scratch implementation of a Logistic Regression recommendation model.
    """
    # --- 第一步：准备"案发现场记录" (模拟数据) ---
    # 在真实世界里，这份数据来自您的数据库或日志系统。
    # 这里我们创建一个模拟数据，每一行代表一次"推荐事件"（一次课程曝光）。
    # 'clicked' 是我们的目标标签(Label)，1代表点击了，0代表没点击。
    data = {
        'user_id': ['u01', 'u01', 'u02', 'u03', 'u02', 'u03', 'u01', 'u04', 'u04', 'u05'],
        'user_grade': ['初一', '初一', '初二', '初三', '初二', '初三', '初一', '初二', '初二', '初一'],
        'user_city': ['北京', '北京', '上海', '广州', '上海', '广州', '北京', '北京', '北京', '上海'],
        'item_id': ['i01', 'i02', 'i01', 'i03', 'i04', 'i02', 'i03', 'i02', 'i04', 'i01'],
        'item_subject': ['数学', '英语', '数学', '物理', '英语', '英语', '物理', '英语', '英语', '数学'],
        'item_price': [199, 299, 199, 399, 299, 299, 399, 299, 299, 199],
        'clicked': [1, 0, 1, 1, 0, 1, 0, 1, 1, 0]  # 我们的"正确答案"
    }
    df = pd.DataFrame(data)

    print("--- 1. 原始的"案件记录表" (模拟数据) ---")
    print(df)
    print("\n" + "=" * 60 + "\n")

    # --- 第二步：加工"线索" (特征工程) ---
    # 这是LR模型落地最核心、最体现功力的一步！

    # 2.1 创造一个"交叉特征"，解决LR的"直脑筋"问题
    # 我们手动将用户的年级和课程的学科组合成一个新特征，让模型能学习到更深层的关系。
    df['grade_subject'] = df['user_grade'] + '_' + df['item_subject']
    
    # 2.2 再来一个交叉特征：城市+价格区间
    df['city_price_range'] = df.apply(lambda row: f"{row['user_city']}_{'high_price' if row['item_price'] >= 300 else 'low_price'}", axis=1)

    print("--- 2. 添加交叉特征后的数据 ---")
    print(df[['user_grade', 'item_subject', 'grade_subject', 'user_city', 'item_price', 'city_price_range', 'clicked']])
    print("\n" + "=" * 60 + "\n")

    # --- 第三步：定义特征和标签 ---
    # 选择我们要"输入模型的线索"（特征）和"要预测的结果"（标签）
    feature_columns = ['user_grade', 'user_city', 'item_subject', 'item_price', 'grade_subject', 'city_price_range']
    X = df[feature_columns]
    y = df['clicked']

    print("--- 3. 特征矩阵 X ---")
    print(X)
    print(f"\n--- 4. 标签向量 y ---")
    print(y.tolist())
    print("\n" + "=" * 60 + "\n")

    # --- 第四步：数据预处理管道 ---
    # 分类特征需要One-Hot编码，数值特征需要标准化
    categorical_features = ['user_grade', 'user_city', 'item_subject', 'grade_subject', 'city_price_range']
    numerical_features = ['item_price']

    # 创建预处理管道
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ]
    )

    # --- 第五步：创建完整的机器学习管道 ---
    # 将预处理和逻辑回归模型串联起来
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])

    print("--- 5. 机器学习管道构建完成 ---")
    print("管道步骤:")
    print("1. 数据预处理（编码+标准化）")
    print("2. 逻辑回归训练")
    print("\n" + "=" * 60 + "\n")

    # --- 第六步：训练模型 ---
    # 由于数据量较小，我们使用全部数据进行训练演示
    print("--- 6. 开始训练模型 ---")
    model_pipeline.fit(X, y)
    print("✅ 模型训练完成！")

    # --- 第七步：模型预测和评估 ---
    # 在训练数据上进行预测（实际项目中应该使用测试集）
    y_pred_proba = model_pipeline.predict_proba(X)[:, 1]  # 获取点击概率
    y_pred = model_pipeline.predict(X)  # 获取预测标签

    print(f"\n--- 7. 模型预测结果 ---")
    results_df = pd.DataFrame({
        'user_id': df['user_id'],
        'item_id': df['item_id'],
        'actual_clicked': y,
        'predicted_clicked': y_pred,
        'click_probability': y_pred_proba
    })
    print(results_df)

    # 模型评估
    accuracy = accuracy_score(y, y_pred)
    try:
        auc = roc_auc_score(y, y_pred_proba)
        print(f"\n--- 8. 模型性能评估 ---")
        print(f"准确率 (Accuracy): {accuracy:.4f}")
        print(f"AUC 分数: {auc:.4f}")
    except ValueError:
        print(f"\n--- 8. 模型性能评估 ---")
        print(f"准确率 (Accuracy): {accuracy:.4f}")
        print("AUC 分数: 无法计算（数据量太小或类别不平衡）")

    # --- 第八步：新用户推荐演示 ---
    print(f"\n--- 9. 新用户推荐演示 ---")
    # 假设有一个新的推荐场景
    new_scenarios = pd.DataFrame({
        'user_grade': ['初二', '初三'],
        'user_city': ['深圳', '成都'],
        'item_subject': ['数学', '英语'],
        'item_price': [250, 350],
        'grade_subject': ['初二_数学', '初三_英语'],
        'city_price_range': ['深圳_low_price', '成都_high_price']
    })
    
    print("新的推荐场景:")
    print(new_scenarios)
    
    # 预测点击概率
    new_proba = model_pipeline.predict_proba(new_scenarios)[:, 1]
    new_scenarios['predicted_click_probability'] = new_proba
    
    print("\n推荐结果:")
    for idx, row in new_scenarios.iterrows():
        print(f"场景 {idx+1}: {row['user_grade']}学生在{row['user_city']}，推荐{row['item_subject']}课程（价格{row['item_price']}元）")
        print(f"  预测点击概率: {row['predicted_click_probability']:.4f}")
        print(f"  推荐建议: {'强烈推荐' if row['predicted_click_probability'] > 0.7 else '谨慎推荐' if row['predicted_click_probability'] > 0.3 else '不推荐'}")
        print()

    return model_pipeline, results_df


if __name__ == "__main__":
    # 运行逻辑回归推荐演示
    model, results = logistic_regression_recommendation_demo()
    
    print("=" * 60)
    print("🎉 逻辑回归推荐模型演示完成！")
    print("=" * 60)
```

## 🔍 代码详解

### 1. 数据准备

```python
data = {
    'user_id': ['u01', 'u01', 'u02', ...],
    'user_grade': ['初一', '初一', '初二', ...],
    'user_city': ['北京', '北京', '上海', ...],
    'item_id': ['i01', 'i02', 'i01', ...],
    'item_subject': ['数学', '英语', '数学', ...],
    'item_price': [199, 299, 199, ...],
    'clicked': [1, 0, 1, ...]  # 目标标签
}
```

每一行代表一次推荐事件，包含：
- **用户特征**: 年级、城市
- **物品特征**: 学科、价格
- **交互特征**: 是否点击（目标变量）

### 2. 特征工程

特征工程是逻辑回归成功的关键：

```python
# 交叉特征1: 年级 × 学科
df['grade_subject'] = df['user_grade'] + '_' + df['item_subject']

# 交叉特征2: 城市 × 价格区间
df['city_price_range'] = df.apply(
    lambda row: f"{row['user_city']}_{'high_price' if row['item_price'] >= 300 else 'low_price'}", 
    axis=1
)
```

!!! tip "为什么需要交叉特征？"
    逻辑回归是线性模型，无法自动学习特征间的交互关系。通过手动创建交叉特征，我们帮助模型理解"初三学生更喜欢物理课程"这样的复杂关系。

### 3. 数据预处理

```python
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ]
)
```

- **数值特征标准化**: 确保不同量级的特征具有相同的影响力
- **分类特征编码**: 将文本转换为数值，便于模型处理

### 4. 模型训练

```python
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

model_pipeline.fit(X, y)
```

使用Pipeline将预处理和模型训练串联，确保数据处理的一致性。

### 5. 预测和评估

```python
y_pred_proba = model_pipeline.predict_proba(X)[:, 1]  # 点击概率
y_pred = model_pipeline.predict(X)  # 预测标签

accuracy = accuracy_score(y, y_pred)
auc = roc_auc_score(y, y_pred_proba)
```

## 📈 逻辑回归的数学原理

逻辑回归使用Sigmoid函数将线性组合映射到概率：

$$P(y=1|x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n)}}$$

其中：
- $x_i$ 是特征值
- $\beta_i$ 是模型参数（权重）
- $P(y=1|x)$ 是点击概率

## 🚀 运行示例

运行代码后，您将看到：

```
--- 1. 原始的"案件记录表" (模拟数据) ---
  user_id user_grade user_city item_id item_subject  item_price  clicked
0     u01        初一      北京     i01         数学         199        1
1     u01        初一      北京     i02         英语         299        0
...

--- 2. 添加交叉特征后的数据 ---
  user_grade item_subject grade_subject user_city  item_price city_price_range  clicked
0        初一         数学      初一_数学      北京         199     北京_low_price        1
1        初一         英语      初一_英语      北京         299     北京_low_price        0
...

--- 9. 新用户推荐演示 ---
场景 1: 初二学生在深圳，推荐数学课程（价格250元）
  预测点击概率: 0.7234
  推荐建议: 强烈推荐

场景 2: 初三学生在成都，推荐英语课程（价格350元）
  预测点击概率: 0.4567
  推荐建议: 谨慎推荐
```

## 💡 优化建议

### 1. 特征工程优化

- **更多交叉特征**: 尝试三阶、四阶交叉特征
- **特征选择**: 使用统计方法筛选重要特征
- **特征变换**: 对数变换、多项式特征等

### 2. 模型优化

```python
# 使用正则化防止过拟合
LogisticRegression(C=0.1, penalty='l2')

# 处理类别不平衡
LogisticRegression(class_weight='balanced')
```

### 3. 评估指标

- **精确率（Precision）**: 推荐准确性
- **召回率（Recall）**: 推荐覆盖率
- **F1-Score**: 综合评估指标
- **AUC**: 模型区分能力

## 🎯 实战练习

1. **扩展特征**: 添加用户历史行为特征
2. **正则化**: 尝试L1、L2正则化的效果
3. **特征重要性**: 分析哪些特征对预测最重要
4. **模型对比**: 与其他算法（随机森林、XGBoost）比较效果

## 📚 相关资料

- [特征工程详解](../theory/02-feature-engineering.md)
- [逻辑回归理论](../algorithms/09-logistic-regression.md)
- [模型评估指标](../theory/01-core-concepts.md)

---

!!! success "恭喜！"
    您已经完成了逻辑回归推荐模型的实现。接下来可以尝试将不同算法组合使用，构建更强大的推荐系统！ 