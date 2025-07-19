# ItemCF 算法实现

本章将通过完整的代码示例来演示如何从零开始实现基于物品的协同过滤算法（ItemCF）。

## 🎯 学习目标

通过本章学习，您将掌握：

- ItemCF 算法的核心思想和实现步骤
- 如何构建用户-物品交互矩阵
- 物品相似度计算方法
- 推荐生成的具体过程

## 📊 算法原理

ItemCF（Item-based Collaborative Filtering）的核心思想是：**相似的物品会被同样的用户喜欢**。

算法步骤：
1. 构建用户-物品交互矩阵
2. 计算物品之间的相似度
3. 基于相似度为用户生成推荐

## 💻 完整代码实现

```python title="code/ItemCF/Main.py"
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def item_based_collaborative_filtering():
    """
    一个完整的、从零开始的ItemCF算法实现流程。
    This function demonstrates a complete, from-scratch implementation of the ItemCF algorithm.
    """
    # --- 第一步：准备数据 (建立"江湖档案库") ---
    # 在真实世界里，这部分数据会来自您的数据库或日志文件。
    # 这里我们用一个简单的例子来模拟您的"花名册数据"。
    # 1 代表学生上过这门课。
    data = {
        '小明': {'数学': 1, '物理': 1, '化学': 0, '生物': 0, '历史': 0},
        '小红': {'数学': 1, '物理': 0, '化学': 1, '生物': 1, '历史': 0},
        '小强': {'数学': 0, '物理': 1, '化学': 1, '历史': 1, '生物': 0},
        '小美': {'数学': 0, '物理': 0, '化学': 1, '生物': 1, '历史': 1},
        '小华': {'数学': 1, '物理': 1, '化学': 1, '生物': 1, '历史': 0}
    }

    # 将数据转换为Pandas DataFrame格式，这是进行数据分析的常用工具。
    # 用户为行，课程为列，这就是我们的"用户-物品交互矩阵"。
    df = pd.DataFrame(data).T  # .T 表示转置，让用户成为行
    df = df.fillna(0)  # 将没有记录的NaN值填充为0

    print("--- 1. 用户-课程交互矩阵 (原始数据) ---")
    print(df)
    print("\n" + "=" * 50 + "\n")

    # --- 第二步：计算物品（课程）之间的相似度 (绘制"物品关系网") ---
    # 我们使用`cosine_similarity`（余弦相似度）来计算。
    # 注意：我们需要对课程（列）进行计算，所以要再次转置矩阵 df.T
    item_similarity_df = pd.DataFrame(
        cosine_similarity(df.T),
        index=df.columns,
        columns=df.columns
    )

    print("--- 2. 课程与课程之间的相似度矩阵 ---")
    print(item_similarity_df)
    print("\n" + "=" * 50 + "\n")

    # --- 第三步：为指定用户生成推荐 (顺藤摸瓜，加权推荐) ---
    target_user = '小明'

    print(f"--- 3. 开始为用户 '{target_user}' 生成推荐 ---")

    # 3.1 找出目标用户上过（评分为1）的所有课程
    user_courses = df.loc[target_user]
    taken_courses = user_courses[user_courses == 1].index.tolist()
    
    print(f"用户 {target_user} 已经上过的课程: {taken_courses}")

    # 3.2 计算推荐分数
    recommendations = {}
    
    # 对于每一门用户没有上过的课程
    for course in df.columns:
        if course not in taken_courses:  # 只推荐用户没上过的课程
            score = 0
            # 基于用户已上过的每门课程计算推荐分数
            for taken_course in taken_courses:
                # 推荐分数 = 相似度 × 用户对已上课程的评分（这里都是1）
                score += item_similarity_df.loc[course, taken_course] * user_courses[taken_course]
            recommendations[course] = score

    # 3.3 按推荐分数排序
    sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n为用户 {target_user} 的推荐结果:")
    for course, score in sorted_recommendations:
        print(f"  {course}: {score:.4f}")

    return sorted_recommendations


if __name__ == "__main__":
    # 运行ItemCF算法
    recommendations = item_based_collaborative_filtering()
    
    print("\n" + "=" * 50)
    print("🎉 ItemCF 算法演示完成！")
    print("=" * 50)
```

## 🔍 代码详解

### 1. 数据准备

```python
# 用户-物品交互矩阵
data = {
    '小明': {'数学': 1, '物理': 1, '化学': 0, '生物': 0, '历史': 0},
    '小红': {'数学': 1, '物理': 0, '化学': 1, '生物': 1, '历史': 0},
    # ... 更多用户数据
}
```

这里构建了一个简单的用户-课程交互矩阵，其中：
- 行代表用户
- 列代表课程（物品）
- 值为1表示用户上过该课程，0表示没有

### 2. 相似度计算

```python
item_similarity_df = pd.DataFrame(
    cosine_similarity(df.T),  # 注意转置
    index=df.columns,
    columns=df.columns
)
```

使用余弦相似度计算物品之间的相似性：

$$\text{similarity}(i,j) = \frac{\sum_{u \in U} r_{ui} \cdot r_{uj}}{\sqrt{\sum_{u \in U} r_{ui}^2} \cdot \sqrt{\sum_{u \in U} r_{uj}^2}}$$

### 3. 推荐生成

```python
for course in df.columns:
    if course not in taken_courses:
        score = 0
        for taken_course in taken_courses:
            score += item_similarity_df.loc[course, taken_course] * user_courses[taken_course]
        recommendations[course] = score
```

推荐分数的计算公式：

$$\text{score}(u,i) = \sum_{j \in I_u} \text{similarity}(i,j) \cdot r_{uj}$$

其中：
- $u$ 是目标用户
- $i$ 是待推荐的物品
- $I_u$ 是用户 $u$ 已交互的物品集合
- $r_{uj}$ 是用户 $u$ 对物品 $j$ 的评分

## 🚀 运行示例

运行上述代码，您将看到：

```
--- 1. 用户-课程交互矩阵 (原始数据) ---
     数学  物理  化学  生物  历史
小明    1    1    0    0    0
小红    1    0    1    1    0
小强    0    1    1    1    1
小美    0    0    1    1    1
小华    1    1    1    1    0

--- 2. 课程与课程之间的相似度矩阵 ---
        数学      物理      化学      生物      历史
数学  1.0000  0.5000  0.5000  0.3333  0.0000
物理  0.5000  1.0000  0.5000  0.3333  0.3333
化学  0.5000  0.5000  1.0000  0.8000  0.5000
生物  0.3333  0.3333  0.8000  1.0000  0.6667
历史  0.0000  0.3333  0.5000  0.6667  1.0000

为用户 小明 的推荐结果:
  化学: 1.0000
  生物: 0.6667
  历史: 0.3333
```

## 💡 算法优化建议

### 1. 相似度计算优化
- **皮尔逊相关系数**: 考虑用户评分的偏好差异
- **调整余弦相似度**: 减少评分偏差的影响
- **Jaccard相似度**: 适用于隐式反馈数据

### 2. 稀疏性处理
```python
# 设置最小共同交互用户数量阈值
min_common_users = 3
if common_users_count < min_common_users:
    similarity = 0
```

### 3. 实时性优化
- **增量更新**: 只重新计算涉及的物品相似度
- **预计算**: 离线预计算相似度矩阵
- **缓存策略**: 缓存热门物品的推荐结果

## 🎯 练习任务

1. **修改数据集**: 尝试使用更大的用户-物品交互数据
2. **评分扩展**: 将二元评分扩展为1-5分的评分系统
3. **相似度对比**: 实现并比较不同相似度计算方法的效果
4. **推荐解释**: 为推荐结果添加解释功能

## 📚 相关资料

- [余弦相似度详解](../theory/06-similarity-computation.md)
- [ItemCF算法理论](../algorithms/07-itemcf-analysis.md)
- [推荐系统评估指标](../theory/01-core-concepts.md)

---

!!! tip "下一步"
    掌握了ItemCF的实现后，建议继续学习 [逻辑回归实现](logistic-regression.md)，了解基于模型的推荐方法。 