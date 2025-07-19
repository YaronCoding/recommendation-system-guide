import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def item_based_collaborative_filtering():
    """
    一个完整的、从零开始的ItemCF算法实现流程。
    This function demonstrates a complete, from-scratch implementation of the ItemCF algorithm.
    """
    # --- 第一步：准备数据 (建立“江湖档案库”) ---
    # 在真实世界里，这部分数据会来自您的数据库或日志文件。
    # 这里我们用一个简单的例子来模拟您的“花名册数据”。
    # 1 代表学生上过这门课。
    data = {
        '小明': {'数学': 1, '物理': 1, '化学': 0, '生物': 0, '历史': 0},
        '小红': {'数学': 1, '物理': 0, '化学': 1, '生物': 1, '历史': 0},
        '小强': {'数学': 0, '物理': 1, '化学': 1, '历史': 1, '生物': 0},
        '小美': {'数学': 0, '物理': 0, '化学': 1, '生物': 1, '历史': 1},
        '小华': {'数学': 1, '物理': 1, '化学': 1, '生物': 1, '历史': 0}
    }

    # 将数据转换为Pandas DataFrame格式，这是进行数据分析的常用工具。
    # 用户为行，课程为列，这就是我们的“用户-物品交互矩阵”。
    df = pd.DataFrame(data).T  # .T 表示转置，让用户成为行
    df = df.fillna(0)  # 将没有记录的NaN值填充为0

    print("--- 1. 用户-课程交互矩阵 (原始数据) ---")
    print(df)
    print("\n" + "=" * 50 + "\n")

    # --- 第二步：计算物品（课程）之间的相似度 (绘制“物品关系网”) ---
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
    courses_taken = user_courses[user_courses == 1].index.tolist()
    print(f"'{target_user}' 上过的课程: {courses_taken}\n")

    # 3.2 初始化一个空的推荐分数记录器
    recommendation_scores = {}

    # 3.3 遍历用户上过的每一门课
    for course_taken in courses_taken:
        # 找出与这门课相似的其他课程（来自我们第二步计算的相似度矩阵）
        similar_courses = item_similarity_df[course_taken]

        # 遍历这些相似的课程
        for course_similar, similarity_score in similar_courses.items():
            # 如果这门相似的课程，用户已经上过了，就跳过
            if course_similar in courses_taken:
                continue

            # 这就是ItemCF的核心：将相似度分数作为权重，进行累加
            # 如果一个课程与多门用户上过的课都相似，它的分数会不断累加，变得更高
            if course_similar not in recommendation_scores:
                recommendation_scores[course_similar] = 0
            recommendation_scores[course_similar] += similarity_score

    # 3.4 对推荐分数进行排序
    # 使用lambda函数，按字典的值（也就是分数）进行降序排序
    sorted_recommendations = sorted(recommendation_scores.items(), key=lambda item: item[1], reverse=True)

    print(f"--- 4. 为 '{target_user}' 生成的最终推荐列表 (按分数高低排序) ---")
    if not sorted_recommendations:
        print("没有可以推荐的课程。")
    else:
        for course, score in sorted_recommendations:
            print(f"课程: {course}, 推荐分数: {score:.4f}")


# 运行主函数
if __name__ == "__main__":
    item_based_collaborative_filtering()
