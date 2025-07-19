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
    # --- 第一步：准备“案发现场记录” (模拟数据) ---
    # 在真实世界里，这份数据来自您的数据库或日志系统。
    # 这里我们创建一个模拟数据，每一行代表一次“推荐事件”（一次课程曝光）。
    # 'clicked' 是我们的目标标签(Label)，1代表点击了，0代表没点击。
    data = {
        'user_id': ['u01', 'u01', 'u02', 'u03', 'u02', 'u03', 'u01', 'u04', 'u04', 'u05'],
        'user_grade': ['初一', '初一', '初二', '初三', '初二', '初三', '初一', '初二', '初二', '初一'],
        'user_city': ['北京', '北京', '上海', '广州', '上海', '广州', '北京', '北京', '北京', '上海'],
        'item_id': ['i01', 'i02', 'i01', 'i03', 'i04', 'i02', 'i03', 'i02', 'i04', 'i01'],
        'item_subject': ['数学', '英语', '数学', '物理', '英语', '英语', '物理', '英语', '英语', '数学'],
        'item_price': [199, 299, 199, 399, 299, 299, 399, 299, 299, 199],
        'clicked': [1, 0, 1, 1, 0, 1, 0, 1, 1, 0]  # 我们的“正确答案”
    }
    df = pd.DataFrame(data)

    print("--- 1. 原始的“案件记录表” (模拟数据) ---")
    print(df)
    print("\n" + "=" * 60 + "\n")

    # --- 第二步：加工“线索” (特征工程) ---
    # 这是LR模型落地最核心、最体现功力的一步！

    # 2.1 创造一个“交叉特征”，解决LR的“直脑筋”问题
    # 我们手动将用户的年级和课程的学科组合成一个新特征，让模型能学习到更深层的关系。
    df['grade_subject'] = df['user_grade'] + '_' + df['item_subject']

    print("--- 2. 特征工程：创建了'年级_学科'交叉特征 ---")
    print(df)
    print("\n" + "=" * 60 + "\n")

    # 2.2 定义哪些是数值特征，哪些是类别特征
    # user_id和item_id通常类别太多，在LR里直接用One-Hot会导致维度爆炸，
    # 这里我们暂时不把它们作为特征，但在更复杂的模型里它们会通过Embedding使用。
    numerical_features = ['item_price']
    categorical_features = ['user_grade', 'user_city', 'item_subject', 'grade_subject']

    # 2.3 分离特征(X)和目标(y)
    X = df[numerical_features + categorical_features]
    y = df['clicked']

    # 2.4 划分训练集和测试集 (80%做练习题，20%做期末考)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # --- 第三步：建立一个“特征处理流水线” (Pipeline) ---
    # 这是scikit-learn中非常优雅和强大的功能，能防止数据泄露，并简化代码。
    # 我们告诉它，对不同类型的特征，要用不同的“工具”去处理。

    # 为数值特征创建一个处理管道：先用StandardScaler进行标准化（让数据更稳定）
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # 为类别特征创建一个处理管道：用OneHotEncoder进行独热编码
    # handle_unknown='ignore' 意味着如果在预测时遇到训练时没见过的类别，就忽略它。
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # 用ColumnTransformer把上面两种处理方式“粘合”起来
    # 告诉它，哪些列用哪个管道处理
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # --- 第四步：正式“聘请”大侦探并开始训练 ---
    # 我们把预处理器和逻辑回归模型本身，也用一个Pipeline串起来！
    # 这样，整个流程就变成了一个全自动的“傻瓜相机”。
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('classifier', LogisticRegression(solver='liblinear'))])

    print("--- 3. 开始训练逻辑回归模型... ---")
    # 一行代码，完成所有特征处理和模型训练！
    model_pipeline.fit(X_train, y_train)
    print("模型训练完成！\n")
    print("=" * 60 + "\n")

    # --- 第五步：评估“破案”能力 (Model Evaluation) ---
    print("--- 4. 在测试集上评估模型性能 ---")
    y_pred = model_pipeline.predict(X_test)
    y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]  # 获取预测为1的概率

    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"AUC (排序能力指标): {auc:.4f}")
    print("混淆矩阵 (Confusion Matrix):")
    print(conf_matrix)
    print("\n" + "=" * 60 + "\n")

    # --- 第六步：模拟线上预测 (Deployment Simulation) ---
    print("--- 5. 模拟一次线上实时预测 ---")
    # 假设来了一个新的推荐请求
    new_data = pd.DataFrame([{
        'user_grade': '初一',
        'user_city': '北京',
        'item_subject': '物理',
        'item_price': 399,
        # 注意：交叉特征也要在这里生成！
        'grade_subject': '初一_物理'
    }])

    print("新请求的数据:")
    print(new_data)

    # 使用训练好的完整流水线进行预测
    prediction_probability = model_pipeline.predict_proba(new_data)[:, 1]

    print(f"\n模型预测该用户点击这门课的概率是: {prediction_probability[0]:.2%}")


# 运行主函数
if __name__ == "__main__":
    logistic_regression_recommendation_demo()