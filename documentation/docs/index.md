# 推荐系统学习指南

!!! info "项目简介"
    一个完整的推荐系统学习指南，从基础理论到实践应用的全面教程。本项目是一个系统性的推荐系统学习资源，旨在帮助学习者从零开始掌握推荐系统的核心概念、算法原理和实际应用。

## 🎯 学习目标

通过本指南的学习，您将能够：

- :material-lightbulb: **理解推荐系统的核心思想和架构设计**
- :material-cog: **掌握特征工程和Embedding技术**  
- :material-algorithm: **学习经典推荐算法（ItemCF、逻辑回归等）**
- :material-brain: **了解现代推荐系统与LLM的结合**
- :material-ruler: **掌握相似度计算和召回排序策略**
- :material-rocket: **能够独立实现和优化推荐算法**

## 📖 学习路径

### :material-book-open: 第一阶段：理论基础

| 章节 | 主题 | 内容概要 |
|-----|------|---------|
| 第1章 | [推荐系统核心思想与架构](theory/01-core-concepts.md) | 推荐系统的基本概念和整体架构 |
| 第2章 | [特征工程：从原始数据到模型输入](theory/02-feature-engineering.md) | 数据预处理和特征构建技术 |
| 第3章 | [揭秘Embedding魔法](theory/03-embedding-magic.md) | 向量化表示和嵌入技术 |

### :material-trending-up: 第二阶段：算法进阶

| 章节 | 主题 | 内容概要 |
|-----|------|---------|
| 第4章 | [当推荐系统遇见最强大脑LLM](advanced/04-llm-recommendation.md) | 大语言模型在推荐系统中的应用 |
| 第5章 | [推荐模型的进化史](theory/05-model-evolution.md) | 推荐算法的发展历程和演进 |
| 第6章 | [相似度计算：推荐的度量衡](theory/06-similarity-computation.md) | 相似度计算方法和度量指标 |

### :material-code-tags: 第三阶段：实践应用

| 章节 | 主题 | 内容概要 |
|-----|------|---------|
| 第7章 | [ItemCF算法深度解析](algorithms/07-itemcf-analysis.md) | 基于物品的协同过滤算法 |
| 第8章 | [推荐系统的两步走战略：召回与排序](algorithms/08-recall-ranking.md) | 召回和排序的策略和实现 |
| 第9章 | [逻辑回归深度解析](algorithms/09-logistic-regression.md) | 逻辑回归在推荐系统中的应用 |

## 🛠️ 技术栈

=== "算法实现"
    - **编程语言**: Python
    - **核心库**: NumPy, Pandas, Scikit-learn
    - **深度学习**: TensorFlow, PyTorch

=== "核心算法"
    - **协同过滤**: ItemCF, UserCF
    - **内容过滤**: 基于内容的推荐
    - **深度学习**: 神经网络推荐模型
    - **混合方法**: 多种算法融合

=== "现代技术"
    - **Embedding**: Word2Vec, Node2Vec
    - **深度学习**: DNN, Wide&Deep
    - **LLM集成**: 大语言模型增强推荐
    - **实时推荐**: 在线学习和更新

## 🚀 快速开始

!!! tip "推荐学习方式"
    建议按照学习路径的顺序进行学习，每章都包含理论讲解和代码实践。

### 环境准备

```bash
# 1. 克隆项目
git clone https://github.com/YaronCoding/recommendation-system-guide.git
cd recommendation-system-guide

# 2. 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt
```

### 开始学习

1. **理论学习**: 从第1章开始，按顺序学习基础理论
2. **代码实践**: 运行每章对应的代码示例
3. **项目实战**: 使用学到的知识构建自己的推荐系统
4. **交流讨论**: 在项目 Issues 中提问和分享心得

## 📁 项目结构

```
recommendation-system-guide/
├── documentation/          # MkDocs 文档项目
│   ├── docs/              # 文档源文件
│   ├── mkdocs.yml         # MkDocs 配置
│   └── requirements.txt   # 文档依赖
├── doc/                   # 原始Markdown文档
├── code/                  # 代码实现
│   ├── ItemCF/           # ItemCF算法实现
│   └── LR/               # 逻辑回归实现
├── requirements.txt       # 项目依赖
└── README.md             # 项目说明
```

## 🌟 特色功能

- **📖 系统化学习路径**: 从基础到高级的完整学习体系
- **💻 代码实战**: 每个算法都有完整的Python实现
- **🎨 可视化展示**: 丰富的图表和示例帮助理解
- **🔗 在线文档**: 支持搜索、主题切换的现代化文档站点
- **📱 响应式设计**: 在手机、平板、电脑上都有完美体验

## 🤝 贡献指南

欢迎所有形式的贡献！

- **🐛 报告问题**: 发现bug或改进建议请提交 Issue
- **💡 功能建议**: 欢迎提出新的功能想法
- **📝 内容贡献**: 改进文档内容或添加新章节
- **🔧 代码贡献**: 优化算法实现或添加新算法

## 📞 联系方式

- **GitHub**: [@YaronCoding](https://github.com/YaronCoding)
- **Email**: yangyaron9@gmail.com
- **Issues**: [项目Issue页面](https://github.com/YaronCoding/recommendation-system-guide/issues)

---

!!! success "开始学习"
    准备好了吗？让我们从 [第1章：推荐系统核心思想与架构](theory/01-core-concepts.md) 开始您的推荐系统学习之旅！ 🚀 