# 中美预测性警务政策文本数据分析
本项目基于Python对中美预测性警务政策文本进行关键词提取、词云绘制、关键词共现分析。

## 项目结构
- data/: 原始政策文本数据
- policy_analysis.py: 核心分析代码
- results/: 分析结果（图表/报告）

## 运行环境
Python 3.8+
依赖库：
- jieba (中文分词)
- wordcloud (词云绘制)
- pandas (数据处理)
- matplotlib (可视化)
- networkx (关键词共现图)

## 运行方法
1. 安装依赖：pip install jieba wordcloud pandas matplotlib networkx
2. 运行脚本：python policy_analysis.py

## 分析结论
- 中国政策核心：公安、警务、数据，侧重服务与治理
- 美国政策核心：数据、算法、犯罪、执法，侧重执法效率与法律约束
