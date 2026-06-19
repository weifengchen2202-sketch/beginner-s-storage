# Tourism Intelligence Analytics Platform

基于 Streamlit 的旅游数据分析平台，适合作为统计学与数据分析课程项目。

## 功能

- 2017-2025 年多国家旅游数据展示
- 世界地图交互分析
- 多国家和地区对比
- 同比增长率、相关性与异常年份分析
- 线性回归、移动平均、多项式回归预测
- 评论分类、好评率与关键词分析
- 自定义 CSV 数据上传
- 自动分析报告生成与下载

## 安装与运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 数据文件

- `tourism_expanded_2017_2025.csv`：默认旅游数据
- `reviews_template.csv`：评论分析示例文件

## 单位

- `Tourists`：人次
- `Revenue`：亿美元

默认数据用于课程项目和功能演示。正式研究时应替换为官方统计数据，预测结果仅供趋势参考。
