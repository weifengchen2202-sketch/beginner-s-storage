import re
from collections import Counter

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

st.set_page_config(page_title="Tourism Intelligence Platform", page_icon="🌏", layout="wide")

st.markdown("""
<style>
.main .block-container {max-width:1280px;padding-top:2rem;padding-bottom:3rem}
[data-testid="stSidebar"] {background:#f4f7fb;border-right:1px solid #e5eaf2}
h1,h2,h3 {color:#172033;letter-spacing:0}
.hero {padding:26px 30px;border-radius:8px;background:linear-gradient(135deg,#0f766e 0%,#2563eb 55%,#1e293b 100%);color:white;margin-bottom:24px}
.hero h1 {color:white;font-size:38px;margin:0;line-height:1.15}
.hero p {color:#e8f3ff;font-size:16px;margin:10px 0 0}
.metric-card {padding:18px 20px;border:1px solid #e6eaf0;border-radius:8px;background:#fff;box-shadow:0 8px 24px rgba(15,23,42,.06);min-height:116px}
.metric-label {font-size:14px;color:#64748b;margin-bottom:8px}
.metric-value {font-size:25px;font-weight:700;color:#172033;line-height:1.2}
.metric-note,.section-note {font-size:13px;color:#64748b;margin-top:8px}
div[data-testid="stDataFrame"] {border:1px solid #e6eaf0;border-radius:8px;overflow:hidden}
.stTabs [data-baseweb="tab-list"] {gap:7px;border-bottom:1px solid #e6eaf0}
.stTabs [data-baseweb="tab"] {height:42px;padding:8px 14px;border-radius:8px 8px 0 0;background:#f8fafc}
.stTabs [aria-selected="true"] {background:#e0f2fe;color:#075985;font-weight:700}
</style>
""", unsafe_allow_html=True)

REVENUE_UNIT = "亿美元"
YEARS = list(range(2017, 2026))

# 课程项目示例参数：2019游客量、2019收入、区域、ISO3、经纬度。
# 正式研究时请通过左侧上传官方统计 CSV 替换。
PROFILES = {
    "China": (65700000, 1310, "Asia", "CHN", 35.86, 104.20),
    "Japan": (31882000, 462, "Asia", "JPN", 36.20, 138.25),
    "Korea": (17503000, 207, "Asia", "KOR", 35.91, 127.77),
    "Thailand": (39916000, 650, "Asia", "THA", 15.87, 100.99),
    "Singapore": (19116000, 290, "Asia", "SGP", 1.35, 103.82),
    "Malaysia": (26100000, 222, "Asia", "MYS", 4.21, 101.98),
    "Australia": (9460000, 470, "Oceania", "AUS", -25.27, 133.78),
    "USA": (79100000, 2610, "North America", "USA", 37.09, -95.71),
    "Canada": (22145000, 240, "North America", "CAN", 56.13, -106.35),
    "Mexico": (45024000, 248, "North America", "MEX", 23.63, -102.55),
    "France": (90914000, 710, "Europe", "FRA", 46.23, 2.21),
    "UK": (40200000, 450, "Europe", "GBR", 55.38, -3.44),
    "Italy": (64513000, 500, "Europe", "ITA", 41.87, 12.57),
    "Spain": (83509000, 735, "Europe", "ESP", 40.46, -3.75),
    "Germany": (39563000, 425, "Europe", "DEU", 51.17, 10.45),
    "Turkey": (51192000, 346, "Europe/Asia", "TUR", 38.96, 35.24),
    "UAE": (16730000, 340, "Middle East", "ARE", 23.42, 53.85),
    "Egypt": (13000000, 130, "Africa", "EGY", 26.82, 30.80),
    "South Africa": (10228000, 90, "Africa", "ZAF", -30.56, 22.94),
    "Brazil": (6353000, 59, "South America", "BRA", -14.24, -51.93),
}

YEAR_FACTOR = {2017: .94, 2018: .97, 2019: 1.0, 2020: .25, 2021: .30, 2022: .62, 2023: .82, 2024: .96, 2025: 1.05}
REVENUE_FACTOR = {2017: .92, 2018: .96, 2019: 1.0, 2020: .30, 2021: .35, 2022: .67, 2023: .86, 2024: 1.0, 2025: 1.08}


def default_data():
    rows = []
    for country, (base_t, base_r, region, iso3, lat, lon) in PROFILES.items():
        country_bias = 1 + ((sum(ord(c) for c in country) % 9) - 4) / 100
        for year in YEARS:
            rows.append({
                "Country": country,
                "Region": region,
                "ISO3": iso3,
                "Latitude": lat,
                "Longitude": lon,
                "Year": year,
                "Tourists": round(base_t * YEAR_FACTOR[year] * country_bias),
                "Revenue": round(base_r * REVENUE_FACTOR[year] * country_bias, 1),
            })
    return pd.DataFrame(rows)


def clean_data(data):
    required = {"Country", "Year", "Tourists", "Revenue"}
    if not required.issubset(data.columns):
        st.error("CSV 至少需要 Country、Year、Tourists、Revenue 四列。")
        st.stop()
    data = data.copy()
    data["Year"] = pd.to_numeric(data["Year"], errors="coerce")
    data["Tourists"] = pd.to_numeric(data["Tourists"], errors="coerce")
    data["Revenue"] = pd.to_numeric(data["Revenue"], errors="coerce")
    data = data.dropna(subset=["Country", "Year", "Tourists", "Revenue"])
    data["Year"] = data["Year"].astype(int)
    meta = pd.DataFrame([
        {"Country": k, "Region": v[2], "ISO3": v[3], "Latitude": v[4], "Longitude": v[5]}
        for k, v in PROFILES.items()
    ])
    for col in ["Region", "ISO3", "Latitude", "Longitude"]:
        if col not in data.columns:
            data = data.merge(meta[["Country", col]], on="Country", how="left")
    return data


def recovery_rate(data):
    before = data[data["Year"] <= 2019]["Tourists"].mean()
    after = data[data["Year"] >= 2021]["Tourists"].mean()
    return after / before * 100 if before > 0 and pd.notna(after) else None


def forecast(data, years, model_name):
    data = data.sort_values("Year")
    future = list(range(int(data["Year"].max()) + 1, int(data["Year"].max()) + years + 1))
    if model_name == "移动平均预测":
        window = min(3, len(data))
        return pd.DataFrame({"Year": future, "Predicted Tourists": data["Tourists"].tail(window).mean(), "Predicted Revenue": data["Revenue"].tail(window).mean()})
    model_data = data[~data["Year"].isin([2020, 2021])] if model_name == "排除疫情年份后线性回归" else data
    if len(model_data) < 2:
        model_data = data
    X, future_x = model_data[["Year"]], pd.DataFrame({"Year": future})
    if model_name == "多项式回归":
        tourist_model = make_pipeline(PolynomialFeatures(2), LinearRegression())
        revenue_model = make_pipeline(PolynomialFeatures(2), LinearRegression())
    else:
        tourist_model, revenue_model = LinearRegression(), LinearRegression()
    tourist_model.fit(X, model_data["Tourists"])
    revenue_model.fit(X, model_data["Revenue"])
    result = pd.DataFrame({"Year": future, "Predicted Tourists": tourist_model.predict(future_x), "Predicted Revenue": revenue_model.predict(future_x)})
    result[["Predicted Tourists", "Predicted Revenue"]] = result[["Predicted Tourists", "Predicted Revenue"]].clip(lower=0)
    return result


def keywords(comments):
    bank = ["风景", "服务", "交通", "价格", "排队", "人多", "干净", "方便", "预约", "体验", "贵", "拥挤", "精彩", "值得", "文化", "环境", "门票", "酒店", "餐饮"]
    count = Counter()
    for text in comments.dropna().astype(str):
        for word in bank:
            if word in text:
                count[word] += 1
        for word in re.findall(r"[A-Za-z]{3,}", text.lower()):
            count[word] += 1
    return pd.DataFrame(count.most_common(12), columns=["关键词", "出现次数"])


def report_text(data, country, total_tourists, total_revenue, model):
    if len(data) < 2:
        return "当前数据不足，无法生成完整报告。"
    growth = (data["Tourists"].iloc[-1] / data["Tourists"].iloc[0] - 1) * 100
    growth_table = data.assign(Growth=data["Tourists"].pct_change() * 100).dropna(subset=["Growth"])
    anomaly = growth_table.sort_values("Growth").iloc[0]
    recovery = recovery_rate(data)
    recovery_info = f"疫情后恢复率为 {recovery:.1f}%" if recovery is not None else "疫情恢复率数据不足"
    return f"""旅游数据分析报告

分析国家：{country}
分析年份：{data['Year'].min()}-{data['Year'].max()}
游客总量：{total_tourists:,.0f} 人次
旅游收入总额：{total_revenue:,.1f} {REVENUE_UNIT}
游客量整体变化：{growth:.1f}%
{recovery_info}
下降最明显年份：{int(anomaly['Year'])} 年（{anomaly['Growth']:.1f}%）
预测模型：{model}

说明：默认数据为课程项目示例数据，预测结果仅供趋势参考。"""


st.sidebar.title("⚙ 控制面板")
st.sidebar.caption("可使用默认数据，也可以上传自己的 CSV")
upload = st.sidebar.file_uploader("上传旅游数据 CSV", type=["csv"])
df = clean_data(pd.read_csv(upload) if upload else default_data())

country = st.sidebar.selectbox("选择国家", sorted(df["Country"].unique()))
year_range = st.sidebar.slider("选择年份范围", int(df["Year"].min()), int(df["Year"].max()), (int(df["Year"].min()), int(df["Year"].max())))
forecast_years = st.sidebar.slider("预测未来几年", 1, 5, 3)
forecast_model = st.sidebar.selectbox("选择预测模型", ["线性回归", "移动平均预测", "多项式回归", "排除疫情年份后线性回归"])
compare_countries = st.sidebar.multiselect("选择对比国家", sorted(df["Country"].unique()), default=[country])

country_data = df[(df["Country"] == country) & df["Year"].between(*year_range)].sort_values("Year").copy()
if country_data.empty:
    st.warning("当前筛选条件下没有数据。")
    st.stop()

st.markdown(f'<div class="hero"><h1>🌏 Tourism Intelligence Analytics Platform</h1><p>{country} · {year_range[0]}-{year_range[1]} · 数据上传、世界地图、统计分析、预测与评论洞察</p></div>', unsafe_allow_html=True)

total_tourists = country_data["Tourists"].sum()
avg_tourists = country_data["Tourists"].mean()
max_tourists = country_data["Tourists"].max()
total_revenue = country_data["Revenue"].sum()

for col, label, value, note in zip(st.columns(4), ["游客总量", "平均游客量", "最高游客量", "旅游收入总额"], [f"{total_tourists:,.0f}", f"{avg_tourists:,.0f}", f"{max_tourists:,.0f}", f"{total_revenue:,.1f}"], ["人次", "人次/年", "人次", REVENUE_UNIT]):
    with col:
        st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">单位：{note}</div></div>', unsafe_allow_html=True)

st.write("")
tabs = st.tabs(["📊 数据概览", "🗺️ 世界地图", "📈 趋势统计", "🌍 国家对比", "🔮 未来预测", "💬 评论口碑", "📘 数据说明", "🤖 自动报告"])

with tabs[0]:
    st.subheader("当前国家数据")
    display = country_data[[c for c in ["Country", "Region", "Year", "Tourists", "Revenue"] if c in country_data]].rename(columns={"Country":"国家", "Region":"地区", "Year":"年份", "Tourists":"游客人数（人次）", "Revenue":f"旅游收入（{REVENUE_UNIT}）"})
    st.dataframe(display, use_container_width=True, hide_index=True)
    latest = df[df["Year"] == df["Year"].max()].sort_values("Tourists", ascending=False)
    st.subheader("最新年份热门目的地排行")
    st.dataframe(latest[["Country", "Region", "Tourists", "Revenue"]], use_container_width=True, hide_index=True)
    st.download_button("下载当前旅游数据", df.to_csv(index=False).encode("utf-8-sig"), "tourism_data.csv", "text/csv")

with tabs[1]:
    st.subheader("世界地图交互分析")
    map_year = st.slider("选择地图年份", int(df["Year"].min()), int(df["Year"].max()), int(df["Year"].max()), key="map_year")
    map_data = df[df["Year"] == map_year].dropna(subset=["Latitude", "Longitude"])
    fig_map = px.scatter_geo(map_data, lat="Latitude", lon="Longitude", size="Tourists", color="Revenue", hover_name="Country", hover_data={"Region":True, "Tourists":":,.0f", "Revenue":":.1f", "Latitude":False, "Longitude":False}, projection="natural earth", color_continuous_scale="Tealrose", title=f"{map_year} 年世界旅游分布")
    fig_map.update_layout(height=570, margin=dict(l=0, r=0, t=45, b=0))
    st.plotly_chart(fig_map, use_container_width=True)
    map_country = st.selectbox("选择地图国家并查看明细", sorted(map_data["Country"].unique()))
    st.dataframe(df[df["Country"] == map_country].sort_values("Year"), use_container_width=True, hide_index=True)

with tabs[2]:
    growth = country_data.copy()
    growth["游客同比增长率（%）"] = growth["Tourists"].pct_change() * 100
    growth["收入同比增长率（%）"] = growth["Revenue"].pct_change() * 100
    st.subheader("同比增长与异常年份")
    st.dataframe(growth, use_container_width=True, hide_index=True)
    valid = growth.dropna(subset=["游客同比增长率（%）"])
    if not valid.empty:
        worst = valid.loc[valid["游客同比增长率（%）"].idxmin()]
        best = valid.loc[valid["游客同比增长率（%）"].idxmax()]
        c1, c2 = st.columns(2)
        c1.metric("下降最明显年份", int(worst["Year"]), f'{worst["游客同比增长率（%）"]:.1f}%')
        c2.metric("增长最明显年份", int(best["Year"]), f'{best["游客同比增长率（%）"]:.1f}%')
    long_data = country_data.melt(id_vars="Year", value_vars=["Tourists", "Revenue"], var_name="指标", value_name="数值")
    st.plotly_chart(px.line(long_data, x="Year", y="数值", color="指标", markers=True, facet_row="指标", title=f"{country} 趋势分析"), use_container_width=True)
    corr = country_data["Tourists"].corr(country_data["Revenue"])
    st.metric("游客量与旅游收入相关系数", f"{corr:.3f}")
    recovery = recovery_rate(country_data)
    if recovery is not None:
        st.metric("疫情后恢复率", f"{recovery:.1f}%")

with tabs[3]:
    st.subheader("多国家对比")
    if compare_countries:
        compare = df[df["Country"].isin(compare_countries) & df["Year"].between(*year_range)]
        summary = compare.groupby("Country", as_index=False).agg(游客总量=("Tourists", "sum"), 平均游客量=("Tourists", "mean"), 旅游收入总额=("Revenue", "sum"))
        st.dataframe(summary.sort_values("游客总量", ascending=False), use_container_width=True, hide_index=True)
        st.plotly_chart(px.line(compare, x="Year", y="Tourists", color="Country", markers=True, title="游客量趋势对比"), use_container_width=True)
        st.plotly_chart(px.line(compare, x="Year", y="Revenue", color="Country", markers=True, title="旅游收入趋势对比"), use_container_width=True)
    else:
        st.info("请在左侧选择至少一个对比国家。")

with tabs[4]:
    st.subheader("未来发展预测")
    predicted = forecast(country_data, forecast_years, forecast_model)
    st.caption(f"当前模型：{forecast_model}。预测仅供趋势参考。")
    st.dataframe(predicted.rename(columns={"Year":"预测年份", "Predicted Tourists":"预测游客人数（人次）", "Predicted Revenue":f"预测收入（{REVENUE_UNIT}）"}), use_container_width=True, hide_index=True)
    actual_chart = country_data[["Year", "Tourists"]].rename(columns={"Tourists":"游客人数"}).assign(类型="实际")
    forecast_chart = predicted[["Year", "Predicted Tourists"]].rename(columns={"Predicted Tourists":"游客人数"}).assign(类型="预测")
    st.plotly_chart(px.line(pd.concat([actual_chart, forecast_chart]), x="Year", y="游客人数", color="类型", markers=True, title=f"{country} 游客量预测"), use_container_width=True)

with tabs[5]:
    st.subheader("景点评论口碑分析")
    review_file = st.file_uploader("上传评论 CSV（Place、Date、Rating、Comment）", type=["csv"], key="reviews")
    if review_file:
        reviews = pd.read_csv(review_file)
        required = {"Place", "Date", "Rating", "Comment"}
        if required.issubset(reviews.columns):
            reviews["Rating"] = pd.to_numeric(reviews["Rating"], errors="coerce")
            reviews = reviews.dropna(subset=["Place", "Rating", "Comment"])
            reviews["评论类型"] = reviews["Rating"].apply(lambda x: "好评" if x >= 4 else ("中评" if x == 3 else "差评"))
            place = st.selectbox("选择景点", sorted(reviews["Place"].unique()))
            selected = reviews[reviews["Place"] == place]
            positive = (selected["评论类型"] == "好评").sum()
            cols = st.columns(4)
            cols[0].metric("评论总数", len(selected)); cols[1].metric("平均评分", f'{selected["Rating"].mean():.2f}'); cols[2].metric("好评数量", positive); cols[3].metric("好评率", f'{positive / len(selected) * 100:.1f}%')
            st.bar_chart(selected["评论类型"].value_counts())
            k1, k2 = st.columns(2)
            k1.write("好评关键词"); k1.dataframe(keywords(selected[selected["评论类型"] == "好评"]["Comment"]), hide_index=True)
            k2.write("差评关键词"); k2.dataframe(keywords(selected[selected["评论类型"] == "差评"]["Comment"]), hide_index=True)
            st.dataframe(selected, use_container_width=True, hide_index=True)
        else:
            st.error("评论文件缺少必要列。")
    else:
        st.info("可上传仓库中的 reviews_template.csv 体验评论分析。")

with tabs[6]:
    st.subheader("数据来源与字段说明")
    st.markdown(f"""
- `Country`：国家名称
- `Year`：年份
- `Tourists`：游客人数，单位为人次
- `Revenue`：旅游收入，当前单位为{REVENUE_UNIT}
- `Region`、`ISO3`、`Latitude`、`Longitude`：用于地区与地图分析

默认数据为课程项目示例数据，由程序内置参数生成。正式研究时，请从官方统计机构或可靠公开数据库取得数据，再通过左侧上传 CSV。预测结果会受到样本量、异常年份和模型选择影响。
""")

with tabs[7]:
    st.subheader("自动分析报告")
    text = report_text(country_data, country, total_tourists, total_revenue, forecast_model)
    st.success(text)
    st.download_button("下载分析报告", text, f"{country}_tourism_report.txt", "text/plain")
