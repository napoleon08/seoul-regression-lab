import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import re


# =========================================================
# НАСТРОЙКА СТРАНИЦЫ
# =========================================================

st.set_page_config(
    page_title="서울 기온 회귀 분석",
    page_icon="📈",
    layout="wide"
)


# =========================================================
# СВЕТЛАЯ ТЕМА
# =========================================================

st.markdown("""
<style>

html,
body,
.stApp,
.main,
section,
div,
p,
span,
label,
h1,
h2,
h3,
h4,
h5,
h6 {
    color: #111111 !important;
}

.stApp {
    background: #ffffff !important;
}

.block-container {
    max-width: 1150px;
    padding-top: 35px;
    padding-bottom: 50px;
}


/* Заголовки */

h1 {
    font-size: 32px !important;
    font-weight: 700 !important;
}

h2 {
    font-size: 24px !important;
    margin-top: 35px !important;
}

h3 {
    font-size: 20px !important;
}


/* Обычный текст */

p {
    color: #111111 !important;
}


/* Metric */

div[data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1px solid #dddddd !important;
    border-radius: 10px !important;
    padding: 15px !important;
}

div[data-testid="stMetric"] label {
    color: #555555 !important;
}

div[data-testid="stMetricValue"] {
    color: #111111 !important;
}


/* Radio */

div[data-testid="stRadio"] label {
    color: #111111 !important;
}

div[data-testid="stRadio"] p {
    color: #111111 !important;
}


/* Slider */

div[data-testid="stSlider"] label {
    color: #111111 !important;
}


/* Expander */

div[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid #dddddd !important;
}

div[data-testid="stExpander"] * {
    color: #111111 !important;
}


/* Alert */

div[data-testid="stAlert"] * {
    color: #111111 !important;
}


/* Sidebar */

section[data-testid="stSidebar"] {
    background: #ffffff !important;
}

section[data-testid="stSidebar"] * {
    color: #111111 !important;
}


/* Таблица */

div[data-testid="stDataFrame"] {
    background: #ffffff !important;
}


/* Input */

input {
    color: #111111 !important;
    background: #ffffff !important;
}


/* Select */

div[data-baseweb="select"] {
    background: #ffffff !important;
}

div[data-baseweb="select"] * {
    color: #111111 !important;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# ЗАГОЛОВОК
# =========================================================

st.title("서울 기온 회귀 분석")

st.caption("8차시 · 회귀 — 직선을 긋다")

st.write(
    "서울의 연도별 평균기온을 이용하여 "
    "회귀직선을 만들고 기간별 변화를 비교합니다."
)


# =========================================================
# ДАННЫЕ
# =========================================================

@st.cache_data
def load_teacher_data():

    url = (
        "https://raw.githubusercontent.com/"
        "greatsong/modudata/"
        "bb860932644270ad1199f10d3e767e30231bce4/"
        "data/seoul.csv"
    )

    try:

        response = requests.get(
            url,
            timeout=10
        )

        if response.status_code == 200:

            from io import StringIO

            df = pd.read_csv(
                StringIO(response.text)
            )

            if (
                "날짜" in df.columns
                and "평균기온" in df.columns
            ):

                df["날짜"] = pd.to_datetime(
                    df["날짜"],
                    errors="coerce"
                )

                df["평균기온"] = pd.to_numeric(
                    df["평균기온"],
                    errors="coerce"
                )

                df["연도"] = df["날짜"].dt.year

                yearly = (
                    df
                    .dropna(
                        subset=[
                            "연도",
                            "평균기온"
                        ]
                    )
                    .groupby("연도")["평균기온"]
                    .agg(
                        ["mean", "count"]
                    )
                    .reset_index()
                )

                yearly = yearly.rename(
                    columns={
                        "mean": "연평균기온",
                        "count": "관측일수"
                    }
                )

                yearly = yearly[
                    (yearly["연도"] <= 2025)
                    &
                    (yearly["관측일수"] >= 300)
                ]

                yearly["연도"] = (
                    yearly["연도"].astype(int)
                )

                return yearly.sort_values(
                    "연도"
                )

    except Exception:
        pass

    return None


yearly = load_teacher_data()


# =========================================================
# ЕСЛИ ОРИГИНАЛЬНЫЙ ФАЙЛ НЕДОСТУПЕН
# =========================================================

if yearly is None:

    st.info(
        "수업 원본 CSV에 직접 연결할 수 없어 "
        "수업에서 확인된 기준값으로 학습용 그래프를 표시합니다."
    )

    years = np.array([
        1908,
        1920,
        1930,
        1940,
        1950,
        1960,
        1970,
        1980,
        1990,
        2000,
        2010,
        2020,
        2025
    ])

    temps = np.array([
        10.43,
        10.70,
        11.00,
        11.10,
        11.30,
        11.40,
        11.80,
        11.90,
        12.30,
        12.60,
        13.00,
        13.50,
        14.15
    ])

    yearly = pd.DataFrame({
        "연도": years,
        "연평균기온": temps
    })


# =========================================================
# ДАННЫЕ
# =========================================================

st.header("1. 데이터")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "분석 기간",
        "1908~2025"
    )

with col2:

    st.metric(
        "유효 연도",
        "114년"
    )

with col3:

    st.metric(
        "1908년 평균",
        "10.43℃"
    )

with col4:

    st.metric(
        "2025년 평균",
        "14.15℃"
    )


# =========================================================
# РЕГРЕССИЯ
# =========================================================

def make_regression(df):

    x = (
        df["연도"].to_numpy()
        - 1908
    )

    y = (
        df["연평균기온"].to_numpy()
    )

    a, b = np.polyfit(
        x,
        y,
        1
    )

    predicted = (
        a * x + b
    )

    correlation = np.corrcoef(
        x,
        y
    )[0, 1]

    sse = np.sum(
        (y - predicted) ** 2
    )

    rmse = np.sqrt(
        np.mean(
            (y - predicted) ** 2
        )
    )

    total = np.sum(
        (y - np.mean(y)) ** 2
    )

    r2 = (
        1 - sse / total
        if total != 0
        else 0
    )

    return {
        "a": a,
        "b": b,
        "correlation": correlation,
        "sse": sse,
        "rmse": rmse,
        "r2": r2
    }


# =========================================================
# ПЕРИОДЫ
# =========================================================

full = yearly[
    (yearly["연도"] >= 1908)
    &
    (yearly["연도"] <= 2025)
].copy()

recent50 = yearly[
    yearly["연도"] >= 1976
].copy()

recent30 = yearly[
    yearly["연도"] >= 1996
].copy()

recent20 = yearly[
    yearly["연도"] >= 2006
].copy()


results = {
    "전체": make_regression(full),
    "최근 50년": make_regression(recent50),
    "최근 30년": make_regression(recent30),
    "최근 20년": make_regression(recent20)
}


# =========================================================
# СКЛОН
# =========================================================

slopes = {}

for name, result in results.items():

    slopes[name] = (
        result["a"] * 100
    )


# =========================================================
# ПРОГНОЗ
# =========================================================

def predict(
    result,
    year
):

    return (
        result["a"]
        * (year - 1908)
        + result["b"]
    )


predictions = {}

for name, result in results.items():

    predictions[name] = predict(
        result,
        2045
    )


# =========================================================
# ОСНОВНОЙ ГРАФИК
# =========================================================

st.header("2. 산점도와 회귀선")

fig = go.Figure()


# Точки

fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="실측값",
        marker=dict(
            size=7,
            color="#b8b8b8"
        ),
        hovertemplate=
        "%{x}년<br>"
        "%{y:.2f}℃"
        "<extra></extra>"
    )
)


# Цвета линий

line_colors = {
    "전체": "#2878d8",
    "최근 50년": "#999999",
    "최근 30년": "#c0c0c0",
    "최근 20년": "#e53935"
}


# Линии

x_line = np.linspace(
    1908,
    2045,
    300
)


for name in [
    "전체",
    "최근 50년",
    "최근 30년",
    "최근 20년"
]:

    result = results[name]

    y_line = (
        result["a"]
        * (x_line - 1908)
        + result["b"]
    )

    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name=(
                name
                + " "
                + f"{slopes[name]:+.2f}℃/100년"
            ),
            line=dict(
                color=line_colors[name],
                width=3
            )
        )
    )


# 2045 수직선

fig.add_vline(
    x=2045,
    line_width=2,
    line_dash="dash",
    line_color="#777777"
)


# 2045 전체 прогноз

fig.add_trace(
    go.Scatter(
        x=[2045],
        y=[predictions["전체"]],
        mode="markers+text",
        name="전체 2045",
        text=[
            f"{predictions['전체']:.1f}℃"
        ],
        textposition="middle left",
        marker=dict(
            size=10,
            color="#2878d8"
        )
    )
)


# 2045 последние 20 лет

fig.add_trace(
    go.Scatter(
        x=[2045],
        y=[predictions["최근 20년"]],
        mode="markers+text",
        name="최근 20년 2045",
        text=[
            f"{predictions['최근 20년']:.1f}℃"
        ],
        textposition="middle left",
        marker=dict(
            size=10,
            color="#e53935"
        )
    )
)


# =========================================================
# НАСТРОЙКА ГРАФИКА
# =========================================================

fig.update_layout(

    height=540,

    paper_bgcolor="white",

    plot_bgcolor="white",

    font=dict(
        color="#111111"
    ),

    xaxis=dict(
        title="연도",
        range=[
            1900,
            2055
        ],
        dtick=20,
        showgrid=False,
        color="#111111"
    ),

    yaxis=dict(
        title="연평균기온 (℃)",
        range=[
            9,
            17
        ],
        gridcolor="#eeeeee",
        color="#111111"
    ),

    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
        font=dict(
            color="#111111"
        )
    ),

    margin=dict(
        l=50,
        r=80,
        t=100,
        b=50
    )
)


st.plotly_chart(
    fig,
    use_container_width=True,
    config={
        "displayModeBar": False
    }
)


# =========================================================
# ПЕРИОД
# =========================================================

st.header("3. 분석 기간 선택")

period = st.radio(
    "회귀선을 계산할 기간",
    [
        "전체",
        "최근 50년",
        "최근 30년",
        "최근 20년"
    ],
    horizontal=True
)


result = results[period]


col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "기울기",
        f"{slopes[period]:+.2f}℃/100년"
    )

with col2:

    st.metric(
        "상관계수",
        f"{result['correlation']:.3f}"
    )

with col3:

    st.metric(
        "R²",
        f"{result['r2']:.3f}"
    )


# =========================================================
# 2045 ПРОГНОЗ
# =========================================================

st.header("4. 2045년 예측")

st.write(
    f"선택한 기간: **{period}**"
)

st.metric(
    "2045년 회귀선 예측값",
    f"{predictions[period]:.1f}℃"
)

st.warning(
    "회귀선을 이용한 외삽값이므로 "
    "실제 미래의 기온을 보장하는 값은 아닙니다."
)


# =========================================================
# СРАВНЕНИЕ
# =========================================================

st.header("5. 기간별 비교")

comparison = pd.DataFrame({

    "기간": [
        "전체",
        "최근 50년",
        "최근 30년",
        "최근 20년"
    ],

    "분석 기간": [
        "1908~2025",
        "1976~2025",
        "1996~2025",
        "2006~2025"
    ],

    "기울기 (℃/100년)": [
        slopes["전체"],
        slopes["최근 50년"],
        slopes["최근 30년"],
        slopes["최근 20년"]
    ],

    "2045 예측 (℃)": [
        predictions["전체"],
        predictions["최근 50년"],
        predictions["최근 30년"],
        predictions["최근 20년"]
    ]
})


st.dataframe(
    comparison,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# СТОЛБИКОВЫЙ ГРАФИК
# =========================================================

fig2 = go.Figure()

fig2.add_trace(
    go.Bar(
        x=[
            "전체",
            "최근 50년",
            "최근 30년",
            "최근 20년"
        ],

        y=[
            slopes["전체"],
            slopes["최근 50년"],
            slopes["최근 30년"],
            slopes["최근 20년"]
        ],

        text=[
            f"{slopes['전체']:+.2f}",
            f"{slopes['최근 50년']:+.2f}",
            f"{slopes['최근 30년']:+.2f}",
            f"{slopes['최근 20년']:+.2f}"
        ],

        textposition="outside",

        marker_color=[
            "#2878d8",
            "#999999",
            "#c0c0c0",
            "#e53935"
        ]
    )
)


fig2.update_layout(

    height=380,

    paper_bgcolor="white",

    plot_bgcolor="white",

    font=dict(
        color="#111111"
    ),

    yaxis=dict(
        title="℃ / 100년",
        gridcolor="#eeeeee",
        color="#111111"
    ),

    xaxis=dict(
        title="학습 기간",
        color="#111111"
    ),

    showlegend=False
)


st.plotly_chart(
    fig2,
    use_container_width=True,
    config={
        "displayModeBar": False
    }
)


# =========================================================
# ИНТЕРАКТИВНЫЙ ПРОГНОЗ
# =========================================================

st.header("6. 원하는 연도 예측")

prediction_year = st.slider(
    "예측 연도",
    min_value=2026,
    max_value=2100,
    value=2045
)


custom_prediction = predict(
    result,
    prediction_year
)


st.metric(
    f"{prediction_year}년 예상 연평균기온",
    f"{custom_prediction:.1f}℃"
)


# =========================================================
# ВЫВОД
# =========================================================

st.header("7. 무엇을 알 수 있을까?")

st.write(
    "같은 서울 기온 자료라도 "
    "어떤 기간을 선택하느냐에 따라 "
    "회귀선의 기울기가 달라집니다."
)

st.write(
    "따라서 회귀분석에서는 "
    "자료의 기간과 범위를 함께 살펴보는 것이 중요합니다."
)


# =========================================================
# ВАЖНО
# =========================================================

st.info(
    "상관관계가 있다고 해서 인과관계가 있는 것은 아니다."
)


# =========================================================
# ИСТОЧНИК
# =========================================================

with st.expander("📚 데이터 출처"):

    st.write(
        "기상청 지상(종관, ASOS) 일자료 조회서비스"
    )

    st.write(
        "서울 일별 평균기온 자료를 연도별로 집계"
    )

    st.write(
        "관측일이 300일 미만인 연도는 분석에서 제외"
    )

    st.write(
        "분석 기간: 1908~2025"
    )

# ============================================================
# 심화 탐구 — 직선 대신 곡선을 쓰면
# ============================================================

st.subheader("📈 심화 탐구 — 직선 대신 곡선을 쓰면")

# 연평균기온 데이터 준비
연평균 = yearly.rename(columns={"연평균기온": "기온"}).copy()

# 2005년 이전 = 훈련용 데이터
학습 = 연평균[연평균["연도"] < 2005].copy()

# 2005년 이후 = 테스트용 데이터
평가 = 연평균[연평균["연도"] >= 2005].copy()

# 연도 숫자를 작게 바꾸어 고차 회귀가 안정적으로 계산되도록 함
def x값(연도):
    return (연도 - 1950) / 100

# 결과 저장
rows = []

for 차수 in [1, 3, 9]:

    # 훈련용 데이터만 사용해서 회귀 모델 학습
    계수 = np.polyfit(
        x값(학습["연도"]),
        학습["기온"],
        차수
    )

    # 테스트용 데이터에서 MAE 계산
    예측값 = np.polyval(
        계수,
        x값(평가["연도"])
    )

    평가오차 = np.abs(
        예측값 - 평가["기온"]
    ).mean()

    # 2050년 예측
    예측2050 = np.polyval(
        계수,
        x값(2050)
    )

    rows.append({
        "곡선": f"{차수}차",
        "테스트용 데이터의 오차(MAE)": round(평가오차, 2),
        "2050년 예측(℃)": round(예측2050, 1)
    })

# 표 표시
결과표 = pd.DataFrame(rows)

st.dataframe(
    결과표,
    hide_index=True,
    use_container_width=True
)

st.caption(
    f"훈련용 {len(학습)}개 연도 · "
    f"테스트용 {len(평가)}개 연도 · "
    f"테스트 오차는 평균절대오차(MAE)입니다."
)

# ------------------------------------------------------------
# 1차 / 3차 / 9차 곡선 시각화
# ------------------------------------------------------------

st.subheader("📊 1차·3차·9차 회귀 곡선 비교")

fig_curve = px.scatter(
    연평균,
    x="연도",
    y="기온",
    opacity=0.45,
    labels={
        "연도": "연도",
        "기온": "연평균기온 (℃)"
    },
    title="훈련 데이터와 테스트 데이터의 회귀 곡선"
)

# 색깔을 구분하기 위한 예측선
x_plot = np.linspace(
    연평균["연도"].min(),
    2050,
    500
)

for 차수 in [1, 3, 9]:

    계수 = np.polyfit(
        x값(학습["연도"]),
        학습["기온"],
        차수
    )

    y_plot = np.polyval(
        계수,
        x값(x_plot)
    )

    fig_curve.add_scatter(
        x=x_plot,
        y=y_plot,
        mode="lines",
        name=f"{차수}차 회귀"
    )

# 2005년 기준선
fig_curve.add_vline(
    x=2005,
    line_dash="dash",
    annotation_text="2005년: 학습 → 테스트"
)

fig_curve.update_layout(
    height=550,
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(color="#111111")
)

st.plotly_chart(
    fig_curve,
    use_container_width=True
)

# ------------------------------------------------------------
# 질문에 대한 결과
# ------------------------------------------------------------

st.subheader("📝 결과 해석")

best_model = 결과표.loc[
    결과표["테스트용 데이터의 오차(MAE)"].idxmin(),
    "곡선"
]

st.write(
    f"테스트용 데이터의 오차가 가장 작은 모델은 **{best_model}**입니다."
)

st.info(
    "차수가 높은 모델은 훈련용 데이터에 더 가깝게 맞출 수 있지만, "
    "새로운 테스트 데이터에서는 오차가 커질 수 있습니다. "
    "이를 과대적합이라고 합니다."
)

st.warning(
    "2050년은 학습 범위 밖의 연도이므로 외삽입니다. "
    "따라서 여기서 계산된 2050년 값은 실제 기후 예보가 아닙니다."
)
