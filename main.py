import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import re
import json


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="서울 기온 회귀 분석",
    page_icon="📈",
    layout="wide"
)


# =========================================================
# STYLE
# =========================================================

st.markdown("""
<style>

.block-container {
    max-width: 1150px;
    padding-top: 35px;
    padding-bottom: 50px;
}

h1 {
    font-size: 32px !important;
    font-weight: 700 !important;
}

h2 {
    margin-top: 32px !important;
    font-size: 23px !important;
}

div[data-testid="stMetric"] {
    border: 1px solid #e6e6e6;
    border-radius: 10px;
    padding: 12px;
    background: #ffffff;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# TITLE
# =========================================================

st.title("📈 서울 기온 회귀 분석")

st.caption("8차시 · 회귀 — 직선을 긋다")

st.write(
    "서울의 연도별 평균기온과 회귀선을 비교합니다."
)


# =========================================================
# DATA LOADER
# =========================================================

@st.cache_data(ttl=3600)
def load_data():

    # -----------------------------------------------------
    # 1. 수업 원본 데이터
    # -----------------------------------------------------

    teacher_url = (
        "https://raw.githubusercontent.com/"
        "greatsong/modudata/"
        "bb860932644270ad1199f10d3e767e30231bce4/"
        "data/seoul.csv"
    )

    try:

        response = requests.get(
            teacher_url,
            timeout=10
        )

        if response.status_code == 200:

            from io import StringIO

            df = pd.read_csv(
                StringIO(
                    response.text
                )
            )

            return make_yearly(df)

    except Exception:
        pass


    # -----------------------------------------------------
    # 2. DATA CLOCK KOREA
    # -----------------------------------------------------

    data_clock_url = (
        "https://www.dataclockkorea.com/"
        "climate/seoul/"
    )

    try:

        response = requests.get(
            data_clock_url,
            timeout=10,
            headers={
                "User-Agent":
                "Mozilla/5.0"
            }
        )

        if response.status_code == 200:

            html = response.text

            # 여러 형태의 JSON 데이터 탐색
            patterns = [

                r'"year"\s*:\s*(19\d{2}|20\d{2})'
                r'\s*,\s*"value"\s*:\s*([0-9.]+)',

                r'"연도"\s*:\s*(19\d{2}|20\d{2})'
                r'\s*,\s*"연평균기온"\s*:\s*([0-9.]+)',

                r'"year"\s*:\s*"(19\d{2}|20\d{2})"'
                r'\s*,\s*"value"\s*:\s*([0-9.]+)'
            ]

            rows = []

            for pattern in patterns:

                matches = re.findall(
                    pattern,
                    html
                )

                if len(matches) >= 20:

                    for year, temp in matches:

                        rows.append({
                            "연도": int(year),
                            "연평균기온": float(temp)
                        })

                    break

            if len(rows) >= 20:

                df = pd.DataFrame(rows)

                df = (
                    df
                    .drop_duplicates(
                        subset=["연도"]
                    )
                    .sort_values("연도")
                )

                return df

    except Exception:
        pass


    return None


# =========================================================
# DAILY → YEARLY
# =========================================================

def make_yearly(df):

    if (
        "날짜" not in df.columns
        or "평균기온" not in df.columns
    ):

        return None

    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce"
    )

    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    df["연도"] = df["날짜"].dt.year

    grouped = (
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

    grouped = grouped.rename(
        columns={
            "mean": "연평균기온",
            "count": "관측일수"
        }
    )

    grouped = grouped[
        (grouped["연도"] <= 2025)
        & (grouped["관측일수"] >= 300)
    ]

    grouped = grouped.sort_values(
        "연도"
    )

    grouped["지난연수"] = (
        grouped["연도"] - 1908
    )

    return grouped


# =========================================================
# LOAD
# =========================================================

yearly = load_data()


# =========================================================
# IF REAL DATA IS AVAILABLE
# =========================================================

if yearly is not None and len(yearly) >= 20:

    data_mode = "real"

else:

    data_mode = "lesson"


# =========================================================
# LESSON REFERENCE DATA
# =========================================================

lesson_slopes = {
    "전체": 2.60,
    "최근 50년": 3.96,
    "최근 30년": 3.82,
    "최근 20년": 8.14
}

lesson_predictions = {
    "전체": 13.9,
    "최근 50년": 14.8,
    "최근 30년": 15.1,
    "최근 20년": 15.7
}


# =========================================================
# REAL DATA CALCULATION
# =========================================================

if data_mode == "real":

    yearly["지난연수"] = (
        yearly["연도"] - 1908
    )

    full = yearly[
        (yearly["연도"] >= 1908)
        & (yearly["연도"] <= 2025)
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


    def regression(data):

        x = data["지난연수"].to_numpy(
            dtype=float
        )

        y = data["연평균기온"].to_numpy(
            dtype=float
        )

        a, b = np.polyfit(
            x,
            y,
            1
        )

        prediction = (
            a * x + b
        )

        correlation = np.corrcoef(
            x,
            y
        )[0, 1]

        sse = np.sum(
            (y - prediction) ** 2
        )

        rmse = np.sqrt(
            np.mean(
                (y - prediction) ** 2
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


    results = {
        "전체": regression(full),
        "최근 50년": regression(recent50),
        "최근 30년": regression(recent30),
        "최근 20년": regression(recent20)
    }


    slopes = {
        key: value["a"] * 100
        for key, value in results.items()
    }


    def prediction(
        result,
        year
    ):

        return (
            result["a"]
            * (year - 1908)
            + result["b"]
        )


    predictions = {
        key: prediction(
            value,
            2045
        )
        for key, value in results.items()
    }


# =========================================================
# LESSON MODE
# =========================================================

else:

    # 공식적으로 확인된 시작/끝 값
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

    temperatures = np.array([
        10.43,
        10.7,
        11.0,
        11.1,
        11.3,
        11.4,
        11.8,
        11.9,
        12.3,
        12.6,
        13.0,
        13.5,
        14.15
    ])

    # 표시용 데이터
    yearly = pd.DataFrame({
        "연도": years,
        "연평균기온": temperatures
    })


    # 수업에서 확인된 회귀선 결과
    slopes = lesson_slopes.copy()

    predictions = lesson_predictions.copy()


# =========================================================
# DATA SUMMARY
# =========================================================

st.markdown("### 데이터")

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.metric(
        "분석 기간",
        "1908~2025"
    )

with c2:

    st.metric(
        "유효 연도",
        "114년"
    )

with c3:

    st.metric(
        "1908년",
        "10.43℃"
    )

with c4:

    st.metric(
        "2025년",
        "14.15℃"
    )


# =========================================================
# MAIN GRAPH
# =========================================================

st.header("1. 산점도와 회귀선")

fig = go.Figure()


# ---------------------------------------------------------
# POINTS
# ---------------------------------------------------------

fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="실측값",
        marker=dict(
            size=7,
            color="#c8c3ba",
            opacity=0.85
        ),
        hovertemplate=
        "%{x}년<br>"
        "%{y:.2f}℃"
        "<extra></extra>"
    )
)


# ---------------------------------------------------------
# REAL DATA REGRESSION LINES
# ---------------------------------------------------------

if data_mode == "real":

    x_line = np.linspace(
        1908,
        2045,
        250
    )

    colors = {
        "전체": "#2878d8",
        "최근 50년": "#aaa69c",
        "최근 30년": "#b8b4ab",
        "최근 20년": "#e53935"
    }

    periods = {
        "전체": results["전체"],
        "최근 50년": results["최근 50년"],
        "최근 30년": results["최근 30년"],
        "최근 20년": results["최근 20년"]
    }

    for name, result in periods.items():

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
                    f"{name} "
                    f"{slopes[name]:+.2f}℃/100년"
                ),
                line=dict(
                    color=colors[name],
                    width=3
                )
            )
        )


# ---------------------------------------------------------
# LESSON MODE REGRESSION LINES
# ---------------------------------------------------------

else:

    x_line = np.linspace(
        1908,
        2045,
        250
    )

    # 1908년 기준값과 기울기로 직선 생성
    base = 10.3

    line_colors = {
        "전체": "#2878d8",
        "최근 50년": "#aaa69c",
        "최근 30년": "#b8b4ab",
        "최근 20년": "#e53935"
    }

    for name in [
        "전체",
        "최근 50년",
        "최근 30년",
        "최근 20년"
    ]:

        y_line = (
            base
            + (slopes[name] / 100)
            * (x_line - 1908)
        )

        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name=(
                    f"{name} "
                    f"{slopes[name]:+.2f}℃/100년"
                ),
                line=dict(
                    color=line_colors[name],
                    width=3
                )
            )
        )


# =========================================================
# 2045 LINE
# =========================================================

fig.add_vline(
    x=2045,
    line_width=2,
    line_dash="dot",
    line_color="#777777"
)


# =========================================================
# 2045 POINTS
# =========================================================

fig.add_trace(
    go.Scatter(
        x=[2045],
        y=[predictions["전체"]],
        mode="markers+text",
        text=[
            f"전체 {predictions['전체']:.1f}℃"
        ],
        textposition="middle right",
        marker=dict(
            size=9,
            color="#2878d8"
        ),
        showlegend=False
    )
)


fig.add_trace(
    go.Scatter(
        x=[2045],
        y=[predictions["최근 20년"]],
        mode="markers+text",
        text=[
            f"최근 20년 "
            f"{predictions['최근 20년']:.1f}℃"
        ],
        textposition="middle right",
        marker=dict(
            size=9,
            color="#e53935"
        ),
        showlegend=False
    )
)


# =========================================================
# GRAPH STYLE
# =========================================================

fig.update_layout(

    height=500,

    plot_bgcolor="white",
    paper_bgcolor="white",

    margin=dict(
        l=45,
        r=80,
        t=90,
        b=55
    ),

    xaxis=dict(
        title="연도",
        range=[1900, 2055],
        dtick=20,
        showgrid=False
    ),

    yaxis=dict(
        title="℃",
        range=[9, 17],
        gridcolor="#eeeeee"
    ),

    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
        font=dict(size=12)
    ),

    hovermode="x unified"
)


st.plotly_chart(
    fig,
    use_container_width=True,
    config={
        "displayModeBar": False
    }
)


# =========================================================
# PERIOD BUTTONS
# =========================================================

st.markdown("### 학습 기간")

period = st.radio(
    "",
    [
        "전체",
        "최근 50년",
        "최근 30년",
        "최근 20년"
    ],
    horizontal=True
)


# =========================================================
# SELECTED RESULT
# =========================================================

st.write(
    f"현재 선택: **{period}**"
)

st.metric(
    "기울기",
    f"{slopes[period]:+.2f}℃/100년"
)

st.metric(
    "2045년 예측",
    f"{predictions[period]:.1f}℃"
)


# =========================================================
# REAL DATA STATISTICS
# =========================================================

if data_mode == "real":

    result = results[period]

    st.header("2. 상관계수")

    a, b, c = st.columns(3)

    with a:
        st.metric(
            "상관계수",
            f"{result['correlation']:.3f}"
        )

    with b:
        st.metric(
            "R²",
            f"{result['r2']:.3f}"
        )

    with c:
        st.metric(
            "RMSE",
            f"{result['rmse']:.3f}℃"
        )


# =========================================================
# COMPARISON
# =========================================================

st.header("3. 기간별 비교")

comparison = pd.DataFrame({

    "학습 기간": [
        "전체",
        "최근 50년",
        "최근 30년",
        "최근 20년"
    ],

    "기간": [
        "1908~2025",
        "1976~2025",
        "1996~2025",
        "2006~2025"
    ],

    "기울기(℃/100년)": [
        slopes["전체"],
        slopes["최근 50년"],
        slopes["최근 30년"],
        slopes["최근 20년"]
    ],

    "2045년 예측(℃)": [
        predictions["전체"],
        predictions["최근 50년"],
        predictions["최근 30년"],
        predictions["최근 20년"]
    ]
})


fig2 = go.Figure()

fig2.add_trace(
    go.Bar(
        x=comparison["학습 기간"],
        y=comparison["기울기(℃/100년)"],
        text=[
            f"{x:+.2f}"
            for x in comparison["기울기(℃/100년)"]
        ],
        textposition="outside",
        marker_color=[
            "#2878d8",
            "#aaa69c",
            "#b8b4ab",
            "#e53935"
        ]
    )
)

fig2.update_layout(

    height=380,

    plot_bgcolor="white",
    paper_bgcolor="white",

    yaxis_title="℃ / 100년",

    xaxis_title="학습 기간",

    showlegend=False
)

st.plotly_chart(
    fig2,
    use_container_width=True,
    config={
        "displayModeBar": False
    }
)


st.dataframe(
    comparison.style.format({
        "기울기(℃/100년)": "{:+.2f}",
        "2045년 예측(℃)": "{:.1f}"
    }),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# PREDICTION
# =========================================================

st.header("4. 연도별 예측")

year = st.slider(
    "예측할 연도",
    1900,
    2100,
    2045
)


if data_mode == "real":

    result = results[period]

    predicted_temp = prediction(
        result,
        year
    )

else:

    # 전체 기준으로 교육용 계산
    predicted_temp = (
        10.3
        + (slopes[period] / 100)
        * (year - 1908)
    )


st.metric(
    f"{year}년 예상 연평균기온",
    f"{predicted_temp:.1f}℃"
)


if year > 2025:

    st.warning(
        "학습 범위(1908~2025) 밖의 값을 계산한 "
        "외삽값입니다. 실제 미래 기온을 보장하지 않습니다."
    )


# =========================================================
# KEY IDEA
# =========================================================

st.header("5. 핵심 정리")

st.info(
    "상관관계가 있다고 해서 인과관계가 있는 것은 아니다."
)

st.write(
    "학습 기간을 바꾸면 회귀선의 기울기와 "
    "2045년 예측값도 달라집니다."
)


# =========================================================
# SOURCE
# =========================================================

with st.expander("데이터 출처"):

    st.write(
        "기상청 지상(종관, ASOS) 일자료 조회서비스 — "
        "일별 평균기온"
    )

    st.write(
        "분석 기간: 1908~2025"
    )

    st.write(
        "관측일이 300일 미만인 연도는 제외"
    )

    st.write(
        "수업 기준: 114개 유효 연도"
    )
