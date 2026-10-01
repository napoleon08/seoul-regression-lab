import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# =========================================================
# 기본 설정
# =========================================================

st.set_page_config(
    page_title="서울 기온 회귀 분석",
    page_icon="📈",
    layout="wide"
)


# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>

.block-container {
    max-width: 1200px;
    padding-top: 35px;
    padding-bottom: 50px;
}

h1 {
    font-size: 32px !important;
    font-weight: 700 !important;
    margin-bottom: 8px !important;
}

h2 {
    font-size: 24px !important;
    margin-top: 35px !important;
}

h3 {
    font-size: 20px !important;
}

.small-text {
    color: #777777;
    font-size: 14px;
}

.info-box {
    padding: 15px 18px;
    border-radius: 10px;
    background: #f7f7f7;
    border: 1px solid #e5e5e5;
    margin: 12px 0;
}

.prediction-box {
    padding: 16px 20px;
    border-radius: 12px;
    background: #fafafa;
    border: 1px solid #dddddd;
    margin-top: 12px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 제목
# =========================================================

st.title("📈 서울 기온 회귀 분석")

st.markdown(
    '<div class="small-text">8차시 · 회귀 — 직선을 긋다</div>',
    unsafe_allow_html=True
)

st.markdown(
    "1908~2025년 서울 연평균기온 데이터를 이용하여 회귀선을 그어 봅니다."
)


# =========================================================
# 데이터 불러오기
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv("seoul.csv", encoding="utf-8")

    # 날짜가 있는 경우
    if "날짜" in df.columns:

        df["날짜"] = pd.to_datetime(
            df["날짜"],
            errors="coerce"
        )

        df["연도"] = df["날짜"].dt.year

        df["평균기온"] = pd.to_numeric(
            df["평균기온"],
            errors="coerce"
        )

        # 연도별 평균 + 관측일수
        yearly = (
            df.dropna(subset=["연도", "평균기온"])
              .groupby("연도")["평균기온"]
              .agg(["mean", "count"])
              .reset_index()
        )

        yearly = yearly.rename(
            columns={
                "mean": "연평균기온",
                "count": "관측일수"
            }
        )

    # 이미 연평균 데이터인 경우
    elif "연도" in df.columns and "연평균기온" in df.columns:

        yearly = df.copy()

        if "관측일수" not in yearly.columns:
            yearly["관측일수"] = 365

    # 혹시 영어 컬럼인 경우
    elif "year" in df.columns and "temperature" in df.columns:

        yearly = pd.DataFrame({
            "연도": pd.to_numeric(df["year"], errors="coerce"),
            "연평균기온": pd.to_numeric(
                df["temperature"],
                errors="coerce"
            ),
            "관측일수": 365
        })

    else:
        raise ValueError(
            "seoul.csv의 열 이름을 확인해주세요. "
            "필요한 열: 날짜, 평균기온"
        )

    # 숫자 변환
    yearly["연도"] = pd.to_numeric(
        yearly["연도"],
        errors="coerce"
    )

    yearly["연평균기온"] = pd.to_numeric(
        yearly["연평균기온"],
        errors="coerce"
    )

    yearly["관측일수"] = pd.to_numeric(
        yearly["관측일수"],
        errors="coerce"
    )

    yearly = yearly.dropna(
        subset=["연도", "연평균기온"]
    )

    # 1908~2025
    yearly = yearly[
        (yearly["연도"] >= 1908) &
        (yearly["연도"] <= 2025)
    ]

    # 관측일수 300일 이상
    yearly = yearly[
        yearly["관측일수"] >= 300
    ]

    yearly = yearly.sort_values("연도").reset_index(drop=True)

    # 1908년부터 지난 연수
    yearly["지난연수"] = yearly["연도"] - 1908

    return yearly


# =========================================================
# 데이터 확인
# =========================================================

try:
    yearly = load_data()

except Exception as e:

    st.error("seoul.csv를 불러오지 못했습니다.")

    st.write(str(e))

    st.info(
        "GitHub 저장소에 main.py와 같은 위치에 "
        "seoul.csv 파일을 넣어주세요."
    )

    st.stop()


# =========================================================
# 회귀 계산 함수
# =========================================================

def regression(data):

    x = data["지난연수"].values
    y = data["연평균기온"].values

    a, b = np.polyfit(x, y, 1)

    y_pred = a * x + b

    # 상관계수
    correlation = np.corrcoef(x, y)[0, 1]

    # SSE
    sse = np.sum((y - y_pred) ** 2)

    # RMSE
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))

    # R²
    ss_total = np.sum((y - np.mean(y)) ** 2)

    if ss_total == 0:
        r2 = 0
    else:
        r2 = 1 - sse / ss_total

    return {
        "a": a,
        "b": b,
        "correlation": correlation,
        "sse": sse,
        "rmse": rmse,
        "r2": r2
    }


# =========================================================
# 기간 데이터
# =========================================================

full = yearly[
    (yearly["연도"] >= 1908) &
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


# =========================================================
# 회귀 계산
# =========================================================

result_full = regression(full)
result_50 = regression(recent50)
result_30 = regression(recent30)
result_20 = regression(recent20)


# =========================================================
# 100년당 변화량
# =========================================================

slope_full = result_full["a"] * 100
slope_50 = result_50["a"] * 100
slope_30 = result_30["a"] * 100
slope_20 = result_20["a"] * 100


# =========================================================
# 2045년 예측
# =========================================================

def predict(result, year):

    x = year - 1908

    return result["a"] * x + result["b"]


pred_full = predict(result_full, 2045)
pred_20 = predict(result_20, 2045)


# =========================================================
# 기본 정보
# =========================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "분석 연도",
        f"{int(full['연도'].min())}~{int(full['연도'].max())}"
    )

with c2:
    st.metric(
        "유효 연도 수",
        f"{len(full)}년"
    )

with c3:
    st.metric(
        "1908년",
        f"{full.iloc[0]['연평균기온']:.2f}℃"
    )

with c4:
    st.metric(
        "2025년",
        f"{full.iloc[-1]['연평균기온']:.2f}℃"
    )


# =========================================================
# 1. 회귀선 그래프
# =========================================================

st.header("1. 산점도와 회귀선")

fig = go.Figure()


# ---------------------------------------------------------
# 실제 데이터
# ---------------------------------------------------------

fig.add_trace(
    go.Scatter(
        x=full["연도"],
        y=full["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(
            size=7,
            color="#c8c3ba",
            opacity=0.9
        ),
        hovertemplate=
            "<b>%{x}년</b><br>"
            "연평균기온: %{y:.2f}℃"
            "<extra></extra>"
    )
)


# ---------------------------------------------------------
# 회귀선 생성
# ---------------------------------------------------------

x_line = np.linspace(1908, 2045, 200)


def make_line(result):

    x = x_line
    y = result["a"] * (x - 1908) + result["b"]

    return x, y


x, y = make_line(result_full)

fig.add_trace(
    go.Scatter(
        x=x,
        y=y,
        mode="lines",
        name=f"전체 1908~2025   {slope_full:+.2f}℃/100년",
        line=dict(
            color="#2878d8",
            width=3
        ),
        hovertemplate=
            "%{x:.0f}년<br>"
            "%{y:.2f}℃"
            "<extra>전체</extra>"
    )
)


x, y = make_line(result_50)

fig.add_trace(
    go.Scatter(
        x=x,
        y=y,
        mode="lines",
        name=f"최근 50년 1976~2025   {slope_50:+.2f}℃/100년",
        line=dict(
            color="#aaa69c",
            width=2
        )
    )
)


x, y = make_line(result_30)

fig.add_trace(
    go.Scatter(
        x=x,
        y=y,
        mode="lines",
        name=f"최근 30년 1996~2025   {slope_30:+.2f}℃/100년",
        line=dict(
            color="#bcb8af",
            width=2
        )
    )
)


x, y = make_line(result_20)

fig.add_trace(
    go.Scatter(
        x=x,
        y=y,
        mode="lines",
        name=f"최근 20년 2006~2025   {slope_20:+.2f}℃/100년",
        line=dict(
            color="#e53935",
            width=3
        )
    )
)


# =========================================================
# 2045 수직선
# =========================================================

fig.add_vline(
    x=2045,
    line_width=2,
    line_dash="dot",
    line_color="#777777"
)


# =========================================================
# 2045 예측점
# =========================================================

fig.add_trace(
    go.Scatter(
        x=[2045],
        y=[pred_full],
        mode="markers+text",
        name="전체 2045 예측",
        text=[f"전체 {pred_full:.1f}℃"],
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
        y=[pred_20],
        mode="markers+text",
        name="최근 20년 2045 예측",
        text=[f"최근 20년 {pred_20:.1f}℃"],
        textposition="middle right",
        marker=dict(
            size=9,
            color="#e53935"
        ),
        showlegend=False
    )
)


# =========================================================
# 그래프 디자인
# =========================================================

fig.update_layout(

    height=520,

    margin=dict(
        l=45,
        r=35,
        t=30,
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
        range=[
            min(full["연평균기온"].min(), 9) - 0.3,
            max(pred_20, pred_full, full["연평균기온"].max()) + 0.8
        ],
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

    plot_bgcolor="white",
    paper_bgcolor="white",

    hovermode="x unified"
)

st.plotly_chart(
    fig,
    use_container_width=True,
    config={
        "displayModeBar": False
    }
)


st.caption(
    "회귀선은 각 기간의 연평균기온과 연도의 선형 관계를 나타냅니다."
)


# =========================================================
# 2. 상관계수
# =========================================================

st.header("2. 상관관계")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "상관계수",
        f"{result_full['correlation']:.3f}"
    )

with col2:

    st.metric(
        "회귀 기울기",
        f"{slope_full:+.2f} ℃/100년"
    )

with col3:

    st.metric(
        "결정계수 R²",
        f"{result_full['r2']:.3f}"
    )


st.markdown(
    """
<div class="info-box">

<strong>해석</strong><br>

연도가 증가할수록 서울의 연평균기온도 증가하는 경향을 확인할 수 있습니다.

</div>
""",
    unsafe_allow_html=True
)


# =========================================================
# 3. 회귀식
# =========================================================

st.header("3. 회귀식")

st.latex(
    f"y = {result_full['a']:.4f}x + {result_full['b']:.4f}"
)

st.write(
    f"여기서 x는 1908년부터 지난 연수이고, "
    f"y는 서울의 연평균기온입니다."
)

st.write(
    f"기울기: **{slope_full:+.2f}℃/100년**"
)


# =========================================================
# 4. 2045년 예측
# =========================================================

st.header("4. 2045년 기온 예측")

p1, p2 = st.columns(2)

with p1:

    st.markdown(
        f"""
        <div class="prediction-box">

        <strong>전체 기간 · 1908~2025</strong>

        <h2>{pred_full:.1f}℃</h2>

        </div>
        """,
        unsafe_allow_html=True
    )

with p2:

    st.markdown(
        f"""
        <div class="prediction-box">

        <strong>최근 20년 · 2006~2025</strong>

        <h2>{pred_20:.1f}℃</h2>

        </div>
        """,
        unsafe_allow_html=True
    )


st.write(
    f"전체 기간으로 계산하면 2045년은 약 **{pred_full:.1f}℃**, "
    f"최근 20년으로 계산하면 약 **{pred_20:.1f}℃**로 예측됩니다."
)


# =========================================================
# 5. 기간별 비교
# =========================================================

st.header("5. 학습 기간에 따른 회귀선 비교")

comparison = pd.DataFrame({

    "기간": [
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
        slope_full,
        slope_50,
        slope_30,
        slope_20
    ],

    "2045년 예측(℃)": [
        pred_full,
        predict(result_50, 2045),
        predict(result_30, 2045),
        pred_20
    ]
})


fig2 = go.Figure()

fig2.add_trace(
    go.Bar(
        x=comparison["기간"],
        y=comparison["기울기(℃/100년)"],
        text=[
            f"{v:+.2f}"
            for v in comparison["기울기(℃/100년)"]
        ],
        textposition="outside",
        marker_color=[
            "#2878d8",
            "#aaa69c",
            "#bcb8af",
            "#e53935"
        ]
    )
)

fig2.update_layout(

    height=400,

    yaxis_title="℃ / 100년",

    xaxis_title="학습 기간",

    plot_bgcolor="white",

    paper_bgcolor="white",

    yaxis=dict(
        gridcolor="#eeeeee"
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
# 비교 표
# =========================================================

st.dataframe(
    comparison.style.format({
        "기울기(℃/100년)": "{:+.2f}",
        "2045년 예측(℃)": "{:.1f}"
    }),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 6. 모델 평가
# =========================================================

st.header("6. 회귀모델 평가")

e1, e2, e3 = st.columns(3)

with e1:

    st.metric(
        "SSE",
        f"{result_full['sse']:.2f}"
    )

with e2:

    st.metric(
        "RMSE",
        f"{result_full['rmse']:.3f}℃"
    )

with e3:

    st.metric(
        "R²",
        f"{result_full['r2']:.3f}"
    )


# =========================================================
# 7. 잔차 그래프
# =========================================================

st.header("7. 잔차 확인")

x = full["연도"].values
y = full["연평균기온"].values

y_pred = (
    result_full["a"] *
    (x - 1908) +
    result_full["b"]
)

residual = y - y_pred


fig3 = go.Figure()

fig3.add_trace(
    go.Scatter(
        x=x,
        y=residual,
        mode="markers",
        marker=dict(
            size=7,
            color="#aaa69c"
        ),
        hovertemplate=
            "%{x}년<br>"
            "잔차: %{y:.2f}℃"
            "<extra></extra>"
    )
)

fig3.add_hline(
    y=0,
    line_dash="dash",
    line_color="#555555"
)

fig3.update_layout(

    height=380,

    xaxis_title="연도",

    yaxis_title="잔차(℃)",

    plot_bgcolor="white",

    paper_bgcolor="white",

    yaxis=dict(
        gridcolor="#eeeeee"
    ),

    showlegend=False
)

st.plotly_chart(
    fig3,
    use_container_width=True,
    config={
        "displayModeBar": False
    }
)


# =========================================================
# 8. 결론
# =========================================================

st.header("8. 결론")

st.markdown(
    """
<div class="info-box">

<strong>핵심 정리</strong>

<ul>
<li>1908~2025년 서울 연평균기온에는 상승하는 경향이 나타납니다.</li>
<li>전체 기간의 회귀선 기울기는 100년당 약 +2.60℃입니다.</li>
<li>최근 기간만 사용하면 회귀선의 기울기가 달라집니다.</li>
<li>따라서 어떤 기간의 데이터를 학습시키느냐에 따라 미래 예측값도 달라질 수 있습니다.</li>
</ul>

</div>
""",
    unsafe_allow_html=True
)


st.warning(
    "상관관계가 있다고 해서 인과관계가 있는 것은 아니다."
)


# =========================================================
# 데이터 출처
# =========================================================

with st.expander("데이터 출처"):

    st.write(
        "기상청 서울 관측소의 일별 평균기온을 연 단위로 집계한 데이터입니다."
    )

    st.write(
        "관측일이 300일 미만인 연도는 제외합니다."
    )

    st.write(
        "분석 기간: 1908~2025"
    )
