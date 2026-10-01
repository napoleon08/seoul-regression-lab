import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# =========================================================
# 페이지 설정
# =========================================================

st.set_page_config(
    page_title="서울 기온 회귀 분석",
    page_icon="📈",
    layout="wide"
)


# =========================================================
# 간단한 디자인
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
}

h2 {
    font-size: 23px !important;
    margin-top: 35px !important;
}

h3 {
    font-size: 19px !important;
}

.metric-box {
    background: #fafafa;
    border: 1px solid #e5e5e5;
    border-radius: 10px;
    padding: 16px;
}

.info-box {
    background: #f8f8f8;
    border: 1px solid #e5e5e5;
    border-radius: 10px;
    padding: 15px 18px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 제목
# =========================================================

st.title("📈 서울 기온 회귀 분석")

st.caption("8차시 · 회귀 — 직선을 긋다")

st.write(
    "1908~2025년 서울 연평균기온 데이터를 이용하여 "
    "회귀선을 그어 봅니다."
)


# =========================================================
# 데이터 불러오기
# =========================================================

@st.cache_data
def load_csv(file):

    df = pd.read_csv(file)

    # 날짜 + 평균기온 형태
    if "날짜" in df.columns and "평균기온" in df.columns:

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
    elif (
        "연도" in df.columns
        and "연평균기온" in df.columns
    ):

        yearly = df.copy()

        if "관측일수" not in yearly.columns:
            yearly["관측일수"] = 365

    else:

        raise ValueError(
            "CSV에 '날짜'와 '평균기온' 열이 필요합니다."
        )

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
        (yearly["연도"] >= 1908)
        & (yearly["연도"] <= 2025)
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
# CSV 선택
# =========================================================

uploaded_file = st.file_uploader(
    "서울 기온 데이터 CSV",
    type=["csv"]
)


# 업로드한 파일이 있으면 사용
if uploaded_file is not None:

    try:
        yearly = load_csv(uploaded_file)

    except Exception as e:

        st.error(f"데이터를 읽을 수 없습니다: {e}")
        st.stop()

else:

    # GitHub에 seoul.csv가 있으면 자동 사용
    try:

        yearly = load_csv("seoul.csv")

    except Exception:

        st.info(
            "seoul.csv 파일을 프로젝트 폴더에 넣거나 "
            "위에서 CSV 파일을 업로드하세요."
        )

        st.stop()


# =========================================================
# 데이터가 충분한지 확인
# =========================================================

if len(yearly) < 10:

    st.error(
        "분석할 수 있는 연도 데이터가 너무 적습니다."
    )

    st.stop()


# =========================================================
# 회귀 계산 함수
# =========================================================

def regression(data):

    x = data["지난연수"].to_numpy(dtype=float)
    y = data["연평균기온"].to_numpy(dtype=float)

    # y = ax + b
    a, b = np.polyfit(x, y, 1)

    predicted = a * x + b

    # 상관계수
    correlation = np.corrcoef(x, y)[0, 1]

    # SSE
    sse = np.sum(
        (y - predicted) ** 2
    )

    # RMSE
    rmse = np.sqrt(
        np.mean(
            (y - predicted) ** 2
        )
    )

    # R²
    total = np.sum(
        (y - np.mean(y)) ** 2
    )

    if total == 0:
        r2 = 0
    else:
        r2 = 1 - (sse / total)

    return {
        "a": a,
        "b": b,
        "correlation": correlation,
        "sse": sse,
        "rmse": rmse,
        "r2": r2
    }


# =========================================================
# 기간별 데이터
# =========================================================

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


# =========================================================
# 기간별 회귀
# =========================================================

r_full = regression(full)
r_50 = regression(recent50)
r_30 = regression(recent30)
r_20 = regression(recent20)


# =========================================================
# 100년당 변화량
# =========================================================

slope_full = r_full["a"] * 100
slope_50 = r_50["a"] * 100
slope_30 = r_30["a"] * 100
slope_20 = r_20["a"] * 100


# =========================================================
# 미래 예측 함수
# =========================================================

def predict(result, year):

    x = year - 1908

    return (
        result["a"] * x
        + result["b"]
    )


pred_full = predict(
    r_full,
    2045
)

pred_50 = predict(
    r_50,
    2045
)

pred_30 = predict(
    r_30,
    2045
)

pred_20 = predict(
    r_20,
    2045
)


# =========================================================
# 기본 정보
# =========================================================

st.markdown("### 데이터")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "분석 기간",
        f"{int(full['연도'].min())}~{int(full['연도'].max())}"
    )

with c2:
    st.metric(
        "유효 연도",
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
# 1. 산점도 + 회귀선
# =========================================================

st.header("1. 산점도와 회귀선")


fig = go.Figure()


# ---------------------------------------------------------
# 실제 연평균기온
# ---------------------------------------------------------

fig.add_trace(
    go.Scatter(
        x=full["연도"],
        y=full["연평균기온"],
        mode="markers",
        name="연평균기온",
        marker=dict(
            size=7,
            color="#c8c3ba",
            opacity=0.85
        ),
        hovertemplate=
        "<b>%{x}년</b><br>"
        "연평균기온: %{y:.2f}℃"
        "<extra></extra>"
    )
)


# ---------------------------------------------------------
# 회귀선 함수
# ---------------------------------------------------------

def regression_line(result):

    x = np.linspace(
        1908,
        2045,
        250
    )

    y = (
        result["a"]
        * (x - 1908)
        + result["b"]
    )

    return x, y


# ---------------------------------------------------------
# 전체
# ---------------------------------------------------------

x, y = regression_line(r_full)

fig.add_trace(
    go.Scatter(
        x=x,
        y=y,
        mode="lines",
        name=f"전체 1908~2025   {slope_full:+.2f}℃/100년",
        line=dict(
            color="#2878d8",
            width=3
        )
    )
)


# ---------------------------------------------------------
# 최근 50년
# ---------------------------------------------------------

x, y = regression_line(r_50)

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


# ---------------------------------------------------------
# 최근 30년
# ---------------------------------------------------------

x, y = regression_line(r_30)

fig.add_trace(
    go.Scatter(
        x=x,
        y=y,
        mode="lines",
        name=f"최근 30년 1996~2025   {slope_30:+.2f}℃/100년",
        line=dict(
            color="#b8b4ab",
            width=2
        )
    )
)


# ---------------------------------------------------------
# 최근 20년
# ---------------------------------------------------------

x, y = regression_line(r_20)

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
# 2045년 수직 점선
# =========================================================

fig.add_vline(
    x=2045,
    line_width=2,
    line_dash="dot",
    line_color="#777777"
)


# =========================================================
# 2045 예측점 - 전체
# =========================================================

fig.add_trace(
    go.Scatter(
        x=[2045],
        y=[pred_full],
        mode="markers+text",
        text=[f"전체 {pred_full:.1f}℃"],
        textposition="middle right",
        marker=dict(
            size=9,
            color="#2878d8"
        ),
        showlegend=False
    )
)


# =========================================================
# 2045 예측점 - 최근 20년
# =========================================================

fig.add_trace(
    go.Scatter(
        x=[2045],
        y=[pred_20],
        mode="markers+text",
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
# 그래프 설정
# =========================================================

min_temp = min(
    full["연평균기온"].min(),
    9
)

max_temp = max(
    full["연평균기온"].max(),
    pred_full,
    pred_20
)

fig.update_layout(

    height=500,

    margin=dict(
        l=45,
        r=80,
        t=80,
        b=55
    ),

    plot_bgcolor="white",
    paper_bgcolor="white",

    xaxis=dict(
        title="연도",
        range=[1900, 2055],
        dtick=20,
        showgrid=False,
        zeroline=False
    ),

    yaxis=dict(
        title="℃",
        range=[
            min_temp - 0.5,
            max_temp + 0.7
        ],
        gridcolor="#eeeeee",
        zeroline=False
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


st.caption(
    "돌 하나의 점은 해당 연도의 연평균기온을 나타냅니다."
)


# =========================================================
# 기간 선택
# =========================================================

st.markdown("### 회귀선 선택")

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


if period == "전체":

    selected = r_full
    selected_slope = slope_full
    selected_prediction = pred_full
    selected_period = "1908~2025"

elif period == "최근 50년":

    selected = r_50
    selected_slope = slope_50
    selected_prediction = pred_50
    selected_period = "1976~2025"

elif period == "최근 30년":

    selected = r_30
    selected_slope = slope_30
    selected_prediction = pred_30
    selected_period = "1996~2025"

else:

    selected = r_20
    selected_slope = slope_20
    selected_prediction = pred_20
    selected_period = "2006~2025"


st.write(
    f"**{selected_period}** 회귀선"
)

st.write(
    f"기울기: **{selected_slope:+.2f}℃/100년**"
)

st.write(
    f"2045년 예측: **{selected_prediction:.1f}℃**"
)


# =========================================================
# 2. 상관계수
# =========================================================

st.header("2. 상관계수")

m1, m2, m3 = st.columns(3)

with m1:

    st.metric(
        "상관계수",
        f"{selected['correlation']:.3f}"
    )

with m2:

    st.metric(
        "기울기",
        f"{selected_slope:+.2f}℃/100년"
    )

with m3:

    st.metric(
        "R²",
        f"{selected['r2']:.3f}"
    )


# =========================================================
# 3. 회귀식
# =========================================================

st.header("3. 회귀식")

st.latex(
    f"y = {selected['a']:.4f}x + {selected['b']:.4f}"
)

st.write(
    "x = 1908년부터 지난 연수"
)

st.write(
    "y = 서울 연평균기온"
)


# =========================================================
# 4. 2045년 예측
# =========================================================

st.header("4. 2045년 예측")


p1, p2 = st.columns(2)


with p1:

    st.metric(
        "전체 1908~2025",
        f"{pred_full:.1f}℃"
    )


with p2:

    st.metric(
        "최근 20년 2006~2025",
        f"{pred_20:.1f}℃"
    )


st.write(
    f"전체 기간으로 계산하면 2045년 예측값은 "
    f"**{pred_full:.1f}℃**입니다."
)

st.write(
    f"최근 20년으로 계산하면 2045년 예측값은 "
    f"**{pred_20:.1f}℃**입니다."
)


# =========================================================
# 5. 기간별 비교
# =========================================================

st.header("5. 학습 기간에 따른 차이")


comparison = pd.DataFrame({

    "구분": [
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

    "기울기": [
        slope_full,
        slope_50,
        slope_30,
        slope_20
    ],

    "2045년 예측": [
        pred_full,
        pred_50,
        pred_30,
        pred_20
    ]
})


# ---------------------------------------------------------
# 비교 그래프
# ---------------------------------------------------------

fig2 = go.Figure()

fig2.add_trace(
    go.Bar(
        x=comparison["구분"],
        y=comparison["기울기"],
        text=[
            f"{value:+.2f}"
            for value in comparison["기울기"]
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

    xaxis_title="학습 기간",

    yaxis_title="℃ / 100년",

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


# ---------------------------------------------------------
# 비교 표
# ---------------------------------------------------------

comparison_display = comparison.copy()

comparison_display["기울기"] = (
    comparison_display["기울기"]
    .map(lambda x: f"{x:+.2f}℃/100년")
)

comparison_display["2045년 예측"] = (
    comparison_display["2045년 예측"]
    .map(lambda x: f"{x:.1f}℃")
)


st.dataframe(
    comparison_display,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 6. 모델 평가
# =========================================================

st.header("6. 모델 평가")


e1, e2, e3 = st.columns(3)


with e1:

    st.metric(
        "SSE",
        f"{selected['sse']:.2f}"
    )


with e2:

    st.metric(
        "RMSE",
        f"{selected['rmse']:.3f}℃"
    )


with e3:

    st.metric(
        "R²",
        f"{selected['r2']:.3f}"
    )


# =========================================================
# 7. 잔차 그래프
# =========================================================

st.header("7. 잔차 그래프")


x = full["연도"].to_numpy()

y = full["연평균기온"].to_numpy()

predicted = (
    r_full["a"]
    * (x - 1908)
    + r_full["b"]
)

residual = y - predicted


fig3 = go.Figure()


fig3.add_trace(
    go.Scatter(
        x=x,
        y=residual,
        mode="markers",
        name="잔차",
        marker=dict(
            size=7,
            color="#aaa69c"
        ),
        hovertemplate=
        "<b>%{x}년</b><br>"
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

    plot_bgcolor="white",
    paper_bgcolor="white",

    xaxis_title="연도",

    yaxis_title="잔차(℃)",

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
# 8. 핵심 정리
# =========================================================

st.header("8. 핵심 정리")


st.markdown(
    """
<div class="info-box">

<strong>회귀 분석에서 알 수 있는 것</strong>

<br><br>

• 연도가 증가하면서 서울 연평균기온도 증가하는 경향이 나타납니다.

<br><br>

• 전체 기간과 최근 기간은 서로 다른 회귀선을 나타냅니다.

<br><br>

• 어떤 기간의 데이터를 학습하느냐에 따라 미래 예측값도 달라집니다.

</div>
""",
    unsafe_allow_html=True
)


st.warning(
    "상관관계가 있다고 해서 인과관계가 있는 것은 아니다."
)


# =========================================================
# 데이터 정보
# =========================================================

with st.expander("데이터 정보"):

    st.write(
        "분석 기간: 1908~2025"
    )

    st.write(
        "관측일수 300일 미만인 연도는 분석에서 제외했습니다."
    )

    st.write(
        f"최종 분석 연도 수: {len(full)}년"
    )
