import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ============================================================
# SEOUL CLIMATE // REGRESSION LAB
# 8차시 · 회귀 — 직선을 긋다
# 송탄고등학교 데이터 과학
# ============================================================

st.set_page_config(
    page_title="SEOUL CLIMATE // REGRESSION LAB",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# SETTINGS
# ============================================================

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e767e30231bce4/data/seoul.csv"
)

BASE_YEAR = 1908
MAX_YEAR = 2025
MIN_DAYS = 300


# ============================================================
# SPACE-X STYLE CSS
# ============================================================

st.markdown(
    """
<style>

@import url(
'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap'
);

:root {
    --black: #050505;
    --panel: #0b0b0b;
    --panel2: #101010;
    --line: #292929;
    --text: #f2f2f2;
    --muted: #858585;
    --red: #ff3b30;
    --red-light: #ff7169;
    --green: #48d597;
}

html, body, [class*="css"] {
    font-family: "Inter", sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 85% 0%,
            rgba(255, 59, 48, 0.09),
            transparent 28%
        ),
        linear-gradient(
            180deg,
            #050505 0%,
            #080808 100%
        );

    color: var(--text);
}

.block-container {
    max-width: 1500px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

[data-testid="stSidebar"] {
    background: #080808;
    border-right: 1px solid #202020;
}

[data-testid="stSidebar"] * {
    color: #eeeeee !important;
}

.hero {
    position: relative;
    overflow: hidden;

    border: 1px solid #2b2b2b;

    background:
        linear-gradient(
            135deg,
            rgba(255,255,255,.045),
            rgba(255,255,255,.008)
        ),
        #090909;

    padding: 42px 42px 36px;
    margin-bottom: 25px;
}

.hero:after {
    content: "";

    position: absolute;

    right: -100px;
    top: -160px;

    width: 330px;
    height: 330px;

    border-radius: 50%;

    border: 1px solid rgba(255,59,48,.25);

    box-shadow:
        0 0 0 30px rgba(255,59,48,.035),
        0 0 0 65px rgba(255,59,48,.018);
}

.eyebrow {
    font-family: "Space Grotesk", sans-serif;

    color: var(--red-light);

    font-size: .72rem;

    font-weight: 700;

    letter-spacing: .20em;

    text-transform: uppercase;
}

.hero h1 {
    font-family: "Space Grotesk", sans-serif;

    font-size: clamp(2.5rem, 5vw, 5.3rem);

    line-height: .9;

    letter-spacing: -.055em;

    margin: 10px 0 16px;

    font-weight: 700;
}

.hero p {
    color: #a5a5a5;

    max-width: 900px;

    line-height: 1.7;

    margin: 0;
}

.status-row {
    display: flex;

    gap: 8px;

    flex-wrap: wrap;

    margin-top: 23px;
}

.status {
    border: 1px solid #303030;

    background: #0e0e0e;

    padding: 7px 11px;

    color: #bdbdbd;

    font-size: .65rem;

    letter-spacing: .1em;

    text-transform: uppercase;
}

.status.red {
    color: #ff7169;

    border-color: rgba(255,59,48,.45);
}

.section-kicker {
    color: #777;

    font-family: "Space Grotesk", sans-serif;

    font-size: .7rem;

    font-weight: 700;

    letter-spacing: .18em;

    text-transform: uppercase;

    margin-top: 32px;
}

.section-title {
    color: #f1f1f1;

    font-family: "Space Grotesk", sans-serif;

    font-size: 1.7rem;

    letter-spacing: -.025em;

    margin: 5px 0;
}

.section-copy {
    color: #858585;

    line-height: 1.65;

    margin-bottom: 17px;
}

.card {
    background: #0b0b0b;

    border: 1px solid #252525;

    padding: 18px;

    min-height: 115px;
}

.card.red {
    border-top: 2px solid var(--red);
}

.card.green {
    border-top: 2px solid var(--green);
}

.label {
    color: #707070;

    font-size: .66rem;

    letter-spacing: .12em;

    text-transform: uppercase;

    font-weight: 700;
}

.value {
    font-family: "Space Grotesk", sans-serif;

    color: #f3f3f3;

    font-size: 2rem;

    line-height: 1.1;

    margin-top: 7px;
}

.detail {
    color: #777;

    font-size: .75rem;

    margin-top: 6px;
}

.insight {
    background: #0b0b0b;

    border: 1px solid #282828;

    border-left: 3px solid var(--red);

    padding: 17px 19px;

    color: #bdbdbd;

    line-height: 1.7;
}

.insight strong {
    color: white;
}

.formula {
    background: #080808;

    border: 1px solid #292929;

    padding: 18px 20px;

    font-family: "Space Grotesk", monospace;

    font-size: 1rem;

    color: #eee;

    margin-bottom: 13px;
}

.badge {
    display: inline-block;

    padding: 4px 8px;

    border: 1px solid #333;

    background: #111;

    color: #999;

    font-size: .63rem;

    letter-spacing: .1em;

    text-transform: uppercase;
}

.badge.red {
    color: #ff7169;

    border-color: rgba(255,59,48,.45);
}

.badge.green {
    color: #69dba5;

    border-color: rgba(72,213,151,.45);
}

.footer {
    border-top: 1px solid #222;

    margin-top: 45px;

    padding-top: 18px;

    color: #5f5f5f;

    font-size: .67rem;

    letter-spacing: .07em;
}

.stButton > button,
.stDownloadButton > button {
    border-radius: 0 !important;

    background: #111 !important;

    border: 1px solid #353535 !important;

    color: #eee !important;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    border-color: #ff3b30 !important;
}

div[data-testid="stMetric"] {
    background: #0b0b0b;

    border: 1px solid #252525;

    padding: 14px 16px;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# DATA
# ============================================================

@st.cache_data(ttl=3600)
def load_data():

    df = pd.read_csv(
        DATA_URL,
        encoding="utf-8"
    )

    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce"
    )

    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["날짜", "평균기온"]
    ).copy()

    df["연도"] = df["날짜"].dt.year

    return df


@st.cache_data
def make_yearly_data(df):

    yearly = (
        df.groupby("연도")["평균기온"]
        .agg(["mean", "count"])
        .reset_index()
    )

    yearly = yearly.rename(
        columns={
            "mean": "연평균기온",
            "count": "관측일수"
        }
    )

    yearly = yearly[
        (yearly["연도"] <= MAX_YEAR)
        &
        (yearly["관측일수"] >= MIN_DAYS)
    ].copy()

    yearly["연평균기온"] = yearly[
        "연평균기온"
    ].round(4)

    yearly["1908년부터 지난 연수"] = (
        yearly["연도"] - BASE_YEAR
    )

    yearly = yearly.sort_values(
        "연도"
    ).reset_index(drop=True)

    return yearly


# ============================================================
# REGRESSION
# ============================================================

def regression(data):

    x = data[
        "1908년부터 지난 연수"
    ].to_numpy(dtype=float)

    y = data[
        "연평균기온"
    ].to_numpy(dtype=float)

    slope, intercept = np.polyfit(
        x,
        y,
        1
    )

    prediction = (
        slope * x
        + intercept
    )

    residual = y - prediction

    sse = np.sum(
        residual ** 2
    )

    rmse = np.sqrt(
        np.mean(
            residual ** 2
        )
    )

    total = np.sum(
        (y - np.mean(y)) ** 2
    )

    r2 = (
        1 - sse / total
        if total != 0
        else np.nan
    )

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "prediction": prediction,
        "residual": residual,
        "sse": float(sse),
        "rmse": float(rmse),
        "r2": float(r2)
    }


def predict(model, year):

    return (
        model["slope"]
        * (year - BASE_YEAR)
        + model["intercept"]
    )


# ============================================================
# PLOT STYLE
# ============================================================

def style_plot(fig, height=560):

    fig.update_layout(

        template="plotly_dark",

        height=height,

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="#080808",

        margin=dict(
            l=25,
            r=25,
            t=60,
            b=45
        ),

        font=dict(
            family="Inter",
            color="#d8d8d8"
        ),

        legend=dict(
            bgcolor="rgba(0,0,0,0)",

            bordercolor="#292929",

            borderwidth=1
        ),

        hoverlabel=dict(
            bgcolor="#111",
            bordercolor="#444",
            font_color="#fff"
        )
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="#1b1b1b",
        zeroline=False,
        linecolor="#333"
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="#1b1b1b",
        zeroline=False,
        linecolor="#333"
    )

    return fig


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
<div class="hero">

<div class="eyebrow">
DATA SCIENCE · SONGTAN HIGH SCHOOL · LESSON 08
</div>

<h1>
SEOUL CLIMATE<br>
REGRESSION LAB
</h1>

<p>
서울의 연도별 평균기온을 산점도와 상관계수로 살펴보고,
단순 선형회귀를 이용해 관계를 하나의 직선으로 요약합니다.
학습 기간을 바꾸었을 때 기울기와 예측값이 어떻게 달라지는지도 비교합니다.
</p>

<div class="status-row">

<span class="status red">
REGRESSION ONLINE
</span>

<span class="status">
SEOUL DAILY DATA
</span>

<span class="status">
DATA THROUGH 2025
</span>

<span class="status">
≥ 300 DAYS
</span>

<span class="status">
BASE YEAR 1908
</span>

</div>

</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## MISSION CONTROL")

    st.caption(
        "8차시 · 회귀 — 직선을 긋다"
    )

    st.divider()

    selected_year = st.slider(
        "예측할 연도",
        1900,
        2100,
        2045,
        1
    )

    recent_period = st.selectbox(
        "비교할 최근 기간",
        [20, 30, 50],
        index=0,
        format_func=lambda x:
            f"최근 {x}년"
    )

    st.divider()

    show_recent_line = st.checkbox(
        "최근 기간 회귀선",
        value=True
    )

    show_extrapolation = st.checkbox(
        "외삽 영역 표시",
        value=True
    )

    st.divider()

    st.markdown(
        "### ANALYSIS RULES"
    )

    st.write(
        f"데이터 종료: **{MAX_YEAR}년**"
    )

    st.write(
        f"최소 관측일: **{MIN_DAYS}일**"
    )

    st.write(
        f"기준 연도: **{BASE_YEAR}년**"
    )


# ============================================================
# LOAD
# ============================================================

try:

    daily = load_data()

    yearly = make_yearly_data(
        daily
    )

except Exception as e:

    st.error(
        "서울 기온 데이터를 불러오지 못했습니다."
    )

    st.code(str(e))

    st.stop()


if yearly.empty:

    st.error(
        "분석 가능한 데이터가 없습니다."
    )

    st.stop()


# ============================================================
# BASIC VALUES
# ============================================================

start_year = int(
    yearly["연도"].min()
)

end_year = int(
    yearly["연도"].max()
)

year_count = len(
    yearly
)

model = regression(
    yearly
)

correlation = float(
    yearly["연도"].corr(
        yearly["연평균기온"]
    )
)

recent_start = (
    end_year
    - recent_period
    + 1
)

recent = yearly[
    yearly["연도"] >= recent_start
].copy()

recent_model = regression(
    recent
)

prediction = float(
    predict(
        model,
        selected_year
    )
)

recent_prediction = float(
    predict(
        recent_model,
        selected_year
    )
)

outside_training = (
    selected_year < start_year
    or
    selected_year > end_year
)


# ============================================================
# 01 DATA
# ============================================================

st.markdown(
    '<div class="section-kicker">'
    '01 / DATA TELEMETRY'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">'
    '데이터를 먼저 확인합니다'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-copy">'
    '2025년까지의 서울 기온 자료를 사용하고, '
    '관측일이 300일 이상인 연도만 최종 분석에 포함합니다.'
    '</div>',
    unsafe_allow_html=True
)


c1, c2, c3, c4 = st.columns(4)


with c1:

    st.markdown(
        f"""
<div class="card red">

<div class="label">
VALID YEARS
</div>

<div class="value">
{year_count}
</div>

<div class="detail">
{start_year}–{end_year}
</div>

</div>
""",
        unsafe_allow_html=True
    )


with c2:

    st.markdown(
        f"""
<div class="card">

<div class="label">
DAILY RECORDS
</div>

<div class="value">
{len(daily):,}
</div>

<div class="detail">
원자료 관측 행
</div>

</div>
""",
        unsafe_allow_html=True
    )


with c3:

    st.markdown(
        f"""
<div class="card">

<div class="label">
CORRELATION
</div>

<div class="value">
{correlation:+.3f}
</div>

<div class="detail">
연도 ↔ 연평균기온
</div>

</div>
""",
        unsafe_allow_html=True
    )


with c4:

    st.markdown(
        f"""
<div class="card green">

<div class="label">
100 YEAR TREND
</div>

<div class="value">
{model["slope"] * 100:+.2f}℃
</div>

<div class="detail">
℃ / 100년
</div>

</div>
""",
        unsafe_allow_html=True
    )


# ============================================================
# 02 SCATTER + REGRESSION
# ============================================================

st.markdown(
    '<div class="section-kicker">'
    '02 / RELATION'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">'
    '산점도 → 상관계수 → 회귀선'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-copy">'
    '각 점은 서울의 실제 연평균기온입니다. '
    '빨간색 회귀선은 전체 학습 기간을 하나의 직선으로 요약합니다.'
    '</div>',
    unsafe_allow_html=True
)


fig = go.Figure()


# Actual observations

fig.add_trace(
    go.Scatter(

        x=yearly["연도"],

        y=yearly["연평균기온"],

        mode="markers",

        name="서울 연평균기온",

        marker=dict(
            size=8,
            color="#eeeeee",
            opacity=.72,
            line=dict(
                color="#555",
                width=.6
            )
        ),

        customdata=yearly[
            ["관측일수"]
        ].to_numpy(),

        hovertemplate=(
            "<b>%{x}년</b><br>"
            "연평균기온: %{y:.2f}℃<br>"
            "관측일수: %{customdata[0]}일"
            "<extra></extra>"
        )
    )
)


# Overall regression line

line_years = np.linspace(
    start_year,
    end_year,
    300
)

line_values = [
    predict(
        model,
        year
    )
    for year in line_years
]


fig.add_trace(
    go.Scatter(

        x=line_years,

        y=line_values,

        mode="lines",

        name="전체 회귀 직선",

        line=dict(
            color="#ff3b30",
            width=3
        ),

        hovertemplate=(
            "회귀선: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


# Recent regression line

if show_recent_line:

    recent_years = np.linspace(
        recent_start,
        end_year,
        200
    )

    recent_values = [
        predict(
            recent_model,
            year
        )
        for year in recent_years
    ]

    fig.add_trace(
        go.Scatter(

            x=recent_years,

            y=recent_values,

            mode="lines",

            name=f"최근 {recent_period}년 회귀선",

            line=dict(
                color="#ff9b94",
                width=2,
                dash="dash"
            )
        )
    )


# Selected year

fig.add_vline(
    x=selected_year,
    line_width=1,
    line_dash="dot",
    line_color="#ffffff"
)


# Extrapolation zone

if (
    show_extrapolation
    and
    outside_training
):

    if selected_year > end_year:

        fig.add_vrect(
            x0=end_year,
            x1=selected_year,
            fillcolor="rgba(255,59,48,.08)",
            line_width=0,
            annotation_text="EXTRAPOLATION",
            annotation_position="top left",
            annotation_font=dict(
                color="#ff7169"
            )
        )

    else:

        fig.add_vrect(
            x0=selected_year,
            x1=start_year,
            fillcolor="rgba(255,59,48,.08)",
            line_width=0,
            annotation_text="EXTRAPOLATION",
            annotation_position="top right",
            annotation_font=dict(
                color="#ff7169"
            )
        )


# Prediction point

fig.add_trace(
    go.Scatter(

        x=[selected_year],

        y=[prediction],

        mode="markers",

        name=f"{selected_year}년 예측",

        marker=dict(
            size=15,
            color="#ff3b30",
            line=dict(
                color="#ffffff",
                width=2
            )
        ),

        hovertemplate=(
            f"<b>{selected_year}년</b><br>"
            f"예측값: {prediction:.2f}℃"
            "<extra></extra>"
        )
    )
)


fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)"
)

style_plot(
    fig,
    590
)

st.plotly_chart(
    fig,
    use_container_width=True
)


m1, m2, m3 = st.columns(3)


with m1:

    st.metric(
        "상관계수",
        f"{correlation:+.3f}"
    )


with m2:

    st.metric(
        "사용한 연도",
        f"{year_count}개",
        f"{start_year}–{end_year}"
    )


with m3:

    st.metric(
        "전체 기간 기울기",
        f"{model['slope'] * 100:+.2f}℃",
        "℃ / 100년"
    )


# ============================================================
# 03 REGRESSION EQUATION
# ============================================================

st.markdown(
    '<div class="section-kicker">'
    '03 / REGRESSION'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">'
    '기울기와 편향으로 직선을 읽기'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-copy">'
    '수업의 기준에 따라 1908년을 0으로 두고 '
    '연도에서 1908을 뺀 값을 독립 변수로 사용합니다.'
    '</div>',
    unsafe_allow_html=True
)


equation = (
    f"예측값 = "
    f"{model['slope']:.5f} × "
    f"(연도 − 1908) + "
    f"{model['intercept']:.2f}"
)


st.markdown(
    f"""
<div class="formula">
{equation}
</div>
""",
    unsafe_allow_html=True
)


e1, e2, e3 = st.columns(3)


with e1:

    st.markdown(
        f"""
<div class="card red">

<div class="label">
기울기
</div>

<div class="value">
{model["slope"]:+.5f}
</div>

<div class="detail">
℃ / 1년
</div>

</div>
""",
        unsafe_allow_html=True
    )


with e2:

    st.markdown(
        f"""
<div class="card">

<div class="label">
100년 기울기
</div>

<div class="value">
{model["slope"] * 100:+.2f}℃
</div>

<div class="detail">
℃ / 100년
</div>

</div>
""",
        unsafe_allow_html=True
    )


with e3:

    st.markdown(
        f"""
<div class="card">

<div class="label">
편향 / 절편
</div>

<div class="value">
{model["intercept"]:.2f}℃
</div>

<div class="detail">
1908년 예측값
</div>

</div>
""",
        unsafe_allow_html=True
    )


st.markdown(
    f"""
<div class="insight">

<strong>기울기 해석:</strong>

연도가 1년 증가할 때 회귀선의 예상 연평균기온은
약 <strong>{model["slope"]:+.5f}℃</strong> 변합니다.

100년 기준으로 환산하면
<strong>{model["slope"] * 100:+.2f}℃ / 100년</strong>입니다.

<br><br>

<strong>편향 / 절편:</strong>

(연도 − 1908)이 0인 1908년에서
회귀선이 예측하는 값은
<strong>{model["intercept"]:.2f}℃</strong>입니다.

</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# 04 PREDICTION
# ============================================================

st.markdown(
    '<div class="section-kicker">'
    '04 / PREDICTION'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">'
    '기온 예측기'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-copy">'
    '왼쪽 Mission Control의 slider로 연도를 선택하면 '
    '회귀식을 이용해 예상 연평균기온을 계산합니다.'
    '</div>',
    unsafe_allow_html=True
)


p1, p2 = st.columns(2)


with p1:

    st.markdown(
        f"""
<div class="card red">

<div class="label">
SELECTED YEAR
</div>

<div class="value">
{selected_year}년
</div>

<div class="detail">
slider 선택값
</div>

</div>
""",
        unsafe_allow_html=True
    )


with p2:

    st.markdown(
        f"""
<div class="card green">

<div class="label">
EXPECTED TEMPERATURE
</div>

<div class="value">
{prediction:.1f}℃
</div>

<div class="detail">
회귀식으로 계산한 예상 연평균기온
</div>

</div>
""",
        unsafe_allow_html=True
    )


if outside_training:

    st.markdown(
        f"""
<div class="insight">

<span class="badge red">
EXTRAPOLATION
</span>

<br><br>

<strong>
{selected_year}년은 학습 범위 밖입니다.
</strong>

<br>

회귀선은
<strong>{start_year}~{end_year}년</strong>
데이터를 이용해 만들었습니다.

따라서 {selected_year}년 예측은
<strong>외삽</strong>입니다.

<br>

계산은 가능하지만 실제 미래 기온을 보장하지 않습니다.

</div>
""",
        unsafe_allow_html=True
    )

else:

    actual_row = yearly[
        yearly["연도"] == selected_year
    ]

    if not actual_row.empty:

        actual = float(
            actual_row[
                "연평균기온"
            ].iloc[0]
        )

        difference = (
            actual - prediction
        )

        st.markdown(
            f"""
<div class="insight">

<span class="badge green">
TRAINING RANGE
</span>

<br><br>

<strong>
{selected_year}년은 학습 범위 안입니다.
</strong>

<br>

실측값:
<strong>{actual:.2f}℃</strong>

<br>

회귀 예측값:
<strong>{prediction:.2f}℃</strong>

<br>

실측값 − 예측값:
<strong>{difference:+.2f}℃</strong>

</div>
""",
            unsafe_allow_html=True
        )


# ============================================================
# 05 PERIOD COMPARISON
# ============================================================

st.markdown(
    '<div class="section-kicker">'
    '05 / COMPARISON'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">'
    '학습 기간에 따라 기울기는 어떻게 달라지는가?'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-copy">'
    '전체 기간과 최근 50년, 30년, 20년을 같은 방식으로 회귀해 '
    '기울기의 차이를 비교합니다.'
    '</div>',
    unsafe_allow_html=True
)


periods = {

    "전체 114개 해":
        yearly,

    "최근 50년":
        yearly[
            yearly["연도"]
            >= end_year - 49
        ],

    "최근 30년":
        yearly[
            yearly["연도"]
            >= end_year - 29
        ],

    "최근 20년":
        yearly[
            yearly["연도"]
            >= end_year - 19
        ]
}


comparison_rows = []


for name, subset in periods.items():

    fitted = regression(
        subset
    )

    comparison_rows.append({

        "학습 기간":
            name,

        "시작 연도":
            int(subset["연도"].min()),

        "끝 연도":
            int(subset["연도"].max()),

        "사용한 해":
            len(subset),

        "기울기 (℃/100년)":
            fitted["slope"] * 100,

        f"{selected_year}년 예측값 (℃)":
            predict(
                fitted,
                selected_year
            )
    })


comparison_df = pd.DataFrame(
    comparison_rows
)


cols = st.columns(4)


for col, row in zip(
    cols,
    comparison_rows
):

    with col:

        color_class = (
            "red"
            if row["학습 기간"]
            == "전체 114개 해"
            else ""
        )

        st.markdown(
            f"""
<div class="card {color_class}">

<div class="label">
{row["학습 기간"]}
</div>

<div class="value">
{row["기울기 (℃/100년)"]:+.2f}℃
</div>

<div class="detail">
/ 100년 · n={row["사용한 해"]}
</div>

</div>
""",
            unsafe_allow_html=True
        )


display_df = comparison_df.copy()

display_df[
    "기울기 (℃/100년)"
] = display_df[
    "기울기 (℃/100년)"
].round(2)

display_df[
    f"{selected_year}년 예측값 (℃)"
] = display_df[
    f"{selected_year}년 예측값 (℃)"
].round(2)


st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# PERIOD COMPARISON GRAPH
# ============================================================

comparison_fig = go.Figure()


for name, subset in periods.items():

    fitted = regression(
        subset
    )

    x_start = int(
        subset["연도"].min()
    )

    x_end = end_year

    xs = np.linspace(
        x_start,
        x_end,
        180
    )

    ys = [
        predict(
            fitted,
            year
        )
        for year in xs
    ]

    comparison_fig.add_trace(
        go.Scatter(

            x=xs,

            y=ys,

            mode="lines",

            name=name,

            line=dict(
                width=3
                if name
                == "전체 114개 해"
                else 2,

                dash="solid"
                if name
                == "전체 114개 해"
                else "dash"
            )
        )
    )


comparison_fig.add_vline(
    x=selected_year,
    line_dash="dot",
    line_color="#ffffff",
    line_width=1
)


comparison_fig.update_layout(
    title=(
        f"학습 기간별 회귀선 "
        f"· 선택 연도 {selected_year}년"
    ),
    xaxis_title="연도",
    yaxis_title="예측 연평균기온 (℃)"
)


style_plot(
    comparison_fig,
    500
)


st.plotly_chart(
    comparison_fig,
    use_container_width=True
)


st.markdown(
    """
<div class="insight">

<strong>핵심:</strong>

같은 단순 선형회귀를 사용하더라도
어떤 기간을 학습에 포함시키느냐에 따라
회귀선의 기울기와 예측값은 달라질 수 있습니다.

따라서 예측값을 제시할 때는
<strong>어떤 기간의 데이터로 만든 회귀선인지</strong>
함께 밝혀야 합니다.

</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# 06 MODEL EVALUATION
# ============================================================

st.markdown(
    '<div class="section-kicker">'
    '06 / MODEL EVALUATION'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">'
    '회귀선은 실제 데이터와 얼마나 가까운가?'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-copy">'
    '실측값과 회귀 예측값의 차이를 잔차라고 합니다. '
    '잔차를 이용해 모델이 데이터를 얼마나 잘 요약하는지 확인합니다.'
    '</div>',
    unsafe_allow_html=True
)


v1, v2, v3 = st.columns(3)


with v1:

    st.markdown(
        f"""
<div class="card">

<div class="label">
SSE
</div>

<div class="value">
{model["sse"]:.2f}
</div>

<div class="detail">
제곱오차 합
</div>

</div>
""",
        unsafe_allow_html=True
    )


with v2:

    st.markdown(
        f"""
<div class="card">

<div class="label">
RMSE
</div>

<div class="value">
{model["rmse"]:.2f}℃
</div>

<div class="detail">
평균적인 오차 규모 참고
</div>

</div>
""",
        unsafe_allow_html=True
    )


with v3:

    st.markdown(
        f"""
<div class="card">

<div class="label">
R²
</div>

<div class="value">
{model["r2"]:.3f}
</div>

<div class="detail">
회귀모델 설명력 참고
</div>

</div>
""",
        unsafe_allow_html=True
    )


# ============================================================
# RESIDUAL GRAPH
# ============================================================

residual_fig = go.Figure()


residual_fig.add_trace(
    go.Scatter(

        x=yearly["연도"],

        y=model["residual"],

        mode="markers",

        name="잔차",

        marker=dict(
            size=8,
            color="#eeeeee",
            opacity=.75
        ),

        hovertemplate=(
            "<b>%{x}년</b><br>"
            "잔차: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


residual_fig.add_hline(
    y=0,
    line_color="#ff3b30",
    line_width=2
)


residual_fig.update_layout(
    title="잔차 · 실측값 − 회귀 예측값",
    xaxis_title="연도",
    yaxis_title="잔차 (℃)"
)


style_plot(
    residual_fig,
    430
)


st.plotly_chart(
    residual_fig,
    use_container_width=True
)


# ============================================================
# 07 DATA AUDIT
# ============================================================

st.markdown(
    '<div class="section-kicker">'
    '07 / DATA AUDIT'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">'
    '전처리 기준'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    f"""
<div class="insight">

<strong>전처리 과정</strong>

<br><br>

① 2025년 이후 데이터 제외

<br>

② 날짜가 정상적으로 읽히지 않는 행 제외

<br>

③ 평균기온이 없는 행 제외

<br>

④ 연도별 관측일 수 계산

<br>

⑤ 관측일이 {MIN_DAYS}일 미만인 연도 제외

<br>

⑥ 남은 데이터를 이용해 연평균기온 계산

<br>

⑦ 1908년을 기준으로
<strong>연도 − 1908</strong> 변수를 생성

</div>
""",
    unsafe_allow_html=True
)


with st.expander(
    "최종 분석 데이터 보기"
):

    st.dataframe(

        yearly[
            [
                "연도",
                "관측일수",
                "연평균기온",
                "1908년부터 지난 연수"
            ]
        ].round(3),

        use_container_width=True,

        hide_index=True
    )


csv_data = yearly.to_csv(
    index=False
).encode(
    "utf-8-sig"
)


st.download_button(

    "최종 분석 데이터 CSV",

    data=csv_data,

    file_name=
        "seoul_regression_yearly_data.csv",

    mime="text/csv"
)


# ============================================================
# 08 CORRELATION != CAUSATION
# ============================================================

st.markdown(
    '<div class="section-kicker">'
    '08 / JUDGEMENT'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">'
    '상관관계 ≠ 인과관계'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    f"""
<div class="insight">

<strong>상관계수 {correlation:+.3f}</strong>은

연도와 연평균기온 사이의
선형적인 관계를 숫자로 요약합니다.

<br><br>

하지만 두 변수가 함께 변화한다는 사실만으로
한 변수가 다른 변수의 원인이라고 결론 내릴 수 없습니다.

<br><br>

이번 회귀선은
<strong>시간에 따른 기온의 추세를 요약하는 모델</strong>이며,

기온 변화의 원인을 증명하는
인과모형은 아닙니다.

</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# 09 CONCEPTS
# ============================================================

st.markdown(
    '<div class="section-kicker">'
    '09 / CONCEPTS'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">'
    '모델 · 학습 · 예측 · 외삽'
    '</div>',
    unsafe_allow_html=True
)


concepts = [

    (
        "모델",
        "기울기와 편향으로 결정되는 직선"
    ),

    (
        "학습",
        "제곱오차의 합이 작아지도록 기울기와 편향을 찾는 과정"
    ),

    (
        "예측",
        "학습된 회귀식을 사용해 선택한 연도의 값을 계산하는 것"
    ),

    (
        "외삽",
        "학습 데이터 범위 밖의 값을 회귀식으로 추정하는 것"
    )
]


concept_cols = st.columns(4)


for col, (title, description) in zip(
    concept_cols,
    concepts
):

    with col:

        st.markdown(
            f"""
<div class="card">

<div class="label">
{title}
</div>

<div class="detail"
style="
font-size:.82rem;
line-height:1.6;
margin-top:10px;
color:#aaa;
">

{description}

</div>

</div>
""",
            unsafe_allow_html=True
        )


# ============================================================
# 10 ASSIGNMENT OUTPUT
# ============================================================

st.markdown(
    '<div class="section-kicker">'
    '10 / ASSIGNMENT OUTPUT'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">'
    '오늘의 산출물'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-copy">'
    '수업에서 제출할 핵심 결과를 자동으로 정리합니다.'
    '</div>',
    unsafe_allow_html=True
)


assignment = f"""
[26데과-7 회귀 — 오늘의 산출물]

1. 직선을 만드는 데 사용한 해의 개수와 기간

- 사용한 해: {year_count}개
- 기간: {start_year}~{end_year}년


2. 내가 슬라이더로 고른 해와 그때 나온 예상 기온

- 선택 연도: {selected_year}년
- 예상 연평균기온: {prediction:.1f}℃


3. 그 해가 학습 기간 안쪽인지 바깥쪽인지

- {"학습 기간 밖 → 외삽값" if outside_training else "학습 기간 안 → 학습 범위 안의 연도"}


[추가 분석]

전체 기간 기울기:
{model["slope"] * 100:+.2f}℃ / 100년

최근 {recent_period}년 기울기:
{recent_model["slope"] * 100:+.2f}℃ / 100년

상관계수:
{correlation:+.3f}

회귀식:
예측값 =
{model["slope"]:.5f}
× (연도 − 1908)
+
{model["intercept"]:.2f}


[해석할 때 주의]

- 회귀선은 데이터를 하나의 직선으로 요약한 모델이다.
- 학습 범위 밖의 외삽값은 실제 미래 기온을 보장하지 않는다.
- 상관관계만으로 인과관계를 결론 내릴 수 없다.
"""


st.code(
    assignment,
    language="text"
)


st.download_button(

    "제출용 산출물 TXT 다운로드",

    data=assignment.encode(
        "utf-8"
    ),

    file_name=
        "26데과-7_회귀_오늘의_산출물.txt",

    mime="text/plain"
)


# ============================================================
# FINAL MISSION STATUS
# ============================================================

st.markdown(
    '<div class="section-kicker">'
    'MISSION STATUS'
    '</div>',
    unsafe_allow_html=True
)

if outside_training:

    st.markdown(
        f"""
<div class="insight">

<span class="badge red">
EXTRAPOLATION ACTIVE
</span>

<br><br>

선택한 연도:
<strong>{selected_year}</strong>

<br>

학습 범위:
<strong>{start_year}–{end_year}</strong>

<br>

회귀 예상값:
<strong>{prediction:.2f}℃</strong>

<br><br>

이 결과는 학습 데이터 범위 밖의
외삽 결과입니다.

</div>
""",
        unsafe_allow_html=True
    )

else:

    st.markdown(
        f"""
<div class="insight">

<span class="badge green">
TRAINING RANGE
</span>

<br><br>

선택한 연도:
<strong>{selected_year}</strong>

<br>

학습 범위:
<strong>{start_year}–{end_year}</strong>

<br>

회귀 예상값:
<strong>{prediction:.2f}℃</strong>

</div>
""",
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    f"""
<div class="footer">

SEOUL CLIMATE // REGRESSION LAB
· SONGTAN HIGH SCHOOL
· DATA SCIENCE LESSON 08
· DATA THROUGH {MAX_YEAR}
· VALIDATION ≥ {MIN_DAYS} DAYS
· BASE YEAR {BASE_YEAR}

</div>
""",
    unsafe_allow_html=True
)
