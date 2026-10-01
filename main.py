import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from io import BytesIO
import requests


# ============================================================
# 0. PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SEOUL TEMPERATURE // REGRESSION LAB",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 1. GLOBAL SETTINGS
# ============================================================

BASE_YEAR = 1908
MIN_DAYS = 300

# Рабочие внешние источники.
# Первый старый URL НЕ используется.
REMOTE_URLS = [
    "https://raw.githubusercontent.com/greatsong/2021dataset/master/seoul.csv",
    "https://raw.githubusercontent.com/greatsong/2021dataset/main/seoul.csv",
]


# ============================================================
# 2. CSS — SPACE / MISSION CONTROL STYLE
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- MAIN ---------- */

    .stApp {
        background:
            radial-gradient(circle at 50% -10%, rgba(255, 70, 70, 0.10), transparent 35%),
            linear-gradient(180deg, #050505 0%, #090909 45%, #030303 100%);
        color: #f2f2f2;
    }

    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }

    [data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, #090909 0%, #050505 100%);
        border-right: 1px solid #292929;
    }

    [data-testid="stSidebar"] * {
        color: #eeeeee !important;
    }

    /* ---------- TEXT ---------- */

    h1, h2, h3 {
        color: #ffffff !important;
        letter-spacing: -0.03em;
    }

    p, li, label {
        color: #dddddd !important;
    }

    .small-gray {
        color: #8e8e8e;
        font-size: 0.82rem;
    }

    .red {
        color: #ff4b4b;
    }

    .green {
        color: #7ee787;
    }

    .yellow {
        color: #f2cc60;
    }

    /* ---------- HERO ---------- */

    .hero {
        border: 1px solid #2d2d2d;
        background:
            linear-gradient(135deg, rgba(255,255,255,0.035), rgba(255,255,255,0.01));
        padding: 32px 34px;
        border-radius: 18px;
        margin-bottom: 24px;
        box-shadow:
            0 0 40px rgba(255, 40, 40, 0.04),
            inset 0 0 40px rgba(255,255,255,0.015);
    }

    .hero-top {
        font-size: 0.75rem;
        letter-spacing: 0.25em;
        color: #ff4b4b;
        font-weight: 700;
        margin-bottom: 12px;
    }

    .hero-title {
        font-size: 3rem;
        line-height: 1;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 12px;
    }

    .hero-sub {
        color: #a7a7a7;
        font-size: 1rem;
        line-height: 1.6;
    }

    /* ---------- SECTION ---------- */

    .section {
        margin-top: 34px;
        margin-bottom: 12px;
        padding-bottom: 9px;
        border-bottom: 1px solid #252525;
    }

    .section-number {
        color: #ff4b4b;
        font-size: 0.75rem;
        letter-spacing: 0.20em;
        font-weight: 800;
    }

    .section-title {
        color: white;
        font-size: 1.55rem;
        font-weight: 750;
    }

    /* ---------- METRIC CARDS ---------- */

    div[data-testid="stMetric"] {
        background: #0b0b0b;
        border: 1px solid #272727;
        border-radius: 14px;
        padding: 16px;
    }

    div[data-testid="stMetricLabel"] {
        color: #8f8f8f !important;
    }

    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
    }

    /* ---------- BUTTONS ---------- */

    .stButton > button {
        background: #111111;
        color: white;
        border: 1px solid #343434;
        border-radius: 9px;
        min-height: 42px;
        font-weight: 650;
    }

    .stButton > button:hover {
        border-color: #ff4b4b;
        color: #ff6666;
        background: #151515;
    }

    /* ---------- INPUTS ---------- */

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background-color: #0d0d0d !important;
        border-color: #333333 !important;
    }

    input {
        color: #ffffff !important;
    }

    /* ---------- DATAFRAME ---------- */

    [data-testid="stDataFrame"] {
        border: 1px solid #292929;
        border-radius: 12px;
        overflow: hidden;
    }

    /* ---------- ALERTS ---------- */

    .stAlert {
        border-radius: 12px;
    }

    /* ---------- CODE ---------- */

    code {
        color: #ff7b7b !important;
    }

    /* ---------- DIVIDER ---------- */

    hr {
        border-color: #242424 !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def section(number, title):
    st.markdown(
        f"""
        <div class="section">
            <div class="section-number">{number}</div>
            <div class="section-title">{title}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def clean_column_name(x):
    return str(x).replace("\ufeff", "").strip()


def read_csv_safely(source):
    """
    CSV를 CP949 / UTF-8-SIG / UTF-8 순서로 시도.
    """

    encodings = [
        "cp949",
        "euc-kr",
        "utf-8-sig",
        "utf-8"
    ]

    last_error = None

    for encoding in encodings:
        try:
            if isinstance(source, (str, Path)):
                return pd.read_csv(source, encoding=encoding)

            if isinstance(source, bytes):
                return pd.read_csv(
                    BytesIO(source),
                    encoding=encoding
                )

        except Exception as e:
            last_error = e

    raise ValueError(
        f"CSV를 읽을 수 없습니다. 마지막 오류: {last_error}"
    )


def find_temperature_column(df):
    """
    평균기온 컬럼 이름 자동 탐색.
    """

    columns = [clean_column_name(c) for c in df.columns]
    df.columns = columns

    candidates = [
        "평균기온(℃)",
        "평균기온",
        "평균기온(°C)",
        "평균기온(도)",
        "Average Temperature",
        "avg_temp"
    ]

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    # 이름에 평균기온이 들어가는 컬럼 자동 탐색
    for col in df.columns:
        if "평균기온" in col:
            return col

    # 영문 탐색
    for col in df.columns:
        lower = col.lower()

        if (
            "average" in lower
            and "temp" in lower
        ):
            return col

    return None


def find_date_column(df):
    """
    날짜 컬럼 자동 탐색.
    """

    candidates = [
        "날짜",
        "일시",
        "date",
        "Date",
        "DATE"
    ]

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    for col in df.columns:
        if "날짜" in col or "일시" in col:
            return col

    for col in df.columns:
        if "date" in str(col).lower():
            return col

    return None


# ============================================================
# 4. LOAD DATA
# ============================================================

@st.cache_data(show_spinner=False)
def load_local_csv(path):
    return read_csv_safely(path)


@st.cache_data(show_spinner=False)
def load_remote_csv(url):
    response = requests.get(
        url,
        timeout=20,
        headers={
            "User-Agent": "Mozilla/5.0"
        }
    )

    response.raise_for_status()

    return read_csv_safely(response.content)


def normalize_weather_data(raw_df):
    """
    Raw daily weather data -> cleaned daily dataframe.
    """

    df = raw_df.copy()

    df.columns = [
        clean_column_name(c)
        for c in df.columns
    ]

    date_column = find_date_column(df)
    temp_column = find_temperature_column(df)

    if date_column is None:
        raise ValueError(
            "날짜 컬럼을 찾지 못했습니다."
        )

    if temp_column is None:
        raise ValueError(
            "평균기온 컬럼을 찾지 못했습니다."
        )

    df["날짜"] = pd.to_datetime(
        df[date_column],
        errors="coerce"
    )

    df["평균기온"] = pd.to_numeric(
        df[temp_column],
        errors="coerce"
    )

    df = df.dropna(
        subset=["날짜", "평균기온"]
    ).copy()

    df["연도"] = df["날짜"].dt.year

    df = df[
        df["연도"] >= BASE_YEAR
    ].copy()

    df = df.sort_values("날짜")

    return df


def make_yearly_data(daily_df):
    """
    Daily -> annual mean.

    300일 이상 관측된 해만 사용.
    """

    grouped = (
        daily_df
        .groupby("연도")
        .agg(
            평균기온=("평균기온", "mean"),
            관측일수=("평균기온", "count")
        )
        .reset_index()
    )

    yearly = grouped[
        grouped["관측일수"] >= MIN_DAYS
    ].copy()

    yearly = yearly.sort_values("연도")

    yearly["평균기온"] = yearly[
        "평균기온"
    ].round(3)

    return yearly


def load_data(uploaded_file=None):
    """
    우선순위:

    1. 사용자가 업로드한 CSV
    2. 같은 폴더의 seoul.csv
    3. GitHub 공개 CSV
    """

    errors = []

    # --------------------------------------------------------
    # 1. Uploaded CSV
    # --------------------------------------------------------

    if uploaded_file is not None:

        try:
            raw = read_csv_safely(
                uploaded_file.getvalue()
            )

            daily = normalize_weather_data(raw)
            yearly = make_yearly_data(daily)

            if len(yearly) >= 5:
                return daily, yearly, "사용자 업로드 CSV"

        except Exception as e:
            errors.append(
                f"업로드 CSV: {e}"
            )

    # --------------------------------------------------------
    # 2. Local seoul.csv
    # --------------------------------------------------------

    possible_files = [
        Path("seoul.csv"),
        Path("./data/seoul.csv"),
        Path("./dataset/seoul.csv")
    ]

    for path in possible_files:

        if path.exists():

            try:
                raw = load_local_csv(
                    str(path)
                )

                daily = normalize_weather_data(raw)
                yearly = make_yearly_data(daily)

                if len(yearly) >= 5:
                    return (
                        daily,
                        yearly,
                        f"로컬 파일: {path}"
                    )

            except Exception as e:
                errors.append(
                    f"{path}: {e}"
                )

    # --------------------------------------------------------
    # 3. Remote GitHub fallback
    # --------------------------------------------------------

    for url in REMOTE_URLS:

        try:
            raw = load_remote_csv(url)

            daily = normalize_weather_data(raw)
            yearly = make_yearly_data(daily)

            if len(yearly) >= 5:
                return (
                    daily,
                    yearly,
                    "GitHub 공개 데이터"
                )

        except Exception as e:
            errors.append(
                f"{url}: {e}"
            )

    # --------------------------------------------------------
    # Nothing worked
    # --------------------------------------------------------

    error_text = "\n".join(errors)

    raise RuntimeError(
        "서울 기온 데이터를 불러오지 못했습니다.\n\n"
        + error_text
    )


# ============================================================
# 5. SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            font-size:0.72rem;
            letter-spacing:0.22em;
            color:#ff4b4b;
            font-weight:800;
        ">
        DATA SCIENCE / 08
        </div>

        <div style="
            font-size:1.35rem;
            font-weight:800;
            margin-top:6px;
            margin-bottom:22px;
        ">
        REGRESSION LAB
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### DATA INPUT")

    uploaded_file = st.file_uploader(
        "서울 기온 CSV",
        type=["csv"],
        help=(
            "seoul.csv가 있다면 여기에 업로드하세요. "
            "업로드하지 않으면 앱이 자동으로 공개 데이터를 찾습니다."
        )
    )

    st.markdown("---")

    st.markdown("### MODEL CONTROL")

    prediction_year = st.slider(
        "예측 연도",
        min_value=2026,
        max_value=2100,
        value=2050,
        step=1
    )

    comparison_window = st.selectbox(
        "최근 기간 비교",
        options=[20, 30, 50],
        index=1,
        format_func=lambda x: f"최근 {x}년"
    )

    st.markdown("---")

    st.markdown("### DISPLAY")

    show_daily_data = st.checkbox(
        "일별 원본 데이터 표시",
        value=False
    )

    show_residuals = st.checkbox(
        "잔차 데이터 표시",
        value=True
    )

    st.markdown("---")

    st.markdown(
        """
        <div class="small-gray">
        8차시 · 회귀<br>
        직선을 긋다<br><br>

        문제 정의 → 데이터 수집 → 전처리/탐색
        → 모델링 → 평가 → 활용
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 6. LOAD DATA
# ============================================================

try:

    daily, yearly, data_source = load_data(
        uploaded_file
    )

except Exception as e:

    st.markdown(
        """
        <div class="hero">

        <div class="hero-top">
        DATA INPUT ERROR
        </div>

        <div class="hero-title">
        SEOUL TEMPERATURE
        </div>

        <div class="hero-sub">
        데이터를 찾지 못했지만 앱 자체는 정상적으로 실행되었습니다.
        아래의 CSV 업로드 버튼을 사용하면 즉시 분석을 시작할 수 있습니다.
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.error(str(e))

    st.info(
        """
        `seoul.csv` 파일을 GitHub에서 `main.py`와 같은 폴더에 넣거나,
        왼쪽 사이드바의 CSV 업로드 버튼으로 업로드하세요.

        필요한 컬럼:
        - 날짜
        - 평균기온 또는 평균기온(℃)
        """
    )

    st.stop()


# ============================================================
# 7. BASIC VARIABLES
# ============================================================

start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())

first_temp = float(
    yearly.iloc[0]["평균기온"]
)

last_temp = float(
    yearly.iloc[-1]["평균기온"]
)

temperature_change = last_temp - first_temp

n_years = len(yearly)

years = yearly["연도"].to_numpy(
    dtype=float
)

temps = yearly["평균기온"].to_numpy(
    dtype=float
)


# ============================================================
# 8. REGRESSION MODEL
# ============================================================

# x = year - 1908
x = years - BASE_YEAR
y = temps

slope, intercept = np.polyfit(
    x,
    y,
    1
)

predicted = (
    slope * x
    + intercept
)

residuals = y - predicted

# correlation
if len(x) >= 2:
    correlation = float(
        np.corrcoef(x, y)[0, 1]
    )
else:
    correlation = np.nan

# SSE
sse = float(
    np.sum(
        residuals ** 2
    )
)

# RMSE
rmse = float(
    np.sqrt(
        np.mean(
            residuals ** 2
        )
    )
)

# R²
sst = float(
    np.sum(
        (y - np.mean(y)) ** 2
    )
)

if sst == 0:
    r2 = np.nan
else:
    r2 = float(
        1 - sse / sst
    )


# prediction
prediction_x = (
    prediction_year
    - BASE_YEAR
)

prediction_value = (
    slope * prediction_x
    + intercept
)


# ============================================================
# 9. HERO
# ============================================================

st.markdown(
    f"""
    <div class="hero">

        <div class="hero-top">
            SEOUL · CLIMATE · DATA SCIENCE · 08
        </div>

        <div class="hero-title">
            REGRESSION LAB
        </div>

        <div class="hero-sub">
            서울 연평균기온 데이터를 이용해
            상관관계 → 선형회귀 → 예측 → 평가까지
            하나의 분석 시스템으로 확인합니다.
        </div>

        <br>

        <div class="small-gray">
            DATA RANGE
            <span style="color:#ffffff;">
                {start_year} — {end_year}
            </span>
            &nbsp;&nbsp;|&nbsp;&nbsp;

            VALID YEARS
            <span style="color:#ffffff;">
                {n_years}
            </span>
            &nbsp;&nbsp;|&nbsp;&nbsp;

            SOURCE
            <span style="color:#ffffff;">
                {data_source}
            </span>
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 10. TOP METRICS
# ============================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "START",
        f"{first_temp:.2f} ℃",
        f"{start_year}"
    )

with c2:
    st.metric(
        "LATEST",
        f"{last_temp:.2f} ℃",
        f"{end_year}"
    )

with c3:
    st.metric(
        "CHANGE",
        f"{temperature_change:+.2f} ℃"
    )

with c4:
    st.metric(
        "CORRELATION r",
        f"{correlation:.4f}"
    )


# ============================================================
# 11. DATA TELEMETRY
# ============================================================

section(
    "01",
    "DATA TELEMETRY · 데이터 구조 확인"
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "관측 데이터",
        f"{len(daily):,} rows"
    )

with col2:
    st.metric(
        "연간 데이터",
        f"{len(yearly):,} years"
    )

with col3:
    st.metric(
        "기준",
        f"{MIN_DAYS}일 이상"
    )

st.caption(
    "일별 평균기온을 연도별로 집계하고, "
    f"관측일수가 {MIN_DAYS}일 이상인 해만 분석에 사용했습니다."
)

st.dataframe(
    yearly,
    use_container_width=True,
    hide_index=True,
    column_config={
        "연도": st.column_config.NumberColumn(
            "연도",
            format="%d"
        ),
        "평균기온": st.column_config.NumberColumn(
            "연평균기온 (℃)",
            format="%.3f"
        ),
        "관측일수": st.column_config.NumberColumn(
            "관측일수",
            format="%d"
        )
    }
)


if show_daily_data:

    with st.expander(
        "일별 원본 데이터 열기"
    ):

        st.dataframe(
            daily,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# 12. ANNUAL TEMPERATURE GRAPH
# ============================================================

section(
    "02",
    "EXPLORATION · 연평균기온 변화"
)

fig_temperature = go.Figure()

fig_temperature.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["평균기온"],
        mode="lines+markers",
        name="연평균기온",
        line=dict(
            width=2
        ),
        marker=dict(
            size=5
        ),
        hovertemplate=
        "<b>%{x}년</b><br>"
        "연평균기온: %{y:.2f} ℃"
        "<extra></extra>"
    )
)

fig_temperature.update_layout(
    height=520,
    template="plotly_dark",
    paper_bgcolor="#050505",
    plot_bgcolor="#050505",
    xaxis_title="연도",
    yaxis_title="평균기온 (℃)",
    hovermode="x unified",
    margin=dict(
        l=20,
        r=20,
        t=45,
        b=20
    )
)

st.plotly_chart(
    fig_temperature,
    use_container_width=True
)


# ============================================================
# 13. RELATIONSHIP / SCATTER
# ============================================================

section(
    "03",
    "RELATION · 연도와 기온의 관계"
)

fig_scatter = go.Figure()

fig_scatter.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["평균기온"],
        mode="markers",
        name="관측값",
        marker=dict(
            size=7
        ),
        hovertemplate=
        "연도: %{x}<br>"
        "평균기온: %{y:.2f} ℃"
        "<extra></extra>"
    )
)

regression_line = (
    slope * (years - BASE_YEAR)
    + intercept
)

fig_scatter.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=regression_line,
        mode="lines",
        name="회귀직선",
        line=dict(
            width=3
        ),
        hovertemplate=
        "회귀값: %{y:.2f} ℃"
        "<extra></extra>"
    )
)

fig_scatter.update_layout(
    height=540,
    template="plotly_dark",
    paper_bgcolor="#050505",
    plot_bgcolor="#050505",
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    hovermode="closest"
)

st.plotly_chart(
    fig_scatter,
    use_container_width=True
)


r1, r2, r3 = st.columns(3)

with r1:
    st.metric(
        "상관계수 r",
        f"{correlation:.4f}"
    )

with r2:
    st.metric(
        "기울기",
        f"{slope:.5f} ℃/year"
    )

with r3:
    st.metric(
        "결정계수 R²",
        f"{r2:.4f}"
    )


st.info(
    f"""
    **상관관계 해석**

    연도와 연평균기온의 피어슨 상관계수는
    **r = {correlation:.4f}** 입니다.

    단, 상관관계가 존재한다고 해서
    연도가 기온 변화의 직접적인 원인이라고
    단정할 수는 없습니다.
    """
)


# ============================================================
# 14. REGRESSION
# ============================================================

section(
    "04",
    "REGRESSION · 직선을 긋다"
)

st.markdown(
    """
    단순선형회귀에서는 다음 형태의 직선을 찾습니다.

    **y = ax + b**

    여기서 x는 기준 연도 이후의 경과 연수,
    y는 연평균기온입니다.
    """
)

formula = (
    f"y = {slope:.6f} × (연도 − {BASE_YEAR}) "
    f"+ {intercept:.4f}"
)

st.code(
    formula,
    language="text"
)

a1, a2, a3 = st.columns(3)

with a1:
    st.metric(
        "a · 기울기",
        f"{slope:.6f}"
    )

with a2:
    st.metric(
        "b · 절편",
        f"{intercept:.4f}"
    )

with a3:
    st.metric(
        "R²",
        f"{r2:.4f}"
    )


# ============================================================
# 15. REGRESSION DIAGRAM
# ============================================================

fig_regression = go.Figure()

fig_regression.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["평균기온"],
        mode="markers",
        name="실제 데이터",
        marker=dict(
            size=6
        )
    )
)

fig_regression.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=regression_line,
        mode="lines",
        name="선형회귀",
        line=dict(
            width=4
        )
    )
)

fig_regression.update_layout(
    height=540,
    template="plotly_dark",
    paper_bgcolor="#050505",
    plot_bgcolor="#050505",
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)"
)

st.plotly_chart(
    fig_regression,
    use_container_width=True
)


# ============================================================
# 16. PREDICTION
# ============================================================

section(
    "05",
    "PREDICTION · 미래값 계산"
)

p1, p2 = st.columns([1, 2])

with p1:

    st.markdown(
        f"""
        <div style="
            background:#0b0b0b;
            border:1px solid #2b2b2b;
            border-radius:16px;
            padding:25px;
        ">

        <div class="small-gray">
        TARGET YEAR
        </div>

        <div style="
            font-size:2.7rem;
            font-weight:800;
            color:white;
        ">
        {prediction_year}
        </div>

        <div class="small-gray">
        회귀모델에 입력
        </div>

        <hr style="border-color:#292929;">

        <div class="small-gray">
        PREDICTED TEMPERATURE
        </div>

        <div style="
            font-size:2.3rem;
            font-weight:800;
            color:#ff5555;
        ">
        {prediction_value:.2f} ℃
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

with p2:

    future_years = np.arange(
        start_year,
        max(end_year, prediction_year) + 1
    )

    future_x = (
        future_years
        - BASE_YEAR
    )

    future_prediction = (
        slope * future_x
        + intercept
    )

    fig_future = go.Figure()

    fig_future.add_trace(
        go.Scatter(
            x=yearly["연도"],
            y=yearly["평균기온"],
            mode="markers",
            name="실제값",
            marker=dict(
                size=5
            )
        )
    )

    fig_future.add_trace(
        go.Scatter(
            x=future_years,
            y=future_prediction,
            mode="lines",
            name="회귀 예측",
            line=dict(
                width=3,
                dash="dash"
            )
        )
    )

    fig_future.add_trace(
        go.Scatter(
            x=[prediction_year],
            y=[prediction_value],
            mode="markers",
            name=f"{prediction_year} 예측",
            marker=dict(
                size=14,
                symbol="diamond"
            )
        )
    )

    fig_future.update_layout(
        height=470,
        template="plotly_dark",
        paper_bgcolor="#050505",
        plot_bgcolor="#050505",
        xaxis_title="연도",
        yaxis_title="예상 연평균기온 (℃)"
    )

    st.plotly_chart(
        fig_future,
        use_container_width=True
    )


st.warning(
    """
    ⚠️ 이것은 실제 기상예보가 아닙니다.

    과거 데이터에 적합한 단순선형회귀식을
    미래 연도까지 외삽한 결과입니다.
    따라서 실제 미래의 기온을 보장하지 않습니다.
    """
)


# ============================================================
# 17. RECENT PERIOD COMPARISON
# ============================================================

section(
    "06",
    "COMPARISON · 전체 기간과 최근 기간"
)

recent_n = comparison_window

recent = yearly.tail(
    recent_n
).copy()

recent_years = recent["연도"].to_numpy(
    dtype=float
)

recent_temps = recent["평균기온"].to_numpy(
    dtype=float
)

if len(recent) >= 2:

    recent_slope, recent_intercept = np.polyfit(
        recent_years - BASE_YEAR,
        recent_temps,
        1
    )

    recent_pred = (
        recent_slope
        * (recent_years - BASE_YEAR)
        + recent_intercept
    )

    recent_corr = float(
        np.corrcoef(
            recent_years,
            recent_temps
        )[0, 1]
    )

    recent_sse = np.sum(
        (recent_temps - recent_pred) ** 2
    )

    recent_sst = np.sum(
        (recent_temps - np.mean(recent_temps)) ** 2
    )

    recent_r2 = (
        1 - recent_sse / recent_sst
        if recent_sst != 0
        else np.nan
    )

else:

    recent_slope = np.nan
    recent_intercept = np.nan
    recent_corr = np.nan
    recent_r2 = np.nan


comparison_df = pd.DataFrame(
    {
        "구간": [
            f"{start_year}~{end_year}",
            f"{int(recent['연도'].min())}~{int(recent['연도'].max())}"
        ],
        "데이터 수": [
            len(yearly),
            len(recent)
        ],
        "기울기 (℃/년)": [
            slope,
            recent_slope
        ],
        "상관계수": [
            correlation,
            recent_corr
        ],
        "R²": [
            r2,
            recent_r2
        ]
    }
)

st.dataframe(
    comparison_df,
    use_container_width=True,
    hide_index=True
)


fig_comparison = go.Figure()

fig_comparison.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["평균기온"],
        mode="lines",
        name="전체 기간",
        line=dict(
            width=2
        )
    )
)

fig_comparison.add_trace(
    go.Scatter(
        x=recent["연도"],
        y=recent["평균기온"],
        mode="lines+markers",
        name=f"최근 {recent_n}년",
        line=dict(
            width=4
        ),
        marker=dict(
            size=7
        )
    )
)

fig_comparison.update_layout(
    height=500,
    template="plotly_dark",
    paper_bgcolor="#050505",
    plot_bgcolor="#050505",
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)"
)

st.plotly_chart(
    fig_comparison,
    use_container_width=True
)


# ============================================================
# 18. RESIDUAL ANALYSIS
# ============================================================

section(
    "07",
    "MODEL EVALUATION · 잔차 분석"
)

residual_df = yearly.copy()

residual_df["예측값"] = predicted.round(3)

residual_df["잔차"] = residuals.round(3)

residual_df["절대잔차"] = (
    np.abs(residuals)
    .round(3)
)

fig_residual = go.Figure()

fig_residual.add_trace(
    go.Scatter(
        x=residual_df["연도"],
        y=residual_df["잔차"],
        mode="lines+markers",
        name="잔차",
        marker=dict(
            size=5
        ),
        hovertemplate=
        "연도: %{x}<br>"
        "잔차: %{y:.3f} ℃"
        "<extra></extra>"
    )
)

fig_residual.add_hline(
    y=0,
    line_width=2
)

fig_residual.update_layout(
    height=450,
    template="plotly_dark",
    paper_bgcolor="#050505",
    plot_bgcolor="#050505",
    xaxis_title="연도",
    yaxis_title="잔차 (실제값 − 예측값)"
)

st.plotly_chart(
    fig_residual,
    use_container_width=True
)

e1, e2, e3 = st.columns(3)

with e1:
    st.metric(
        "RMSE",
        f"{rmse:.3f} ℃"
    )

with e2:
    st.metric(
        "SSE",
        f"{sse:.2f}"
    )

with e3:
    st.metric(
        "R²",
        f"{r2:.4f}"
    )

if show_residuals:

    with st.expander(
        "잔차 데이터 보기"
    ):

        st.dataframe(
            residual_df[
                [
                    "연도",
                    "평균기온",
                    "예측값",
                    "잔차",
                    "절대잔차"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# 19. MODEL INTERPRETATION
# ============================================================

section(
    "08",
    "MODEL INTERPRETATION · 모델 읽기"
)

if slope > 0:
    trend_text = (
        f"회귀모델의 기울기는 "
        f"{slope:.5f} ℃/년으로 양수입니다."
    )
else:
    trend_text = (
        f"회귀모델의 기울기는 "
        f"{slope:.5f} ℃/년으로 음수입니다."
    )

st.markdown(
    f"""
    ### ① 기울기

    {trend_text}

    즉, 단순선형회귀 모델에서는
    시간이 지나면서 연평균기온이 변화하는
    선형 경향을 나타냅니다.

    ### ② 상관관계

    상관계수는 **{correlation:.4f}** 입니다.

    상관계수는 두 변수의 선형적인 관계 정도를
    나타내지만, 상관관계만으로 인과관계를 증명할 수는 없습니다.

    ### ③ 결정계수

    R² = **{r2:.4f}**

    회귀모델이 관측된 연평균기온의 변동을
    어느 정도 설명하는지를 나타내는 지표입니다.

    ### ④ 오차

    RMSE = **{rmse:.3f} ℃**

    실제값과 회귀모델 예측값의 차이를
    평균적인 크기로 해석할 수 있습니다.
    """
)


# ============================================================
# 20. DATA QUALITY AUDIT
# ============================================================

section(
    "09",
    "DATA AUDIT · 데이터 품질 확인"
)

audit1, audit2, audit3, audit4 = st.columns(4)

missing_temperature = int(
    daily["평균기온"].isna().sum()
)

duplicate_dates = int(
    daily["날짜"].duplicated().sum()
)

min_observation_days = int(
    yearly["관측일수"].min()
)

max_observation_days = int(
    yearly["관측일수"].max()
)

with audit1:
    st.metric(
        "결측 기온",
        missing_temperature
    )

with audit2:
    st.metric(
        "중복 날짜",
        duplicate_dates
    )

with audit3:
    st.metric(
        "최소 관측일수",
        min_observation_days
    )

with audit4:
    st.metric(
        "최대 관측일수",
        max_observation_days
    )

st.success(
    "전처리 완료: 날짜/기온 결측값을 제거하고 "
    f"{MIN_DAYS}일 이상 관측된 연도만 연평균 계산에 사용했습니다."
)


# ============================================================
# 21. HISTOGRAM
# ============================================================

section(
    "10",
    "DISTRIBUTION · 연평균기온 분포"
)

fig_hist = px.histogram(
    yearly,
    x="평균기온",
    nbins=20,
    template="plotly_dark"
)

fig_hist.update_layout(
    height=420,
    paper_bgcolor="#050505",
    plot_bgcolor="#050505",
    xaxis_title="연평균기온 (℃)",
    yaxis_title="연도 수"
)

st.plotly_chart(
    fig_hist,
    use_container_width=True
)


# ============================================================
# 22. HIGHEST / LOWEST YEARS
# ============================================================

section(
    "11",
    "RECORDS · 극값 확인"
)

highest = yearly.loc[
    yearly["평균기온"].idxmax()
]

lowest = yearly.loc[
    yearly["평균기온"].idxmin()
]

q1, q2 = st.columns(2)

with q1:

    st.markdown(
        f"""
        <div style="
            background:#0b0b0b;
            border:1px solid #2b2b2b;
            border-radius:15px;
            padding:25px;
        ">

        <div class="small-gray">
        HIGHEST ANNUAL MEAN
        </div>

        <div style="
            font-size:2.4rem;
            font-weight:800;
            color:#ff5555;
        ">
        {highest['평균기온']:.2f} ℃
        </div>

        <div>
        {int(highest['연도'])}년
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

with q2:

    st.markdown(
        f"""
        <div style="
            background:#0b0b0b;
            border:1px solid #2b2b2b;
            border-radius:15px;
            padding:25px;
        ">

        <div class="small-gray">
        LOWEST ANNUAL MEAN
        </div>

        <div style="
            font-size:2.4rem;
            font-weight:800;
            color:#7aa2ff;
        ">
        {lowest['평균기온']:.2f} ℃
        </div>

        <div>
        {int(lowest['연도'])}년
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 23. TOP 10 WARMEST YEARS
# ============================================================

section(
    "12",
    "TOP RECORDS · 가장 높은 연평균기온"
)

top10 = (
    yearly
    .sort_values(
        "평균기온",
        ascending=False
    )
    .head(10)
    .copy()
)

fig_top10 = px.bar(
    top10.sort_values("평균기온"),
    x="평균기온",
    y="연도",
    orientation="h",
    text="평균기온",
    template="plotly_dark"
)

fig_top10.update_traces(
    texttemplate="%{text:.2f} ℃",
    textposition="outside"
)

fig_top10.update_layout(
    height=500,
    paper_bgcolor="#050505",
    plot_bgcolor="#050505",
    xaxis_title="연평균기온 (℃)",
    yaxis_title="연도"
)

st.plotly_chart(
    fig_top10,
    use_container_width=True
)


# ============================================================
# 24. RECENT VS EARLY
# ============================================================

section(
    "13",
    "PERIOD ANALYSIS · 시대별 비교"
)

periods = []

for label, data in [
    ("초기 20년", yearly.head(20)),
    ("중간 20년", yearly.iloc[
        max(0, len(yearly)//2 - 10):
        min(len(yearly), len(yearly)//2 + 10)
    ]),
    ("최근 20년", yearly.tail(20))
]:

    if len(data) > 0:

        periods.append(
            {
                "기간": label,
                "시작연도": int(
                    data["연도"].min()
                ),
                "종료연도": int(
                    data["연도"].max()
                ),
                "평균기온": float(
                    data["평균기온"].mean()
                )
            }
        )

period_df = pd.DataFrame(
    periods
)

st.dataframe(
    period_df,
    use_container_width=True,
    hide_index=True
)

fig_period = px.bar(
    period_df,
    x="기간",
    y="평균기온",
    text="평균기온",
    template="plotly_dark"
)

fig_period.update_traces(
    texttemplate="%{text:.2f} ℃",
    textposition="outside"
)

fig_period.update_layout(
    height=400,
    paper_bgcolor="#050505",
    plot_bgcolor="#050505",
    yaxis_title="평균기온 (℃)"
)

st.plotly_chart(
    fig_period,
    use_container_width=True
)


# ============================================================
# 25. CONCEPTS
# ============================================================

section(
    "14",
    "CONCEPTS · 핵심 개념"
)

concept1, concept2 = st.columns(2)

with concept1:

    with st.expander(
        "상관관계 ≠ 인과관계",
        expanded=True
    ):

        st.markdown(
            """
            두 변수의 움직임이 함께 나타난다고 해서
            한 변수가 다른 변수의 원인이라는 뜻은 아닙니다.

            이 분석에서 연도와 기온 사이에
            선형적인 관계가 나타나더라도,
            그것만으로 특정 원인을 증명할 수 없습니다.
            """
        )

with concept2:

    with st.expander(
        "외삽(Extrapolation)",
        expanded=True
    ):

        st.markdown(
            """
            관측된 데이터 범위 밖으로
            회귀식을 확장하여 값을 계산하는 것입니다.

            따라서 2050년, 2100년 등의 값은
            실제 관측값이 아니라
            현재 모델을 미래까지 연장한 계산값입니다.
            """
        )


# ============================================================
# 26. ASSIGNMENT OUTPUT
# ============================================================

section(
    "15",
    "ASSIGNMENT OUTPUT · 제출용 결과"
)

submission_text = f"""
[26데과-7] 회귀 — 오늘의 산출물

주제:
서울 연도와 연평균기온의 관계를 선형회귀로 분석하였다.

데이터 기간:
{start_year}년 ~ {end_year}년

사용한 연간 데이터:
{n_years}개

전처리:
일별 평균기온을 연도별로 평균하였다.
관측일수가 {MIN_DAYS}일 미만인 연도는 제외하였다.

상관계수:
r = {correlation:.4f}

회귀식:
y = {slope:.6f} × (연도 − {BASE_YEAR}) + {intercept:.4f}

기울기:
{slope:.6f} ℃/년

결정계수:
R² = {r2:.4f}

RMSE:
{rmse:.3f} ℃

예측 연도:
{prediction_year}

회귀모델 예측값:
{prediction_value:.2f} ℃

해석:
연도와 서울 연평균기온 사이의 선형적 관계를
단순선형회귀 모델로 확인하였다.
다만 상관관계가 인과관계를 의미하는 것은 아니며,
미래 예측값은 실제 기상예보가 아니라
과거 자료에 기반한 회귀모델의 외삽값이다.
"""

st.text_area(
    "제출용 텍스트",
    submission_text,
    height=430
)


# ============================================================
# 27. DOWNLOAD CSV
# ============================================================

section(
    "16",
    "EXPORT · 분석 결과 저장"
)

csv_data = residual_df.to_csv(
    index=False,
    encoding="utf-8-sig"
)

download_col1, download_col2 = st.columns(2)

with download_col1:

    st.download_button(
        label="📥 분석 결과 CSV 다운로드",
        data=csv_data,
        file_name="seoul_regression_result.csv",
        mime="text/csv",
        use_container_width=True
    )

with download_col2:

    st.download_button(
        label="📄 제출용 텍스트 다운로드",
        data=submission_text,
        file_name="regression_submission.txt",
        mime="text/plain",
        use_container_width=True
    )


# ============================================================
# 28. FINAL JUDGEMENT
# ============================================================

section(
    "17",
    "JUDGEMENT · 데이터 과학적 결론"
)

st.markdown(
    f"""
    ### 분석 결과

    **1. 문제 정의**

    연도에 따라 서울의 연평균기온이 어떻게 변화했는지
    데이터로 확인하였다.

    **2. 데이터 수집**

    서울 관측소의 일별 평균기온 데이터를 사용하였다.

    **3. 전처리**

    날짜와 평균기온을 정리하고,
    관측일수가 {MIN_DAYS}일 이상인 연도만
    연평균 계산에 사용하였다.

    **4. 탐색**

    연도와 연평균기온의 산점도 및
    시계열 그래프를 확인하였다.

    **5. 모델링**

    단순선형회귀를 적용하여

    `{formula}`

    의 회귀식을 얻었다.

    **6. 평가**

    상관계수:
    **{correlation:.4f}**

    R²:
    **{r2:.4f}**

    RMSE:
    **{rmse:.3f} ℃**

    **7. 활용**

    회귀식을 이용하여 선택한 미래 연도의
    예상값을 계산하였다.

    단, 이 값은 실제 미래 기온을 의미하지 않으며
    과거 관측자료를 기반으로 한 모델의 외삽 결과이다.
    """
)


# ============================================================
# 29. SOURCE / FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    f"""
    <div style="
        text-align:center;
        color:#666;
        font-size:0.78rem;
        line-height:1.8;
        padding:25px;
    ">

    SEOUL TEMPERATURE REGRESSION LAB<br>

    Data Science · Lesson 08 · Regression<br>

    Analysis period: {start_year} — {end_year}<br>

    Daily observations → Annual aggregation →
    Correlation → Linear Regression → Evaluation → Prediction

    </div>
    """,
    unsafe_allow_html=True
)
