import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from io import BytesIO

# =========================================================
# SEOUL TEMPERATURE REGRESSION LAB
# 데이터 과학 송탄고 · 8차시 회귀
# =========================================================

st.set_page_config(
    page_title="서울 기온 회귀 분석",
    page_icon="📈",
    layout="wide"
)

# =========================================================
# CONSTANTS
# =========================================================

START_YEAR = 1908
END_YEAR = 2025
MIN_DAYS = 300

KMA_API_URL = (
    "https://apis.data.go.kr/1360000/"
    "AsosDalyInfoService/getWthrDataList"
)

# Seoul ASOS station
SEOUL_STATION = "108"


# =========================================================
# BASIC STYLE
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #050505;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3 {
        color: white !important;
    }

    p, li, label {
        color: #dddddd !important;
    }

    [data-testid="stMetricValue"] {
        color: white !important;
    }

    [data-testid="stMetricLabel"] {
        color: #aaaaaa !important;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #333333;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# TITLE
# =========================================================

st.title("📈 서울 기온 회귀 분석")

st.write(
    "서울의 연도별 연평균기온을 산점도와 단순 선형회귀로 분석합니다."
)

st.caption(
    "데이터 기준: 1908~2025 · 관측일 300일 이상인 연도만 사용 · "
    "연도 − 1908을 독립변수로 사용"
)


# =========================================================
# FUNCTIONS
# =========================================================

def find_column(df, candidates):
    """
    여러 가능한 컬럼명 중 실제 데이터에 존재하는 컬럼을 찾는다.
    """
    for col in candidates:
        if col in df.columns:
            return col
    return None


def prepare_daily_data(df):
    """
    KMA/교과서 CSV 등 여러 형식의 데이터를 통일한다.
    """

    df = df.copy()

    # 컬럼명 공백 제거
    df.columns = [str(c).strip() for c in df.columns]

    # -----------------------------------------------------
    # 날짜 컬럼 찾기
    # -----------------------------------------------------

    date_col = find_column(
        df,
        [
            "날짜",
            "일시",
            "date",
            "Date",
            "DATE"
        ]
    )

    if date_col is None:
        raise ValueError(
            "날짜 컬럼을 찾을 수 없습니다. "
            "예: 날짜 또는 일시"
        )

    # -----------------------------------------------------
    # temperature column
    # -----------------------------------------------------

    temp_col = find_column(
        df,
        [
            "평균기온",
            "평균기온(℃)",
            "평균 기온",
            "avgTa",
            "mean",
            "temperature"
        ]
    )

    if temp_col is None:
        raise ValueError(
            "평균기온 컬럼을 찾을 수 없습니다."
        )

    # -----------------------------------------------------
    # convert
    # -----------------------------------------------------

    df["날짜_정리"] = pd.to_datetime(
        df[date_col],
        errors="coerce"
    )

    df["기온_정리"] = pd.to_numeric(
        df[temp_col],
        errors="coerce"
    )

    df = df.dropna(
        subset=["날짜_정리", "기온_정리"]
    ).copy()

    df["연도"] = df["날짜_정리"].dt.year

    # 수업 범위
    df = df[
        (df["연도"] >= START_YEAR) &
        (df["연도"] <= END_YEAR)
    ].copy()

    return df


def make_yearly_data(df):
    """
    일자료 → 연평균기온

    관측일 300일 이상인 연도만 사용.
    """

    yearly = (
        df.groupby("연도")["기온_정리"]
        .agg(
            연평균기온="mean",
            관측일수="count"
        )
        .reset_index()
    )

    yearly = yearly[
        yearly["관측일수"] >= MIN_DAYS
    ].copy()

    yearly = yearly.sort_values("연도")

    yearly["1908년부터_지난_연수"] = (
        yearly["연도"] - START_YEAR
    )

    return yearly


@st.cache_data
def load_local_file(file_bytes):
    """
    업로드된 CSV 읽기.
    """

    errors = []

    for encoding in [
        "utf-8",
        "utf-8-sig",
        "cp949",
        "euc-kr"
    ]:
        try:
            df = pd.read_csv(
                BytesIO(file_bytes),
                encoding=encoding
            )
            return prepare_daily_data(df)
        except Exception as e:
            errors.append(str(e))

    raise ValueError(
        "CSV를 읽을 수 없습니다.\n" +
        "\n".join(errors)
    )


@st.cache_data
def load_local_seoul_csv():
    """
    GitHub Repository에 seoul.csv가 있는 경우 사용.
    """

    errors = []

    for encoding in [
        "utf-8",
        "utf-8-sig",
        "cp949",
        "euc-kr"
    ]:

        try:
            df = pd.read_csv(
                "seoul.csv",
                encoding=encoding
            )

            return prepare_daily_data(df)

        except Exception as e:
            errors.append(str(e))

    return None


def kma_request(
    service_key,
    start_date,
    end_date,
    page_no=1,
    num_rows=1000
):
    """
    기상청 ASOS API 호출.
    """

    params = {
        "ServiceKey": service_key,
        "pageNo": page_no,
        "numOfRows": num_rows,
        "dataType": "JSON",
        "dataCd": "ASOS",
        "dateCd": "DAY",
        "startDt": start_date,
        "endDt": end_date,
        "stnIds": SEOUL_STATION
    }

    response = requests.get(
        KMA_API_URL,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    return response.json()


@st.cache_data
def load_kma_data(service_key):
    """
    1908~2025 서울 일자료를 KMA API에서 가져온다.

    API 호출 횟수를 줄이기 위해 연도 단위로 요청한다.
    """

    all_rows = []

    progress = st.progress(
        0,
        text="기상청 데이터를 가져오는 중..."
    )

    total_years = END_YEAR - START_YEAR + 1

    for i, year in enumerate(
        range(START_YEAR, END_YEAR + 1)
    ):

        start_date = f"{year}0101"
        end_date = f"{year}1231"

        try:

            page = 1

            while True:

                data = kma_request(
                    service_key,
                    start_date,
                    end_date,
                    page_no=page,
                    num_rows=1000
                )

                response = data.get(
                    "response",
                    {}
                )

                header = response.get(
                    "header",
                    {}
                )

                if str(
                    header.get("resultCode", "")
                ) not in ["00", "0"]:

                    message = header.get(
                        "resultMsg",
                        "KMA API 오류"
                    )

                    raise RuntimeError(
                        f"{year}: {message}"
                    )

                body = response.get(
                    "body",
                    {}
                )

                items = (
                    body
                    .get("items", {})
                    .get("item", [])
                )

                if isinstance(items, dict):
                    items = [items]

                if not items:
                    break

                all_rows.extend(items)

                total_count = int(
                    body.get(
                        "totalCount",
                        len(items)
                    )
                )

                if page * 1000 >= total_count:
                    break

                page += 1

        except Exception as e:

            progress.empty()

            raise RuntimeError(
                f"{year}년 데이터를 가져오는 중 오류:\n{e}"
            )

        progress.progress(
            (i + 1) / total_years,
            text=f"{year}년 데이터 수집 중..."
        )

    progress.empty()

    if not all_rows:
        raise RuntimeError(
            "기상청에서 데이터를 받지 못했습니다."
        )

    raw = pd.DataFrame(all_rows)

    # KMA field
    if "tm" not in raw.columns:
        raise RuntimeError(
            "KMA 응답에 날짜(tm)가 없습니다."
        )

    if "avgTa" not in raw.columns:
        raise RuntimeError(
            "KMA 응답에 평균기온(avgTa)이 없습니다."
        )

    df = pd.DataFrame({
        "날짜": raw["tm"],
        "평균기온": raw["avgTa"]
    })

    return prepare_daily_data(df)


def regression(yearly):
    """
    단순 선형회귀.
    x = 연도 - 1908
    y = 연평균기온
    """

    x = yearly["1908년부터_지난_연수"].to_numpy(
        dtype=float
    )

    y = yearly["연평균기온"].to_numpy(
        dtype=float
    )

    slope, intercept = np.polyfit(
        x,
        y,
        1
    )

    predicted = slope * x + intercept

    residuals = y - predicted

    sse = np.sum(
        residuals ** 2
    )

    mse = np.mean(
        residuals ** 2
    )

    rmse = np.sqrt(mse)

    ss_total = np.sum(
        (y - np.mean(y)) ** 2
    )

    r2 = (
        1 - sse / ss_total
        if ss_total != 0
        else np.nan
    )

    correlation = np.corrcoef(
        x,
        y
    )[0, 1]

    result = yearly.copy()

    result["예측기온"] = predicted
    result["잔차"] = residuals
    result["제곱오차"] = residuals ** 2

    return {
        "slope": slope,
        "intercept": intercept,
        "correlation": correlation,
        "sse": sse,
        "rmse": rmse,
        "r2": r2,
        "data": result
    }


def fit_period(yearly, start_year):
    """
    특정 시작연도부터 2025년까지 회귀.
    """

    subset = yearly[
        yearly["연도"] >= start_year
    ].copy()

    if len(subset) < 2:
        return None

    x = subset["연도"].to_numpy(
        dtype=float
    )

    y = subset["연평균기온"].to_numpy(
        dtype=float
    )

    slope, intercept = np.polyfit(
        x,
        y,
        1
    )

    return {
        "start": int(subset["연도"].min()),
        "end": int(subset["연도"].max()),
        "n": len(subset),
        "slope": slope,
        "slope100": slope * 100,
        "intercept": intercept,
        "data": subset
    }


def prediction(
    year,
    slope,
    intercept
):
    """
    x = year - 1908
    """

    return (
        slope * (year - START_YEAR)
        + intercept
    )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙ 데이터 설정")

    uploaded = st.file_uploader(
        "서울 기온 CSV 업로드",
        type=["csv"]
    )

    st.divider()

    st.subheader("기상청 API")

    st.write(
        "CSV가 없으면 KMA ServiceKey를 "
        "Streamlit Secrets에 넣어 사용할 수 있습니다."
    )

    api_key_input = st.text_input(
        "KMA ServiceKey",
        type="password",
        help=(
            "공공데이터포털에서 발급받은 "
            "기상청 ASOS API 인증키"
        )
    )

    st.divider()

    st.subheader("예측 연도")

    selected_year = st.slider(
        "연도를 선택하세요",
        min_value=1900,
        max_value=2100,
        value=2045,
        step=1
    )

    st.divider()

    st.subheader("학습 기간")

    period_choice = st.radio(
        "회귀선 학습 기간",
        [
            "전체 기간",
            "최근 50년",
            "최근 30년",
            "최근 20년"
        ],
        index=0
    )


# =========================================================
# LOAD DATA
# =========================================================

daily = None
source_name = None

# ---------------------------------------------------------
# 1. Uploaded CSV
# ---------------------------------------------------------

if uploaded is not None:

    try:

        daily = load_local_file(
            uploaded.getvalue()
        )

        source_name = "사용자가 업로드한 CSV"

    except Exception as e:

        st.error(
            f"업로드한 CSV를 읽지 못했습니다.\n\n{e}"
        )

        st.stop()


# ---------------------------------------------------------
# 2. Local seoul.csv
# ---------------------------------------------------------

if daily is None:

    try:

        local = load_local_seoul_csv()

        if local is not None:

            daily = local
            source_name = "Repository의 seoul.csv"

    except Exception:
        pass


# ---------------------------------------------------------
# 3. KMA API
# ---------------------------------------------------------

if daily is None:

    secret_key = None

    try:
        secret_key = st.secrets[
            "KMA_SERVICE_KEY"
        ]
    except Exception:
        pass

    service_key = (
        api_key_input.strip()
        if api_key_input.strip()
        else secret_key
    )

    if service_key:

        try:

            daily = load_kma_data(
                service_key
            )

            source_name = (
                "기상청 ASOS OpenAPI"
            )

        except Exception as e:

            st.error(
                "기상청 API에서 데이터를 가져오지 못했습니다."
            )

            st.code(
                str(e)
            )

            st.stop()


# ---------------------------------------------------------
# NO DATA
# ---------------------------------------------------------

if daily is None:

    st.warning(
        "서울 기온 데이터가 아직 없습니다."
    )

    st.info(
        """
        ### 가장 쉬운 방법

        Repository에 다음 파일을 넣으세요.

        `seoul.csv`

        또는 왼쪽 사이드바에서 CSV를 업로드하세요.

        ### CSV의 필수 컬럼

        날짜 + 평균기온

        예:

        날짜 | 평균기온
        --- | ---
        1908-01-01 | -2.4
        1908-01-02 | -3.1
        ...

        ### 또는

        Streamlit Secrets에:

        `KMA_SERVICE_KEY`

        를 넣으면 기상청 ASOS API에서 자동으로 데이터를 가져옵니다.
        """
    )

    st.stop()


# =========================================================
# YEARLY DATA
# =========================================================

yearly = make_yearly_data(
    daily
)

if len(yearly) < 2:

    st.error(
        "회귀 분석을 수행할 수 있는 연도 데이터가 부족합니다."
    )

    st.stop()


# =========================================================
# MAIN REGRESSION
# =========================================================

result = regression(
    yearly
)

model_data = result["data"]

slope = result["slope"]
intercept = result["intercept"]

correlation = result["correlation"]
sse = result["sse"]
rmse = result["rmse"]
r2 = result["r2"]


# =========================================================
# DATA STATUS
# =========================================================

st.success(
    f"데이터 로드 완료 · {source_name}"
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "사용한 연도",
    f"{len(yearly)}개"
)

c2.metric(
    "시작 연도",
    f"{yearly['연도'].min()}년"
)

c3.metric(
    "마지막 연도",
    f"{yearly['연도'].max()}년"
)

c4.metric(
    "평균 관측일",
    f"{yearly['관측일수'].mean():.1f}일"
)


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "① 산점도 + 회귀",
        "② 상관계수",
        "③ 학습기간 비교",
        "④ 예측",
        "⑤ 모델 평가",
        "⑥ 데이터 확인"
    ]
)


# =========================================================
# TAB 1
# =========================================================

with tab1:

    st.header(
        "① 서울 연도와 연평균기온의 관계"
    )

    st.write(
        "각 점은 관측일 300일 이상인 한 해의 연평균기온입니다."
    )

    fig = go.Figure()

    # Scatter
    fig.add_trace(
        go.Scatter(
            x=yearly["연도"],
            y=yearly["연평균기온"],
            mode="markers",
            name="실측 연평균기온",
            marker=dict(
                size=8,
                opacity=0.75
            ),
            hovertemplate=(
                "<b>%{x}년</b><br>"
                "연평균기온: %{y:.2f}℃"
                "<extra></extra>"
            )
        )
    )

    # Regression line
    x_line = np.linspace(
        yearly["연도"].min(),
        yearly["연도"].max(),
        300
    )

    y_line = prediction(
        x_line,
        slope,
        intercept
    )

    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name="회귀 직선",
            line=dict(
                width=3
            ),
            hovertemplate=(
                "%{x:.0f}년<br>"
                "회귀값: %{y:.2f}℃"
                "<extra></extra>"
            )
        )
    )

    fig.update_layout(
        height=620,
        template="plotly_dark",
        xaxis_title="연도",
        yaxis_title="연평균기온 (℃)",
        hovermode="x unified",
        legend_title=""
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("회귀식")

    st.latex(
        rf"""
        \hat{{y}} =
        {slope:.5f}(연도-1908)
        + {intercept:.3f}
        """
    )

    a, b, c = st.columns(3)

    a.metric(
        "기울기",
        f"{slope * 100:+.2f} ℃ / 100년"
    )

    b.metric(
        "편향 / 절편",
        f"{intercept:.2f} ℃"
    )

    c.metric(
        "사용한 연도",
        f"{len(yearly)}개"
    )


# =========================================================
# TAB 2
# =========================================================

with tab2:

    st.header(
        "② 상관계수"
    )

    st.write(
        "연도와 연평균기온의 선형적 관계를 상관계수로 나타냅니다."
    )

    m1, m2 = st.columns(2)

    m1.metric(
        "상관계수 r",
        f"{correlation:.3f}"
    )

    m2.metric(
        "결정계수 R²",
        f"{r2:.3f}"
    )

    st.divider()

    st.write(
        f"""
        **현재 데이터에서 r = {correlation:.3f}**

        상관계수는 −1부터 +1 사이의 값이며,
        두 변수의 선형적 관계의 방향과 강도를 나타냅니다.

        단, 상관관계만으로 인과관계를 증명할 수는 없습니다.
        """
    )

    fig_corr = go.Figure()

    fig_corr.add_trace(
        go.Scatter(
            x=yearly["연도"],
            y=yearly["연평균기온"],
            mode="markers",
            name="연평균기온",
            marker=dict(
                size=9
            )
        )
    )

    fig_corr.update_layout(
        template="plotly_dark",
        height=520,
        xaxis_title="연도",
        yaxis_title="연평균기온 (℃)"
    )

    st.plotly_chart(
        fig_corr,
        use_container_width=True
    )

    st.info(
        "중요: 상관관계 ≠ 인과관계"
    )


# =========================================================
# TAB 3
# =========================================================

with tab3:

    st.header(
        "③ 학습 기간에 따라 회귀선은 어떻게 달라지는가?"
    )

    st.write(
        "교과서 실습에서는 전체 기간과 최근 50·30·20년을 비교합니다."
    )

    periods = {
        "전체 기간": START_YEAR,
        "최근 50년": END_YEAR - 49,
        "최근 30년": END_YEAR - 29,
        "최근 20년": END_YEAR - 19
    }

    comparison_rows = []

    for name, start in periods.items():

        fitted = fit_period(
            yearly,
            start
        )

        if fitted:

            comparison_rows.append(
                {
                    "학습 기간": name,
                    "시작 연도": fitted["start"],
                    "끝 연도": fitted["end"],
                    "사용 연도 수": fitted["n"],
                    "기울기 (℃/100년)": fitted["slope100"]
                }
            )

    comparison = pd.DataFrame(
        comparison_rows
    )

    st.dataframe(
        comparison.style.format(
            {
                "기울기 (℃/100년)": "{:+.2f}"
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    fig_period = go.Figure()

    # actual
    fig_period.add_trace(
        go.Scatter(
            x=yearly["연도"],
            y=yearly["연평균기온"],
            mode="markers",
            name="실측값",
            marker=dict(
                size=6,
                opacity=0.45
            )
        )
    )

    line_specs = [
        ("전체 기간", START_YEAR),
        ("최근 50년", END_YEAR - 49),
        ("최근 30년", END_YEAR - 29),
        ("최근 20년", END_YEAR - 19)
    ]

    for name, start in line_specs:

        fitted = fit_period(
            yearly,
            start
        )

        if fitted is None:
            continue

        x_values = np.linspace(
            fitted["start"],
            END_YEAR,
            200
        )

        y_values = (
            fitted["slope"] * x_values
            + fitted["intercept"]
        )

        fig_period.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines",
                name=(
                    f"{name} "
                    f"({fitted['slope100']:+.2f}℃/100년)"
                ),
                line=dict(
                    width=3
                )
            )
        )

    fig_period.update_layout(
        template="plotly_dark",
        height=650,
        xaxis_title="연도",
        yaxis_title="연평균기온 (℃)",
        hovermode="x unified"
    )

    st.plotly_chart(
        fig_period,
        use_container_width=True
    )

    selected_start = periods[
        period_choice
    ]

    selected_model = fit_period(
        yearly,
        selected_start
    )

    st.subheader(
        f"현재 선택: {period_choice}"
    )

    if selected_model:

        x1, x2, x3 = st.columns(3)

        x1.metric(
            "학습 시작",
            f"{selected_model['start']}년"
        )

        x2.metric(
            "학습 연도 수",
            f"{selected_model['n']}개"
        )

        x3.metric(
            "기울기",
            f"{selected_model['slope100']:+.2f} ℃/100년"
        )


# =========================================================
# TAB 4
# =========================================================

with tab4:

    st.header(
        "④ 연도별 예상 연평균기온"
    )

    if period_choice == "전체 기간":

        active_model = fit_period(
            yearly,
            START_YEAR
        )

    elif period_choice == "최근 50년":

        active_model = fit_period(
            yearly,
            END_YEAR - 49
        )

    elif period_choice == "최근 30년":

        active_model = fit_period(
            yearly,
            END_YEAR - 29
        )

    else:

        active_model = fit_period(
            yearly,
            END_YEAR - 19
        )

    predicted_temperature = (
        active_model["slope"] * selected_year
        + active_model["intercept"]
    )

    st.metric(
        f"{selected_year}년 예상 연평균기온",
        f"{predicted_temperature:.1f}℃"
    )

    st.caption(
        f"학습 범위: "
        f"{active_model['start']}~"
        f"{active_model['end']}년"
    )

    if (
        selected_year <
        active_model["start"]
        or
        selected_year >
        active_model["end"]
    ):

        st.warning(
            "학습 범위 밖의 값입니다. "
            "이것은 외삽(extrapolation)입니다."
        )

        st.write(
            "회귀선으로 계산한 값일 뿐 실제 미래 기온을 보장하지 않습니다."
        )

    else:

        st.success(
            "선택한 연도가 학습 범위 안에 있습니다."
        )

        actual_row = yearly[
            yearly["연도"] == selected_year
        ]

        if not actual_row.empty:

            actual = float(
                actual_row.iloc[0]["연평균기온"]
            )

            difference = (
                actual -
                predicted_temperature
            )

            a, b, c = st.columns(3)

            a.metric(
                "실측값",
                f"{actual:.2f}℃"
            )

            b.metric(
                "예측값",
                f"{predicted_temperature:.2f}℃"
            )

            c.metric(
                "실측 - 예측",
                f"{difference:+.2f}℃"
            )

    # Prediction curve
    prediction_years = np.arange(
        1900,
        2101
    )

    prediction_values = (
        active_model["slope"]
        * prediction_years
        + active_model["intercept"]
    )

    fig_pred = go.Figure()

    fig_pred.add_trace(
        go.Scatter(
            x=prediction_years,
            y=prediction_values,
            mode="lines",
            name="회귀선"
        )
    )

    fig_pred.add_vline(
        x=selected_year,
        line_dash="dash",
        annotation_text=str(
            selected_year
        )
    )

    fig_pred.update_layout(
        template="plotly_dark",
        height=560,
        xaxis_title="연도",
        yaxis_title="예상 연평균기온 (℃)"
    )

    st.plotly_chart(
        fig_pred,
        use_container_width=True
    )


# =========================================================
# TAB 5
# =========================================================

with tab5:

    st.header(
        "⑤ 모델 평가"
    )

    st.write(
        "실측값과 회귀선의 차이를 이용하여 모델의 오차를 확인합니다."
    )

    a, b, c, d = st.columns(4)

    a.metric(
        "SSE",
        f"{sse:.2f}"
    )

    b.metric(
        "RMSE",
        f"{rmse:.3f}℃"
    )

    c.metric(
        "R²",
        f"{r2:.3f}"
    )

    d.metric(
        "상관계수",
        f"{correlation:.3f}"
    )

    st.divider()

    # Residual graph
    fig_res = go.Figure()

    fig_res.add_trace(
        go.Scatter(
            x=model_data["연도"],
            y=model_data["잔차"],
            mode="markers",
            name="잔차",
            hovertemplate=(
                "%{x}년<br>"
                "잔차: %{y:.2f}℃"
                "<extra></extra>"
            )
        )
    )

    fig_res.add_hline(
        y=0,
        line_dash="dash"
    )

    fig_res.update_layout(
        template="plotly_dark",
        height=550,
        xaxis_title="연도",
        yaxis_title="잔차 = 실측값 − 예측값"
    )

    st.plotly_chart(
        fig_res,
        use_container_width=True
    )

    st.subheader(
        "최근 3개 연도의 실측값과 예측값"
    )

    recent_rows = model_data.tail(3)[
        [
            "연도",
            "연평균기온",
            "예측기온",
            "잔차",
            "제곱오차"
        ]
    ].copy()

    st.dataframe(
        recent_rows.style.format(
            {
                "연평균기온": "{:.2f}",
                "예측기온": "{:.2f}",
                "잔차": "{:+.2f}",
                "제곱오차": "{:.3f}"
            }
        ),
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# TAB 6
# =========================================================

with tab6:

    st.header(
        "⑥ 데이터 확인"
    )

    st.write(
        "전처리 후 실제 회귀에 사용된 연도입니다."
    )

    st.dataframe(
        yearly[
            [
                "연도",
                "연평균기온",
                "관측일수",
                "1908년부터_지난_연수"
            ]
        ].style.format(
            {
                "연평균기온": "{:.2f}"
            }
        ),
        use_container_width=True,
        height=600,
        hide_index=True
    )

    st.divider()

    st.subheader(
        "데이터 전처리 기준"
    )

    st.write(
        f"""
        - 분석 기간: {START_YEAR}~{END_YEAR}
        - 관측일 수 기준: {MIN_DAYS}일 이상
        - 사용한 연도: {len(yearly)}개
        - 회귀 독립변수: 연도 − {START_YEAR}
        - 종속변수: 연평균기온
        """
    )

    csv = yearly.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        "📥 전처리된 연평균 데이터 다운로드",
        data=csv,
        file_name="seoul_yearly_temperature_1908_2025.csv",
        mime="text/csv"
    )


# =========================================================
# ASSIGNMENT OUTPUT
# =========================================================

st.divider()

st.header(
    "📝 오늘의 산출물"
)

selected_prediction = prediction(
    selected_year,
    slope,
    intercept
)

assignment_text = f"""
[26데과-7] 회귀 — 오늘의 산출물

데이터:
서울 일별 평균기온 데이터

분석 기간:
{yearly["연도"].min()}~{yearly["연도"].max()}년

전처리 기준:
관측일 300일 이상

회귀에 사용한 연도:
{len(yearly)}개

상관계수:
{correlation:.3f}

회귀식:
예측값 = {slope:.5f} × (연도 - 1908) + {intercept:.3f}

전체 기간 기울기:
{slope * 100:+.2f}℃ / 100년

선택한 학습 기간:
{period_choice}

선택한 연도:
{selected_year}년

예측 연평균기온:
{selected_prediction:.2f}℃

RMSE:
{rmse:.3f}℃

R²:
{r2:.3f}

주의:
학습 범위 밖의 연도는 외삽값이며
실제 미래 기온을 보장하지 않는다.

또한 상관관계만으로 인과관계를 판단할 수 없다.
"""

st.code(
    assignment_text,
    language="text"
)

st.download_button(
    "📄 산출물 텍스트 다운로드",
    data=assignment_text,
    file_name="회귀_오늘의_산출물.txt",
    mime="text/plain"
)


# =========================================================
# SOURCE
# =========================================================

st.divider()

st.caption(
    "수업 기준: 데이터 과학 송탄고 · 8차시 회귀"
)

st.caption(
    "기상청 ASOS 일자료 / 공공데이터포털"
)
