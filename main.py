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
# 간단한 CSS
# =========================================================

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1200px;
        padding-top: 35px;
        padding-bottom: 60px;
    }

    h1 {
        font-size: 42px !important;
        margin-bottom: 5px;
    }

    h2 {
        margin-top: 35px;
    }

    .small-text {
        color: #777777;
        font-size: 15px;
    }

    .result-box {
        padding: 18px;
        border: 1px solid #dddddd;
        border-radius: 10px;
        background: #fafafa;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 제목
# =========================================================

st.title("📈 서울 기온 회귀 분석")

st.write(
    "서울의 연도별 연평균기온과 연도의 관계를 "
    "산점도와 단순 선형회귀로 분석합니다."
)

st.caption(
    "분석 기준: 1908~2025년 · 관측일 300일 이상인 연도 사용"
)


# =========================================================
# 데이터 불러오기
# =========================================================

@st.cache_data
def load_data():

    # Repository에 있는 seoul.csv
    df = pd.read_csv(
        "seoul.csv",
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
    )

    df["연도"] = df["날짜"].dt.year

    # 1908~2025
    df = df[
        (df["연도"] >= 1908) &
        (df["연도"] <= 2025)
    ]

    # 연평균
    yearly = (
        df.groupby("연도")["평균기온"]
        .agg(
            연평균기온="mean",
            관측일수="count"
        )
        .reset_index()
    )

    # 300일 이상
    yearly = yearly[
        yearly["관측일수"] >= 300
    ].copy()

    yearly = yearly.sort_values(
        "연도"
    )

    # 1908년을 0으로
    yearly["x"] = (
        yearly["연도"] - 1908
    )

    return yearly


# =========================================================
# 데이터 오류 처리
# =========================================================

try:

    yearly = load_data()

except FileNotFoundError:

    st.error(
        "seoul.csv 파일을 찾을 수 없습니다."
    )

    st.info(
        """
        GitHub Repository에

        `main.py`

        와

        `seoul.csv`

        를 같은 폴더에 넣어주세요.
        """
    )

    st.stop()

except Exception as e:

    st.error(
        f"데이터를 읽는 중 오류가 발생했습니다: {e}"
    )

    st.stop()


# =========================================================
# 회귀 계산
# =========================================================

x = yearly["x"].to_numpy()
y = yearly["연평균기온"].to_numpy()

slope, intercept = np.polyfit(
    x,
    y,
    1
)

yearly["예측기온"] = (
    slope * yearly["x"]
    + intercept
)

yearly["잔차"] = (
    yearly["연평균기온"]
    - yearly["예측기온"]
)

# 상관계수
correlation = (
    yearly["연도"]
    .corr(yearly["연평균기온"])
)

# 제곱오차합
sse = np.sum(
    yearly["잔차"] ** 2
)

# RMSE
rmse = np.sqrt(
    np.mean(yearly["잔차"] ** 2)
)

# R²
ss_total = np.sum(
    (
        yearly["연평균기온"]
        - yearly["연평균기온"].mean()
    ) ** 2
)

r2 = 1 - (
    sse / ss_total
)


# =========================================================
# 기본 데이터 정보
# =========================================================

st.success(
    f"데이터 로드 완료 · {len(yearly)}개 연도 사용"
)


a, b, c = st.columns(3)

a.metric(
    "사용한 연도",
    f"{len(yearly)}개"
)

b.metric(
    "시작 연도",
    f"{yearly['연도'].min()}년"
)

c.metric(
    "마지막 연도",
    f"{yearly['연도'].max()}년"
)


# =========================================================
# 1. 산점도 + 회귀선
# =========================================================

st.header("1. 산점도와 회귀선")

st.write(
    "각 점은 한 해의 연평균기온입니다. "
    "직선은 모든 데이터를 가장 잘 대표하도록 계산한 회귀선입니다."
)


fig = go.Figure()


# 실제 데이터
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="실측값",
        marker=dict(
            size=8
        ),
        hovertemplate=(
            "%{x}년<br>"
            "연평균기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


# 회귀선
line_x = np.linspace(
    yearly["연도"].min(),
    yearly["연도"].max(),
    300
)

line_x_relative = (
    line_x - 1908
)

line_y = (
    slope * line_x_relative
    + intercept
)


fig.add_trace(
    go.Scatter(
        x=line_x,
        y=line_y,
        mode="lines",
        name="회귀선",
        line=dict(
            width=3
        )
    )
)


fig.update_layout(
    height=600,
    template="plotly_white",
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    hovermode="x unified"
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# =========================================================
# 2. 상관계수
# =========================================================

st.header("2. 상관계수")

st.write(
    "연도와 연평균기온이 얼마나 함께 변하는지를 나타냅니다."
)


a, b = st.columns(2)

a.metric(
    "상관계수",
    f"{correlation:.3f}"
)

b.metric(
    "결정계수 R²",
    f"{r2:.3f}"
)


st.info(
    "상관관계가 있다고 해서 인과관계가 있다는 뜻은 아닙니다."
)


# =========================================================
# 3. 회귀식
# =========================================================

st.header("3. 회귀 직선")

st.write(
    "수업에서는 1908년을 0으로 두고 "
    "연도에서 1908을 뺀 값을 사용합니다."
)

st.latex(
    rf"""
    \hat{{y}}
    =
    {slope:.5f}
    (연도-1908)
    +
    {intercept:.3f}
    """
)


a, b = st.columns(2)

a.metric(
    "기울기",
    f"{slope * 100:+.2f} ℃ / 100년"
)

b.metric(
    "편향",
    f"{intercept:.2f} ℃"
)


st.write(
    f"""
    **기울기 해석**

    이 데이터에서 회귀선의 기울기는
    **{slope * 100:+.2f}℃ / 100년**입니다.

    즉, 전체 1908~2025년 데이터를 하나의 직선으로 요약했을 때
    100년당 평균적으로 약 {abs(slope * 100):.2f}℃의 변화량을
    나타내는 회귀선입니다.
    """
)


# =========================================================
# 4. 예측
# =========================================================

st.header("4. 연평균기온 예측")

year = st.slider(
    "예측할 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2045
)


prediction = (
    slope * (year - 1908)
    + intercept
)


st.metric(
    f"{year}년 예상 연평균기온",
    f"{prediction:.1f}℃"
)


# 실제값이 존재하는 경우
actual = yearly[
    yearly["연도"] == year
]

if not actual.empty:

    actual_value = float(
        actual.iloc[0]["연평균기온"]
    )

    difference = (
        actual_value
        - prediction
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "실제값",
        f"{actual_value:.2f}℃"
    )

    c2.metric(
        "예측값",
        f"{prediction:.2f}℃"
    )

    c3.metric(
        "실제값 - 예측값",
        f"{difference:+.2f}℃"
    )

else:

    if (
        year < yearly["연도"].min()
        or
        year > yearly["연도"].max()
    ):

        st.warning(
            "이 연도는 학습 범위 밖입니다. "
            "외삽값이므로 실제 미래 기온을 보장하지 않습니다."
        )


# =========================================================
# 예측 그래프
# =========================================================

prediction_years = np.arange(
    1900,
    2101
)

prediction_values = (
    slope
    * (prediction_years - 1908)
    + intercept
)


fig2 = go.Figure()


fig2.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="실측값",
        marker=dict(
            size=6,
            opacity=0.55
        )
    )
)


fig2.add_trace(
    go.Scatter(
        x=prediction_years,
        y=prediction_values,
        mode="lines",
        name="회귀선",
        line=dict(
            width=3
        )
    )
)


fig2.add_trace(
    go.Scatter(
        x=[year],
        y=[prediction],
        mode="markers",
        name=f"{year}년 예측",
        marker=dict(
            size=14
        )
    )
)


fig2.update_layout(
    height=550,
    template="plotly_white",
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)"
)


st.plotly_chart(
    fig2,
    use_container_width=True
)


# =========================================================
# 5. 학습 기간 비교
# =========================================================

st.header("5. 학습 기간에 따른 회귀선 비교")

st.write(
    "같은 방법으로 전체 기간, 최근 50년, 최근 30년, 최근 20년을 비교합니다."
)


def make_regression(data):

    xx = data["연도"].to_numpy()
    yy = data["연평균기온"].to_numpy()

    aa, bb = np.polyfit(
        xx,
        yy,
        1
    )

    return aa, bb


periods = {
    "전체": 1908,
    "최근 50년": 1976,
    "최근 30년": 1996,
    "최근 20년": 2006
}


comparison = []


for name, start_year in periods.items():

    temp = yearly[
        yearly["연도"] >= start_year
    ].copy()

    aa, bb = make_regression(
        temp
    )

    slope100 = aa * 100

    pred_2045 = (
        aa * 2045
        + bb
    )

    comparison.append(
        {
            "학습 기간": name,
            "시작": int(temp["연도"].min()),
            "끝": int(temp["연도"].max()),
            "사용 연도": len(temp),
            "기울기": slope100,
            "2045년 예측": pred_2045
        }
    )


comparison_df = pd.DataFrame(
    comparison
)


st.dataframe(
    comparison_df.style.format(
        {
            "기울기": "{:+.2f} ℃/100년",
            "2045년 예측": "{:.1f}℃"
        }
    ),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 비교 그래프
# =========================================================

fig3 = go.Figure()


# 실제값
fig3.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="실측값",
        marker=dict(
            size=5,
            opacity=0.35
        )
    )
)


for name, start_year in periods.items():

    temp = yearly[
        yearly["연도"] >= start_year
    ]

    aa, bb = make_regression(
        temp
    )

    xx = np.linspace(
        start_year,
        2025,
        150
    )

    yy = (
        aa * xx
        + bb
    )

    fig3.add_trace(
        go.Scatter(
            x=xx,
            y=yy,
            mode="lines",
            name=name,
            line=dict(
                width=3
            )
        )
    )


fig3.update_layout(
    height=600,
    template="plotly_white",
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)"
)


st.plotly_chart(
    fig3,
    use_container_width=True
)


# =========================================================
# 6. 모델 평가
# =========================================================

st.header("6. 모델 평가")

a, b, c = st.columns(3)

a.metric(
    "SSE",
    f"{sse:.2f}"
)

b.metric(
    "RMSE",
    f"{rmse:.2f}℃"
)

c.metric(
    "R²",
    f"{r2:.3f}"
)


# =========================================================
# 잔차 그래프
# =========================================================

st.subheader("잔차")

fig4 = go.Figure()


fig4.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["잔차"],
        mode="markers",
        name="잔차"
    )
)


fig4.add_hline(
    y=0,
    line_dash="dash"
)


fig4.update_layout(
    height=450,
    template="plotly_white",
    xaxis_title="연도",
    yaxis_title="실측값 - 예측값 (℃)"
)


st.plotly_chart(
    fig4,
    use_container_width=True
)


# =========================================================
# 7. 마지막 3년 확인
# =========================================================

st.header("7. 최근 3개 연도 확인")

last_three = yearly.tail(3)[
    [
        "연도",
        "연평균기온",
        "예측기온",
        "잔차"
    ]
].copy()


st.dataframe(
    last_three.style.format(
        {
            "연평균기온": "{:.2f}℃",
            "예측기온": "{:.2f}℃",
            "잔차": "{:+.2f}℃"
        }
    ),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 8. 오늘의 산출물
# =========================================================

st.header("8. 오늘의 산출물")

st.write(
    "과제 제출에 필요한 핵심 숫자입니다."
)


st.code(
    f"""
서울 기온 회귀 분석

사용한 데이터:
서울 연평균기온

분석 기간:
1908~2025년

사용한 연도:
{len(yearly)}개

상관계수:
{correlation:.3f}

전체 기간 기울기:
{slope * 100:+.2f}℃ / 100년

회귀식:
예측값 = {slope:.5f} × (연도 - 1908) + {intercept:.3f}

{year}년 예상 연평균기온:
{prediction:.1f}℃

RMSE:
{rmse:.2f}℃

R²:
{r2:.3f}

주의:
학습 범위 밖의 값은 외삽값이며
실제 미래 기온을 보장하지 않는다.

상관관계만으로 인과관계를 판단할 수 없다.
""",
    language="text"
)


# =========================================================
# 데이터 표
# =========================================================

with st.expander("전체 연평균 데이터 보기"):

    st.dataframe(
        yearly[
            [
                "연도",
                "연평균기온",
                "관측일수"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )
