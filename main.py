import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error

# ============================================================
# 기본 설정
# ============================================================

st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 서울 기온 예측기")
st.write(
    "서울의 연평균기온 데이터를 이용하여 회귀 직선을 만들고, "
    "1차·3차·9차 다항 회귀 모델을 비교합니다."
)

# ============================================================
# 데이터 불러오기
# ============================================================

URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

@st.cache_data
def load_data():
    df = pd.read_csv(URL, encoding="utf-8")

    # 날짜 변환
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")

    # 기온을 숫자로 변환
    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    # 필요한 값이 없는 행 제거
    df = df.dropna(
        subset=["날짜", "평균기온"]
    ).copy()

    # 연도 만들기
    df["연도"] = df["날짜"].dt.year

    return df


df = load_data()

# ============================================================
# 2025년까지 + 관측일 300일 이상인 연도만 사용
# ============================================================

year_count = (
    df.groupby("연도")
    .size()
    .reset_index(name="관측일수")
)

year_mean = (
    df.groupby("연도")["평균기온"]
    .mean()
    .reset_index(name="연평균기온")
)

yearly = pd.merge(
    year_mean,
    year_count,
    on="연도"
)

# 2025년 이후 제거
# 관측일이 300일 미만인 연도 제거
yearly = yearly[
    (yearly["연도"] <= 2025) &
    (yearly["관측일수"] >= 300)
].copy()

yearly = yearly.sort_values("연도").reset_index(drop=True)

# ============================================================
# 기본 정보
# ============================================================

st.subheader("📌 데이터 정보")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "사용한 연도 수",
        f"{len(yearly)}개"
    )

with col2:
    st.metric(
        "시작 연도",
        f"{int(yearly['연도'].min())}년"
    )

with col3:
    st.metric(
        "끝 연도",
        f"{int(yearly['연도'].max())}년"
    )

st.write(
    "2025년까지의 데이터 중 관측일수가 300일 이상인 연도만 사용했습니다."
)

# ============================================================
# 상관계수
# ============================================================

correlation = yearly["연도"].corr(
    yearly["연평균기온"]
)

st.subheader("📈 연도와 연평균기온의 상관관계")

st.metric(
    "상관계수",
    f"{correlation:.3f}"
)

# ============================================================
# 기존 수업 내용 — 회귀 직선
# ============================================================

st.subheader("📏 연도와 연평균기온의 회귀 직선")

# 1908년부터 몇 년이 지났는지를 독립변수로 사용
yearly["경과연수"] = yearly["연도"] - 1908

# 1차 회귀
linear_coef = np.polyfit(
    yearly["경과연수"],
    yearly["연평균기온"],
    1
)

linear_slope = linear_coef[0]
linear_intercept = linear_coef[1]

yearly["회귀예측"] = (
    linear_slope * yearly["경과연수"]
    + linear_intercept
)

# 회귀식
st.write(
    f"회귀식: "
    f"연평균기온 = "
    f"{linear_slope:.4f} × (연도 - 1908) "
    f"+ {linear_intercept:.2f}"
)

# ============================================================
# 기존 산점도 + 회귀 직선
# ============================================================

fig_linear = go.Figure()

fig_linear.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="실제 연평균기온"
    )
)

line_years = np.linspace(
    yearly["연도"].min(),
    yearly["연도"].max(),
    300
)

line_x = line_years - 1908

line_y = (
    linear_slope * line_x
    + linear_intercept
)

fig_linear.add_trace(
    go.Scatter(
        x=line_years,
        y=line_y,
        mode="lines",
        name="회귀 직선"
    )
)

fig_linear.update_layout(
    title="서울 연평균기온과 회귀 직선",
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    height=550
)

st.plotly_chart(
    fig_linear,
    use_container_width=True
)

# ============================================================
# 연도 슬라이더
# ============================================================

st.subheader("🎚️ 연도를 선택해서 기온 예측하기")

selected_year = st.slider(
    "예측할 연도",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)

selected_elapsed = selected_year - 1908

selected_prediction = (
    linear_slope * selected_elapsed
    + linear_intercept
)

st.metric(
    f"{selected_year}년 예상 연평균기온",
    f"{selected_prediction:.2f} ℃"
)

# ============================================================
# ============================================================
# 심화 탐구
# 직선 대신 곡선을 쓰면
# ============================================================
# ============================================================

st.divider()

st.header("🔬 심화 탐구 프로젝트")
st.subheader("직선 대신 곡선을 쓰면")

st.write(
    "2005년 이전 데이터를 훈련용으로 사용하고, "
    "2005년부터의 데이터를 테스트용으로 사용합니다."
)

st.write(
    "1차·3차·9차 다항 회귀 모델을 각각 훈련한 뒤 "
    "같은 테스트용 데이터에서 평균절대오차(MAE)를 비교합니다."
)

# ============================================================
# 훈련용 / 테스트용 분리
# ============================================================

학습 = yearly[
    yearly["연도"] < 2005
].copy()

평가 = yearly[
    yearly["연도"] >= 2005
].copy()

# 데이터가 충분한지 확인
if len(학습) < 10:
    st.warning(
        f"훈련용 데이터가 {len(학습)}개입니다. "
        "다항 회귀 결과가 불안정할 수 있습니다."
    )

if len(평가) == 0:
    st.error(
        "2005년 이후 테스트용 데이터가 없습니다."
    )
    st.stop()

# ============================================================
# 고차 다항식용 연도 변환
# ============================================================
#
# 2005^9 같은 큰 숫자를 그대로 계산하지 않도록
# 연도를 작은 값으로 변환합니다.
#
# x = (연도 - 1950) / 100
#
# ============================================================

def 변환연도(year):
    return (year - 1950) / 100


# ============================================================
# 1차 / 3차 / 9차 모델 계산
# ============================================================

rows = []

models = {}

for degree in [1, 3, 9]:

    # 훈련용 데이터로만 모델 학습
    x_train = 변환연도(
        학습["연도"].to_numpy()
    )

    y_train = 학습["연평균기온"].to_numpy()

    coefficients = np.polyfit(
        x_train,
        y_train,
        degree
    )

    models[degree] = coefficients

    # 테스트용 데이터 예측
    x_test = 변환연도(
        평가["연도"].to_numpy()
    )

    y_test = 평가["연평균기온"].to_numpy()

    test_prediction = np.polyval(
        coefficients,
        x_test
    )

    # MAE
    test_mae = mean_absolute_error(
        y_test,
        test_prediction
    )

    # 2050년 예측
    prediction_2050 = np.polyval(
        coefficients,
        변환연도(2050)
    )

    rows.append(
        {
            "곡선": f"{degree}차",
            "테스트용 데이터의 오차(MAE)": round(
                test_mae,
                2
            ),
            "2050년 예측(℃)": round(
                prediction_2050,
                1
            )
        }
    )

# 결과표
result_table = pd.DataFrame(rows)

# ============================================================
# 결과 표
# ============================================================

st.subheader("📊 1차·3차·9차 모델 비교")

st.dataframe(
    result_table,
    hide_index=True,
    use_container_width=True
)

st.caption(
    f"훈련용 {len(학습)}개 연도 · "
    f"테스트용 {len(평가)}개 연도 · "
    "테스트 오차는 평균절대오차(MAE)입니다."
)

# ============================================================
# 가장 좋은 테스트 모델
# ============================================================

best_index = result_table[
    "테스트용 데이터의 오차(MAE)"
].idxmin()

best_model = result_table.loc[
    best_index,
    "곡선"
]

best_error = result_table.loc[
    best_index,
    "테스트용 데이터의 오차(MAE)"
]

st.success(
    f"테스트용 데이터에서 가장 작은 오차를 보인 모델은 "
    f"**{best_model}**이며, MAE는 **{best_error:.2f}℃**입니다."
)

# ============================================================
# 1차·3차·9차 회귀곡선 그래프
# ============================================================

st.subheader("📈 1차·3차·9차 회귀 곡선 비교")

fig_curve = go.Figure()

# 실제 데이터
fig_curve.add_trace(
    go.Scatter(
        x=학습["연도"],
        y=학습["연평균기온"],
        mode="markers",
        name="훈련용 데이터"
    )
)

fig_curve.add_trace(
    go.Scatter(
        x=평가["연도"],
        y=평가["연평균기온"],
        mode="markers",
        name="테스트용 데이터"
    )
)

# 곡선 그리기
curve_years = np.linspace(
    yearly["연도"].min(),
    2050,
    700
)

for degree in [1, 3, 9]:

    coefficients = models[degree]

    curve_x = 변환연도(curve_years)

    curve_y = np.polyval(
        coefficients,
        curve_x
    )

    fig_curve.add_trace(
        go.Scatter(
            x=curve_years,
            y=curve_y,
            mode="lines",
            name=f"{degree}차 회귀"
        )
    )

# 2005년 경계선
fig_curve.add_vline(
    x=2005,
    line_dash="dash",
    annotation_text="2005년: 훈련 → 테스트"
)

fig_curve.update_layout(
    title="1차·3차·9차 회귀 곡선",
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    height=600
)

st.plotly_chart(
    fig_curve,
    use_container_width=True
)

# ============================================================
# 2050년 예측 비교
# ============================================================

st.subheader("🔮 2050년 예측값 비교")

prediction_2050_table = result_table[
    ["곡선", "2050년 예측(℃)"]
].copy()

st.dataframe(
    prediction_2050_table,
    hide_index=True,
    use_container_width=True
)

# ============================================================
# 과대적합 설명
# ============================================================

st.subheader("🧠 결과 해석")

st.write(
    "차수가 높아지면 훈련용 데이터의 점들에 "
    "더 가깝게 맞출 수 있습니다."
)

st.write(
    "하지만 훈련용 데이터에 지나치게 맞춰진 모델은 "
    "새로운 테스트용 데이터에서 오히려 오차가 커질 수 있습니다."
)

st.info(
    "이러한 현상을 **과대적합(과대적합, Overfitting)**이라고 합니다."
)

# ============================================================
# 2050년 외삽 주의
# ============================================================

st.warning(
    "⚠️ 2050년 예측값은 학습 범위 밖의 외삽(Extrapolation)입니다. "
    "따라서 이 값은 실제 기후 예보가 아니라 "
    "회귀 모델을 이용해 계산한 예시값입니다."
)

# ============================================================
# 학습 범위 정보
# ============================================================

st.subheader("📚 학습 범위와 테스트 범위")

info_col1, info_col2 = st.columns(2)

with info_col1:
    st.write(
        f"**훈련용 데이터:** "
        f"{int(학습['연도'].min())}년 ~ "
        f"{int(학습['연도'].max())}년"
    )

    st.write(
        f"**훈련용 연도 수:** {len(학습)}개"
    )

with info_col2:
    st.write(
        f"**테스트용 데이터:** "
        f"{int(평가['연도'].min())}년 ~ "
        f"{int(평가['연도'].max())}년"
    )

    st.write(
        f"**테스트용 연도 수:** {len(평가)}개"
    )

# ============================================================
# 원본 연평균 데이터 확인
# ============================================================

with st.expander("📋 사용한 연평균기온 데이터 보기"):

    st.dataframe(
        yearly[
            [
                "연도",
                "연평균기온",
                "관측일수"
            ]
        ],
        hide_index=True,
        use_container_width=True
    )
