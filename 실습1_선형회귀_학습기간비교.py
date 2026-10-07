import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================================
# 페이지 설정
# ============================================================

st.set_page_config(
    page_title="실습1 · 선형회귀 학습기간 비교",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# 화면 스타일
# ============================================================

st.markdown("""
<style>

html, body, .stApp {
    background-color: white !important;
    color: #111111 !important;
}

h1, h2, h3, h4, p, span, label {
    color: #111111 !important;
}

.block-container {
    max-width: 1200px;
    padding-top: 35px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# 제목
# ============================================================

st.title("📊 실습 1 · 선형회귀 학습기간 비교")

st.write(
    "서울 연평균기온 데이터를 이용하여 "
    "최근 50년과 최근 100년의 선형회귀 모델을 만들고, "
    "공통 테스트 데이터에서 예측 성능을 비교합니다."
)

# ============================================================
# 데이터 불러오기
# ============================================================

URL = (
    "https://raw.githubusercontent.com/"
    "greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/"
    "data/seoul.csv"
)


@st.cache_data
def load_data():

    df = pd.read_csv(
        URL,
        encoding="utf-8"
    )

    # 날짜 변환
    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce"
    )

    # 평균기온 숫자 변환
    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    # 결측값 제거
    df = df.dropna(
        subset=[
            "날짜",
            "평균기온"
        ]
    ).copy()

    # 연도 만들기
    df["연도"] = df["날짜"].dt.year

    return df


df = load_data()

# ============================================================
# 연도별 평균기온 계산
# ============================================================

yearly_mean = (
    df.groupby("연도")["평균기온"]
    .mean()
    .reset_index()
)

yearly_count = (
    df.groupby("연도")
    .size()
    .reset_index(name="관측일수")
)

yearly = pd.merge(
    yearly_mean,
    yearly_count,
    on="연도"
)

yearly = yearly.rename(
    columns={
        "평균기온": "연평균기온"
    }
)

# ============================================================
# 수업 기준 적용
# 2025년 이후 제거
# 관측일 300일 미만 제거
# ============================================================

yearly = yearly[
    (yearly["연도"] <= 2025) &
    (yearly["관측일수"] >= 300)
].copy()

yearly = yearly.sort_values(
    "연도"
).reset_index(drop=True)

# ============================================================
# 데이터 정보
# ============================================================

st.subheader("1. 데이터 확인")

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

st.caption(
    "2025년까지의 데이터 중 관측일수가 300일 이상인 연도만 사용했습니다."
)

# ============================================================
# 학습 / 테스트 데이터
# ============================================================

# 최근 50년
# 1956 ~ 2005

train_50 = yearly[
    (yearly["연도"] >= 1956) &
    (yearly["연도"] <= 2005)
].copy()

# 최근 100년
# 1906 ~ 2005
#
# 실제 데이터에서 1906~1907년이 없거나
# 300일 기준을 통과하지 못하면 자동으로 제외됩니다.

train_100 = yearly[
    (yearly["연도"] >= 1906) &
    (yearly["연도"] <= 2005)
].copy()

# 공통 테스트 데이터
# 2006 ~ 2025

test = yearly[
    (yearly["연도"] >= 2006) &
    (yearly["연도"] <= 2025)
].copy()

# ============================================================
# 데이터 개수 표시
# ============================================================

st.subheader("2. 학습 데이터와 테스트 데이터")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "최근 50년 학습",
        f"{len(train_50)}개 연도"
    )
    st.write(
        f"{int(train_50['연도'].min())} ~ "
        f"{int(train_50['연도'].max())}"
    )

with col2:
    st.metric(
        "최근 100년 학습",
        f"{len(train_100)}개 연도"
    )
    st.write(
        f"{int(train_100['연도'].min())} ~ "
        f"{int(train_100['연도'].max())}"
    )

with col3:
    st.metric(
        "공통 테스트",
        f"{len(test)}개 연도"
    )
    st.write(
        f"{int(test['연도'].min())} ~ "
        f"{int(test['연도'].max())}"
    )

# ============================================================
# 테스트 데이터가 없는 경우
# ============================================================

if len(test) < 2:
    st.error(
        "2006~2025 테스트 데이터가 충분하지 않습니다."
    )
    st.stop()

# ============================================================
# 선형회귀 함수
# ============================================================

def train_linear_regression(train_data):

    X_train = train_data[
        ["연도"]
    ]

    y_train = train_data[
        "연평균기온"
    ]

    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )

    return model


# ============================================================
# 모델 학습
# ============================================================

model_50 = train_linear_regression(
    train_50
)

model_100 = train_linear_regression(
    train_100
)

# ============================================================
# 테스트 데이터 예측
# ============================================================

X_test = test[
    ["연도"]
]

y_test = test[
    "연평균기온"
]

prediction_50 = model_50.predict(
    X_test
)

prediction_100 = model_100.predict(
    X_test
)

# ============================================================
# 평가 함수
# ============================================================

def evaluate_model(
    y_true,
    y_pred
):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    mse = mean_squared_error(
        y_true,
        y_pred
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    return mae, mse, r2


mae_50, mse_50, r2_50 = evaluate_model(
    y_test,
    prediction_50
)

mae_100, mse_100, r2_100 = evaluate_model(
    y_test,
    prediction_100
)

# ============================================================
# 기울기
# ============================================================

slope_50 = model_50.coef_[0]
slope_100 = model_100.coef_[0]

# ============================================================
# 회귀식
# ============================================================

intercept_50 = model_50.intercept_
intercept_100 = model_100.intercept_

# ============================================================
# 결과 표
# ============================================================

st.subheader("3. 테스트 데이터 예측 성능 비교")

results = pd.DataFrame({

    "학습 기간": [
        "최근 50년 (1956~2005)",
        "최근 100년 (1906~2005)"
    ],

    "학습 연도 수": [
        len(train_50),
        len(train_100)
    ],

    "기울기 (℃/년)": [
        slope_50,
        slope_100
    ],

    "기울기 (℃/100년)": [
        slope_50 * 100,
        slope_100 * 100
    ],

    "MAE (℃)": [
        mae_50,
        mae_100
    ],

    "MSE (℃²)": [
        mse_50,
        mse_100
    ],

    "R²": [
        r2_50,
        r2_100
    ]
})

st.dataframe(
    results.style.format({
        "기울기 (℃/년)": "{:.5f}",
        "기울기 (℃/100년)": "{:.2f}",
        "MAE (℃)": "{:.3f}",
        "MSE (℃²)": "{:.3f}",
        "R²": "{:.3f}"
    }),
    hide_index=True,
    use_container_width=True
)

# ============================================================
# 주요 지표 카드
# ============================================================

st.subheader("4. 핵심 결과")

col1, col2 = st.columns(2)

with col1:

    st.markdown("### 🔵 최근 50년 학습")

    st.metric(
        "기울기",
        f"{slope_50 * 100:+.2f} ℃/100년"
    )

    st.metric(
        "MAE",
        f"{mae_50:.3f} ℃"
    )

    st.metric(
        "MSE",
        f"{mse_50:.3f} ℃²"
    )

    st.metric(
        "R²",
        f"{r2_50:.3f}"
    )

with col2:

    st.markdown("### 🟢 최근 100년 학습")

    st.metric(
        "기울기",
        f"{slope_100 * 100:+.2f} ℃/100년"
    )

    st.metric(
        "MAE",
        f"{mae_100:.3f} ℃"
    )

    st.metric(
        "MSE",
        f"{mse_100:.3f} ℃²"
    )

    st.metric(
        "R²",
        f"{r2_100:.3f}"
    )

# ============================================================
# 회귀선 그래프
# ============================================================

st.subheader("5. 학습 기간에 따른 회귀선 비교")

fig = go.Figure()

# 실제 연평균기온
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(
            size=6,
            color="#999999"
        )
    )
)

# 50년 회귀선
x_line_50 = np.linspace(
    1956,
    2025,
    300
)

y_line_50 = model_50.predict(
    pd.DataFrame({
        "연도": x_line_50
    })
)

fig.add_trace(
    go.Scatter(
        x=x_line_50,
        y=y_line_50,
        mode="lines",
        name="최근 50년 회귀선",
        line=dict(
            width=3,
            color="#2878d8"
        )
    )
)

# 100년 회귀선
x_line_100 = np.linspace(
    max(1906, int(yearly["연도"].min())),
    2025,
    300
)

y_line_100 = model_100.predict(
    pd.DataFrame({
        "연도": x_line_100
    })
)

fig.add_trace(
    go.Scatter(
        x=x_line_100,
        y=y_line_100,
        mode="lines",
        name="최근 100년 회귀선",
        line=dict(
            width=3,
            color="#2ca02c"
        )
    )
)

# 테스트 시작점
fig.add_vline(
    x=2006,
    line_dash="dash",
    annotation_text="2006년 테스트 시작"
)

fig.update_layout(
    title="서울 연평균기온과 학습 기간별 회귀선",
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    height=600,
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(
        color="#111111"
    ),
    xaxis=dict(
        showgrid=False
    ),
    yaxis=dict(
        gridcolor="#eeeeee"
    )
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ============================================================
# 테스트 데이터에서 실제값 vs 예측값
# ============================================================

st.subheader("6. 공통 테스트 데이터(2006~2025) 비교")

fig2 = go.Figure()

fig2.add_trace(
    go.Scatter(
        x=test["연도"],
        y=y_test,
        mode="lines+markers",
        name="실제 기온"
    )
)

fig2.add_trace(
    go.Scatter(
        x=test["연도"],
        y=prediction_50,
        mode="lines",
        name="50년 학습 예측"
    )
)

fig2.add_trace(
    go.Scatter(
        x=test["연도"],
        y=prediction_100,
        mode="lines",
        name="100년 학습 예측"
    )
)

fig2.update_layout(
    title="2006~2025 테스트 데이터의 실제값과 예측값",
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    height=500,
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(
        color="#111111"
    )
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

# ============================================================
# 가장 좋은 모델
# ============================================================

st.subheader("7. 어떤 학습 기간이 더 잘 예측했을까?")

# MAE는 작을수록 좋음
if mae_50 < mae_100:
    mae_best = "최근 50년"
else:
    mae_best = "최근 100년"

# MSE는 작을수록 좋음
if mse_50 < mse_100:
    mse_best = "최근 50년"
else:
    mse_best = "최근 100년"

# R²는 클수록 좋음
if r2_50 > r2_100:
    r2_best = "최근 50년"
else:
    r2_best = "최근 100년"

st.write(
    f"**MAE가 더 작은 모델:** {mae_best}"
)

st.write(
    f"**MSE가 더 작은 모델:** {mse_best}"
)

st.write(
    f"**R²가 더 큰 모델:** {r2_best}"
)

# ============================================================
# 기울기 비교
# ============================================================

st.subheader("8. 회귀선의 기울기 비교")

slope_difference = (
    (slope_50 - slope_100) * 100
)

st.write(
    f"최근 50년 회귀선의 기울기: "
    f"**{slope_50 * 100:+.2f} ℃/100년**"
)

st.write(
    f"최근 100년 회귀선의 기울기: "
    f"**{slope_100 * 100:+.2f} ℃/100년**"
)

st.write(
    f"두 기울기의 차이: "
    f"**{abs(slope_difference):.2f} ℃/100년**"
)

# ============================================================
# 자동 해석
# ============================================================

st.subheader("9. 결과 해석")

if abs(slope_50) > abs(slope_100):

    st.write(
        "최근 50년을 학습한 회귀선의 기울기가 "
        "최근 100년을 학습한 회귀선보다 더 큽니다. "
        "즉, 최근 기간의 기온 변화 추세가 더 가파르게 나타납니다."
    )

elif abs(slope_50) < abs(slope_100):

    st.write(
        "최근 100년을 학습한 회귀선의 기울기가 "
        "최근 50년을 학습한 회귀선보다 더 큽니다. "
        "학습 기간을 길게 잡았을 때 전체적인 장기 추세가 더 크게 나타납니다."
    )

else:

    st.write(
        "두 학습 기간의 회귀선 기울기가 거의 같습니다."
    )

st.write(
    "두 모델은 같은 2006~2025년 데이터를 테스트 데이터로 사용했기 때문에 "
    "MAE, MSE, R²를 공통 기준으로 비교할 수 있습니다."
)

st.info(
    "MAE와 MSE는 작을수록 예측 오차가 작고, "
    "R²는 일반적으로 클수록 테스트 데이터의 변동을 더 잘 설명합니다."
)

# ============================================================
# 학습 데이터 확인
# ============================================================

with st.expander("📋 학습 데이터와 테스트 데이터 확인"):

    st.write("### 최근 50년 학습 데이터")

    st.dataframe(
        train_50[
            ["연도", "연평균기온", "관측일수"]
        ],
        hide_index=True,
        use_container_width=True
    )

    st.write("### 최근 100년 학습 데이터")

    st.dataframe(
        train_100[
            ["연도", "연평균기온", "관측일수"]
        ],
        hide_index=True,
        use_container_width=True
    )

    st.write("### 공통 테스트 데이터")

    test_display = test.copy()

    test_display["50년 예측"] = prediction_50
    test_display["100년 예측"] = prediction_100

    st.dataframe(
        test_display[
            [
                "연도",
                "연평균기온",
                "50년 예측",
                "100년 예측",
                "관측일수"
            ]
        ],
        hide_index=True,
        use_container_width=True
    )
