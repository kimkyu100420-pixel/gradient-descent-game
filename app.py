import os
import io
import random
import hashlib
import datetime

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# PostgreSQL
try:
    import psycopg2
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False


# =========================================================
# 1. 페이지 설정
# =========================================================
st.set_page_config(
    page_title="AI 경사하강법 미니게임",
    page_icon="⚽",
    layout="wide"
)


# =========================================================
# 2. 스타일
# =========================================================
st.markdown("""
<style>
    .challenge-box {
        padding: 25px;
        border-radius: 18px;
        background: linear-gradient(135deg, #eef4ff, #f8fbff);
        border: 1px solid #dce7f7;
        margin-bottom: 20px;
    }

    .big-number {
        font-size: 32px;
        font-weight: 700;
    }

    .game-card {
        padding: 18px;
        border-radius: 15px;
        background-color: #f7f7f7;
        margin-bottom: 10px;
    }

    .rank-1 {
        font-size: 20px;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. PostgreSQL 연결
# =========================================================
DATABASE_URL = os.getenv("DATABASE_URL")


def get_connection():
    if not DATABASE_URL:
        return None

    if not POSTGRES_AVAILABLE:
        return None

    return psycopg2.connect(DATABASE_URL)


def init_database():
    """
    최초 실행 시 실험 데이터를 저장할 테이블 생성
    """
    conn = get_connection()

    if conn is None:
        return False

    try:
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id SERIAL PRIMARY KEY,
                nickname TEXT NOT NULL,
                learning_rate DOUBLE PRECISION NOT NULL,
                start_x DOUBLE PRECISION NOT NULL,
                epochs INTEGER NOT NULL,
                final_x DOUBLE PRECISION NOT NULL,
                final_loss DOUBLE PRECISION NOT NULL,
                result TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        cur.close()
        conn.close()

        return True

    except Exception:
        try:
            conn.rollback()
            conn.close()
        except Exception:
            pass

        return False


DB_READY = init_database()


# =========================================================
# 4. 실험 데이터 저장
# =========================================================
def save_experiment(
    nickname,
    learning_rate,
    start_x,
    epochs,
    final_x,
    final_loss,
    result
):
    conn = get_connection()

    if conn is None:
        return False

    try:
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO experiments (
                nickname,
                learning_rate,
                start_x,
                epochs,
                final_x,
                final_loss,
                result
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            nickname,
            learning_rate,
            start_x,
            epochs,
            final_x,
            final_loss,
            result
        ))

        conn.commit()

        cur.close()
        conn.close()

        return True

    except Exception:
        try:
            conn.rollback()
            conn.close()
        except Exception:
            pass

        return False


# =========================================================
# 5. 전체 데이터 불러오기
# =========================================================
def load_experiments():
    conn = get_connection()

    if conn is None:
        return pd.DataFrame()

    try:
        query = """
            SELECT
                id,
                nickname AS "닉네임",
                learning_rate AS "학습률(α)",
                start_x AS "시작위치(x₀)",
                epochs AS "발걸음수",
                final_x AS "최종위치",
                final_loss AS "최종Loss",
                result AS "결과상태",
                created_at AS "참여시간"
            FROM experiments
            ORDER BY id DESC
        """

        df = pd.read_sql_query(query, conn)

        conn.close()

        return df

    except Exception:
        try:
            conn.close()
        except Exception:
            pass

        return pd.DataFrame()


# =========================================================
# 6. Session State
# =========================================================
if "trajectory" not in st.session_state:
    st.session_state.trajectory = []

if "game_result" not in st.session_state:
    st.session_state.game_result = None

if "current_x" not in st.session_state:
    st.session_state.current_x = 0.0

if "current_loss" not in st.session_state:
    st.session_state.current_loss = 0.0

if "last_saved" not in st.session_state:
    st.session_state.last_saved = False


# =========================================================
# 7. 경사하강법
# =========================================================
def run_gradient_descent(start_x, learning_rate, steps):

    x = float(start_x)

    trajectory = [x]

    diverged = False

    for _ in range(steps):

        gradient = 2 * x

        x = x - learning_rate * gradient

        # 오버플로우 방지
        if abs(x) > 1e5:
            x = 1e5 if x > 0 else -1e5

            trajectory.append(x)

            diverged = True

            break

        trajectory.append(x)

    return np.array(trajectory), diverged


# =========================================================
# 8. 결과 판정
# =========================================================
def determine_result(alpha, final_x, diverged):

    if alpha < 1.0 and abs(final_x) < 0.1:
        return "성공"

    if abs(alpha - 1.0) < 1e-5:
        return "진동"

    if alpha >= 1.0 or diverged or abs(final_x) >= 1e5:
        return "폭발"

    return "실패"


def get_result_message(result):

    if result == "성공":
        return (
            "🎉 **착륙 성공!** 적절한 보폭으로 "
            "최적 정답(0, 0)에 무사히 도착했습니다!"
        )

    if result == "진동":
        return (
            "⚠️ **무한 핑퐁!** 공이 제자리에서 "
            "좌우로 계속 튕기고 있습니다."
        )

    return (
        "💥 **우주로 튕겨나감!** 보폭이 너무 커서 "
        "공이 안드로메다로 날아갔습니다!"
    )


# =========================================================
# 9. 헤더
# =========================================================
st.markdown("""
<div class="challenge-box">

# ⚽ AI 경사하강법: 공 골짜기 착륙 미니게임

### 🎯 미션

AI가 정답을 찾아가는 **보폭(학습률)** 을 조절해서
공을 안드로메다로 날려버리지 않고
**골짜기 맨 아래 (0, 0)** 에 착륙시키세요!

</div>
""", unsafe_allow_html=True)

st.info(
    "💡 Tip: **보폭이 너무 작으면 답답하고, 너무 크면 공이 폭발합니다!**"
)


# =========================================================
# 10. DB 상태 표시
# =========================================================
if not DB_READY:

    st.warning(
        "⚠️ 현재 중앙 데이터베이스가 연결되지 않았습니다. "
        "Railway PostgreSQL과 DATABASE_URL을 확인해주세요."
    )


# =========================================================
# 11. 사이드바
# =========================================================
st.sidebar.header("🎮 컨트롤러")

nickname = st.sidebar.text_input(
    "플레이어 닉네임",
    value="익명",
    max_chars=20
)

x0 = st.sidebar.slider(
    "📍 공의 시작 위치 (x₀)",
    -10.0,
    10.0,
    8.0,
    0.5
)

alpha = st.sidebar.slider(
    "👟 한 걸음 보폭 크기 (학습률 α)",
    0.01,
    1.50,
    0.10,
    0.05
)

epochs = st.sidebar.slider(
    "🏃‍♂️ 발걸음 수 (Epochs)",
    5,
    50,
    20,
    5
)

st.sidebar.markdown("---")

st.sidebar.markdown("### 🏆 챌린지")

st.sidebar.caption(
    "공을 성공적으로 착륙시켜\n"
    "우리 반 기록실에 이름을 올려보세요!"
)

run_game = st.sidebar.button(
    "🚀 공 던지기 & 기록 제출",
    use_container_width=True,
    type="primary"
)


# =========================================================
# 12. 게임 실행
# =========================================================
if run_game:

    trajectory, diverged = run_gradient_descent(
        x0,
        alpha,
        epochs
    )

    final_x = float(trajectory[-1])

    final_loss = float(final_x ** 2)

    result = determine_result(
        alpha,
        final_x,
        diverged
    )

    player_name = nickname.strip()

    if not player_name:
        player_name = "익명"

    # Session State
    st.session_state.trajectory = trajectory.tolist()
    st.session_state.current_x = final_x
    st.session_state.current_loss = final_loss
    st.session_state.game_result = result

    # 중앙 DB 저장
    saved = save_experiment(
        nickname=player_name,
        learning_rate=alpha,
        start_x=x0,
        epochs=epochs,
        final_x=final_x,
        final_loss=final_loss,
        result=result
    )

    st.session_state.last_saved = saved


# =========================================================
# 13. 결과 알림
# =========================================================
if run_game:

    if st.session_state.last_saved:

        st.success(
            "📊 실험 결과가 우리 반 전체 기록실에 저장되었습니다!"
        )

    else:

        st.warning(
            "⚠️ 게임은 완료되었지만 중앙 DB에 저장되지 않았습니다."
        )


# =========================================================
# 14. 메인 화면
# =========================================================
left_col, right_col = st.columns([2, 1])


# =========================================================
# 15. 그래프
# =========================================================
with left_col:

    x_curve = np.linspace(-12, 12, 500)

    y_curve = x_curve ** 2

    fig = go.Figure()

    # 골짜기
    fig.add_trace(
        go.Scatter(
            x=x_curve,
            y=y_curve,
            mode="lines",
            name="골짜기 f(x)=x²",
            line=dict(
                color="gray",
                width=3
            )
        )
    )

    # 공의 궤적
    if st.session_state.trajectory:

        trajectory = np.array(
            st.session_state.trajectory,
            dtype=float
        )

        losses = trajectory ** 2

        # 그래프 밖으로 너무 크게 튀는 값 때문에
        # Plotly 그래프가 망가지지 않도록 시각화용 값만 제한
        visual_losses = np.clip(
            losses,
            -5,
            120
        )

        fig.add_trace(
            go.Scatter(
                x=trajectory,
                y=visual_losses,
                mode="lines+markers",
                name="⚽ 공의 이동 궤적",
                line=dict(
                    color="red",
                    width=3,
                    dash="dot"
                ),
                marker=dict(
                    color="red",
                    size=12,
                    symbol="circle",
                    line=dict(
                        color="darkred",
                        width=2
                    )
                )
            )
        )

        # 현재 공
        fig.add_trace(
            go.Scatter(
                x=[trajectory[-1]],
                y=[min(losses[-1], 120)],
                mode="markers",
                name="⚽ 현재 공",
                marker=dict(
                    color="red",
                    size=20,
                    symbol="circle",
                    line=dict(
                        color="white",
                        width=3
                    )
                )
            )
        )

    # 목표 지점
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[0],
            mode="markers",
            name="🎯 목표 지점",
            marker=dict(
                color="green",
                size=15,
                symbol="star"
            )
        )
    )

    fig.update_layout(
        title="⚽ 공이 골짜기를 따라 이동하는 모습",
        xaxis_title="가중치 위치 (x)",
        yaxis_title="오차 / 손실값 (Loss)",

        xaxis=dict(
            range=[-12, 12],
            zeroline=True
        ),

        yaxis=dict(
            range=[-5, 120],
            zeroline=True
        ),

        height=600,

        margin=dict(
            l=40,
            r=20,
            t=70,
            b=40
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# 16. 스코어보드
# =========================================================
with right_col:

    st.subheader("🏆 실시간 스코어보드")

    result = st.session_state.game_result

    if result is None:

        st.info(
            "🎮 사이드바에서 설정을 고르고\n"
            "**「🚀 공 던지기 & 기록 제출」**을 눌러주세요!"
        )

    elif result == "성공":

        st.success(get_result_message(result))

    elif result == "진동":

        st.warning(get_result_message(result))

    else:

        st.error(get_result_message(result))

    st.metric(
        "현재 남아있는 오차 (Loss)",
        f"{st.session_state.current_loss:.6g}"
    )

    st.metric(
        "현재 공의 위치 (x)",
        f"{st.session_state.current_x:.6g}"
    )

    st.markdown("---")

    st.markdown("### 🧠 이번 플레이")

    st.write(f"📍 시작 위치: **{x0}**")

    st.write(f"👟 학습률 α: **{alpha:.2f}**")

    st.write(f"🏃 발걸음 수: **{epochs}회**")


# =========================================================
# 17. 게임 원리 설명
# =========================================================
st.markdown("---")

with st.expander("🔍 이 게임에서 AI는 무엇을 하고 있을까?"):

    st.markdown("""
### ⚽ 공의 움직임 = 경사하강법

골짜기의 모양은

**f(x) = x²**

입니다.

골짜기의 기울기는

**f'(x) = 2x**

입니다.

AI는 매 발걸음마다

**다음 위치 = 현재 위치 − 학습률 × 기울기**

를 계산합니다.

즉,

**xₙ₊₁ = xₙ − α × 2xₙ**

입니다.

### 👟 학습률에 따라 어떻게 달라질까?

- 🐢 너무 작음 → 천천히 이동
- 🎯 적절함 → 골짜기 아래로 수렴
- 🏓 α = 1 → 계속 좌우로 핑퐁
- 🚀 α > 1 → 점점 멀리 튕겨나감

이것이 머신러닝에서 사용하는 **경사하강법의 핵심 아이디어**입니다.
""")


# =========================================================
# 18. 우리 반 전체 기록실
# =========================================================
st.markdown("---")

st.header("🏆 우리 반 전체 참여자 기록실")

df = load_experiments()


if df.empty:

    st.info(
        "아직 실험 기록이 없습니다.\n\n"
        "친구들에게 링크를 공유하고 첫 번째 실험을 시작해보세요! 🚀"
    )

else:

    # 숫자 반올림
    display_df = df.copy()

    display_df["학습률(α)"] = display_df["학습률(α)"].round(3)
    display_df["최종위치"] = display_df["최종위치"].round(6)
    display_df["최종Loss"] = display_df["최종Loss"].round(6)

    # =====================================================
    # 통계 카드
    # =====================================================
    total = len(df)

    success_count = int(
        (df["결과상태"] == "성공").sum()
    )

    vibration_count = int(
        (df["결과상태"] == "진동").sum()
    )

    explosion_count = int(
        (df["결과상태"] == "폭발").sum()
    )

    success_rate = success_count / total * 100
    explosion_rate = explosion_count / total * 100

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "👥 전체 실험",
        f"{total}회"
    )

    c2.metric(
        "🎉 착륙 성공",
        f"{success_count}회",
        f"{success_rate:.1f}%"
    )

    c3.metric(
        "⚠️ 무한 핑퐁",
        f"{vibration_count}회"
    )

    c4.metric(
        "💥 우주 폭발",
        f"{explosion_count}회",
        f"{explosion_rate:.1f}%"
    )

    st.markdown("---")

    # =====================================================
    # 기록 / 차트
    # =====================================================
    left, right = st.columns([1.5, 1])

    with left:

        st.subheader("📋 실험 기록")

        st.dataframe(
            display_df[
                [
                    "닉네임",
                    "학습률(α)",
                    "시작위치(x₀)",
                    "발걸음수",
                    "최종위치",
                    "최종Loss",
                    "결과상태",
                    "참여시간"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

        # CSV 다운로드
        csv_buffer = io.StringIO()

        df.to_csv(
            csv_buffer,
            index=False
        )

        st.download_button(
            "📥 우리 반 실험 데이터 CSV 다운로드",
            data=csv_buffer.getvalue().encode("utf-8-sig"),
            file_name="gradient_descent_class_data.csv",
            mime="text/csv",
            use_container_width=True
        )

    with right:

        st.subheader("📊 결과 비율")

        chart_data = pd.DataFrame({
            "결과상태": [
                "성공",
                "진동",
                "폭발"
            ],
            "인원": [
                success_count,
                vibration_count,
                explosion_count
            ]
        })

        fig_donut = px.pie(
            chart_data,
            names="결과상태",
            values="인원",
            hole=0.55,
            color="결과상태",
            color_discrete_map={
                "성공": "green",
                "진동": "orange",
                "폭발": "red"
            }
        )

        fig_donut.update_traces(
            textinfo="label+percent"
        )

        st.plotly_chart(
            fig_donut,
            use_container_width=True
        )

    # =====================================================
    # 학습률별 분석
    # =====================================================
    st.markdown("---")

    st.subheader("🔬 학습률에 따른 실험 결과")

    analysis_df = (
        df.groupby("learning_rate")
        .agg(
            실험횟수=("result", "count"),
            성공횟수=("result", lambda x: (x == "성공").sum()),
            평균Loss=("final_loss", "mean")
        )
        .reset_index()
    )

    analysis_df["성공률"] = (
        analysis_df["성공횟수"]
        / analysis_df["실험횟수"]
        * 100
    )

    analysis_df["learning_rate"] = analysis_df[
        "learning_rate"
    ].round(3)

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:

        fig_success = px.bar(
            analysis_df,
            x="learning_rate",
            y="성공률",
            title="👟 학습률별 성공률",
            labels={
                "learning_rate": "학습률 α",
                "성공률": "성공률 (%)"
            }
        )

        fig_success.update_yaxes(
            range=[0, 100]
        )

        st.plotly_chart(
            fig_success,
            use_container_width=True
        )

    with chart_col2:

        fig_loss = px.scatter(
            df,
            x="learning_rate",
            y="final_loss",
            color="result",
            title="📉 학습률과 최종 Loss",
            labels={
                "learning_rate": "학습률 α",
                "final_loss": "최종 Loss",
                "result": "결과"
            },
            color_discrete_map={
                "성공": "green",
                "진동": "orange",
                "폭발": "red"
            }
        )

        st.plotly_chart(
            fig_loss,
            use_container_width=True
        )

    # =====================================================
    # 참여자 통계 문구
    # =====================================================
    st.markdown("---")

    st.markdown(
        f"""
### 🔥 현재까지 우리 반 실험 결과

**총 {total}회**의 실험이 진행되었습니다.

🎉 착륙 성공: **{success_count}회 ({success_rate:.1f}%)**

⚠️ 무한 핑퐁: **{vibration_count}회**

💥 우주 폭발: **{explosion_count}회 ({explosion_rate:.1f}%)**

> 🚀 현재까지 참여한 실험 중 **{explosion_rate:.1f}%**가
> 공을 우주로 날려버렸습니다!
"""
    )


# =========================================================
# 19. 참여 안내
# =========================================================
st.markdown("---")

st.info(
    "🎁 **챌린지 안내:** 친구들에게 이 페이지 링크를 공유하고 "
    "각자 원하는 학습률로 실험해보세요. "
    "모든 실험 결과는 우리 반 전체 기록실에 자동으로 누적됩니다!"
)
