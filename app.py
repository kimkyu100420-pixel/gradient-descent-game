
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# =========================================================
# 1. 페이지 설정
# =========================================================
st.set_page_config(
    page_title="AI 경사하강법 미니게임",
    page_icon="⚽",
    layout="wide",
)

START_X = 8.0                 # 모든 플레이어의 출발 위치 고정
SUCCESS_THRESHOLD = 0.00001   # 엄격한 착륙 성공 기준

# =========================================================
# 2. 디자인
# =========================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Jua&display=swap');

    [data-testid="stAppViewContainer"] > .main {
        background: linear-gradient(
            180deg, #DDF2FF 0%, #F0F8FF 48%, #F4F8EC 100%
        );
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #D5F2D5 0%, #A8DDB0 100%);
    }
    h1, h2, h3 {
        font-family: 'Jua', sans-serif;
    }
    div[data-testid="stButton"] button,
    div[data-testid="stDownloadButton"] button {
        border-radius: 18px;
        font-family: 'Jua', sans-serif;
        font-weight: 700;
    }
    div[data-testid="stMetric"] {
        background-color: rgba(255,255,255,0.7);
        border-radius: 14px;
        padding: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# 3. 게임 소개 및 초보자 안내
# =========================================================
st.markdown(
    """
    <div style="
        background:linear-gradient(135deg,#234B87,#3877B8);
        padding:26px;
        border-radius:18px;
        margin-bottom:14px;
        box-shadow:0 5px 15px rgba(0,0,0,0.15);
    ">
        <h1 style="color:white;margin:0;">
            ⚽ AI 경사하강법: 골짜기 착륙 챌린지
        </h1>
        <p style="color:#FFF0B5;font-size:18px;margin:12px 0 0 0;">
            공을 골짜기 바닥에 정확하게 착륙시켜라!
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "🎯 미션: 모든 플레이어는 같은 위치에서 출발해요. "
    "학습률과 발걸음 수를 조절해 공을 목표 지점 0에 최대한 정확하게 보내세요!"
)

with st.expander("🎮 게임 방법 — 처음이라면 꼭 읽어 보세요", expanded=True):
    guide1, guide2, guide3 = st.columns(3)

    with guide1:
        st.markdown(
            """
            ### 📍 출발 위치
            모든 플레이어가 **8.0**에서 출발해요.

            출발 위치는 바꿀 수 없어요.
            """
        )

    with guide2:
        st.markdown(
            """
            ### 👟 학습률
            공이 한 번 움직일 때의 **보폭**이에요.

            - 너무 작으면 천천히 이동해요.
            - 너무 크면 목표를 지나칠 수 있어요.
            """
        )

    with guide3:
        st.markdown(
            """
            ### 🔁 발걸음 수
            공을 움직이는 **횟수**예요.

            횟수가 부족하면 목표에 가까워지기 전에 끝날 수 있어요.
            """
        )

    st.markdown("---")
    st.markdown(
        """
        **🏁 성공 조건**

        마지막 공의 위치가 0에서 `0.00001`보다 가까워야 착륙 성공이에요.

        **💡 도전 팁:** 학습률을 바꾸면 공의 움직임이 달라져요.
        발걸음 수도 조절하면서 어떤 조합이 가장 좋은지 찾아보세요!
        """
    )

# =========================================================
# 4. 세션 상태
# =========================================================
if "experiment_data" not in st.session_state:
    st.session_state.experiment_data = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

# =========================================================
# 5. 경사하강법 계산
# =========================================================
def loss_function(x):
    return x ** 2


def run_gradient_descent(x0, alpha, epochs):
    x = float(x0)
    x_history = [x]
    y_history = [loss_function(x)]
    overflow = False

    for _ in range(int(epochs)):
        # f(x) = x², f'(x) = 2x
        x_next = x - alpha * (2 * x)

        if not np.isfinite(x_next) or abs(x_next) > 1e5:
            overflow = True
            break

        x = x_next
        x_history.append(x)
        y_history.append(loss_function(x))

    return (
        np.array(x_history),
        np.array(y_history),
        overflow,
    )


def classify_result(alpha, final_x, overflow):
    if overflow or not np.isfinite(final_x):
        return "우주폭발", "explode"

    if abs(final_x) < SUCCESS_THRESHOLD:
        return "착륙성공", "success"

    # 이론적으로 학습률이 0.5보다 크면 진동이 커지고 발산해요.
    if alpha >= 0.5:
        return "우주폭발", "explode"

    return "착륙중", "landing"


def calc_score(final_x):
    # 목표 위치에 가까울수록 높은 점수
    score = 100 - round(abs(final_x) * 10000)
    return max(0, min(100, score))


def safe_text(value):
    """닉네임을 HTML 문맥에 안전하게 표시하기 위한 처리."""
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


# =========================================================
# 6. 게임 컨트롤러
# =========================================================
st.sidebar.header("🎮 게임 컨트롤러")
st.sidebar.caption("출발 위치는 모두 동일해요!")

nickname_input = st.sidebar.text_input(
    "🏷️ 플레이어 닉네임",
    value="익명",
    max_chars=20,
)

st.sidebar.metric("📍 공의 출발 위치", "8.0 (고정)")

st.sidebar.markdown("---")
st.sidebar.subheader("👟 1. 학습률 선택")

alpha_input = st.sidebar.slider(
    "공의 보폭",
    min_value=0.05,
    max_value=0.45,
    value=0.20,
    step=0.01,
    format="%.2f",
    help="공이 한 번에 이동하는 보폭이에요.",
)

if alpha_input < 0.15:
    st.sidebar.caption("🐢 작은 보폭: 목표에 도달하기까지 더 많은 이동이 필요할 수 있어요.")
elif alpha_input < 0.30:
    st.sidebar.caption("⚖️ 보폭을 조절해 최적의 결과를 찾아보세요.")
else:
    st.sidebar.caption("🚀 큰 보폭: 빠르게 이동하지만 목표를 지나칠 수도 있어요.")

st.sidebar.markdown("---")
st.sidebar.subheader("🔁 2. 발걸음 수 선택")

epochs_input = st.sidebar.slider(
    "공을 움직일 횟수",
    min_value=1,
    max_value=10,
    value=5,
    step=1,
    help="공을 몇 번 움직일지 정해요.",
)

st.sidebar.caption(
    f"현재 설정: 학습률 {alpha_input:.2f} · "
    f"발걸음 {epochs_input}회"
)

run_button = st.sidebar.button(
    "⚽ 도전 시작!",
    width="stretch",
    type="primary",
)

# =========================================================
# 7. 게임 실행 및 기록 저장
# =========================================================
if run_button:
    safe_nickname = (
        nickname_input.strip()
        if nickname_input.strip()
        else "익명"
    )

    x_hist, y_hist, overflow = run_gradient_descent(
        START_X,
        alpha_input,
        epochs_input,
    )

    final_x = float(x_hist[-1])
    final_loss = float(y_hist[-1])

    status_label, status_code = classify_result(
        alpha_input,
        final_x,
        overflow,
    )

    record = {
        "닉네임": safe_nickname,
        "학습률(α)": round(float(alpha_input), 3),
        "시작위치(x0)": START_X,
        "발걸음수": int(epochs_input),
        "최종위치": final_x,
        "최종Loss": final_loss,
        "결과상태": status_label,
    }

    st.session_state.experiment_data.append(record)

    st.session_state.last_result = {
        "nickname": safe_nickname,
        "x_history": x_hist,
        "y_history": y_hist,
        "final_x": final_x,
        "final_loss": final_loss,
        "initial_loss": float(y_hist[0]),
        "alpha": float(alpha_input),
        "epochs": int(epochs_input),
        "status_code": status_code,
        "status_label": status_label,
    }

    st.toast("실험 결과가 기록되었어요!", icon="⚽")

    if status_code == "success":
        st.balloons()

# =========================================================
# 8. 그래프와 스코어보드
# =========================================================
st.divider()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🏔️ 공의 이동 경로")

    fig = go.Figure()

    x_curve = np.linspace(-12, 12, 400)
    fig.add_trace(
        go.Scatter(
            x=x_curve,
            y=x_curve ** 2,
            mode="lines",
            line=dict(color="#8B929A", width=3),
            name="골짜기",
            hoverinfo="skip",
        )
    )

    result = st.session_state.last_result

    if result is not None:
        fig.add_trace(
            go.Scatter(
                x=result["x_history"],
                y=result["y_history"],
                mode="lines+markers",
                line=dict(color="#E74C3C", width=2, dash="dot"),
                marker=dict(
                    size=10,
                    color="#E74C3C",
                    line=dict(color="#922B21", width=1),
                ),
                name="⚽ 공의 이동",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[0],
                mode="markers",
                marker=dict(
                    size=16,
                    color="#F1C40F",
                    symbol="star",
                    line=dict(color="#D68910", width=1),
                ),
                name="🏁 목표 지점",
            )
        )

    fig.update_layout(
        xaxis_title="공의 위치",
        yaxis_title="오차 (Loss)",
        xaxis=dict(range=[-12, 12], zeroline=True),
        yaxis=dict(range=[-5, 120], zeroline=True),
        height=500,
        margin=dict(l=10, r=10, t=35, b=10),
        legend=dict(orientation="h", y=1.08, x=0),
        hovermode="closest",
        plot_bgcolor="rgba(255,255,255,0.65)",
        paper_bgcolor="rgba(255,255,255,0)",
    )

    st.plotly_chart(fig, width="stretch")

with col2:
    st.subheader("📟 내 결과")

    result = st.session_state.last_result

    if result is None:
        st.info(
            "👈 왼쪽 사이드바에서 학습률과 발걸음 수를 선택한 뒤 "
            "'도전 시작!'을 눌러 보세요."
        )
    else:
        code = result["status_code"]

        if code == "success":
            st.success(
                "🎉 **착륙 성공!**\n\n"
                "공이 목표 지점에 매우 정확하게 도착했어요!"
            )
        elif code == "explode":
            st.error(
                "💥 **우주폭발!**\n\n"
                "공이 안정적으로 목표에 도착하지 못했어요."
            )
        else:
            st.warning(
                "🐢 **아직 착륙 중!**\n\n"
                "목표에 가까워졌지만 정확도가 부족해요. "
                "설정을 바꿔 다시 도전해 보세요."
            )

        score = calc_score(result["final_x"])

        st.metric("🏅 점수", f"{score} / 100")

        st.metric(
            "📍 최종 위치",
            f"{result['final_x']:.8f}",
        )

        st.metric(
            "📉 최종 오차 (Loss)",
            f"{result['final_loss']:.8f}",
        )

        st.metric(
            "👟 적용한 학습률",
            f"{result['alpha']:.2f}",
        )

        st.metric(
            "🔁 실제 이동 횟수",
            f"{len(result['x_history']) - 1}회",
        )

        if code != "success":
            st.caption(
                "💡 학습률과 발걸음 수 중 하나만 바꿔서 "
                "결과 차이를 비교해 보세요."
            )

# =========================================================
# 9. 우리 반 기록실
# =========================================================
st.divider()
st.header("🏆 우리 반 도전 기록실")

st.caption(
    "현재 실행 세션에 누적된 기록이에요. "
    "모든 기기의 기록을 공유하려면 별도의 공용 데이터베이스가 필요해요."
)

df = pd.DataFrame(st.session_state.experiment_data)

if df.empty:
    st.info(
        "📭 아직 도전 기록이 없어요. "
        "첫 번째 도전자가 되어 보세요!"
    )
else:
    best_idx = df["최종Loss"].idxmin()
    best_row = df.loc[best_idx]
    best_score = calc_score(best_row["최종위치"])

    st.markdown(
        f"### 🥇 명예의 전당: {safe_text(best_row['닉네임'])}"
    )
    st.write(
        f"최고 기록 Loss: **{best_row['최종Loss']:.8f}** · "
        f"점수: **{best_score}점**"
    )

    stat_col1, stat_col2 = st.columns([2, 1])

    with stat_col1:
        display_df = df.copy()
        display_df["최종위치"] = display_df["최종위치"].map(
            lambda v: f"{v:.8f}"
        )
        display_df["최종Loss"] = display_df["최종Loss"].map(
            lambda v: f"{v:.8f}"
        )

        st.dataframe(
            display_df,
            width="stretch",
            hide_index=True,
        )

        csv_bytes = df.to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            label="📥 실험 기록 CSV 다운로드",
            data=csv_bytes,
            file_name="gradient_descent_game_data.csv",
            mime="text/csv",
            width="stretch",
        )

    with stat_col2:
        status_counts = (
            df["결과상태"]
            .value_counts()
            .rename_axis("결과")
            .reset_index(name="횟수")
        )

        color_map = {
            "착륙성공": "#2ECC71",
            "우주폭발": "#E74C3C",
            "착륙중": "#F1C40F",
        }

        fig_pie = px.pie(
            status_counts,
            names="결과",
            values="횟수",
            color="결과",
            color_discrete_map=color_map,
            hole=0.45,
            title="도전 결과 비율",
        )

        fig_pie.update_traces(
            textposition="inside",
            textinfo="percent+label",
        )
        fig_pie.update_layout(
            height=360,
            margin=dict(l=5, r=5, t=50, b=5),
        )

        st.plotly_chart(fig_pie, width="stretch")

    total_count = len(df)
    success_count = int((df["결과상태"] == "착륙성공").sum())
    explosion_count = int((df["결과상태"] == "우주폭발").sum())

    success_rate = round(success_count / total_count * 100, 1)
    explosion_rate = round(explosion_count / total_count * 100, 1)

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric("👥 총 도전 횟수", f"{total_count}회")
    metric2.metric("🎯 착륙 성공률", f"{success_rate}%")
    metric3.metric("💥 우주폭발 비율", f"{explosion_rate}%")

    st.markdown(
        f"🔥 현재까지 **{explosion_rate}%**의 도전이 우주폭발로 끝났어요!"
    )
    st.markdown(
        f"🎯 착륙 성공은 **{success_count}회**예요. "
        "학습률과 발걸음 수를 조절해 더 좋은 조합을 찾아보세요!"
    )

# =========================================================
# 10. 하단 안내
# =========================================================
st.divider()
st.caption(
    "학습 목표: 경사하강법, 학습률, 반복 횟수의 관계를 "
    "직접 실험하며 이해하기"
)

