
import html

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =========================================================
# 1. 페이지 설정
# =========================================================
st.set_page_config(
    page_title="AI 경사하강법 미니게임",
    page_icon="⚽",
    layout="wide",
)

START_X = 8.0
SUCCESS_THRESHOLD = 1e-5
CLIP_LIMIT = 1e5

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
        background: linear-gradient(
            180deg, #D5F2D5 0%, #A8DDB0 100%
        );
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
# 3. 제목과 게임 안내
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
            보폭을 잘못 조절하면 공이 우주까지 날아갑니다!
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "🎯 미션: 공을 골짜기 바닥의 목표 지점 0에 정확히 착륙시키세요. "
    "시작 위치는 모두 같고, 학습률과 발걸음 수만 조절할 수 있어요."
)

with st.expander("🎮 게임 방법 — 처음이라면 꼭 읽어 보세요", expanded=True):
    guide1, guide2, guide3 = st.columns(3)

    with guide1:
        st.markdown(
            """
            ### 📍 시작 위치

            모두 **8.0**에서 출발해요.

            공정한 경쟁을 위해 출발 위치는 바꿀 수 없어요.
            """
        )

    with guide2:
        st.markdown(
            """
            ### 👟 학습률 = 보폭

            공이 한 번 움직일 때의 크기예요.

            - 너무 작으면 느려요.
            - 적당하면 목표에 가까워져요.
            - 너무 크면 목표를 지나쳐 튕겨 나가요.
            """
        )

    with guide3:
        st.markdown(
            """
            ### 🔁 발걸음 수

            공을 움직이는 횟수예요.

            횟수가 적으면 목표에 도착하기 전에 끝날 수 있어요.
            """
        )

    st.markdown("---")
    st.markdown(
        """
        **실험 힌트**

        - 학습률 `0.10`: 조금씩 목표에 접근
        - 학습률 `0.50`: 좌우로 같은 거리만큼 튕김
        - 학습률 `1.00`: 공이 매번 반대편으로 이동하며 거리가 유지됨
        - 학습률 `1.20`: 매번 더 멀리 튕겨 나감

        **🏁 성공 조건:** 마지막 위치가 0에서 `0.00001`보다 가까워야 해요.
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
# 5. 경사하강법 엔진
# =========================================================
def loss_function(x):
    return x ** 2


def run_gradient_descent(x0, alpha, epochs):
    """
    f(x) = x²
    f'(x) = 2x
    x_next = x - alpha * 2x
    """
    x = float(x0)
    x_history = [x]
    y_history = [loss_function(x)]
    overflow = False

    for _ in range(int(epochs)):
        x_next = x - alpha * (2.0 * x)

        if not np.isfinite(x_next) or abs(x_next) > CLIP_LIMIT:
            overflow = True

            # 그래프에 발산 방향이 드러나도록 마지막 위치를 제한
            if not np.isfinite(x_next):
                direction = np.sign(x) * np.sign(1 - 2 * alpha)
                if direction == 0:
                    direction = 1
            else:
                direction = np.sign(x_next)

            x_next = float(direction * CLIP_LIMIT)

            x_history.append(x_next)
            y_history.append(CLIP_LIMIT ** 2)
            break

        x = float(x_next)
        x_history.append(x)
        y_history.append(loss_function(x))

    return (
        np.array(x_history, dtype=float),
        np.array(y_history, dtype=float),
        overflow,
    )


def classify_result(alpha, final_x, overflow):
    if overflow or not np.isfinite(final_x):
        return "우주폭발", "explode"

    # α = 1이면 x_next = -x.
    # 위치가 좌우로 바뀌지만 거리는 줄어들지 않음.
    if np.isclose(alpha, 1.0, atol=1e-8):
        return "무한핑퐁", "pingpong"

    # α > 1이면 위치의 절댓값이 매번 증가
    if alpha > 1.0:
        return "우주폭발", "explode"

    # α = 0.5이면 x_next = 0.
    # 단, 0.5는 학습률 슬라이더에서 선택 가능.
    if abs(final_x) < SUCCESS_THRESHOLD:
        return "착륙성공", "success"

    return "착륙중", "landing"


def calc_score(final_x):
    """최종 위치가 목표 0에 가까울수록 높은 점수."""
    if not np.isfinite(final_x):
        return 0

    score = 100 - int(abs(final_x) * 10)
    return max(0, min(100, score))


def safe_nickname(value):
    value = str(value).strip() or "익명"
    return html.escape(value[:20])


# =========================================================
# 6. 게임 컨트롤러
# =========================================================
st.sidebar.header("🎮 게임 컨트롤러")
st.sidebar.caption("모두 같은 위치에서 출발해요!")

nickname_input = st.sidebar.text_input(
    "🏷️ 플레이어 닉네임",
    value="익명",
    max_chars=20,
)

st.sidebar.metric("📍 공의 시작 위치", "8.0 (고정)")

st.sidebar.markdown("---")
st.sidebar.subheader("👟 1. 학습률 — 공의 보폭")

alpha_input = st.sidebar.slider(
    "학습률",
    min_value=0.05,
    max_value=1.50,
    value=0.50,
    step=0.05,
    format="%.2f",
    help="공이 한 번에 움직이는 크기예요. 너무 크면 점점 멀리 튕겨 나가요.",
)

if alpha_input < 0.15:
    st.sidebar.caption("🐢 보폭이 작아요. 여러 번 움직여야 목표에 가까워질 수 있어요.")
elif alpha_input < 0.45:
    st.sidebar.caption("⚖️ 조금씩 이동해요. 발걸음 수를 조절해 보세요.")
elif np.isclose(alpha_input, 0.50):
    st.sidebar.caption("↔️ 공이 반대편으로 튕기지만 점점 목표에 가까워지진 않아요.")
elif np.isclose(alpha_input, 1.00):
    st.sidebar.caption("🏓 무한 핑퐁! 좌우로 튕기며 거리도 그대로 유지돼요.")
elif alpha_input < 1.00:
    st.sidebar.caption("⚠️ 보폭이 커지고 있어요. 공의 이동 경로를 확인하세요.")
else:
    st.sidebar.caption("💥 위험! 공이 매번 더 멀리 튕겨 나가는 구간이에요.")

st.sidebar.markdown("---")
st.sidebar.subheader("🔁 2. 발걸음 수 — 이동 횟수")

epochs_input = st.sidebar.slider(
    "공을 움직일 횟수",
    min_value=1,
    max_value=10,
    value=5,
    step=1,
    help="공을 몇 번 움직일지 정해요.",
)

st.sidebar.caption(
    f"현재 설정: 학습률 {alpha_input:.2f} · {epochs_input}회 이동"
)

run_button = st.sidebar.button(
    "⚽ 도전 시작!",
    width="stretch",
    type="primary",
)

# =========================================================
# 7. 실행 및 기록 저장
# =========================================================
if run_button:
    nickname = safe_nickname(nickname_input)

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

    st.session_state.experiment_data.append(
        {
            "닉네임": nickname,
            "학습률(α)": float(alpha_input),
            "시작위치(x0)": START_X,
            "발걸음수": int(epochs_input),
            "최종위치": final_x,
            "최종Loss": final_loss,
            "결과상태": status_label,
        }
    )

    st.session_state.last_result = {
        "nickname": nickname,
        "x_history": x_hist,
        "y_history": y_hist,
        "final_x": final_x,
        "final_loss": final_loss,
        "initial_loss": float(y_hist[0]),
        "alpha": float(alpha_input),
        "epochs": int(epochs_input),
        "status_code": status_code,
        "status_label": status_label,
        "overflow": overflow,
    }

    if status_code == "success":
        st.balloons()
        st.toast("착륙 성공! 정확하게 도착했어요!", icon="🎉")
    elif status_code == "explode":
        st.toast("공이 우주로 튕겨 나갔어요!", icon="💥")
    elif status_code == "pingpong":
        st.toast("무한 핑퐁! 공이 같은 거리로 튕기고 있어요.", icon="🏓")
    else:
        st.toast("아직 착륙 중이에요. 설정을 바꿔 다시 도전하세요!", icon="⚽")

# =========================================================
# 8. 그래프와 스코어보드
# =========================================================
st.divider()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🏔️ 공의 이동 경로")

    fig = go.Figure()

    # 골짜기 모양
    x_curve = np.linspace(-12, 12, 400)
    fig.add_trace(
        go.Scatter(
            x=x_curve,
            y=x_curve ** 2,
            mode="lines",
            line=dict(color="#8B929A", width=3),
            name="골짜기 f(x) = x²",
            hoverinfo="skip",
        )
    )

    result = st.session_state.last_result

    if result is not None:
        # 실제 이동 경로
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
                hovertemplate="위치: %{x:.4f}<br>Loss: %{y:.4f}<extra></extra>",
            )
        )

        # 목표 지점
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

    if result is not None and (
        result["overflow"] or abs(result["final_x"]) > 12
    ):
        st.warning(
            "공이 그래프의 표시 범위를 벗어났어요. "
            "최종 위치와 Loss 지표에서 발산 정도를 확인하세요."
        )

with col2:
    st.subheader("📟 내 결과")

    result = st.session_state.last_result

    if result is None:
        st.info(
            "👈 사이드바에서 학습률과 발걸음 수를 정하고 "
            "'도전 시작!'을 눌러 보세요."
        )
    else:
        code = result["status_code"]

        if code == "success":
            st.success(
                "🎉 **착륙 성공!**\n\n"
                "공이 목표 지점에 아주 정확하게 도착했어요."
            )
        elif code == "pingpong":
            st.warning(
                "🏓 **무한 핑퐁!**\n\n"
                "공이 좌우로 튕기지만 거리는 줄어들지 않아요."
            )
        elif code == "explode":
            st.error(
                "💥 **우주폭발!**\n\n"
                "공이 목표에서 점점 멀어지고 있어요."
            )
        else:
            st.info(
                "🐢 **아직 착륙 중!**\n\n"
                "목표에 가까워졌지만 정확도가 부족해요."
            )

        st.metric("🏅 점수", f"{calc_score(result['final_x'])} / 100")
        st.metric("📍 최종 위치", f"{result['final_x']:.8g}")
        st.metric("📉 최종 Loss", f"{result['final_loss']:.8g}")
        st.metric("👟 적용한 학습률", f"{result['alpha']:.2f}")
        st.metric(
            "🔁 실제 이동 횟수",
            f"{len(result['x_history']) - 1}회",
        )

        st.caption(
            "출발 위치 8.0 · 목표 위치 0 · "
            f"성공 기준 |x| < {SUCCESS_THRESHOLD:g}"
        )

# =========================================================
# 9. 우리 반 도전 기록실
# =========================================================
st.divider()
st.header("🏆 우리 반 도전 기록실")

st.caption(
    "현재 브라우저 세션의 기록이에요. "
    "모든 친구의 기록을 한곳에 모으려면 공용 데이터베이스가 필요해요."
)

df = pd.DataFrame(st.session_state.experiment_data)

if df.empty:
    st.info("📭 아직 기록이 없어요. 첫 번째 도전자가 되어 보세요!")
else:
    best_idx = df["최종Loss"].idxmin()
    best_row = df.loc[best_idx]

    st.markdown(
        f"### 🥇 명예의 전당: {best_row['닉네임']}"
    )
    st.write(
        f"최고 기록 Loss: **{best_row['최종Loss']:.8g}** · "
        f"점수: **{calc_score(best_row['최종위치'])}점**"
    )

    stat_col1, stat_col2 = st.columns([2, 1])

    with stat_col1:
        display_df = df.copy()
        display_df["최종위치"] = display_df["최종위치"].map(
            lambda v: f"{v:.8g}"
        )
        display_df["최종Loss"] = display_df["최종Loss"].map(
            lambda v: f"{v:.8g}"
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
            "무한핑퐁": "#F39C12",
            "우주폭발": "#E74C3C",
            "착륙중": "#95A5A6",
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
    pingpong_count = int((df["결과상태"] == "무한핑퐁").sum())

    success_rate = round(success_count / total_count * 100, 1)
    explosion_rate = round(explosion_count / total_count * 100, 1)

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric("👥 총 도전 횟수", f"{total_count}회")
    metric2.metric("🎯 착륙 성공률", f"{success_rate}%")
    metric3.metric("💥 우주폭발 비율", f"{explosion_rate}%")

    st.markdown(
        f"🔥 **{explosion_count}회** 우주폭발 · "
        f"🏓 **{pingpong_count}회** 무한 핑퐁 · "
        f"🎯 **{success_count}회** 착륙 성공"
    )

# =========================================================
# 10. 하단 안내
# =========================================================
st.divider()

st.caption(
    "학습 목표: 학습률과 반복 횟수가 경사하강법의 결과에 "
    "어떤 영향을 주는지 직접 실험하며 이해하기."
)
