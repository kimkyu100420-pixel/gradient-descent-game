"""
⚽ AI 경사하강법: 공 골짜기 착륙 미니게임 & 실시간 데이터 수집기
-----------------------------------------------------------------------------
고등학생들이 어려운 수학 공식 대신 "U자형 골짜기 위의 공을 보폭(학습률)을
조절해 중앙(0, 0)에 착륙시키는" 게임으로 경사하강법(Gradient Descent)과
학습률(α)의 개념을 직관적으로 체험할 수 있도록 만든 Streamlit 웹 앱입니다.

실행 방법:
    streamlit run app.py

참고: st.metric의 border/chart_data(스파크라인), st.button 등의 width
     파라미터를 사용하므로 Streamlit 1.54.0 이상 버전이 필요합니다.
     (구버전이라면: pip install --upgrade streamlit)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# =============================================================================
# 1. 페이지 설정
# =============================================================================
st.set_page_config(
    page_title="AI 경사하강법 미니게임",
    page_icon="⚽",
    layout="wide",
)

# 게임 감성을 위한 커스텀 스타일. 순수 장식용 CSS라서 셀렉터가 일부 버전에서
# 안 먹더라도 앱 동작(로직)에는 전혀 영향을 주지 않는다.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Jua&display=swap');

    [data-testid="stAppViewContainer"] > .main {
        background: linear-gradient(180deg, #6EC6FF 0%, #BEE9E8 45%, #EAF7E0 100%);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #66BB6A 0%, #2E7D32 100%);
    }
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] p {
        color: #FFFFFF !important;
    }
    h1, h2, h3 {
        font-family: 'Jua', sans-serif;
    }
    div[data-testid="stButton"] button,
    div[data-testid="stDownloadButton"] button {
        border-radius: 24px;
        font-family: 'Jua', sans-serif;
        font-weight: 700;
        transition: transform 0.15s ease;
    }
    div[data-testid="stButton"] button:hover,
    div[data-testid="stDownloadButton"] button:hover {
        transform: scale(1.03);
    }
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.55);
        border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# 2. 상단 대형 챌린지 배너
# =============================================================================
st.markdown(
    """
    <div style="
        background: linear-gradient(135deg, #1E3C72 0%, #2A5298 100%);
        padding: 26px 30px;
        border-radius: 18px;
        margin-bottom: 14px;
        box-shadow: 0 6px 16px rgba(0,0,0,0.25);
    ">
        <h1 style="color:#FFFFFF; margin:0; font-size:2.1rem;">
            ⚽ AI 경사하강법: 공 골짜기 착륙 미니게임
        </h1>
        <p style="color:#FFE8A3; font-size:1.1rem; margin-top:12px; margin-bottom:0;">
            🎯 <b>미션:</b> AI가 정답을 찾아가는 보폭(학습률)을 조절하여, 공을 안드로메다로
            날려버리지 않고 골짜기 맨 아래(0, 0)에 착륙시키세요!
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info("💡 Tip: 보폭이 너무 작으면 답답하고, 너무 크면 공이 폭발합니다!")

# =============================================================================
# 3. 세션 상태 관리
# =============================================================================
if "experiment_data" not in st.session_state:
    st.session_state.experiment_data = []  # 모든 참여자의 실험 기록 누적 리스트

if "last_result" not in st.session_state:
    st.session_state.last_result = None  # 그래프/스코어보드 표시용 최신 결과


# =============================================================================
# 경사하강법 연산 엔진 (Math Engine) + 재미 요소 헬퍼 함수
# =============================================================================
def loss_function(x: float) -> float:
    """손실 함수 f(x) = x^2 (골짜기의 높이)."""
    return x ** 2


def gradient_function(x: float) -> float:
    """도함수(기울기) f'(x) = 2x."""
    return 2.0 * x


def run_gradient_descent(x0: float, alpha: float, epochs: int):
    """
    공을 보폭(학습률 α)만큼씩 굴려 골짜기 아래로 보내는 경사하강법 시뮬레이션.

    수치 안정성(Overflow Safeguard): |x| > 1e5 를 초과하면 즉시 1e5(-1e5)로
    캡핑하고 반복을 종료해, 값이 무한히 커지는 것을 방지한다.

    Returns
    -------
    x_history, y_history : np.ndarray  스텝별 공의 (x, y) 좌표 (초기값 포함)
    is_overflow : bool                 공이 튕겨나가 캡핑되었는지 여부
    """
    CLIP_LIMIT = 1e5

    # 데이터 타입 안전성: 입력값을 안전하게 숫자로 변환
    try:
        x_current = float(x0)
        alpha = float(alpha)
        epochs = int(epochs)
    except (TypeError, ValueError):
        x_current, alpha, epochs = 0.0, 0.1, 20

    x_history = [x_current]
    y_history = [loss_function(x_current)]
    is_overflow = False

    for _ in range(max(epochs, 0)):
        grad = gradient_function(x_current)
        x_next = x_current - alpha * grad

        overflow_now = (not np.isfinite(x_next)) or (abs(x_next) > CLIP_LIMIT)
        if overflow_now:
            x_next = -CLIP_LIMIT if (np.isfinite(x_next) and x_next < 0) else CLIP_LIMIT
            is_overflow = True

        x_current = x_next
        x_history.append(x_current)
        y_history.append(loss_function(x_current))

        if overflow_now:
            break

    return np.array(x_history, dtype=float), np.array(y_history, dtype=float), is_overflow


def classify_result(alpha: float, final_x: float, is_overflow: bool):
    """
    결과를 [착륙성공 / 무한핑퐁 / 우주폭발 / 착륙중] 중 하나로 판정한다.

    - 착륙성공: α < 1.0 이고 최종 |x| < 0.1
    - 무한핑퐁: |α - 1.0| < 1e-5 (보폭이 정확히 1.0 근처)
    - 우주폭발: α > 1.0 이거나 공이 튕겨나간(오버플로우) 경우
    - 착륙중  : 위 조건에 해당하지 않는 경우
               (보폭은 안전하지만 발걸음 수가 부족해 아직 도착하지 못한 경우)
    """
    if alpha < 1.0 and abs(final_x) < 0.1:
        return "착륙성공", "success"
    if abs(alpha - 1.0) < 1e-5:
        return "무한핑퐁", "pingpong"
    if alpha > 1.0 or is_overflow:
        return "우주폭발", "explode"
    return "착륙중", "landing"


def calc_score(final_loss: float) -> int:
    """재미 요소: 최종 Loss가 0에 가까울수록(=중앙에 정확히 착륙) 높은 0~100점."""
    try:
        raw_score = 100.0 - float(final_loss)
    except (TypeError, ValueError):
        raw_score = 0.0
    return int(max(0, min(100, round(raw_score))))


def alpha_feedback(alpha: float) -> str:
    """재미 요소: 현재 보폭(α) 값에 대한 실시간 피드백 문구."""
    if alpha < 0.3:
        return "🟢 안전제일! 느긋느긋 신중한 걸음이에요."
    elif alpha < 0.9:
        return "🟡 적당히 대담한 보폭이네요!"
    elif alpha <= 1.1:
        return "🟠 아슬아슬... 핑퐁 위험 구간 진입!"
    else:
        return "🔴 위험해요! 폭발 각입니다!!"


def md_escape(text: str) -> str:
    """닉네임 등 사용자 입력이 마크다운 서식을 깨뜨리지 않도록 특수문자를 이스케이프."""
    text = str(text)
    for ch in ["\\", "*", "_", "`", "["]:
        text = text.replace(ch, "\\" + ch)
    return text


# =============================================================================
# 4. 게임화된 사이드바 조작창
# =============================================================================
st.sidebar.header("🎮 컨트롤러")

nickname_input = st.sidebar.text_input("플레이어 닉네임", value="익명")

x0_input = st.sidebar.slider(
    "📍 공의 시작 위치 (x₀)", -10.0, 10.0, 8.0, 0.5, format="%.1f",
)

alpha_input = st.sidebar.slider(
    "👟 한 걸음 보폭 크기 (학습률 α)", 0.01, 1.50, 0.10, 0.05, format="%.2f",
)
st.sidebar.caption(alpha_feedback(alpha_input))

epochs_input = st.sidebar.slider(
    "🏃‍♂️ 발걸음 수 (Epochs)", 5, 50, 20, 5,
)

run_button = st.sidebar.button("🚀 공 던지기 & 데이터 제출", width="stretch")

# =============================================================================
# 버튼 클릭 시 시뮬레이션 실행 & 데이터 누적
# =============================================================================
if run_button:
    try:
        safe_nickname = (
            nickname_input.strip()
            if isinstance(nickname_input, str) and nickname_input.strip()
            else "익명"
        )

        x_hist, y_hist, overflow_flag = run_gradient_descent(x0_input, alpha_input, epochs_input)
        final_x = float(x_hist[-1])
        final_loss = float(y_hist[-1])

        status_label, status_code = classify_result(alpha_input, final_x, overflow_flag)

        # 실험 기록 누적 저장 (7개 필드)
        st.session_state.experiment_data.append({
            "닉네임": safe_nickname,
            "학습률(α)": round(float(alpha_input), 4),
            "시작위치(x0)": round(float(x0_input), 4),
            "발걸음수": int(epochs_input),
            "최종위치": round(final_x, 4),
            "최종Loss": round(final_loss, 4),
            "결과상태": status_label,
        })

        # 메인 화면(그래프/스코어보드) 표시용 최신 결과 저장
        st.session_state.last_result = {
            "nickname": safe_nickname,
            "x_history": x_hist,
            "y_history": y_hist,
            "final_x": final_x,
            "final_loss": final_loss,
            "initial_loss": float(y_hist[0]),
            "status_code": status_code,
        }

        st.toast(f"{md_escape(safe_nickname)}님의 공이 날아갔습니다!", icon="⚽")
        if status_code == "success":
            st.balloons()

    except Exception as e:
        st.sidebar.error(f"⚠️ 오류가 발생했습니다: {e}")


# =============================================================================
# 7. 메인 시각화 화면 (2:1 비율)
# =============================================================================
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🏔️ 골짜기 착륙 궤적")

    fig = go.Figure()

    # 골짜기 곡선: f(x) = x^2 (회색 라인)
    x_curve = np.linspace(-12, 12, 400)
    y_curve = x_curve ** 2
    fig.add_trace(go.Scatter(
        x=x_curve, y=y_curve,
        mode="lines",
        line=dict(color="gray", width=2),
        name="골짜기 f(x) = x²",
        hoverinfo="skip",
    ))

    result = st.session_state.last_result
    if result is not None:
        # 공의 이동 궤적: 빨간 점선 + 굵은 원형 마커
        fig.add_trace(go.Scatter(
            x=result["x_history"], y=result["y_history"],
            mode="lines+markers",
            line=dict(color="red", width=2, dash="dot"),
            marker=dict(size=10, color="red", symbol="circle",
                        line=dict(color="darkred", width=1)),
            name="⚽ 공의 이동 경로",
        ))
        # 재미 요소: 목표 지점(0, 0)을 별 모양으로 표시
        fig.add_trace(go.Scatter(
            x=[0], y=[0],
            mode="markers",
            marker=dict(size=15, color="gold", symbol="star",
                        line=dict(color="darkorange", width=1)),
            name="🏁 목표 지점",
        ))

    fig.update_layout(
        xaxis_title="가중치 위치 (x)",
        yaxis_title="오차/손실값 (Loss)",
        xaxis=dict(range=[-12, 12]),
        yaxis=dict(range=[-5, 120]),
        height=520,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest",
        plot_bgcolor="rgba(255,255,255,0.55)",
    )

    st.plotly_chart(fig, width="stretch")

with col2:
    st.subheader("📟 실시간 스코어보드")

    result = st.session_state.last_result
    if result is None:
        st.info("👈 사이드바에서 보폭을 설정하고 공을 던져보세요!")
    else:
        status_code = result["status_code"]

        if status_code == "success":
            st.success("🎉 **착륙 성공!** 적절한 보폭으로 최적 정답(0, 0)에 무사히 도착했습니다!")
        elif status_code == "pingpong":
            st.warning("⚠️ **무한 핑퐁!** 공이 제자리에서 좌우로 무한히 튕기기만 합니다.")
        elif status_code == "explode":
            st.error("💥 **우주로 튕겨나감!** 보폭이 너무 커서 공이 안드로메다로 날아가 버렸습니다!")
        else:
            st.info("🐢 **아직 착륙 중...** 너무 신중한 걸음이라 아직 도착 못했어요. 발걸음 수를 늘려보세요!")

        score = calc_score(result["final_loss"])
        delta_loss_value = result["final_loss"] - result["initial_loss"]

        st.metric(
            "🏅 이번 판 점수",
            f"{score} / 100",
            border=True,
        )
        st.metric(
            "현재 남아있는 오차 (Loss)",
            f"{result['final_loss']:.4f}",
            delta=f"{delta_loss_value:+.2f}",
            delta_color="inverse",
            border=True,
            chart_data=np.clip(result["y_history"], -5, 120).tolist(),
            chart_type="area",
        )
        st.metric(
            "현재 공의 위치 (x)",
            f"{result['final_x']:.4f}",
            border=True,
            chart_data=np.clip(result["x_history"], -12, 12).tolist(),
            chart_type="line",
        )


# =============================================================================
# 8. 우리 반 실시간 통계 및 승부욕 유발 섹션
# =============================================================================
st.divider()
st.header("🏆 우리 반 전체 참여자 기록실")

if len(st.session_state.experiment_data) == 0:
    st.info("📭 아직 아무도 공을 던지지 않았어요! 사이드바에서 첫 번째 도전자가 되어보세요.")
else:
    try:
        df = pd.DataFrame(st.session_state.experiment_data)
    except Exception:
        df = pd.DataFrame(
            columns=["닉네임", "학습률(α)", "시작위치(x0)", "발걸음수",
                     "최종위치", "최종Loss", "결과상태"]
        )

    # ---- 재미 요소: 명예의 전당 (승부욕 유발) ----
    if not df.empty and "최종Loss" in df.columns:
        best_row = df.loc[df["최종Loss"].idxmin()]
        best_score = calc_score(best_row["최종Loss"])
        st.markdown(
            f"##### 🥇 명예의 전당: **{md_escape(best_row['닉네임'])}**님이 "
            f"Loss `{best_row['최종Loss']:.4f}` ({best_score}점)로 최고 기록을 세웠습니다!"
        )

    stat_col1, stat_col2 = st.columns([2, 1])

    with stat_col1:
        st.dataframe(df, width="stretch", hide_index=True)

        try:
            csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
        except Exception:
            csv_bytes = "".encode("utf-8-sig")

        st.download_button(
            label="📥 우리 반 실험 데이터 CSV 다운로드",
            data=csv_bytes,
            file_name="gradient_descent_ball_game_data.csv",
            mime="text/csv",
            width="stretch",
        )

    with stat_col2:
        if "결과상태" in df.columns and not df.empty:
            status_counts = (
                df["결과상태"]
                .value_counts()
                .rename_axis("결과상태")
                .reset_index(name="참여자 수")
            )

            color_map = {
                "착륙성공": "#2ECC71",  # 초록
                "무한핑퐁": "#F39C12",  # 주황
                "우주폭발": "#E74C3C",  # 빨강
                "착륙중": "#95A5A6",    # 회색 (보조 상태)
            }

            fig_pie = px.pie(
                status_counts,
                names="결과상태",
                values="참여자 수",
                color="결과상태",
                color_discrete_map=color_map,
                hole=0.45,
                title="착륙 성공 vs 우주 폭발 비율",
            )
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")
            fig_pie.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10))

            st.plotly_chart(fig_pie, width="stretch")
        else:
            st.info("통계를 표시할 데이터가 없습니다.")

    # ---- 위트 있는 자동 요약 텍스트 (하단) ----
    total_count = len(df)
    explode_count = int((df["결과상태"] == "우주폭발").sum()) if "결과상태" in df.columns else 0
    success_count = int((df["결과상태"] == "착륙성공").sum()) if "결과상태" in df.columns else 0
    explode_pct = round(explode_count / total_count * 100, 1) if total_count else 0.0
    success_pct = round(success_count / total_count * 100, 1) if total_count else 0.0

    st.markdown(f"🔥 현재까지 참여자의 **{explode_pct}%**가 공을 우주로 날려버렸습니다!")
    st.markdown(
        f"🎯 착륙 성공률은 **{success_pct}%** 입니다! · 👥 총 **{total_count}명**이 도전했습니다!"
    )
