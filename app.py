import html
import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# =========================================================
# 1. 앱 설정
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

    .stApp, [data-testid="stAppViewContainer"] {
        background: #F1F5F9;
        color: #172033;
    }
    [data-testid="stHeader"] { background: rgba(241,245,249,.95); }
    [data-testid="stSidebar"] {
        background: #DCE7F3;
        border-right: 1px solid #B6C8DC;
    }
    [data-testid="stSidebar"] * { color: #172033 !important; }
    h1, h2, h3 {
        font-family: 'Jua', sans-serif !important;
        color: #142B4A !important;
    }
    p, li, label, .stMarkdown { color: #243247; }
    [data-testid="stMetric"] {
        background: #fff; border: 1px solid #CAD5E2;
        border-radius: 12px; padding: 12px;
    }
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"],
    [data-testid="stMetricDelta"] { color: #172033 !important; }
    div[data-testid="stButton"] button,
    div[data-testid="stDownloadButton"] button {
        border-radius: 12px; font-weight: 700; border: 1px solid #244B78;
    }
    [data-testid="stExpander"] {
        background: #fff; border: 1px solid #CAD5E2; border-radius: 12px;
    }
    input, textarea { color: #172033 !important; background: #fff !important; }
    [data-testid="stCaptionContainer"] { color: #45566D !important; }
    .hero {
        background: linear-gradient(135deg, #15345B, #285C91);
        padding: 26px; border-radius: 16px; margin-bottom: 16px;
    }
    .hero h1 { color: #fff !important; margin: 0; }
    .hero p { color: #fff !important; font-size: 17px; margin-bottom: 0; }

    /* Full-screen effects: readable, restrained flashing */
    @keyframes fullShake {
        0%, 100% { transform: translate(0, 0); }
        12% { transform: translate(-7px, 3px); }
        24% { transform: translate(7px, -2px); }
        36% { transform: translate(-5px, -3px); }
        48% { transform: translate(5px, 2px); }
        60% { transform: translate(-3px, 1px); }
        72% { transform: translate(3px, -1px); }
    }
    @keyframes overlayIn {
        0% { opacity: 0; }
        8% { opacity: 1; }
        78% { opacity: 1; }
        100% { opacity: 0; }
    }
    @keyframes cardIn {
        0% { transform: scale(.65) rotate(-4deg); }
        12% { transform: scale(1.04) rotate(1deg); }
        20%, 78% { transform: scale(1) rotate(0); }
        100% { transform: scale(.96); }
    }
    @keyframes particleBurst {
        0% { transform: translate(-50%, -50%) scale(.35) rotate(0); opacity: 0; }
        12% { opacity: 1; }
        100% { transform: translate(calc(-50% + var(--dx)), calc(-50% + var(--dy))) scale(1.2) rotate(360deg); opacity: 0; }
    }
    @keyframes turtleTraverse {
        0% { transform: translateX(-45vw) rotate(-8deg); opacity: 0; }
        12% { opacity: 1; }
        35% { transform: translateX(-10vw) rotate(7deg); }
        60% { transform: translateX(15vw) rotate(-5deg); }
        88% { opacity: 1; }
        100% { transform: translateX(45vw) rotate(0); opacity: 0; }
    }
    .gd-fullscreen-overlay {
        position: fixed !important;
        inset: 0 !important;
        width: 100vw !important;
        height: 100vh !important;
        z-index: 2147483000 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        overflow: hidden !important;
        pointer-events: none !important;
        animation: overlayIn 2.5s ease-in-out both;
        background: radial-gradient(ellipse at center, rgba(135, 12, 0, .70), rgba(35, 0, 0, .82));
    }
    .gd-fullscreen-overlay.gd-failure {
        background: radial-gradient(ellipse at center, rgba(145, 18, 0, .72), rgba(25, 0, 0, .84));
    }
    .gd-effect-card {
        position: relative;
        width: min(88vw, 760px);
        max-height: 82vh;
        overflow: hidden;
        padding: clamp(20px, 4vw, 42px);
        text-align: center;
        color: #fff !important;
        border: 3px solid #ff7658;
        border-radius: 26px;
        background: linear-gradient(145deg, #310b12 0%, #151523 58%, #4a1009 100%);
        box-shadow: 0 0 45px rgba(255, 67, 20, .65), inset 0 0 30px rgba(255, 120, 45, .12);
        animation: cardIn 2.5s ease-out both;
    }
    .gd-effect-card * { color: #fff !important; }
    .gd-effect-kicker {
        font-size: clamp(22px, 4vw, 42px);
        font-weight: 1000;
        line-height: 1.15;
        color: #ff7768 !important;
        text-shadow: 0 2px 12px rgba(255, 40, 20, .45);
        margin-bottom: 12px;
    }
    .gd-effect-emoji {
        font-size: clamp(58px, 11vw, 112px);
        line-height: 1.2;
        margin: 8px 0 14px;
        filter: drop-shadow(0 5px 18px rgba(255, 91, 0, .55));
    }
    .gd-effect-headline {
        font-size: clamp(19px, 3.2vw, 30px);
        font-weight: 900;
        line-height: 1.3;
        margin: 10px 0;
    }
    .gd-effect-copy {
        font-size: clamp(15px, 2vw, 19px);
        line-height: 1.5;
        color: #ffe1d6 !important;
    }
    .gd-effect-particle {
        position: absolute;
        left: 50%;
        top: 50%;
        font-size: clamp(24px, 3.5vw, 40px);
        animation: particleBurst 2.3s cubic-bezier(.15,.65,.2,1) both;
    }
    .gd-turtle-overlay {
        background: radial-gradient(ellipse at center, rgba(15, 90, 35, .72), rgba(4, 35, 18, .84));
    }
    .gd-turtle-card {
        background: linear-gradient(145deg, #123b28, #214b25 60%, #6a5520);
        border-color: #b4e57b;
        box-shadow: 0 0 40px rgba(150, 220, 80, .45);
    }
    .gd-turtle-card .gd-effect-emoji {
        animation: turtleTraverse 1.5s ease-in-out both;
    }

    @keyframes successOverlayIn {
        0% { opacity: 0; }
        10%, 82% { opacity: 1; }
        100% { opacity: 0; }
    }
    @keyframes confettiLeft {
        0% { transform: translate(0, 0) rotate(0) scale(.6); opacity: 0; }
        12% { opacity: 1; }
        100% { transform: translate(var(--dx), var(--dy)) rotate(720deg) scale(1.15); opacity: 0; }
    }
    @keyframes confettiRight {
        0% { transform: translate(0, 0) rotate(0) scale(.6); opacity: 0; }
        12% { opacity: 1; }
        100% { transform: translate(var(--dx), var(--dy)) rotate(-720deg) scale(1.15); opacity: 0; }
    }
    @keyframes successCardPop {
        0% { transform: scale(.55) rotate(-5deg); opacity: 0; }
        18% { transform: scale(1.08) rotate(2deg); opacity: 1; }
        30%, 82% { transform: scale(1) rotate(0); opacity: 1; }
        100% { transform: scale(.97); opacity: 0; }
    }
    @keyframes successPulse {
        0%, 100% { text-shadow: 0 0 8px rgba(255,255,255,.3); }
        50% { text-shadow: 0 0 22px rgba(255,235,120,.95), 0 0 36px rgba(255,120,180,.65); }
    }
    .gd-success-overlay {
        position: fixed !important;
        inset: 0 !important;
        width: 100vw !important;
        height: 100vh !important;
        z-index: 2147483000 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        overflow: hidden !important;
        pointer-events: none !important;
        animation: successOverlayIn 2.8s ease-out both;
        background: radial-gradient(ellipse at center, rgba(16, 95, 64, .62), rgba(7, 38, 47, .78));
    }
    .gd-success-card {
        position: relative;
        width: min(88vw, 760px);
        max-height: 82vh;
        overflow: hidden;
        padding: clamp(20px, 4vw, 42px);
        text-align: center;
        color: #fff !important;
        border: 3px solid #f9dc75;
        border-radius: 26px;
        background: linear-gradient(145deg, #064e3b 0%, #123b5b 55%, #51256c 100%);
        box-shadow: 0 0 42px rgba(255, 218, 90, .5), inset 0 0 30px rgba(255,255,255,.12);
        animation: successCardPop 2.8s ease-out both;
    }
    .gd-success-card * { color: #fff !important; }
    .gd-success-kicker {
        font-size: clamp(25px, 4.5vw, 46px);
        font-weight: 1000;
        line-height: 1.2;
        color: #ffe88a !important;
        animation: successPulse .65s ease-in-out infinite;
    }
    .gd-success-emoji {
        font-size: clamp(58px, 11vw, 112px);
        line-height: 1.3;
        margin: 10px 0;
        filter: drop-shadow(0 4px 16px rgba(255, 230, 100, .6));
    }
    .gd-success-headline {
        font-size: clamp(20px, 3.4vw, 31px);
        font-weight: 900;
        margin: 10px 0;
    }
    .gd-success-copy {
        font-size: clamp(15px, 2vw, 19px);
        line-height: 1.5;
        color: #e4fff4 !important;
    }
    .gd-success-particle {
        position: absolute;
        top: 50%;
        font-size: clamp(24px, 3.8vw, 42px);
        opacity: 0;
    }
    .gd-success-particle.left {
        left: 8%;
        animation: confettiLeft 2.35s cubic-bezier(.12,.65,.2,1) both;
    }
    .gd-success-particle.right {
        right: 8%;
        animation: confettiRight 2.35s cubic-bezier(.12,.65,.2,1) both;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>⚽ AI 경사하강법: 골짜기 착륙 챌린지</h1>
        <p>학습률과 이동 횟수를 조절해 공을 목표 지점에 착륙시키세요.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# 3. 게임 안내
# =========================================================
with st.expander("🎮 게임 방법", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**📍 시작 위치**\n\n모든 플레이어는 8.0에서 출발해요.")
    with c2:
        st.markdown("**👟 학습률 = 보폭**\n\n학습률이 크면 목표를 지나칠 수 있고, 너무 크면 발산해요.")
    with c3:
        st.markdown("**🔁 이동 횟수**\n\n한 번의 이동은 위치를 한 번 갱신하는 계산 1회를 뜻해요.")
    st.markdown(
        """
        **학습률별 움직임**
        - `0.05~0.49`: 목표를 향해 한쪽에서 접근
        - `0.50`: 한 번의 업데이트로 목표에 도착
        - `0.51~0.99`: 목표를 넘나들며 수렴
        - `1.00`: 같은 거리로 계속 튕김
        - `1.00 초과`: 점점 멀리 발산
        """
    )

# =========================================================
# 4. 세션 상태
# =========================================================
if "personal_records" not in st.session_state:
    st.session_state.personal_records = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "attempt_id" not in st.session_state:
    st.session_state.attempt_id = 0
if "effect_shown_for_attempt" not in st.session_state:
    st.session_state.effect_shown_for_attempt = -1

# =========================================================
# 5. 경사하강법과 결과 판정
# =========================================================
def run_gradient_descent(x0, alpha, epochs):
    """
    actual_steps는 실제로 수행한 위치 업데이트 횟수.
    목표 허용 오차에 도달하면 즉시 중단하므로, 도착 후 0에서
    불필요하게 반복한 계산은 이동 횟수에 포함하지 않는다.
    """
    x = float(x0)
    x_history = [x]
    y_history = [x * x]
    overflow = False
    actual_steps = 0

    # 시작점이 이미 목표에 있다면 업데이트하지 않는다.
    if abs(x) < SUCCESS_THRESHOLD:
        return np.array(x_history), np.array(y_history), overflow, actual_steps

    for _ in range(int(epochs)):
        x_next = x - float(alpha) * (2.0 * x)

        if not np.isfinite(x_next) or abs(x_next) > CLIP_LIMIT:
            overflow = True
            if np.isfinite(x_next):
                direction = 1.0 if x_next >= 0 else -1.0
            else:
                direction = 1.0 if x >= 0 else -1.0
                if alpha > 1:
                    direction *= -1.0
            x_next = direction * CLIP_LIMIT
            x_history.append(float(x_next))
            y_history.append(float(CLIP_LIMIT ** 2))
            actual_steps += 1
            break

        x = float(x_next)
        x_history.append(x)
        y_history.append(x * x)
        actual_steps += 1

        # 도착하면 즉시 종료해 실제 이동 횟수를 정확하게 표시
        if abs(x) < SUCCESS_THRESHOLD:
            break

    return np.array(x_history), np.array(y_history), overflow, actual_steps


def classify_result(alpha, final_x, overflow):
    if overflow or not np.isfinite(final_x) or alpha > 1.0:
        return "우주폭발", "explode"
    if np.isclose(alpha, 1.0, atol=1e-8):
        return "무한 핑퐁", "pingpong"
    if abs(final_x) < SUCCESS_THRESHOLD:
        return "착륙 성공", "success"
    if alpha > 0.5:
        return "지그재그 수렴 중", "oscillating"
    return "아직 착륙 중", "landing"


def calc_score(final_x, status_code):
    if status_code == "success":
        return 100
    if not np.isfinite(final_x):
        return 0
    return max(0, min(99, int(99 * math.exp(-abs(final_x) / 2.0))))


# =========================================================
# 6. 이펙트: iframe에서 HTML을 직접 렌더링해 코드 노출 방지
# =========================================================
def show_failure_effect(alpha, attempt_id):
    """앱 전체 화면을 덮는 3초짜리 실패 오버레이. 깜빡임은 줄이고 문구 가독성을 우선한다."""
    if alpha > 1.0:
        message = "학습률이 너무 공격적이었습니다!"
        headline = "🚨 학습률 폭주 🚨"
    elif np.isclose(alpha, 1.0, atol=1e-8):
        message = "공이 같은 거리로 계속 튕깁니다. 수렴하지 않습니다!"
        headline = "🔁 무한 핑퐁 대참사 🔁"
    else:
        message = "정해진 이동 횟수 안에 목표에 도착하지 못했습니다!"
        headline = "착지 실패! 경사하강법 대참사 발생!"

    particles = [
        ("🔥", "-42vw", "-28vh"), ("💥", "38vw", "-30vh"),
        ("☄️", "-43vw", "18vh"), ("🧨", "42vw", "20vh"),
        ("⚡", "-23vw", "32vh"), ("🔥", "25vw", "34vh"),
        ("💣", "0vw", "-38vh"), ("✨", "44vw", "-3vh"),
        ("💥", "-44vw", "-4vh"), ("🪨", "14vw", "-33vh"),
        ("🌋", "-30vw", "25vh"), ("🚨", "32vw", "-20vh"),
        ("⭐", "37vw", "29vh"), ("🔥", "-10vw", "35vh"),
    ]
    particle_html = "".join(
        f'<span class="gd-effect-particle" style="--dx:{dx};--dy:{dy};animation-delay:{i * 0.035}s">{emoji}</span>'
        for i, (emoji, dx, dy) in enumerate(particles)
    )

    st.markdown(
        f"""
        <div class="gd-fullscreen-overlay gd-failure" id="failure-{attempt_id}">
          <div class="gd-effect-card">
            {particle_html}
            <div class="gd-effect-kicker">☢️ CRITICAL FAILURE ☢️</div>
            <div class="gd-effect-emoji">💣 💥 🔥</div>
            <div class="gd-effect-headline">{html.escape(headline)}</div>
            <div class="gd-effect-copy">{html.escape(message)}</div>
            <div class="gd-effect-copy" style="margin-top:12px;font-weight:800">
              SYSTEM ALERT · 궤도 이탈 · 착지 프로토콜 실패
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_success_effect(attempt_id):
    """성공 시 화면 전체에 양쪽 폭죽과 색종이가 터지는 축하 효과."""
    confetti = [
        ("🎆", "left", "-5vw", "-35vh"), ("🎇", "left", "18vw", "-28vh"),
        ("🎉", "left", "32vw", "20vh"), ("✨", "left", "15vw", "34vh"),
        ("🎊", "left", "38vw", "-5vh"), ("⭐", "left", "28vw", "-18vh"),
        ("🎆", "right", "5vw", "-35vh"), ("🎇", "right", "-18vw", "-28vh"),
        ("🎉", "right", "-32vw", "20vh"), ("✨", "right", "-15vw", "34vh"),
        ("🎊", "right", "-38vw", "-5vh"), ("⭐", "right", "-28vw", "-18vh"),
        ("🪅", "left", "24vw", "8vh"), ("🥳", "right", "-24vw", "8vh"),
        ("🌟", "left", "8vw", "-12vh"), ("🌟", "right", "-8vw", "-12vh"),
    ]
    particle_html = "".join(
        f'<span class="gd-success-particle {side}" style="--dx:{dx};--dy:{dy};animation-delay:{i * 0.035}s">{emoji}</span>'
        for i, (emoji, side, dx, dy) in enumerate(confetti)
    )
    st.markdown(
        f"""
        <div class="gd-success-overlay" id="success-{attempt_id}">
          <div class="gd-success-card">
            {particle_html}
            <div class="gd-success-kicker">🎉 LANDING SUCCESS! 🎉</div>
            <div class="gd-success-emoji">🏆 ⚽ 🥳</div>
            <div class="gd-success-headline">완벽 착륙! 경사하강법 마스터!</div>
            <div class="gd-success-copy">
              양쪽에서 폭죽 발사! 🎆 목표 지점에 정확히 도착했어요!<br>
              축하합니다 — 최고 점수 100점!
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_turtle_effect(attempt_id):
    """느린 이동 때 앱 전체에 1.5초 동안 재미있는 거북이 연출을 표시한다."""
    st.markdown(
        f"""
        <div class="gd-fullscreen-overlay gd-turtle-overlay" id="turtle-{attempt_id}">
          <div class="gd-effect-card gd-turtle-card">
            <div class="gd-effect-kicker">🐢 TURTLE MODE ACTIVATED 🐢</div>
            <div class="gd-effect-emoji">🐢</div>
            <div class="gd-effect-headline">느림보 경사하강법 출동!</div>
            <div class="gd-effect-copy">
              목표까지 천천히... 한 걸음씩... 너무 서두르지 마세요!
            </div>
            <div class="gd-effect-copy" style="margin-top:10px;font-weight:800">
              🐌 속도는 느려도 포기하지 않는 중! 🚧
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# 7. 사이드바 컨트롤
# =========================================================
st.sidebar.header("🎮 게임 컨트롤러")
st.sidebar.caption("출발 위치는 모든 플레이어가 동일해요.")
nickname_input = st.sidebar.text_input("플레이어 닉네임", value="익명", max_chars=20)
st.sidebar.metric("📍 시작 위치", "8.0 (고정)")
st.sidebar.markdown("---")
st.sidebar.subheader("👟 학습률 — 보폭")

alpha_input = st.sidebar.slider(
    "한 번에 움직이는 크기",
    min_value=0.05, max_value=1.50, value=0.80, step=0.05, format="%.2f",
)
if alpha_input < 0.5:
    st.sidebar.caption("🟢 목표를 향해 한쪽에서 접근해요.")
elif np.isclose(alpha_input, 0.5):
    st.sidebar.caption("🎯 이론상 한 번에 목표에 도착해요.")
elif alpha_input < 1.0:
    st.sidebar.caption("🟠 목표를 넘나들지만 이동 거리는 줄어들어요.")
elif np.isclose(alpha_input, 1.0):
    st.sidebar.caption("🔁 같은 거리로 좌우 핑퐁!")
else:
    st.sidebar.caption("💥 매번 더 멀리 튕겨 나가요.")

st.sidebar.markdown("---")
st.sidebar.subheader("🔁 발걸음 수")
epochs_input = st.sidebar.slider("최대 이동 횟수", min_value=1, max_value=10, value=5, step=1)
st.sidebar.caption(f"학습률 {alpha_input:.2f} · 최대 {epochs_input}회 이동")
run_button = st.sidebar.button("⚽ 도전 시작!", type="primary", width="stretch")

# =========================================================
# 8. 게임 실행 및 기록 저장
# =========================================================
if run_button:
    nickname = (nickname_input.strip() or "익명")[:20]
    x_hist, y_hist, overflow, actual_steps = run_gradient_descent(
        START_X, alpha_input, epochs_input
    )
    final_x = float(x_hist[-1])
    final_loss = float(y_hist[-1])
    status_label, status_code = classify_result(alpha_input, final_x, overflow)
    score = calc_score(final_x, status_code)

    # 버튼을 누를 때마다 고유 ID를 올려 매 도전마다 이펙트를 다시 재생
    st.session_state.attempt_id += 1
    result = {
        "nickname": nickname,
        "alpha": float(alpha_input),
        "epochs": int(epochs_input),
        "actual_steps": int(actual_steps),
        "x_history": x_hist,
        "y_history": y_hist,
        "final_x": final_x,
        "final_loss": final_loss,
        "status_label": status_label,
        "status_code": status_code,
        "score": score,
        "overflow": overflow,
        "attempt_id": st.session_state.attempt_id,
    }
    st.session_state.last_result = result
    st.session_state.personal_records.append({
        "닉네임": nickname,
        "학습률": alpha_input,
        "최대 이동 횟수": epochs_input,
        "실제 이동 횟수": actual_steps,
        "최종 위치": final_x,
        "최종 Loss": final_loss,
        "점수": score,
        "결과": status_label,
    })

    if status_code == "success":
        st.balloons()
        st.toast("착륙 성공! 100점!", icon="🎉")
    elif status_code in ("explode", "pingpong"):
        st.toast("실패! 대참사가 발생했습니다!", icon="💥")
    else:
        st.toast("실험 기록을 저장했어요.", icon="⚽")

# =========================================================
# 9. 그래프와 결과
# =========================================================
st.divider()
graph_col, score_col = st.columns([2, 1])

with graph_col:
    st.subheader("🏔️ 공의 이동 경로")
    fig = go.Figure()
    x_curve = np.linspace(-12, 12, 400)
    fig.add_trace(go.Scatter(
        x=x_curve, y=x_curve ** 2, mode="lines",
        line=dict(color="#667085", width=3),
        name="골짜기 f(x)=x²", hoverinfo="skip",
    ))

    result = st.session_state.last_result
    if result is not None:
        fig.add_trace(go.Scatter(
            x=result["x_history"], y=result["y_history"],
            mode="lines+markers",
            line=dict(color="#D62728", width=3, dash="dot"),
            marker=dict(size=10, color="#D62728",
                        line=dict(color="#7F1D1D", width=1)),
            name="공의 이동",
            hovertemplate="위치: %{x:.5f}<br>Loss: %{y:.5f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[0], y=[0], mode="markers",
            marker=dict(size=16, color="#F4B400", symbol="star",
                        line=dict(color="#8A5A00", width=1)),
            name="목표 지점",
        ))

    fig.update_layout(
        xaxis_title="공의 위치 (x)", yaxis_title="오차 (Loss)",
        xaxis=dict(range=[-12, 12], zeroline=True),
        yaxis=dict(range=[-5, 120], zeroline=True),
        height=500, margin=dict(l=15, r=15, t=35, b=15),
        legend=dict(orientation="h", y=1.08, x=0),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font=dict(color="#172033"),
    )
    st.plotly_chart(fig, width="stretch")
    if result is not None and abs(result["final_x"]) > 12:
        st.warning("공이 그래프 표시 범위 밖으로 나갔어요. 오른쪽 결과에서 최종 위치를 확인하세요.")

with score_col:
    st.subheader("📟 이번 도전 결과")
    result = st.session_state.last_result
    if result is None:
        st.info("사이드바에서 설정하고 도전을 시작해 보세요.")
    else:
        code = result["status_code"]

        # 도전할 때마다 한 번만 효과를 재생한다. 다른 UI 상호작용으로 재실행되지는 않는다.
        if st.session_state.effect_shown_for_attempt != result["attempt_id"]:
            if code == "success":
                show_success_effect(result["attempt_id"])
                st.session_state.effect_shown_for_attempt = result["attempt_id"]
            elif code in ("explode", "pingpong"):
                show_failure_effect(result["alpha"], result["attempt_id"])
                st.session_state.effect_shown_for_attempt = result["attempt_id"]
            elif (
                code == "landing"
                and result["alpha"] <= 0.30
                and abs(result["final_x"]) >= SUCCESS_THRESHOLD
            ):
                show_turtle_effect(result["attempt_id"])
                st.session_state.effect_shown_for_attempt = result["attempt_id"]

        if code == "success":
            st.success("🎉 착륙 성공! 목표 위치에 도착했어요.")
        elif code == "explode":
            st.error("💥 우주폭발! 공이 목표에서 멀어졌어요.")
        elif code == "pingpong":
            st.warning("🔁 무한 핑퐁! 거리가 줄어들지 않아요.")
        elif code == "oscillating":
            st.warning("↔️ 지그재그 수렴 중! 튕기지만 점점 가까워져요.")
        else:
            st.info("🐢 아직 착륙 중이에요. 더 많은 이동이 필요할 수 있어요.")

        st.metric("🏅 점수", f"{result['score']} / 100",
                  help="성공 판정을 받은 경우에만 100점이에요.")
        st.metric("📍 최종 위치", f"{result['final_x']:.8g}")
        st.metric("📉 최종 Loss", f"{result['final_loss']:.8g}")
        st.metric("🔁 실제 이동 횟수", f"{result['actual_steps']}회",
                  help="실제로 위치 업데이트를 실행한 횟수예요. 목표에 도착하면 조기 종료합니다.")
        st.caption(
            f"최대 이동 횟수: {result['epochs']}회 · "
            f"성공 기준: |x| < {SUCCESS_THRESHOLD:g}"
        )

# =========================================================
# 10. 개인 기록실
# =========================================================
st.divider()
st.header("📒 내 개인 기록")
st.caption("현재 브라우저 세션에서 내가 실행한 실험만 표시돼요.")
df = pd.DataFrame(st.session_state.personal_records)

if df.empty:
    st.info("아직 개인 기록이 없어요. 게임을 한 번 실행하면 결과가 여기에 저장돼요.")
else:
    best_score = int(df["점수"].max())
    total_games = len(df)
    success_games = int((df["결과"] == "착륙 성공").sum())
    m1, m2, m3 = st.columns(3)
    m1.metric("🎮 총 플레이", f"{total_games}회")
    m2.metric("🏅 최고 점수", f"{best_score}점")
    m3.metric("🎯 성공 횟수", f"{success_games}회")

    st.markdown("### 🧾 실험 기록")
    st.dataframe(df.iloc[::-1], width="stretch", hide_index=True)
    csv_data = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "📥 내 기록 CSV 다운로드",
        data=csv_data,
        file_name="my_gradient_descent_records.csv",
        mime="text/csv",
        width="stretch",
    )

    st.markdown("### 📈 내 점수 변화")
    chart_df = df.copy()
    chart_df["도전 번호"] = np.arange(1, len(chart_df) + 1)
    fig_score = px.line(
        chart_df, x="도전 번호", y="점수", markers=True,
        title="도전할수록 점수가 어떻게 달라질까?",
    )
    fig_score.update_layout(
        yaxis=dict(range=[0, 100]),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font=dict(color="#172033"),
    )
    st.plotly_chart(fig_score, width="stretch")

st.divider()
st.caption(
    "AI 학습 원리 체험: 학습률과 반복 횟수가 "
    "수렴·진동·발산에 미치는 영향을 관찰해 보세요."
)
