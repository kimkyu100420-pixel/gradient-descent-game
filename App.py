import html
import math
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# =========================================================
# 1. 설정
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
# 2. 가독성 중심 디자인 + 효과 애니메이션 CSS
# =========================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Jua&display=swap');

    .stApp, [data-testid="stAppViewContainer"] {
        background: #F1F5F9;
        color: #172033;
    }
    [data-testid="stHeader"] { background: rgba(241,245,249,0.95); }
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
        background: #FFFFFF;
        border: 1px solid #CAD5E2;
        border-radius: 12px;
        padding: 12px;
    }
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"],
    [data-testid="stMetricDelta"] { color: #172033 !important; }
    [data-testid="stDataFrame"] { background: #FFFFFF; }
    div[data-testid="stButton"] button,
    div[data-testid="stDownloadButton"] button {
        border-radius: 12px;
        font-weight: 700;
        border: 1px solid #244B78;
    }
    [data-testid="stExpander"] {
        background: #FFFFFF;
        border: 1px solid #CAD5E2;
        border-radius: 12px;
    }
    input, textarea {
        color: #172033 !important;
        background: #FFFFFF !important;
    }
    [data-testid="stCaptionContainer"] { color: #45566D !important; }

    .hero {
        background: linear-gradient(135deg, #15345B, #285C91);
        padding: 26px;
        border-radius: 16px;
        margin-bottom: 16px;
    }
    .hero h1 { color: #FFFFFF !important; margin: 0; }
    .hero p { color: #FFFFFF !important; font-size: 17px; margin-bottom: 0; }

    /* 실패 오버레이 */
    @keyframes failureShake {
        0%, 100% { transform: translate(0, 0); }
        10% { transform: translate(-9px, 4px); }
        20% { transform: translate(8px, -3px); }
        30% { transform: translate(-7px, -5px); }
        40% { transform: translate(9px, 3px); }
        50% { transform: translate(-6px, 2px); }
        60% { transform: translate(6px, -4px); }
        70% { transform: translate(-4px, 3px); }
        80% { transform: translate(4px, 0); }
    }
    @keyframes dangerFlash {
        0%, 100% { opacity: 0; }
        12%, 36% { opacity: .45; }
        24%, 48% { opacity: 0; }
    }
    @keyframes boomPop {
        0% { transform: scale(.08) rotate(-20deg); opacity: 0; }
        35% { transform: scale(1.25) rotate(8deg); opacity: 1; }
        65% { transform: scale(.95) rotate(-5deg); opacity: 1; }
        100% { transform: scale(1) rotate(0); opacity: 1; }
    }
    @keyframes particleFly {
        0% { transform: translate(0, 0) scale(.3) rotate(0); opacity: 0; }
        15% { opacity: 1; }
        100% { transform: translate(var(--dx), var(--dy)) scale(1.2) rotate(220deg); opacity: 0; }
    }
    @keyframes warningPulse {
        0%, 100% { text-shadow: 0 0 5px #ff2525; }
        50% { text-shadow: 0 0 22px #ff2525, 0 0 38px #ff5a00; }
    }
    @keyframes turtleWalk {
        0% { opacity: 0; transform: translateX(-35px); }
        15%, 75% { opacity: 1; }
        100% { opacity: 0; transform: translateX(35px); }
    }
    .failure-overlay {
        position: fixed;
        inset: 0;
        z-index: 99999;
        pointer-events: none;
        display: flex;
        justify-content: center;
        align-items: center;
        background: radial-gradient(circle, rgba(255,40,0,.15), rgba(100,0,0,.55));
        animation: dangerFlash 1.2s ease-out both;
    }
    .failure-card {
        position: relative;
        overflow: hidden;
        width: min(90vw, 650px);
        padding: 28px 20px;
        text-align: center;
        color: white;
        border: 3px solid #ff3b30;
        border-radius: 22px;
        background: linear-gradient(160deg, #380606, #100d18 65%, #6d1000);
        box-shadow: 0 0 35px #ff2400, inset 0 0 35px rgba(255,40,0,.25);
        animation: boomPop .65s cubic-bezier(.2,.8,.3,1.3) both;
    }
    .failure-title {
        font-size: clamp(28px, 5vw, 46px);
        font-weight: 1000;
        color: #ff4438;
        animation: warningPulse .35s infinite;
    }
    .failure-boom {
        font-size: clamp(60px, 12vw, 100px);
        line-height: 1.3;
        filter: drop-shadow(0 0 18px #ff4b00);
    }
    .failure-message {
        font-size: clamp(18px, 3vw, 25px);
        font-weight: 900;
        margin-top: 12px;
        color: #FFFFFF;
    }
    .failure-subtitle {
        margin-top: 12px;
        color: #ffd4c7;
        font-size: 15px;
    }
    .failure-particle {
        position: absolute;
        left: 50%;
        top: 50%;
        font-size: 28px;
        animation: particleFly 1.15s ease-out both;
    }
    .turtle-wrap {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 12px;
        padding: 18px;
        margin: 12px 0;
        border-radius: 16px;
        background: #e4f4d7;
        border: 2px solid #76a85b;
        color: #244b1c;
        font-weight: 800;
    }
    .turtle-emoji {
        font-size: 48px;
        animation: turtleWalk 1.5s ease-in-out both;
    }
    .turtle-text { font-size: 18px; }
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
        st.markdown("**👟 학습률 = 보폭**\n\n한 번 이동할 때 얼마나 크게 움직일지 정해요. 너무 크면 목표를 지나칠 수 있어요.")
    with c3:
        st.markdown("**🔁 발걸음 수 = 이동 기회**\n\n공의 위치를 최대 몇 번 갱신할지 정해요.")
    st.markdown(
        """
        **학습률별 움직임**
        - `0.05~0.49`: 목표를 향해 수렴
        - `0.50`: 한 번에 목표에 도착
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
if "effect_nonce" not in st.session_state:
    st.session_state.effect_nonce = 0

# =========================================================
# 5. 경사하강법 계산
# =========================================================
def run_gradient_descent(x0, alpha, epochs):
    x = float(x0)
    x_history = [x]
    y_history = [x * x]
    overflow = False
    actual_steps = 0

    for _ in range(int(epochs)):
        x_next = x - alpha * (2.0 * x)

        if not np.isfinite(x_next) or abs(x_next) > CLIP_LIMIT:
            overflow = True
            if np.isfinite(x_next):
                direction = 1.0 if x_next >= 0 else -1.0
            else:
                direction = 1.0 if x >= 0 else -1.0
                if alpha > 1:
                    direction *= -1.0
            x_next = direction * CLIP_LIMIT
            x_history.append(x_next)
            y_history.append(CLIP_LIMIT ** 2)
            actual_steps += 1
            break

        x = float(x_next)
        x_history.append(x)
        y_history.append(x * x)
        actual_steps += 1

    return np.array(x_history), np.array(y_history), overflow, actual_steps


def classify_result(alpha, final_x, overflow):
    if overflow or not np.isfinite(final_x):
        return "우주폭발", "explode"
    if alpha > 1.0:
        return "우주폭발", "explode"
    if np.isclose(alpha, 1.0, atol=1e-8):
        return "무한 핑퐁", "pingpong"
    if abs(final_x) < SUCCESS_THRESHOLD:
        return "착륙 성공", "success"
    if alpha > 0.5:
        return "지그재그 수렴 중", "oscillating"
    return "아직 착륙 중", "landing"


def calc_score(final_x, status_code):
    """성공일 때만 100점, 그 외에는 최종 위치 오차에 따라 0~99점."""
    if status_code == "success":
        return 100
    if not np.isfinite(final_x):
        return 0
    score = int(99 * math.exp(-abs(final_x) / 2.0))
    return max(0, min(99, score))


def show_failure_effect(alpha, effect_nonce):
    """components.html에서 독립 HTML로 폭발 애니메이션을 렌더링한다."""
    if alpha > 1:
        message = "학습률이 너무 공격적이었습니다!"
    elif np.isclose(alpha, 1.0, atol=1e-8):
        message = "공이 끝없이 튕깁니다! 수렴하지 않습니다!"
    else:
        message = "정해진 이동 횟수 안에 착륙하지 못했습니다!"

    particles = [
        ("🔥", "-220px", "-160px"), ("💥", "210px", "-150px"),
        ("☄️", "-240px", "90px"), ("🧨", "230px", "110px"),
        ("⚡", "-120px", "190px"), ("🔥", "130px", "200px"),
        ("💣", "0px", "-220px"), ("✨", "260px", "-10px"),
        ("💥", "-260px", "-20px"), ("🪨", "80px", "-190px"),
    ]
    particle_html = "".join(
        f'<span class="particle" style="--dx:{dx};--dy:{dy};animation-delay:{i * 0.035}s">{emoji}</span>'
        for i, (emoji, dx, dy) in enumerate(particles)
    )
    safe_message = html.escape(message)

    html_doc = f"""
    <!doctype html><html><head><meta charset="utf-8">
    <style>
      * {{ box-sizing: border-box; }}
      html, body {{ margin:0; padding:0; width:100%; height:100%; overflow:hidden; background:transparent; font-family:Arial,sans-serif; }}
      @keyframes shake {{
        0%,100% {{ transform:translate(0,0) }}
        10% {{ transform:translate(-9px,4px) }} 20% {{ transform:translate(8px,-3px) }}
        30% {{ transform:translate(-7px,-5px) }} 40% {{ transform:translate(9px,3px) }}
        50% {{ transform:translate(-6px,2px) }} 60% {{ transform:translate(6px,-4px) }}
        70% {{ transform:translate(-4px,3px) }} 80% {{ transform:translate(4px,0) }}
      }}
      @keyframes flash {{ 0%,100% {{opacity:0}} 12%,36% {{opacity:.5}} 24%,48% {{opacity:0}} }}
      @keyframes pop {{ 0% {{transform:scale(.05) rotate(-20deg);opacity:0}} 35% {{transform:scale(1.12) rotate(8deg);opacity:1}} 65% {{transform:scale(.95) rotate(-5deg);opacity:1}} 100% {{transform:scale(1);opacity:1}} }}
      @keyframes fly {{ 0% {{transform:translate(0,0) scale(.3) rotate(0);opacity:0}} 15% {{opacity:1}} 100% {{transform:translate(var(--dx),var(--dy)) scale(1.2) rotate(220deg);opacity:0}} }}
      @keyframes pulse {{ 0%,100% {{text-shadow:0 0 5px #ff2525}} 50% {{text-shadow:0 0 22px #ff2525,0 0 38px #ff5a00}} }}
      .overlay {{ position:relative; width:100%; height:340px; display:flex; align-items:center; justify-content:center; background:radial-gradient(circle,rgba(255,40,0,.2),rgba(100,0,0,.65)); animation:flash 1.2s ease-out both; border-radius:18px; overflow:hidden; }}
      .card {{ position:relative; overflow:hidden; width:min(94%,650px); padding:24px 16px; text-align:center; color:white; border:3px solid #ff3b30; border-radius:22px; background:linear-gradient(160deg,#380606,#100d18 65%,#6d1000); box-shadow:0 0 35px #ff2400,inset 0 0 35px rgba(255,40,0,.25); animation:pop .65s cubic-bezier(.2,.8,.3,1.3) both,shake .55s ease-in-out .1s 2; }}
      .title {{ font-size:clamp(24px,4vw,42px); font-weight:1000; color:#ff4438; animation:pulse .35s infinite; }}
      .boom {{ font-size:clamp(48px,9vw,86px); line-height:1.25; filter:drop-shadow(0 0 18px #ff4b00); }}
      .message {{ font-size:clamp(17px,2.8vw,24px); font-weight:900; margin-top:8px; }}
      .subtitle {{ margin-top:10px; color:#ffd4c7; font-size:14px; }}
      .particle {{ position:absolute; left:50%; top:50%; font-size:27px; animation:fly 1.15s ease-out both; }}
    </style></head><body>
      <div class="overlay"><div class="card">
        {particle_html}
        <div class="title">☢️ CRITICAL FAILURE ☢️</div>
        <div class="boom">💣💥🔥</div>
        <div class="message">착지 실패! 경사하강법 대참사 발생!</div>
        <div class="subtitle">{safe_message}<br>SYSTEM ALERT · 다음 도전에서는 설정을 조절하세요.</div>
      </div></div>
      <script>
        // iframe 안의 애니메이션은 이 독립 문서에서 실행됩니다.
        document.documentElement.dataset.effect = "{effect_nonce}";
      </script>
    </body></html>
    """
    components.html(html_doc, height=350, scrolling=False)


def show_turtle_effect():
    """느린 수렴 결과에 거북이 애니메이션을 1.5초 표시한다."""
    components.html(
        """
        <!doctype html><html><head><meta charset="utf-8"><style>
        @keyframes turtleWalk {
          0% {opacity:0;transform:translateX(-45px)}
          15%,75% {opacity:1}
          100% {opacity:0;transform:translateX(45px)}
        }
        body {margin:0;font-family:Arial,sans-serif}
        .wrap {height:90px;display:flex;align-items:center;justify-content:center;gap:12px;
          border-radius:15px;background:#e4f4d7;border:2px solid #76a85b;color:#244b1c;font-weight:800}
        .turtle {font-size:45px;animation:turtleWalk 1.5s ease-in-out both}
        .text {font-size:17px}
        </style></head><body>
        <div class="wrap"><span class="turtle">🐢</span>
        <span class="text">느릿느릿 이동 중... 조금만 더 기다려 주세요!</span></div>
        </body></html>
        """,
        height=100,
        scrolling=False,
    )

# =========================================================
# 6. 게임 컨트롤러
# =========================================================
st.sidebar.header("🎮 게임 컨트롤러")
st.sidebar.caption("출발 위치는 모두 같아요.")
nickname_input = st.sidebar.text_input("플레이어 닉네임", value="익명", max_chars=20)
st.sidebar.metric("📍 시작 위치", "8.0 (고정)")
st.sidebar.markdown("---")
st.sidebar.subheader("👟 학습률 — 보폭")

alpha_input = st.sidebar.slider(
    "한 번에 움직이는 크",
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
# 7. 실행 및 개인 기록 저장
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

    st.session_state.effect_nonce += 1
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
        "effect_nonce": st.session_state.effect_nonce,
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
# 8. 그래프
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
        xs = result["x_history"]
        ys = result["y_history"]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers",
            line=dict(color="#D62728", width=3, dash="dot"),
            marker=dict(size=10, color="#D62728", line=dict(color="#7F1D1D", width=1)),
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

        # 효과를 점수판에 표시: HTML을 Markdown 코드로 출력하지 않고 별도 iframe으로 렌더링
        if code in ("explode", "pingpong"):
            show_failure_effect(result["alpha"], result["effect_nonce"])
        elif (
            code == "landing"
            and result["alpha"] <= 0.30
            and abs(result["final_x"]) >= SUCCESS_THRESHOLD
        ):
            show_turtle_effect()

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
                  help="공의 위치를 실제로 갱신한 횟수예요.")
        st.caption(
            f"최대 이동 횟수: {result['epochs']}회 · "
            f"성공 기준: |x| < {SUCCESS_THRESHOLD:g}"
        )

# =========================================================
# 9. 개인 기록실
# =========================================================
st.divider()
st.header("📒 내 개인 기록")
st.caption("이 기록실에는 현재 브라우저 세션에서 내가 실행한 실험만 표시돼요.")
records = st.session_state.personal_records
df = pd.DataFrame(records)

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
        "📥 내 기록 CSV 다운로드", data=csv_data,
        file_name="my_gradient_descent_records.csv",
        mime="text/csv", width="stretch",
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
