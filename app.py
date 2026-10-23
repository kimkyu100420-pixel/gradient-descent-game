import html
import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="AI 경사하강법 연구소", page_icon="⚽", layout="wide")

START_X = 8.0
SUCCESS_THRESHOLD = 1e-5
CLIP_LIMIT = 1e5

# --------------------------- Design ---------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Jua&display=swap');
    .stApp,[data-testid="stAppViewContainer"]{background:#F1F5F9;color:#172033}
    [data-testid="stHeader"]{background:rgba(241,245,249,.95)}
    [data-testid="stSidebar"]{background:#DCE7F3;border-right:1px solid #B6C8DC}
    [data-testid="stSidebar"] *{color:#172033!important}
    h1,h2,h3{font-family:'Jua',sans-serif!important;color:#142B4A!important}
    p,li,label,.stMarkdown{color:#243247}
    [data-testid="stMetric"]{background:#fff;border:1px solid #CAD5E2;border-radius:12px;padding:12px}
    [data-testid="stMetricLabel"],[data-testid="stMetricValue"],[data-testid="stMetricDelta"]{color:#172033!important}
    div[data-testid="stButton"] button,div[data-testid="stDownloadButton"] button{border-radius:12px;font-weight:700;border:1px solid #244B78}
    [data-testid="stExpander"]{background:#fff;border:1px solid #CAD5E2;border-radius:12px}
    input,textarea{color:#172033!important;background:#fff!important}
    [data-testid="stCaptionContainer"]{color:#45566D!important}
    .hero{background:linear-gradient(135deg,#15345B,#285C91);padding:26px;border-radius:16px;margin-bottom:16px}
    .hero h1{color:#fff!important;margin:0}.hero p{color:#fff!important;font-size:17px;margin-bottom:0}
    .quest-card{background:#fff;border:1px solid #C8D6E5;border-radius:16px;padding:18px;margin:8px 0}
    .locked{background:#E8EDF3;border:1px dashed #9BAEC2;border-radius:12px;padding:12px;color:#34445A}
    .unlocked{background:#E6F7EC;border:1px solid #78C995;border-radius:12px;padding:12px;color:#14532D}
    @keyframes overlayIn{0%{opacity:0}8%,78%{opacity:1}100%{opacity:0}}
    @keyframes cardIn{0%{transform:scale(.65) rotate(-4deg)}12%{transform:scale(1.04) rotate(1deg)}20%,78%{transform:scale(1) rotate(0)}100%{transform:scale(.96)}}
    @keyframes particleBurst{0%{transform:translate(-50%,-50%) scale(.35) rotate(0);opacity:0}12%{opacity:1}100%{transform:translate(calc(-50% + var(--dx)),calc(-50% + var(--dy))) scale(1.2) rotate(360deg);opacity:0}}
    @keyframes turtleTraverse{0%{transform:translateX(-45vw) rotate(-8deg);opacity:0}12%{opacity:1}35%{transform:translateX(-10vw) rotate(7deg)}60%{transform:translateX(15vw) rotate(-5deg)}88%{opacity:1}100%{transform:translateX(45vw);opacity:0}}
    @keyframes confettiLeft{0%{transform:translate(0,0) rotate(0) scale(.6);opacity:0}12%{opacity:1}100%{transform:translate(var(--dx),var(--dy)) rotate(720deg) scale(1.15);opacity:0}}
    @keyframes confettiRight{0%{transform:translate(0,0) rotate(0) scale(.6);opacity:0}12%{opacity:1}100%{transform:translate(var(--dx),var(--dy)) rotate(-720deg) scale(1.15);opacity:0}}
    .gd-overlay{position:fixed!important;inset:0!important;width:100vw!important;height:100vh!important;z-index:2147483000!important;display:flex!important;align-items:center!important;justify-content:center!important;overflow:hidden!important;pointer-events:none!important;animation:overlayIn 2.5s ease-in-out both;animation-delay:var(--effect-delay,0s);background:radial-gradient(ellipse at center,rgba(135,12,0,.70),rgba(35,0,0,.82))}
    .gd-card{position:relative;overflow:hidden;width:min(88vw,760px);max-height:82vh;padding:clamp(20px,4vw,42px);text-align:center;color:#fff!important;border:3px solid #ff7658;border-radius:26px;background:linear-gradient(145deg,#310b12,#151523 58%,#4a1009);box-shadow:0 0 45px rgba(255,67,20,.65);animation:cardIn 2.5s ease-out both;animation-delay:var(--effect-delay,0s)}
    .gd-card *{color:#fff!important}.gd-kicker{font-size:clamp(22px,4vw,42px);font-weight:1000;color:#ffb0a0!important;margin-bottom:12px}
    .gd-emoji{font-size:clamp(58px,11vw,112px);line-height:1.2;margin:8px 0}.gd-head{font-size:clamp(19px,3.2vw,30px);font-weight:900;line-height:1.3;margin:10px 0}.gd-copy{font-size:clamp(15px,2vw,19px);line-height:1.5;color:#ffe1d6!important}
    .gd-particle{position:absolute;left:50%;top:50%;font-size:clamp(24px,3.5vw,40px);animation:particleBurst 2s cubic-bezier(.15,.65,.2,1) both;animation-delay:calc(var(--effect-delay,0s) + 0s)}
    .gd-success{background:radial-gradient(ellipse at center,rgba(16,95,64,.72),rgba(7,38,47,.84));animation-duration:2.8s}
    .gd-success-card{border-color:#f9dc75;background:linear-gradient(145deg,#064e3b,#123b5b 55%,#51256c);box-shadow:0 0 42px rgba(255,218,90,.5);animation:successCard 2.8s ease-out both;animation-delay:var(--effect-delay,0s)}
    @keyframes successCard{0%{transform:scale(.55) rotate(-5deg);opacity:0}18%{transform:scale(1.08) rotate(2deg);opacity:1}30%,82%{transform:scale(1);opacity:1}100%{transform:scale(.97);opacity:0}}
    .gd-success-particle{position:absolute;top:50%;font-size:clamp(24px,3.8vw,42px);opacity:0}
    .gd-success-particle.left{left:8%;animation:confettiLeft 2.35s ease-out both;animation-delay:var(--effect-delay,0s)}
    .gd-success-particle.right{right:8%;animation:confettiRight 2.35s ease-out both;animation-delay:var(--effect-delay,0s)}
    .gd-turtle{background:radial-gradient(ellipse at center,rgba(15,90,35,.72),rgba(4,35,18,.84));animation-duration:1.5s}
    .gd-turtle-card{background:linear-gradient(145deg,#123b28,#214b25 60%,#6a5520);border-color:#b4e57b;box-shadow:0 0 40px rgba(150,220,80,.45)}
    .gd-turtle-card .gd-emoji{animation:turtleTraverse 1.5s ease-in-out both;animation-delay:var(--effect-delay,0s)}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero"><h1>⚽ AI 경사하강법 연구소</h1><p>직접 실험해 규칙을 발견하고, 퀘스트를 깨며 연구 도감을 완성하세요.</p></div>',
    unsafe_allow_html=True,
)

# --------------------------- Session State ---------------------------
defaults = {
    "personal_records": [],
    "last_result": None,
    "attempt_id": 0,
    "effect_shown_for_attempt": -1,
    "quests_done": [],
    "unlocked_rules": [],
    "xp": 0,
    "quest_notice": None,
    "quests_visible": True,
    "animation_token": 0,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --------------------------- Game Math ---------------------------
def run_gradient_descent(x0, alpha, epochs):
    x = float(x0)
    xs, ys = [x], [x * x]
    overflow = False
    steps = 0
    if abs(x) < SUCCESS_THRESHOLD:
        return np.array(xs), np.array(ys), overflow, steps

    for _ in range(int(epochs)):
        x_next = x - float(alpha) * (2.0 * x)
        if not np.isfinite(x_next) or abs(x_next) > CLIP_LIMIT:
            overflow = True
            if np.isfinite(x_next):
                direction = 1.0 if x_next >= 0 else -1.0
            else:
                direction = (1.0 if x >= 0 else -1.0) * (-1.0 if alpha > 1 else 1.0)
            x_next = direction * CLIP_LIMIT
            xs.append(float(x_next))
            ys.append(float(CLIP_LIMIT ** 2))
            steps += 1
            break
        x = float(x_next)
        xs.append(x)
        ys.append(x * x)
        steps += 1
        if abs(x) < SUCCESS_THRESHOLD:
            break
    return np.array(xs), np.array(ys), overflow, steps


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


def show_failure_effect(alpha, attempt_id, delay_seconds=0):
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
        ("🔥","-42vw","-28vh"),("💥","38vw","-30vh"),("☄️","-43vw","18vh"),
        ("🧨","42vw","20vh"),("⚡","-23vw","32vh"),("🔥","25vw","34vh"),
        ("💣","0vw","-38vh"),("✨","44vw","-3vh"),("💥","-44vw","-4vh"),
        ("🪨","14vw","-33vh"),("🌋","-30vw","25vh"),("🚨","32vw","-20vh"),
    ]
    p_html = "".join(
        f'<span class="gd-particle" style="--dx:{dx};--dy:{dy};animation-delay:{delay_seconds + i*.025}s">{emoji}</span>'
        for i,(emoji,dx,dy) in enumerate(particles)
    )
    st.markdown(
        f"""<div class="gd-overlay" id="failure-{attempt_id}" style="--effect-delay:{delay_seconds:.2f}s;opacity:0">
        <div class="gd-card">{p_html}
        <div class="gd-kicker">☢️ CRITICAL FAILURE ☢️</div>
        <div class="gd-emoji">💣 💥 🔥</div>
        <div class="gd-head">{html.escape(headline)}</div>
        <div class="gd-copy">{html.escape(message)}</div>
        <div class="gd-copy" style="margin-top:12px;font-weight:800">SYSTEM ALERT · 궤도 이탈 · 착지 프로토콜 실패</div>
        </div></div>""",
        unsafe_allow_html=True,
    )


def show_success_effect(attempt_id, delay_seconds=0):
    bits = [
        ("🎆","left","-5vw","-35vh"),("🎇","left","18vw","-28vh"),
        ("🎉","left","32vw","20vh"),("✨","left","15vw","34vh"),
        ("🎊","left","38vw","-5vh"),("⭐","left","28vw","-18vh"),
        ("🎆","right","5vw","-35vh"),("🎇","right","-18vw","-28vh"),
        ("🎉","right","-32vw","20vh"),("✨","right","-15vw","34vh"),
        ("🎊","right","-38vw","-5vh"),("⭐","right","-28vw","-18vh"),
        ("🪅","left","24vw","8vh"),("🥳","right","-24vw","8vh"),
    ]
    p_html = "".join(
        f'<span class="gd-success-particle {side}" style="--dx:{dx};--dy:{dy};animation-delay:{delay_seconds + i*.03}s">{emoji}</span>'
        for i,(emoji,side,dx,dy) in enumerate(bits)
    )
    st.markdown(
        f"""<div class="gd-overlay gd-success" id="success-{attempt_id}" style="--effect-delay:{delay_seconds:.2f}s;opacity:0">
        <div class="gd-card gd-success-card">{p_html}
        <div class="gd-kicker">🎉 LANDING SUCCESS! 🎉</div>
        <div class="gd-emoji">🏆 ⚽ 🥳</div>
        <div class="gd-head">완벽 착륙! 경사하강법 마스터!</div>
        <div class="gd-copy">양쪽에서 폭죽 발사! 🎆 목표 지점에 정확히 도착했어요!<br>축하합니다 — 최고 점수 100점!</div>
        </div></div>""",
        unsafe_allow_html=True,
    )


def show_turtle_effect(attempt_id, delay_seconds=0):
    st.markdown(
        f"""<div class="gd-overlay gd-turtle" id="turtle-{attempt_id}" style="--effect-delay:{delay_seconds:.2f}s;opacity:0">
        <div class="gd-card gd-turtle-card">
        <div class="gd-kicker">🐢 TURTLE MODE ACTIVATED 🐢</div>
        <div class="gd-emoji">🐢</div>
        <div class="gd-head">느림보 경사하강법 출동!</div>
        <div class="gd-copy">목표까지 천천히... 한 걸음씩...<br>🐌 속도는 느려도 포기하지 않는 중! 🚧</div>
        </div></div>""",
        unsafe_allow_html=True,
    )


# --------------------------- Sidebar ---------------------------
st.sidebar.header("🎮 연구소 컨트롤러")
nickname_input = st.sidebar.text_input("연구원 닉네임", value="익명", max_chars=20)
st.sidebar.metric("📍 시작 위치", "8.0 (고정)")
alpha_input = st.sidebar.slider("학습률 α", 0.05, 1.50, 0.80, 0.05, format="%.2f")
epochs_input = st.sidebar.slider("최대 이동 횟수", 1, 10, 5, 1)
playback_speed = 0.65  # 자동 애니메이션 속도(사용자 조절 없음)
run_button = st.sidebar.button("⚽ 실험 실행!", type="primary", width="stretch")

# --------------------------- Hidden rules / quests ---------------------------
RULES = {
    "slow": {
        "title": "느린 수렴",
        "description": "학습률이 작으면 공은 목표를 향해 한쪽에서 조금씩 접근한다.",
        "condition": "0 < α < 0.5",
    },
    "one_step": {
        "title": "완벽한 한 번의 착륙",
        "description": "특정 학습률에서는 첫 업데이트만으로 목표 지점에 도착한다.",
        "condition": "α = 0.5",
    },
    "zigzag": {
        "title": "지그재그 수렴",
        "description": "공이 목표를 넘나들지만, 매번 목표와의 거리가 줄어든다.",
        "condition": "0.5 < α < 1",
    },
    "pingpong": {
        "title": "무한 핑퐁",
        "description": "공은 양쪽으로 번갈아 움직이지만 거리의 크기가 줄어들지 않는다.",
        "condition": "α = 1",
    },
    "diverge": {
        "title": "발산 폭주",
        "description": "공은 목표를 지나칠 뿐 아니라 매번 더 멀리 튕겨 나간다.",
        "condition": "α > 1",
    },
}

QUESTS = [
    {"id":"first_success","title":"첫 번째 착륙","desc":"공을 목표 지점에 성공적으로 착륙시켜라.","reward":100},
    {"id":"slow","title":"거북이의 비밀","desc":"학습률을 바꾸어 느린 수렴 현상을 발견하라.","reward":80},
    {"id":"one_step","title":"원샷 착륙","desc":"한 번의 실제 이동으로 목표에 도착하라.","reward":100},
    {"id":"zigzag","title":"지그재그 탐정","desc":"목표를 넘나들며 수렴하는 조건을 발견하라.","reward":100},
    {"id":"pingpong","title":"무한 핑퐁 실험","desc":"공이 같은 거리로 계속 튕기는 현상을 재현하라.","reward":100},
    {"id":"diverge","title":"폭주 경보","desc":"공이 점점 더 멀리 발산하는 조건을 발견하라.","reward":120},
    {"id":"master","title":"연구소 마스터","desc":"모든 다섯 가지 움직임 규칙을 해금하라.","reward":250},
]

# --------------------------- Run and record ---------------------------
if run_button:
    nickname = (nickname_input.strip() or "익명")[:20]
    xs, ys, overflow, steps = run_gradient_descent(START_X, alpha_input, epochs_input)
    final_x, final_loss = float(xs[-1]), float(ys[-1])
    label, code = classify_result(alpha_input, final_x, overflow)
    score = calc_score(final_x, code)
    st.session_state.attempt_id += 1
    st.session_state.animation_token += 1
    attempt_id = st.session_state.attempt_id

    result = {
        "nickname": nickname, "alpha": float(alpha_input), "epochs": int(epochs_input),
        "actual_steps": int(steps), "x_history": xs, "y_history": ys,
        "final_x": final_x, "final_loss": final_loss, "status_label": label,
        "status_code": code, "score": score, "overflow": overflow, "attempt_id": attempt_id,
    }
    st.session_state.last_result = result
    st.session_state.personal_records.append({
        "닉네임": nickname, "학습률": alpha_input, "최대 이동 횟수": epochs_input,
        "실제 이동 횟수": steps, "최종 위치": final_x, "최종 Loss": final_loss,
        "점수": score, "결과": label,
    })

    # Quest discovery is based on observed experiment results and selected conditions.
    newly_unlocked = []
    if code == "success" and "first_success" not in st.session_state.quests_done:
        st.session_state.quests_done.append("first_success")
        st.session_state.xp += 100
        newly_unlocked.append("첫 번째 착륙 퀘스트 완료! +100 XP")

    if 0 < alpha_input < 0.5 and steps > 0:
        if "slow" not in st.session_state.unlocked_rules:
            st.session_state.unlocked_rules.append("slow")
            newly_unlocked.append("규칙 발견: 느린 수렴")
        if "slow" not in st.session_state.quests_done:
            st.session_state.quests_done.append("slow")
            st.session_state.xp += 80
            newly_unlocked.append("거북이의 비밀 퀘스트 완료! +80 XP")

    if np.isclose(alpha_input, 0.5) and code == "success" and steps == 1:
        if "one_step" not in st.session_state.unlocked_rules:
            st.session_state.unlocked_rules.append("one_step")
            newly_unlocked.append("규칙 발견: 한 번의 착륙")
        if "one_step" not in st.session_state.quests_done:
            st.session_state.quests_done.append("one_step")
            st.session_state.xp += 100
            newly_unlocked.append("원샷 착륙 퀘스트 완료! +100 XP")

    if 0.5 < alpha_input < 1.0 and steps >= 2:
        if "zigzag" not in st.session_state.unlocked_rules:
            st.session_state.unlocked_rules.append("zigzag")
            newly_unlocked.append("규칙 발견: 지그재그 수렴")
        if "zigzag" not in st.session_state.quests_done:
            st.session_state.quests_done.append("zigzag")
            st.session_state.xp += 100
            newly_unlocked.append("지그재그 탐정 퀘스트 완료! +100 XP")

    if np.isclose(alpha_input, 1.0):
        if "pingpong" not in st.session_state.unlocked_rules:
            st.session_state.unlocked_rules.append("pingpong")
            newly_unlocked.append("규칙 발견: 무한 핑퐁")
        if "pingpong" not in st.session_state.quests_done:
            st.session_state.quests_done.append("pingpong")
            st.session_state.xp += 100
            newly_unlocked.append("무한 핑퐁 퀘스트 완료! +100 XP")

    if alpha_input > 1.0:
        if "diverge" not in st.session_state.unlocked_rules:
            st.session_state.unlocked_rules.append("diverge")
            newly_unlocked.append("규칙 발견: 발산 폭주")
        if "diverge" not in st.session_state.quests_done:
            st.session_state.quests_done.append("diverge")
            st.session_state.xp += 120
            newly_unlocked.append("폭주 경보 퀘스트 완료! +120 XP")

    rule_ids = {"slow", "one_step", "zigzag", "pingpong", "diverge"}
    if rule_ids.issubset(set(st.session_state.unlocked_rules)):
        if "master" not in st.session_state.quests_done:
            st.session_state.quests_done.append("master")
            st.session_state.xp += 250
            newly_unlocked.append("연구소 마스터 달성! +250 XP")

    st.session_state.quest_notice = newly_unlocked

    if code == "success":
        st.balloons()
        st.toast("착륙 성공! 100점!", icon="🎉")
    elif code in ("explode", "pingpong"):
        st.toast("실패! 결과를 분석하고 다음 실험을 해 보세요.", icon="💥")
    else:
        st.toast("실험 결과를 기록했어요.", icon="🔬")

# --------------------------- Quest dashboard ---------------------------
st.divider()
quest_head, quest_toggle = st.columns([4, 1])
with quest_head:
    st.header("🧪 연구 퀘스트")
    st.write("퀘스트를 숨겨도 실험과 점수 기록은 계속 이용할 수 있어요.")
with quest_toggle:
    toggle_label = "🙈 퀘스트 숨기기" if st.session_state.quests_visible else "🧪 퀘스트 열기"
    if st.button(toggle_label, key="toggle_quests", width="stretch"):
        st.session_state.quests_visible = not st.session_state.quests_visible
        st.rerun()

if st.session_state.quests_visible:
    st.write("처음에는 규칙이 숨겨져 있어요. 직접 실험해 현상을 발견하면 해당 규칙이 해금됩니다.")
    xp_col, done_col, rule_col = st.columns(3)
    xp_col.metric("✨ 연구 경험치", f"{st.session_state.xp} XP")
    done_col.metric("🏁 완료 퀘스트", f"{len(st.session_state.quests_done)} / {len(QUESTS)}")
    rule_col.metric("🔓 발견한 규칙", f"{len(st.session_state.unlocked_rules)} / 5")

    if st.session_state.quest_notice:
        for notice in st.session_state.quest_notice:
            st.success(notice)
        st.session_state.quest_notice = None

    for quest in QUESTS:
        completed = quest["id"] in st.session_state.quests_done
        with st.container(border=True):
            top1, top2 = st.columns([4, 1])
            top1.markdown(f"### {'✅' if completed else '🔒'} {quest['title']}")
            top2.markdown(f"**{quest['reward']} XP**")
            st.write(quest["desc"])
            if completed:
                st.success("퀘스트 완료")
            else:
                st.caption("미완료 — 사이드바에서 실험 조건을 바꾸어 도전하세요.")

    # --------------------------- Rule encyclopedia ---------------------------
    st.subheader("📚 학습률 움직임 연구 도감")
    st.caption("발견한 규칙만 공개됩니다. 아직 발견하지 못한 항목은 잠겨 있어요.")
    for rule_id, rule in RULES.items():
        if rule_id in st.session_state.unlocked_rules:
            with st.container(border=True):
                st.markdown(f"### 🔓 {rule['title']}")
                st.write(rule["description"])
                st.code(rule["condition"])
        else:
            with st.container(border=True):
                st.markdown(f"### 🔒 미확인 현상 — {len(st.session_state.unlocked_rules)+1:02d}")
                st.write("규칙이 잠겨 있습니다. 학습률과 이동 횟수를 바꾸어 실험해 보세요.")
else:
    st.info("퀘스트와 연구 도감이 숨겨져 있습니다. 다시 보려면 ‘🧪 퀘스트 열기’를 눌러 주세요.")

# --------------------------- Current experiment graph/results ---------------------------
st.divider()
graph_col, score_col = st.columns([2, 1])
with graph_col:
    st.subheader("🏔️ 공의 이동 경로")
    result = st.session_state.last_result

    if result is None:
        fig = go.Figure()
        curve_x = np.linspace(-12, 12, 500)
        fig.add_trace(go.Scatter(
            x=curve_x, y=curve_x**2, mode="lines",
            line=dict(color="#667085", width=3), name="골짜기 f(x)=x²",
            hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=[0], y=[0], mode="markers",
            marker=dict(size=16, color="#F4B400", symbol="star"),
            name="목표 지점"))
        fig.update_layout(
            xaxis_title="공의 위치 (x)", yaxis_title="오차 (Loss)",
            xaxis=dict(range=[-12, 12], zeroline=True, fixedrange=True),
            yaxis=dict(range=[-5, 120], zeroline=True, fixedrange=True),
            height=540, margin=dict(l=15, r=15, t=35, b=15),
            plot_bgcolor="#fff", paper_bgcolor="#fff",
            font=dict(color="#172033"), uirevision="fixed-gradient-graph")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.caption("실험을 실행하면 공이 포물선 위를 따라 이동하는 애니메이션이 자동으로 재생됩니다.")
    else:
        xs_anim = np.asarray(result["x_history"], dtype=float)
        ys_anim = np.asarray(xs_anim ** 2, dtype=float)
        # The ball travels along the parabola y=x²; the red trace shows the path traced so far.
        # Axis limits are identical for every frame, so the graph itself stays fixed.
        curve_x = np.linspace(-12, 12, 500)
        graph_width = 760
        graph_height = 510
        frame_duration = max(120, int(420 / playback_speed))
        transition_duration = max(80, int(260 / playback_speed))
        frame_payload = []
        for i in range(len(xs_anim)):
            frame_payload.append({
                "x": xs_anim[:i + 1].tolist(),
                "y": ys_anim[:i + 1].tolist(),
                "ball_x": float(xs_anim[i]),
                "ball_y": float(ys_anim[i]),
                "step": i,
            })

        # Use Plotly's native animated frames and trigger autoplay when the graph is rendered.
        # Plotly's axes remain fixed throughout the animation.
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=curve_x, y=curve_x**2, mode="lines",
            line=dict(color="#667085", width=3), name="골짜기 f(x)=x²",
            hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=frame_payload[0]["x"], y=frame_payload[0]["y"],
            mode="lines+markers",
            line=dict(color="#D62728", width=3),
            marker=dict(size=9, color="#D62728", symbol="circle", line=dict(color="#7F1D1D", width=1)),
            name="공이 지나간 경로 · 방문 지점",
            hovertemplate="위치: %{x:.5f}<br>Loss: %{y:.5f}<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=[frame_payload[0]["ball_x"]], y=[frame_payload[0]["ball_y"]],
            mode="markers+text",
            marker=dict(size=22, color="#FF8A00", symbol="circle",
                        line=dict(color="#7F1D1D", width=2)),
            text=["⚽"], textposition="top center", textfont=dict(size=20),
            name="공의 현재 위치",
            hovertemplate="현재 위치: %{x:.5f}<br>Loss: %{y:.5f}<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=[0], y=[0], mode="markers",
            marker=dict(size=16, color="#F4B400", symbol="star",
                        line=dict(color="#8A5A00", width=1)),
            name="목표 지점"))

        frames = []
        for i, frame in enumerate(frame_payload):
            frames.append(go.Frame(
                name=str(i),
                data=[
                    go.Scatter(
                        x=frame["x"], y=frame["y"], mode="lines+markers",
                        line=dict(color="#D62728", width=3),
                        marker=dict(size=9, color="#D62728", symbol="circle", line=dict(color="#7F1D1D", width=1)),
                        name="공이 지나간 경로 · 방문 지점"),
                    go.Scatter(
                        x=[frame["ball_x"]], y=[frame["ball_y"]],
                        mode="markers+text",
                        marker=dict(size=22, color="#FF8A00", symbol="circle",
                                    line=dict(color="#7F1D1D", width=2)),
                        text=["⚽"], textposition="top center", textfont=dict(size=20),
                        name="공의 현재 위치"),
                ],
                traces=[1, 2],
                layout=go.Layout(title_text=f"경사하강법 이동 단계 {frame['step']} / {len(frame_payload)-1}"),
            ))
        fig.frames = frames
        fig.update_layout(
            title_text=f"경사하강법 이동 단계 0 / {len(frame_payload)-1}",
            xaxis_title="공의 위치 (x)", yaxis_title="오차 (Loss)",
            xaxis=dict(range=[-12, 12], zeroline=True, fixedrange=True,
                       constrain="domain"),
            yaxis=dict(range=[-5, 120], zeroline=True, fixedrange=True,
                       scaleanchor="x", scaleratio=0.35),
            height=540, margin=dict(l=15, r=15, t=55, b=15),
            plot_bgcolor="#fff", paper_bgcolor="#fff",
            font=dict(color="#172033"),
            showlegend=True,
            legend=dict(orientation="h", y=1.12, x=0),
            uirevision="fixed-gradient-graph",
            updatemenus=[{
                "type": "buttons", "showactive": False, "x": 1.0, "y": 1.18,
                "xanchor": "right",
                "buttons": [
                    {"label": "▶ 다시 재생", "method": "animate",
                     "args": [None, {"frame": {"duration": frame_duration, "redraw": True},
                                     "transition": {"duration": transition_duration},
                                     "fromcurrent": False, "mode": "immediate"}]},
                    {"label": "⏸ 정지", "method": "animate",
                     "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                       "mode": "immediate", "transition": {"duration": 0}}]},
                ],
            }],
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        # Autoplay this attempt's frames from a tiny adjacent component after Plotly is rendered.
        # The animation itself is rendered in the fixed Plotly graph above; this bridge locates its
        # Plotly div and calls Plotly.animate once for each attempt token.
        components.html(
            f"""
            <div id="autoplay-status" style="font:12px sans-serif;color:#64748b;padding:2px 0">
              ⚽ 이동 애니메이션 자동 재생 준비 중…
            </div>
            <script>
            const token = {int(result["attempt_id"])};
            const totalFrames = {len(frame_payload)};
            const frameMs = {frame_duration};
            const transitionMs = {transition_duration};
            let tries = 0;
            function startAnimation() {{
              tries += 1;
              const parentDoc = window.parent.document;
              const plots = parentDoc.querySelectorAll('.stPlotlyChart .js-plotly-plot');
              const plot = plots.length ? plots[plots.length - 1] : null;
              const status = document.getElementById('autoplay-status');
              if (plot && parentDoc.defaultView.Plotly && plot._fullLayout) {{
                if (plot.dataset.autoplayToken !== String(token)) {{
                  plot.dataset.autoplayToken = String(token);
                  parentDoc.defaultView.Plotly.animate(
                    plot, null,
                    {{frame: {{duration: frameMs, redraw: true}},
                     transition: {{duration: transitionMs}},
                     fromcurrent: false, mode: 'immediate'}}
                  );
                }}
                status.textContent = '⚽ 공이 포물선을 따라 자동으로 이동 중…';
              }} else if (tries < 40) {{
                setTimeout(startAnimation, 150);
              }} else {{
                status.textContent = '자동 재생을 시작하지 못했어요. 그래프의 “▶ 다시 재생”을 눌러 주세요.';
              }}
            }}
            setTimeout(startAnimation, 250);
            </script>
            """,
            height=30,
            scrolling=False,
        )
        st.caption("축구공은 이차함수 f(x)=x² 그래프 위를 따라 자동으로 이동합니다. 지나간 위치마다 점이 찍히며, 그래프 축은 고정됩니다.")
    if result is not None and abs(result["final_x"]) > 12:
        st.warning("공이 그래프 표시 범위 밖으로 나갔어요. 오른쪽 결과에서 최종 위치를 확인하세요.")

with score_col:
    st.subheader("📟 이번 실험 결과")
    result = st.session_state.last_result
    if result is None:
        st.info("사이드바에서 설정하고 실험을 시작하세요.")
    else:
        code = result["status_code"]
        if st.session_state.effect_shown_for_attempt != result["attempt_id"]:
            if code == "success":
                show_success_effect(result["attempt_id"], delay_seconds=max(0.8, result["actual_steps"] * frame_duration / 1000))
                st.session_state.effect_shown_for_attempt = result["attempt_id"]
            elif code in ("explode", "pingpong"):
                show_failure_effect(result["alpha"], result["attempt_id"], delay_seconds=max(0.8, result["actual_steps"] * frame_duration / 1000))
                st.session_state.effect_shown_for_attempt = result["attempt_id"]
            elif code == "landing" and result["alpha"] <= 0.30 and abs(result["final_x"]) >= SUCCESS_THRESHOLD:
                show_turtle_effect(result["attempt_id"], delay_seconds=max(0.8, result["actual_steps"] * frame_duration / 1000))
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

        st.metric("🏅 점수", f"{result['score']} / 100")
        st.metric("📍 최종 위치", f"{result['final_x']:.8g}")
        st.metric("📉 최종 Loss", f"{result['final_loss']:.8g}")
        st.metric("🔁 실제 이동 횟수", f"{result['actual_steps']}회",
                  help="실제로 위치 업데이트를 실행한 횟수예요. 목표에 도착하면 조기 종료합니다.")
        st.caption(f"최대 이동 횟수: {result['epochs']}회 · 성공 기준: |x| < {SUCCESS_THRESHOLD:g}")

# --------------------------- Personal records ---------------------------
st.divider()
st.header("📒 내 개인 기록")
st.caption("현재 브라우저 세션에서 실행한 실험 기록입니다.")
df = pd.DataFrame(st.session_state.personal_records)
if df.empty:
    st.info("아직 개인 기록이 없어요. 실험을 실행하면 결과가 저장됩니다.")
else:
    a, b, c = st.columns(3)
    a.metric("🎮 총 플레이", f"{len(df)}회")
    b.metric("🏅 최고 점수", f"{int(df['점수'].max())}점")
    c.metric("🎯 성공 횟수", f"{int((df['결과'] == '착륙 성공').sum())}회")
    st.dataframe(df.iloc[::-1], width="stretch", hide_index=True)
    st.download_button("📥 내 기록 CSV 다운로드",
                       data=df.to_csv(index=False).encode("utf-8-sig"),
                       file_name="my_gradient_descent_records.csv",
                       mime="text/csv", width="stretch")
    chart_df = df.copy()
    chart_df["도전 번호"] = np.arange(1, len(chart_df) + 1)
    fig_score = px.line(chart_df, x="도전 번호", y="점수", markers=True,
                        title="도전할수록 점수가 어떻게 달라질까?")
    fig_score.update_layout(yaxis=dict(range=[0, 100]), plot_bgcolor="#fff",
                            paper_bgcolor="#fff", font=dict(color="#172033"))
    st.plotly_chart(fig_score, width="stretch")

st.divider()
st.caption("경사하강법 연구소 · 실험하고 발견하고 해금하세요!")
