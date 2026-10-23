import html
import json
import math

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="AI 경사하강법 연구소", page_icon="⚽", layout="wide")

START_X = 8.0
SUCCESS_THRESHOLD = 1e-5
TARGET_XP = 10000
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
    div[data-testid="stButton"] button,div[data-testid="stDownloadButton"] button{border-radius:10px;font-weight:850;border:1px solid #55F4FF;background:linear-gradient(135deg,#C8FCFF,#58E8F4)!important;color:#062A40!important;min-height:2.7rem;box-shadow:0 0 12px rgba(50,235,255,.18)}
    div[data-testid="stButton"] button p,div[data-testid="stDownloadButton"] button p{color:#062A40!important}
    div[data-testid="stButton"] button[kind="primary"]{background:linear-gradient(135deg,#63F5FF,#19BBD5)!important;color:#03263B!important;border:1px solid #A1FBFF!important;box-shadow:0 0 16px rgba(54,235,255,.38)}
    div[data-testid="stButton"] button[kind="primary"] p{color:#03263B!important}
    div[data-testid="stButton"] button p,div[data-testid="stDownloadButton"] button p{color:#163354!important}
    div[data-testid="stButton"] button[kind="primary"]{background:linear-gradient(135deg,#1769AA,#2946A0)!important;color:#FFFFFF!important;border:1px solid #164E83!important}
    div[data-testid="stButton"] button[kind="primary"] p{color:#FFFFFF!important}
    [data-testid="stAlert"]{background:#FFFFFF!important;border:1px solid #A8BED6!important;border-radius:12px}
    [data-testid="stAlert"] p,[data-testid="stAlert"] div,[data-testid="stAlert"] span{color:#172B45!important}
    [data-testid="stToast"]{background:#FFFFFF!important;color:#172033!important;border:1px solid #9BB5D2!important}
    [data-testid="stToast"] *{color:#172033!important}
    [data-testid="stNotification"]{color:#172033!important}
    [data-testid="stTextInput"] input,[data-testid="stNumberInput"] input{background:#FFFFFF!important;color:#172033!important}
    [data-testid="stSelectbox"] div,[data-testid="stSlider"] label{color:#172033!important}
    [data-testid="stExpander"]{background:#fff;border:1px solid #CAD5E2;border-radius:12px}
    input,textarea{color:#172033!important;background:#fff!important}
    [data-testid="stCaptionContainer"]{color:#45566D!important}
    .hero{background:linear-gradient(135deg,#15345B,#285C91);padding:26px;border-radius:16px;margin-bottom:16px}
    .hero h1{color:#fff!important;margin:0}.hero p{color:#fff!important;font-size:17px;margin-bottom:0}
    .quest-card{background:#fff;border:1px solid #C8D6E5;border-radius:16px;padding:18px;margin:8px 0}
    .locked{background:#E8EDF3;border:1px dashed #9BAEC2;border-radius:12px;padding:12px;color:#34445A}
    .unlocked{background:#E6F7EC;border:1px solid #78C995;border-radius:12px;padding:12px;color:#14532D}
    .research-panel{background:linear-gradient(145deg,#062238,#0B354D 55%,#09263D);border:2px solid #4CF4FF;border-radius:20px;padding:clamp(18px,3vw,32px);box-shadow:0 0 18px rgba(36,226,255,.22),inset 0 0 24px rgba(47,217,255,.07);margin:14px 0;color:#E8FBFF}
    .panel-heading{background:linear-gradient(120deg,#0D3A53,#10435B);border:1px solid #38DDEC;border-radius:12px;padding:16px 20px;margin-bottom:14px;box-shadow:inset 0 0 16px rgba(51,226,255,.1)}
    .panel-heading h2{margin:0;color:#E8FDFF!important;text-shadow:0 0 10px rgba(65,241,255,.45)}
    .panel-heading p{margin:6px 0 0;color:#B7EDF6!important}
    .research-panel [data-testid="stMarkdownContainer"] p,.research-panel [data-testid="stCaptionContainer"],.research-panel label{color:#D9F5FB!important}
    .research-panel [data-testid="stVerticalBlockBorderWrapper"]{background:linear-gradient(145deg,#0A2B41,#103B52);border:1px solid #2BAFC6;border-radius:12px}
    .progress-hud{background:linear-gradient(115deg,#061B30,#0B3850,#09253E);border:2px solid #48EFFF;border-radius:18px;padding:18px 22px;box-shadow:0 0 18px rgba(49,227,255,.2);margin:12px 0 20px;color:#E9FCFF}
    .progress-hud h3{color:#B8FAFF!important;margin:0 0 8px!important}
    .progress-hud p{color:#D9F7FC!important;margin:0}
    .master-banner{position:relative;overflow:hidden;text-align:center;padding:clamp(24px,5vw,52px);margin:18px 0 24px;border-radius:24px;border:3px solid #ffe68a;background:linear-gradient(120deg,#101b4b,#542b83,#a34a7c,#174e72);box-shadow:0 0 28px rgba(255,194,69,.65),inset 0 0 30px rgba(255,255,255,.12);animation:masterGlow 2.2s ease-in-out infinite alternate}
    .master-banner h1{font-size:clamp(38px,7vw,76px)!important;line-height:1.12;margin:0;color:#fff!important;text-shadow:0 0 8px #ffe68a,0 0 22px #ff8de1,0 4px 2px #17113b!important}
    .master-banner p{font-size:clamp(16px,2.4vw,24px);font-weight:900;color:#fff!important;margin:14px 0 0;text-shadow:0 2px 8px #17113b}
    @keyframes masterGlow{from{filter:saturate(1);box-shadow:0 0 22px rgba(255,194,69,.45),inset 0 0 25px rgba(255,255,255,.1)}to{filter:saturate(1.45);box-shadow:0 0 42px rgba(255,194,69,.85),inset 0 0 38px rgba(255,255,255,.18)}}
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
    "quests_claimed": [],
    "rules_claimed": [],
    "quest_firework": False,
    "unlocked_rules": [],
    "xp": 0,
    "quest_notice": None,
    "quests_visible": True,
    "research_visible": True,
    "animation_token": 0,
    "researcher_mode": False,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# 연구소 마스터: 별도의 퀘스트가 아니라 모든 일반 퀘스트와 도감 규칙 완성 시 표시
REQUIRED_QUEST_IDS = ["first_success", "slow", "one_step", "zigzag", "pingpong", "diverge"]
ALL_RULE_IDS = ["slow", "one_step", "zigzag", "pingpong", "diverge"]
is_research_lab_master = (
    all(qid in st.session_state.quests_done for qid in REQUIRED_QUEST_IDS)
    and all(rid in st.session_state.unlocked_rules for rid in ALL_RULE_IDS)
    and all(qid in st.session_state.quests_claimed for qid in REQUIRED_QUEST_IDS)
    and all(rid in st.session_state.rules_claimed for rid in ALL_RULE_IDS)
)
if is_research_lab_master:
    st.markdown("""
    <div class="master-banner">
      <div style="font-size:clamp(28px,5vw,54px);margin-bottom:8px">✨ 🎆 🏆 🎇 ✨</div>
      <h1>연구소 마스터</h1>
      <p>모든 퀘스트와 움직임 연구도감을 완성했습니다!</p>
      <div style="font-size:clamp(24px,4vw,42px);margin-top:12px">🌟 ⚽ 🌟</div>
    </div>
    """, unsafe_allow_html=True)

# --------------------------- Researcher mode + navigation ---------------------------
nav_mode, nav_spacer, nav_research, nav_quests = st.columns([1.8, 3.8, 2.2, 1.6])
with nav_mode:
    mode_label = "🧑‍🔬 연구자 모드 ON" if st.session_state.researcher_mode else "🔬 연구자 모드 OFF"
    if st.button(mode_label, key="researcher_mode_toggle", width="stretch"):
        st.session_state.researcher_mode = not st.session_state.researcher_mode
        st.rerun()
with nav_research:
    research_label = "📚 도감 숨기기" if st.session_state.research_visible else "📚 움직임 연구도감"
    if st.button(research_label, key="top_toggle_research", width="stretch"):
        st.session_state.research_visible = not st.session_state.research_visible
        st.rerun()
with nav_quests:
    quests_label = "🧪 퀘스트 숨기기" if st.session_state.quests_visible else "🧪 퀘스트"
    if st.button(quests_label, key="top_toggle_quests", type="primary", width="stretch"):
        st.session_state.quests_visible = not st.session_state.quests_visible
        st.rerun()

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
    {"id":"first_success","title":"첫 번째 착륙","desc":"공을 목표 지점에 성공적으로 착륙시켜라.","reward":1200},
    {"id":"slow","title":"거북이의 비밀","desc":"학습률을 바꾸어 느린 수렴 현상을 발견하라.","reward":1300},
    {"id":"one_step","title":"원샷 착륙","desc":"한 번의 실제 이동으로 목표에 도착하라.","reward":1500},
    {"id":"zigzag","title":"지그재그 탐정","desc":"목표를 넘나들며 수렴하는 조건을 발견하라.","reward":1500},
    {"id":"pingpong","title":"무한 핑퐁 실험","desc":"공이 같은 거리로 계속 튕기는 현상을 재현하라.","reward":1500},
    {"id":"diverge","title":"폭주 경보","desc":"공이 점점 더 멀리 발산하는 조건을 발견하라.","reward":1500},
]
RULE_REWARDS = {rule_id: 300 for rule_id in RULES}

# Always-visible progress HUD for normal play; researcher mode shows answer sheets instead.
if not st.session_state.researcher_mode:
    completed_quest_count = sum(q["id"] in st.session_state.quests_done for q in QUESTS)
    discovered_rule_count = sum(rid in st.session_state.unlocked_rules for rid in RULES)
    st.markdown(f"""
    <div class="progress-hud">
      <h3>🏆 연구소 진행 현황</h3>
      <p><b>🎯 목표 경험치: {TARGET_XP:,} XP</b> · 모든 퀘스트와 도감 보상을 수령해 목표를 달성하세요.</p>
    </div>
    """, unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    p1.metric("✨ 연구 경험치", f"{st.session_state.xp:,} / {TARGET_XP:,} XP")
    p2.metric("🏁 완료 퀘스트", f"{completed_quest_count} / {len(QUESTS)}")
    p3.metric("🔓 발견한 규칙", f"{discovered_rule_count} / {len(RULES)}")


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
        newly_unlocked.append("첫 번째 착륙 퀘스트 완료! +100 XP")

    if 0 < alpha_input < 0.5 and steps > 0:
        if "slow" not in st.session_state.unlocked_rules:
            st.session_state.unlocked_rules.append("slow")
            newly_unlocked.append("규칙 발견: 느린 수렴")
        if "slow" not in st.session_state.quests_done:
            st.session_state.quests_done.append("slow")
            newly_unlocked.append("거북이의 비밀 퀘스트 완료! +80 XP")

    if np.isclose(alpha_input, 0.5) and code == "success" and steps == 1:
        if "one_step" not in st.session_state.unlocked_rules:
            st.session_state.unlocked_rules.append("one_step")
            newly_unlocked.append("규칙 발견: 한 번의 착륙")
        if "one_step" not in st.session_state.quests_done:
            st.session_state.quests_done.append("one_step")
            newly_unlocked.append("원샷 착륙 퀘스트 완료! +100 XP")

    if 0.5 < alpha_input < 1.0 and steps >= 2:
        if "zigzag" not in st.session_state.unlocked_rules:
            st.session_state.unlocked_rules.append("zigzag")
            newly_unlocked.append("규칙 발견: 지그재그 수렴")
        if "zigzag" not in st.session_state.quests_done:
            st.session_state.quests_done.append("zigzag")
            newly_unlocked.append("지그재그 탐정 퀘스트 완료! +100 XP")

    if np.isclose(alpha_input, 1.0):
        if "pingpong" not in st.session_state.unlocked_rules:
            st.session_state.unlocked_rules.append("pingpong")
            newly_unlocked.append("규칙 발견: 무한 핑퐁")
        if "pingpong" not in st.session_state.quests_done:
            st.session_state.quests_done.append("pingpong")
            newly_unlocked.append("무한 핑퐁 퀘스트 완료! +100 XP")

    if alpha_input > 1.0:
        if "diverge" not in st.session_state.unlocked_rules:
            st.session_state.unlocked_rules.append("diverge")
            newly_unlocked.append("규칙 발견: 발산 폭주")
        if "diverge" not in st.session_state.quests_done:
            st.session_state.quests_done.append("diverge")
            newly_unlocked.append("폭주 경보 퀘스트 완료! +120 XP")

    st.session_state.quest_notice = newly_unlocked
    completed_notices = [n for n in newly_unlocked if "퀘스트 완료" in n or "마스터 달성" in n]
    for notice in completed_notices:
        st.toast(f"퀘스트 달성! {notice} — 퀘스트 버튼에서 XP를 받아 주세요.", icon="🏆")

    if code == "success":
        st.balloons()
        st.toast("착륙 성공! 100점!", icon="🎉")
    elif code in ("explode", "pingpong"):
        st.toast("실패! 결과를 분석하고 다음 실험을 해 보세요.", icon="💥")
    else:
        st.toast("실험 결과를 기록했어요.", icon="🔬")

# --------------------------- Quest dashboard ---------------------------
if st.session_state.quests_visible:
    with st.container(border=True):
        st.markdown('<div class="panel-heading"><h2>🧪 QUEST CENTER · 퀘스트 센터</h2><p>미션을 완료하고 보상을 수령하세요.</p></div>', unsafe_allow_html=True)
        st.info(f"🎯 목표 경험치: {TARGET_XP:,} XP — 일반 퀘스트와 연구도감 보상을 모두 수령하면 목표 경험치에 도달합니다.")
        if st.session_state.researcher_mode:
            st.markdown("### 🧑‍🔬 연구자 모드 · 퀘스트 정답")
            quest_answers = [
                ("첫 번째 착륙", "실험 결과가 착륙 성공이어야 합니다. 최종 위치 |x| < 1e-5."),
                ("거북이의 비밀", "학습률 0 < α < 0.5로 설정해 실험합니다."),
                ("원샷 착륙", "학습률 α = 0.5로 설정합니다. 한 번의 업데이트로 x=0에 도착합니다."),
                ("지그재그 탐정", "학습률 0.5 < α < 1로 설정합니다."),
                ("무한 핑퐁 실험", "학습률 α = 1.0으로 설정합니다. 부호는 바뀌지만 위치의 절댓값은 줄지 않습니다."),
                ("폭주 경보", "학습률 α > 1.0으로 설정합니다. 목표에서 점점 멀어집니다."),
            ]
            for n, (name, answer) in enumerate(quest_answers, 1):
                st.markdown(f"**{n}. {name}**")
                st.write(answer)
                if n < len(quest_answers):
                    st.divider()
        else:
            for quest in QUESTS:
                completed = quest["id"] in st.session_state.quests_done
                claimed = quest["id"] in st.session_state.quests_claimed
                with st.container(border=True):
                    top1, top2 = st.columns([4, 1])
                    top1.markdown(f"### {'🏆' if claimed else '✅' if completed else '🔒'} {quest['title']}")
                    top2.markdown(f"**{quest['reward']:,} XP**")
                    st.write(quest["desc"])
                    if claimed:
                        st.success("보상 수령 완료")
                    elif completed:
                        st.success("미션 성공! 보상을 수령할 수 있습니다.")
                        if st.button(f"🎁 {quest['reward']:,} XP 받기", key=f"claim_{quest['id']}", type="primary", width="stretch"):
                            st.session_state.quests_claimed.append(quest["id"])
                            st.session_state.xp += quest["reward"]
                            st.session_state.quest_firework = True
                            st.rerun()
                    else:
                        st.caption("미완료 — 실험 조건을 바꾸어 도전하세요.")

            if st.session_state.quest_firework:
                st.balloons()
                st.toast("퀘스트 보상 수령 완료!", icon="🎆")
                components.html("""
                <div style="position:relative;height:220px;overflow:hidden;border-radius:14px;background:linear-gradient(180deg,#06182c,#10425a);">
                  <canvas id="quest-fireworks" style="width:100%;height:220px;display:block;"></canvas>
                  <div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:#C7FCFF;font:900 25px system-ui;text-shadow:0 0 12px #16DDF5;">🎆 REWARD CLAIMED! 🎆</div>
                </div>
                <script>
                (() => {const c=document.getElementById('quest-fireworks'),x=c.getContext('2d'),d=window.devicePixelRatio||1,w=c.clientWidth,h=220;c.width=w*d;c.height=h*d;x.scale(d,d);const cs=['#4CF4FF','#FFD166','#FF6B9A','#B8A0FF','#FFFFFF'];let a=[];for(let b=0;b<7;b++){let cx=30+Math.random()*(w-60),cy=20+Math.random()*120;for(let i=0;i<38;i++){let t=2*Math.PI*i/38,v=1+Math.random()*3;a.push({x:cx,y:cy,vx:Math.cos(t)*v,vy:Math.sin(t)*v,l:75,col:cs[b%cs.length]})}}let f=0;function draw(){x.clearRect(0,0,w,h);for(let p of a){p.x+=p.vx;p.y+=p.vy;p.vy+=.025;p.l--;if(p.l>0){x.globalAlpha=Math.min(1,p.l/20);x.fillStyle=p.col;x.beginPath();x.arc(p.x,p.y,2,0,7);x.fill()}}x.globalAlpha=1;if(++f<100)requestAnimationFrame(draw)}draw()})();
                </script>
                """, height=230)
                st.session_state.quest_firework = False

# --------------------------- Rule encyclopedia ---------------------------
if st.session_state.research_visible:
    with st.container(border=True):
        st.markdown('<div class="panel-heading"><h2>📚 MOVEMENT CODEX · 움직임 연구도감</h2><p>현상을 발견하고 규칙별 연구 보상을 수령하세요.</p></div>', unsafe_allow_html=True)
        if st.session_state.researcher_mode:
            st.markdown("### 🧑‍🔬 연구자 모드 · 도감 정답")
            for n, (rule_id, rule) in enumerate(RULES.items(), 1):
                st.markdown(f"**{n}. {rule['title']}**")
                st.code(rule["condition"])
                st.write(rule["description"])
                if n < len(RULES):
                    st.divider()
        else:
            for rule_id, rule in RULES.items():
                discovered = rule_id in st.session_state.unlocked_rules
                claimed = rule_id in st.session_state.rules_claimed
                with st.container(border=True):
                    if discovered:
                        st.markdown(f"### {'🏆' if claimed else '🔓'} {rule['title']}")
                        st.write(rule["description"])
                        st.code(rule["condition"])
                        reward = RULE_REWARDS[rule_id]
                        if claimed:
                            st.success("연구 보상 수령 완료")
                        else:
                            st.success(f"규칙 발견! 연구 보상 {reward} XP를 받을 수 있습니다.")
                            if st.button(f"💎 연구 보상 {reward} XP 받기", key=f"claim_rule_{rule_id}", type="primary", width="stretch"):
                                st.session_state.rules_claimed.append(rule_id)
                                st.session_state.xp += reward
                                st.session_state.quest_firework = True
                                st.rerun()
                    else:
                        st.markdown("### 🔒 미확인 현상")
                        st.write("이 규칙은 아직 발견되지 않았습니다. 실험을 통해 해금하세요.")

# --------------------------- Current experiment graph/results ---------------------------
st.divider()
graph_col, score_col = st.columns([2, 1])
with graph_col:
    st.subheader("🏔️ 공의 이동 경로")
    result = st.session_state.last_result

    if result is None:
        st.info("사이드바에서 실험을 실행하면 이차함수 그래프와 자동 애니메이션이 표시됩니다.")
        st.caption("그래프의 좌표축은 고정되며, 축구공은 y = x² 곡선 위를 따라 이동합니다.")
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

        # Canvas animation: no external CDN dependency; points are drawn on the exact y=x² curve.
        animation_points = [{"x": float(x), "y": float(x) ** 2} for x in xs_anim]
        points_json = json.dumps(animation_points, ensure_ascii=False)
        components.html(
            f"""
            <div id="gd-wrap" style="width:100%;height:555px;background:#fff;border-radius:12px;overflow:hidden;position:relative;">
              <canvas id="gd-canvas" style="width:100%;height:100%;display:block;"></canvas>
              <div id="gd-status" style="position:absolute;left:12px;bottom:8px;background:rgba(255,255,255,.92);padding:5px 9px;border-radius:8px;font:13px sans-serif;color:#34445e;">⚽ 준비 중…</div>
            </div>
            <script>
            (() => {{
              const points = {points_json};
              const wrap = document.getElementById('gd-wrap');
              const canvas = document.getElementById('gd-canvas');
              const status = document.getElementById('gd-status');
              const ctx = canvas.getContext('2d');
              const xMin=-12, xMax=12, yMin=-5, yMax=150;
              let frame=0, lastTick=0, finished=false;
              const frameMs=550;
              function dims() {{
                const r=wrap.getBoundingClientRect();
                return {{w:r.width,h:r.height,left:62,right:20,top:28,bottom:48}};
              }}
              function px(x,d) {{ return d.left+(x-xMin)/(xMax-xMin)*(d.w-d.left-d.right); }}
              function py(y,d) {{ return d.top+(yMax-y)/(yMax-yMin)*(d.h-d.top-d.bottom); }}
              function resize() {{
                const dpr=window.devicePixelRatio||1, r=wrap.getBoundingClientRect();
                canvas.width=Math.max(1,Math.round(r.width*dpr));
                canvas.height=Math.max(1,Math.round(r.height*dpr));
                ctx.setTransform(dpr,0,0,dpr,0,0); draw();
              }}
              function draw() {{
                const d=dims(), w=d.w, h=d.h;
                ctx.clearRect(0,0,w,h); ctx.fillStyle='#fff'; ctx.fillRect(0,0,w,h);
                ctx.font='12px sans-serif'; ctx.textAlign='center'; ctx.textBaseline='middle';
                ctx.strokeStyle='#e5eaf0'; ctx.lineWidth=1;
                for(let y=0;y<=140;y+=20) {{
                  const yy=py(y,d); ctx.beginPath();ctx.moveTo(d.left,yy);ctx.lineTo(w-d.right,yy);ctx.stroke();
                  ctx.fillStyle='#68758a';ctx.fillText(String(y),d.left-22,yy);
                }}
                for(let x=-10;x<=10;x+=5) {{
                  const xx=px(x,d);ctx.beginPath();ctx.moveTo(xx,d.top);ctx.lineTo(xx,h-d.bottom);ctx.stroke();
                  ctx.fillStyle='#68758a';ctx.fillText(String(x),xx,h-d.bottom+17);
                }}
                ctx.strokeStyle='#475569';ctx.lineWidth=1.4;
                ctx.beginPath();ctx.moveTo(px(0,d),d.top);ctx.lineTo(px(0,d),h-d.bottom);ctx.stroke();
                ctx.beginPath();ctx.moveTo(d.left,py(0,d));ctx.lineTo(w-d.right,py(0,d));ctx.stroke();
                ctx.beginPath();
                for(let i=0;i<=500;i++) {{
                  const x=xMin+(xMax-xMin)*i/500,y=x*x;
                  if(i===0)ctx.moveTo(px(x,d),py(y,d));else ctx.lineTo(px(x,d),py(y,d));
                }}
                ctx.strokeStyle='#68758a';ctx.lineWidth=3;ctx.stroke();
                const gx=px(0,d),gy=py(0,d);ctx.beginPath();
                for(let i=0;i<10;i++) {{
                  const rr=i%2===0?10:4.5, a=-Math.PI/2+i*Math.PI/5;
                  const sx=gx+Math.cos(a)*rr,sy=gy+Math.sin(a)*rr;
                  if(i===0)ctx.moveTo(sx,sy);else ctx.lineTo(sx,sy);
                }}
                ctx.closePath();ctx.fillStyle='#f4b400';ctx.fill();ctx.strokeStyle='#8a5a00';ctx.stroke();
                const shown=points.slice(0,Math.min(frame+1,points.length));
                if(shown.length) {{
                  ctx.beginPath();
                  shown.forEach((p,i)=>{{if(i===0)ctx.moveTo(px(p.x,d),py(p.y,d));else ctx.lineTo(px(p.x,d),py(p.y,d));}});
                  ctx.strokeStyle='#d62828';ctx.lineWidth=2.5;ctx.stroke();
                  shown.forEach(p=>{{ctx.beginPath();ctx.arc(px(p.x,d),py(p.y,d),4.5,0,Math.PI*2);ctx.fillStyle='#d62828';ctx.fill();ctx.strokeStyle='#7f1d1d';ctx.lineWidth=1;ctx.stroke();}});
                  const ball=shown[shown.length-1];
                  ctx.font='26px "Apple Color Emoji","Segoe UI Emoji",sans-serif';ctx.textAlign='center';ctx.textBaseline='middle';
                  ctx.fillText('⚽',px(ball.x,d),py(ball.y,d));
                }}
                ctx.fillStyle='#34445e';ctx.font='bold 13px sans-serif';ctx.textAlign='center';
                ctx.fillText('공의 위치 x',(d.left+w-d.right)/2,h-12);
                ctx.save();ctx.translate(16,(d.top+h-d.bottom)/2);ctx.rotate(-Math.PI/2);ctx.fillText('함숫값 y = x²',0,0);ctx.restore();
                ctx.fillStyle='#172033';ctx.font='bold 14px sans-serif';ctx.textAlign='left';
                ctx.fillText('이동 단계 '+Math.min(frame+1,points.length)+' / '+points.length,d.left,14);
              }}
              function advanceOneStep() {{
                if (finished) return;
                if (frame < points.length - 1) frame += 1;
                draw();
                status.textContent = frame >= points.length - 1
                  ? '✅ 이동 애니메이션 완료'
                  : '⚽ 공이 포물선을 따라 이동 중…';
                if (frame >= points.length - 1) {{
                  finished = true;
                  if (timerHandle) window.clearInterval(timerHandle);
                }}
              }}
              let timerHandle = null;
              try {{
                resize();
                status.textContent = points.length > 1
                  ? '⚽ 공이 포물선을 따라 이동 중…'
                  : '✅ 이동 완료';
                if (points.length > 1) {{
                  timerHandle = window.setInterval(advanceOneStep, frameMs);
                }} else {{
                  finished = true;
                }}
                if (window.ResizeObserver) new ResizeObserver(resize).observe(wrap);
                else window.addEventListener('resize', resize);
              }} catch (err) {{
                status.textContent = '애니메이션 오류: ' + (err && err.message ? err.message : String(err));
              }}
            }})();
            </script>
            """,
            height=565,
            scrolling=False,
        )
        st.caption("실험 실행 시 자동 재생됩니다. 축구공은 이차함수 y=x² 위를 따라 이동하고, 방문한 위치마다 점이 남습니다. 좌표축은 고정됩니다.")

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
    st.subheader("도전할수록 점수가 어떻게 달라질까?")
    score_chart = chart_df.set_index("도전 번호")[["점수"]]
    st.line_chart(score_chart, y="점수")

st.divider()
st.caption("경사하강법 연구소 · 실험하고 발견하고 해금하세요!")
