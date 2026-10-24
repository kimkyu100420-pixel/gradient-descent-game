import html
import json
import math

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="경사하강법 시뮬레이터", page_icon="⚽", layout="wide")

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
    div[data-testid="stButton"] button,div[data-testid="stDownloadButton"] button{background:#FFFFFF!important;color:#163354!important}
    div[data-testid="stButton"] button[kind="primary"]{background:#1769AA!important;color:#FFFFFF!important}
    div[data-testid="stButton"] button[kind="primary"] p{background:transparent!important;color:#FFFFFF!important}
    div[data-testid="stSelectbox"] [data-baseweb="select"] *,div[data-testid="stRadio"] label,div[data-testid="stCheckbox"] label{color:#163354!important}
    div[data-testid="stSlider"] [data-testid="stMarkdownContainer"] p{color:#163354!important}
    div[data-testid="stButton"] button[kind="primary"]{background:linear-gradient(135deg,#1769AA,#2946A0)!important;color:#FFFFFF!important;border:1px solid #164E83!important}
    div[data-testid="stButton"] button[kind="primary"] p{color:#FFFFFF!important}
    div[data-testid="stButton"] button,div[data-testid="stDownloadButton"] button{
      color:#102A43!important;-webkit-text-fill-color:#102A43!important;
      background:#F8FAFC!important;opacity:1!important;text-shadow:none!important;
    }
    div[data-testid="stButton"] button *,div[data-testid="stDownloadButton"] button *{
      color:inherit!important;-webkit-text-fill-color:inherit!important;
      opacity:1!important;visibility:visible!important;text-shadow:none!important;
    }
    div[data-testid="stButton"] button[kind="primary"]{
      background:linear-gradient(135deg,#1769AA,#2946A0)!important;
      color:#FFFFFF!important;-webkit-text-fill-color:#FFFFFF!important;
    }
    div[data-testid="stButton"] button[kind="primary"] *{
      color:#FFFFFF!important;-webkit-text-fill-color:#FFFFFF!important;
    }
    div[data-testid="stDialog"] div[data-testid="stButton"] button{
      background:#E7F8FC!important;color:#082B43!important;
      -webkit-text-fill-color:#082B43!important;border:1px solid #59DDEB!important;
    }
    div[data-testid="stDialog"] div[data-testid="stButton"] button *{
      color:#082B43!important;-webkit-text-fill-color:#082B43!important;
    }
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
    .progress-hud{background:#FFFFFF;border:1px solid #D7E0EA;border-radius:16px;padding:18px 22px;box-shadow:0 4px 14px rgba(20,43,74,.06);margin:12px 0 20px;color:#243247}
    .progress-hud h3{color:#17395E!important;margin:0 0 8px!important}
    .progress-hud p{color:#40536B!important;margin:0}

    [data-testid="stDialog"]>div{width:70vw!important;max-width:1100px!important;min-width:min(70vw,320px)!important}
    [data-testid="stDialog"] [data-testid="stDialogContent"]{max-height:78vh!important;overflow-y:auto!important;background:linear-gradient(145deg,#062238,#0B354D 55%,#09263D)!important;border:2px solid #4CF4FF!important;border-radius:18px!important;box-shadow:0 0 28px rgba(36,226,255,.28)!important}
    [data-testid="stDialog"] [data-testid="stMarkdownContainer"] p,[data-testid="stDialog"] label,[data-testid="stDialog"] [data-testid="stCaptionContainer"]{color:#D9F5FB!important}
    [data-testid="stDialog"] h1,[data-testid="stDialog"] h2,[data-testid="stDialog"] h3{color:#E8FDFF!important}
    [data-testid="stDialog"] [data-testid="stVerticalBlockBorderWrapper"]{background:linear-gradient(145deg,#0A2B41,#103B52)!important;border:1px solid #2BAFC6!important;border-radius:12px!important}
    [data-testid="stDialog"] [data-testid="stAlert"]{font-size:1.08rem!important}
    [data-testid="stDialog"] [data-testid="stAlert"]{
      background:#EAF7FF!important;border:2px solid #4CA8D8!important;
      border-radius:12px!important;padding:12px 16px!important;
    }
    [data-testid="stDialog"] [data-testid="stAlert"] *{
      color:#102A43!important;-webkit-text-fill-color:#102A43!important;
      font-weight:800!important;opacity:1!important;
    }
    [data-testid="stDialog"] [data-testid="stButton"] button{
      background:#E7F8FC!important;color:#082B43!important;
      -webkit-text-fill-color:#082B43!important;border:2px solid #58DCEB!important;
      box-shadow:0 2px 8px rgba(0,0,0,.18)!important;
    }
    [data-testid="stDialog"] [data-testid="stButton"] button *,
    [data-testid="stDialog"] [data-testid="stButton"] button p{
      color:#082B43!important;-webkit-text-fill-color:#082B43!important;
      opacity:1!important;visibility:visible!important;
    }
    [data-testid="stDialog"] [data-testid="stButton"] button[kind="primary"]{
      background:linear-gradient(135deg,#A7F3D0,#6EE7F9)!important;
      color:#073344!important;-webkit-text-fill-color:#073344!important;
    }
    [data-testid="stDialog"] [data-testid="stButton"] button[kind="primary"] *,
    [data-testid="stDialog"] [data-testid="stButton"] button[kind="primary"] p{
      color:#073344!important;-webkit-text-fill-color:#073344!important;
    }
    [data-testid="stDialog"] [data-testid="stButton"] button{font-size:1.05rem!important}
    @media(max-width:700px){[data-testid="stDialog"]>div{width:92vw!important;min-width:92vw!important}[data-testid="stDialog"] [data-testid="stDialogContent"]{max-height:82vh!important}}

    .progress-hud p{color:#40536B!important;margin:0}
    .master-banner{position:relative;overflow:hidden;text-align:center;padding:clamp(24px,5vw,48px);margin:18px 0 24px;border-radius:22px;border:1px solid #C8D8E8;background:linear-gradient(135deg,#FFFFFF,#EDF4FA);box-shadow:0 10px 28px rgba(20,43,74,.09)}
    .master-banner h1{font-size:clamp(36px,6vw,68px)!important;line-height:1.12;margin:0;color:#17395E!important;text-shadow:none!important}
    .master-banner p{font-size:clamp(16px,2.2vw,22px);font-weight:750;color:#40536B!important;margin:14px 0 0;text-shadow:none}
    .gd-overlay{pointer-events:auto!important;cursor:pointer!important}
    .gd-overlay::after{content:"화면을 터치하면 닫힙니다";position:absolute;bottom:7vh;left:0;width:100%;text-align:center;color:#fff;font:700 16px system-ui;text-shadow:0 0 10px #00dfff}
    @keyframes masterGlow{from{opacity:1}to{opacity:1}}
    @keyframes overlayIn{0%{opacity:0}8%,78%{opacity:1}100%{opacity:0}}
    @keyframes gdSuccessFlash{0%,100%{background-color:rgba(20,90,170,.2);filter:brightness(1)}12%{background-color:rgba(255,255,255,.98);filter:brightness(2.2)}24%{background-color:rgba(255,205,65,.82);filter:brightness(1.5)}38%{background-color:rgba(90,220,255,.75);filter:brightness(1.7)}52%{background-color:rgba(255,90,170,.72);filter:brightness(1.4)}68%{background-color:rgba(255,255,255,.8);filter:brightness(1.8)}}
    @keyframes gdScreenShake{0%,100%{transform:translate(0,0) rotate(0)}10%{transform:translate(-9px,4px) rotate(-.4deg)}20%{transform:translate(8px,-5px) rotate(.4deg)}30%{transform:translate(-7px,-4px)}40%{transform:translate(7px,5px)}50%{transform:translate(-5px,3px)}60%{transform:translate(5px,-3px)}70%{transform:translate(-3px,2px)}80%{transform:translate(3px,-2px)}}
    @keyframes cardIn{0%{transform:scale(.65) rotate(-4deg)}12%{transform:scale(1.04) rotate(1deg)}20%,78%{transform:scale(1) rotate(0)}100%{transform:scale(.96)}}
    @keyframes particleBurst{0%{transform:translate(-50%,-50%) scale(.35) rotate(0);opacity:0}12%{opacity:1}100%{transform:translate(calc(-50% + var(--dx)),calc(-50% + var(--dy))) scale(1.2) rotate(360deg);opacity:0}}
    @keyframes turtleTraverse{0%{transform:translateX(-45vw) rotate(-8deg);opacity:0}12%{opacity:1}35%{transform:translateX(-10vw) rotate(7deg)}60%{transform:translateX(15vw) rotate(-5deg)}88%{opacity:1}100%{transform:translateX(45vw);opacity:0}}
    @keyframes confettiLeft{0%{transform:translate(0,0) rotate(0) scale(.6);opacity:0}12%{opacity:1}100%{transform:translate(var(--dx),var(--dy)) rotate(720deg) scale(1.15);opacity:0}}
    @keyframes confettiRight{0%{transform:translate(0,0) rotate(0) scale(.6);opacity:0}12%{opacity:1}100%{transform:translate(var(--dx),var(--dy)) rotate(-720deg) scale(1.15);opacity:0}}
    .gd-overlay{position:fixed!important;inset:0!important;width:100vw!important;height:100vh!important;min-height:100dvh!important;z-index:2147483000!important;display:flex!important;align-items:center!important;justify-content:center!important;overflow:hidden!important;pointer-events:auto!important;cursor:pointer!important;animation:overlayIn 2.5s ease-in-out both;animation-delay:var(--effect-delay,0s);background:radial-gradient(ellipse at center,rgba(135,12,0,.70),rgba(35,0,0,.82))}.gd-overlay.gd-success{background:radial-gradient(circle at center,rgba(20,90,170,.78),rgba(22,12,65,.94))!important;animation:overlayIn 2s ease-in-out both,gdSuccessFlash 2s ease-out both!important;animation-delay:var(--effect-delay,0s),var(--effect-delay,0s)!important}.gd-overlay.gd-turtle{background:radial-gradient(circle at center,rgba(13,95,53,.92),rgba(3,35,27,.97))!important;animation:overlayIn 2s ease-in-out both!important;animation-delay:var(--effect-delay,0s)!important}.gd-overlay[style*="opacity: 0"]{pointer-events:auto}
    .gd-overlay.gd-success .gd-card{border:3px solid #FFE680;background:linear-gradient(145deg,rgba(8,36,93,.96),rgba(48,20,106,.96),rgba(115,20,91,.94));box-shadow:0 0 55px rgba(255,255,255,.55),0 0 95px rgba(76,220,255,.65);animation:cardIn 2s ease-out both,gdScreenShake .3s linear 4;animation-delay:var(--effect-delay,0s),calc(var(--effect-delay,0s) + .25s)}
    .gd-overlay.gd-failure{animation:overlayIn 2s ease-in-out both!important}
    .gd-overlay.gd-failure .gd-card{animation:cardIn 2s ease-out both,gdScreenShake .12s linear 12;animation-delay:var(--effect-delay,0s),var(--effect-delay,0s)}
    .gd-overlay.gd-turtle .gd-card{border-color:#8FFFC0!important;background:linear-gradient(145deg,rgba(3,56,38,.97),rgba(8,88,56,.96),rgba(4,35,30,.97))!important;box-shadow:0 0 50px rgba(65,255,153,.55)!important;animation:cardIn 2s ease-out both,gdScreenShake .35s ease-in-out 2;animation-delay:var(--effect-delay,0s),calc(var(--effect-delay,0s) + .4s)}
    .gd-card{position:relative;overflow:hidden;width:min(88vw,760px);max-height:82vh;padding:clamp(20px,4vw,42px);text-align:center;color:#fff!important;border:3px solid #ff7658;border-radius:26px;background:linear-gradient(145deg,#310b12,#151523 58%,#4a1009);box-shadow:0 0 45px rgba(255,67,20,.65);animation:cardIn 2.5s ease-out both;animation-delay:var(--effect-delay,0s)}
    .gd-card *{color:#fff!important}.gd-kicker{font-size:clamp(22px,4vw,42px);font-weight:1000;color:#ffb0a0!important;margin-bottom:12px}
    .gd-emoji{font-size:clamp(58px,11vw,112px);line-height:1.2;margin:8px 0}.gd-head{font-size:clamp(19px,3.2vw,30px);font-weight:900;line-height:1.3;margin:10px 0}.gd-copy{font-size:clamp(15px,2vw,19px);line-height:1.5;color:#ffe1d6!important}
    .gd-particle{position:absolute;left:50%;top:50%;font-size:clamp(28px,4.2vw,52px);animation:particleBurst 2.15s cubic-bezier(.15,.65,.2,1) both;animation-delay:var(--particle-delay,0s)}
    .gd-bomb-chaos{animation:bombShake .16s linear infinite alternate;filter:drop-shadow(0 0 16px rgba(255,80,20,.75))}
    .gd-overlay.gd-failure{
      overflow:hidden!important;
      background:rgba(15,3,3,.93)!important;
      /* overlayIn controls opacity; gdFailureFlash controls the flash colors.
         Keeping both animations is essential because the HTML starts at opacity:0. */
      animation:overlayIn 3.2s ease-in-out both,gdFailureFlash 3.2s ease-out both!important;
      animation-delay:var(--effect-delay,0s),var(--effect-delay,0s)!important;
      isolation:isolate;
    }
    .gd-overlay.gd-failure::before{
      content:"";position:absolute;inset:-10%;pointer-events:none;z-index:0;
      background:radial-gradient(circle at 50% 48%,rgba(255,255,255,.98) 0%,rgba(255,230,170,.9) 5%,rgba(255,68,0,.52) 18%,rgba(100,0,0,.2) 42%,transparent 68%);
      animation:gdFlashBurst .72s cubic-bezier(.1,.8,.2,1) both;
    }
    .gd-overlay.gd-failure::after{
      content:"";position:absolute;inset:0;pointer-events:none;z-index:0;
      background:repeating-linear-gradient(0deg,rgba(255,255,255,.04) 0 2px,transparent 2px 5px);
      animation:gdWarningFlicker .18s steps(2,end) 7;
    }
    .gd-overlay.gd-failure .gd-card{
      position:relative;z-index:2;
      animation:gdCardImpact 2.8s cubic-bezier(.2,.8,.2,1) both;
      box-shadow:0 0 0 2px rgba(255,90,30,.6),0 0 50px rgba(255,55,0,.7),0 0 120px rgba(255,0,0,.35)!important;
    }
    .gd-overlay.gd-failure .gd-particle{z-index:1;font-size:clamp(34px,5vw,64px);text-shadow:0 0 18px rgba(255,95,0,.95)}
    @keyframes gdFailureFlash{
      0%{background:#fff!important;filter:brightness(3)}
      5%{background:rgba(255,235,210,.98)!important;filter:brightness(2.2)}
      12%{background:rgba(255,40,0,.92)!important;filter:brightness(1.8)}
      19%{background:rgba(25,0,0,.96)!important;filter:brightness(.8)}
      28%{background:rgba(255,75,0,.72)!important;filter:brightness(1.4)}
      38%,100%{background:rgba(15,3,3,.93)!important;filter:brightness(1)}
    }
    @keyframes gdFlashBurst{
      0%{opacity:1;transform:scale(.05)}
      18%{opacity:1;transform:scale(1.25)}
      42%{opacity:.72;transform:scale(1.7)}
      100%{opacity:0;transform:scale(2.1)}
    }
    @keyframes gdWarningFlicker{
      0%,100%{opacity:.1}
      50%{opacity:.8;background-color:rgba(255,0,0,.3)}
    }
    @keyframes gdCardImpact{
      0%{opacity:0;transform:translate3d(0,0,0) scale(.5) rotate(-4deg)}
      8%{opacity:1;transform:translate3d(-12px,5px,0) scale(1.04) rotate(2deg)}
      14%{transform:translate3d(13px,-7px,0) scale(1.02) rotate(-2deg)}
      20%{transform:translate3d(-10px,6px,0) scale(1) rotate(1deg)}
      26%{transform:translate3d(8px,-4px,0) scale(1) rotate(-1deg)}
      34%,100%{opacity:1;transform:translate3d(0,0,0) scale(1) rotate(0)}
    }

    @keyframes bombShake{from{transform:translate(-5px,-3px) rotate(-4deg) scale(1)}to{transform:translate(5px,3px) rotate(4deg) scale(1.08)}}
    .gd-success{background:radial-gradient(ellipse at center,rgba(16,95,64,.72),rgba(7,38,47,.84));animation-duration:2.8s}
    .gd-success-card{border-color:#f9dc75;background:linear-gradient(145deg,#064e3b,#123b5b 55%,#51256c);box-shadow:0 0 42px rgba(255,218,90,.5);animation:successCard 2.8s ease-out both;animation-delay:var(--effect-delay,0s)}
    @keyframes successCard{0%{transform:scale(.55) rotate(-5deg);opacity:0}18%{transform:scale(1.08) rotate(2deg);opacity:1}30%,82%{transform:scale(1);opacity:1}100%{transform:scale(.97);opacity:0}}
    .gd-success-particle{position:absolute;top:50%;font-size:clamp(24px,3.8vw,42px);opacity:0}
    .gd-success-particle.left{left:8%;animation:confettiLeft 2.35s ease-out both;animation-delay:var(--effect-delay,0s)}
    .gd-success-particle.right{right:8%;animation:confettiRight 2.35s ease-out both;animation-delay:var(--effect-delay,0s)}
    .gd-turtle{background:radial-gradient(ellipse at center,rgba(35,125,48,.88),rgba(4,35,18,.96));animation-duration:5.8s!important}
    .gd-turtle::before{content:"";position:absolute;inset:0;pointer-events:none;background:repeating-linear-gradient(135deg,transparent 0 42px,rgba(190,255,120,.09) 43px 45px);animation:turtleMeadow 5.8s linear both}
    .gd-turtle-card{z-index:2;background:linear-gradient(145deg,#123b28,#214b25 55%,#6a5520);border-color:#b4e57b;box-shadow:0 0 40px rgba(150,220,80,.45),0 0 100px rgba(93,220,65,.22);animation:turtleCardBounce 5.8s ease-in-out both!important;animation-delay:var(--effect-delay,0s)}
    .gd-turtle-card .gd-emoji{display:block;animation:turtleTraverse 5.2s ease-in-out both!important;animation-delay:var(--effect-delay,0s)}
    .gd-turtle-particle{position:absolute;left:50%;top:50%;font-size:clamp(26px,4vw,48px);animation:turtleParticleFloat 4.8s ease-in-out both;animation-delay:calc(var(--particle-delay,0s) + var(--effect-delay,0s))}
    @keyframes turtleCardBounce{0%{transform:scale(.62) rotate(-4deg);opacity:0}10%{transform:scale(1.04) rotate(2deg);opacity:1}18%,82%{transform:scale(1) rotate(0);opacity:1}92%{transform:scale(1.02) rotate(-1deg);opacity:1}100%{transform:scale(.96);opacity:0}}
    @keyframes turtleParticleFloat{0%{transform:translate(-50%,-30%) scale(.5) rotate(-10deg);opacity:0}15%{opacity:1}50%{transform:translate(calc(-50% + var(--dx)),calc(-50% + var(--dy))) scale(1.12) rotate(12deg);opacity:1}100%{transform:translate(calc(-50% + var(--dx)),calc(-50% + var(--dy) - 18px)) scale(.85) rotate(-8deg);opacity:0}}
    @keyframes turtleMeadow{0%{transform:translateX(-6%);opacity:.1}30%{opacity:1}100%{transform:translateX(6%);opacity:.35}}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero"><h1>⚽ 경사하강법 시뮬레이터</h1><p>직접 실험해 규칙을 발견하고, 퀘스트를 깨며 연구 도감을 완성하세요.</p></div>',
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
    "master_popup_pending": False,
    "master_celebration_pending": False,
    "unlocked_rules": [],
    "xp": 0,
    "quest_notice": None,
    "quests_visible": False,
    "research_visible": False,
    "animation_token": 0,
    "researcher_mode": False,
    "intro_popup_shown": False,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Migrate sessions from the older research-rule reward system to quest-only XP.
if "rules_claimed" in st.session_state:
    st.session_state.rules_claimed = []

# 연구소 마스터: 별도의 퀘스트가 아니라 모든 일반 퀘스트와 도감 규칙 완성 시 표시
REQUIRED_QUEST_IDS = ["first_success", "slow", "one_step", "zigzag", "pingpong", "diverge"]
ALL_RULE_IDS = ["slow", "one_step", "zigzag", "pingpong", "diverge"]
is_research_lab_master = (
    all(qid in st.session_state.quests_done for qid in REQUIRED_QUEST_IDS)
    and all(qid in st.session_state.quests_claimed for qid in REQUIRED_QUEST_IDS)
)
if is_research_lab_master:
    st.markdown("""
    <div class="master-banner">
      <div style="display:flex;justify-content:center;align-items:center;gap:16px;margin:0 auto 18px;flex-wrap:wrap">
        <span style="font-size:clamp(30px,5vw,48px)">📈</span>
        <span style="width:clamp(78px,12vw,112px);height:clamp(78px,12vw,112px);display:flex;align-items:center;justify-content:center;border-radius:50%;background:linear-gradient(145deg,#DCEBFA,#FFFFFF);border:1px solid #B7CDE4;font-size:clamp(44px,7vw,68px);box-shadow:inset 0 2px 10px #FFFFFF,0 8px 18px rgba(20,43,74,.12)">🏆</span>
        <span style="font-size:clamp(30px,5vw,48px)">⚽</span>
      </div>
      <h1>경사하강법 마스터</h1>
      <p>여섯 개의 퀘스트를 모두 완료하고, 움직임 연구를 마쳤습니다.</p>
      <div style="margin-top:18px;display:inline-block;background:#E2F1E8;color:#216343;padding:8px 16px;border-radius:999px;font-weight:800">ALL QUESTS COMPLETE · 10,000 XP</div>
    </div>
    """, unsafe_allow_html=True)

# --------------------------- Researcher mode + navigation ---------------------------
nav_mode, nav_spacer, nav_quests = st.columns([1.8, 5.8, 2.0])
with nav_mode:
    mode_label = "🧑‍🔬 연구자 모드 ON" if st.session_state.researcher_mode else "🔬 연구자 모드 OFF"
    if st.button(mode_label, key="researcher_mode_toggle", width="stretch"):
        st.session_state.researcher_mode = not st.session_state.researcher_mode
        st.rerun()
with nav_quests:
    if st.button("🧪 퀘스트", key="top_toggle_quests", type="primary", width="stretch"):
        st.session_state.quests_visible = True
        st.session_state.research_visible = False
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



def get_experiment_notification():
    """Return the separate white top-left notification after the full-screen effect closes."""
    notices = st.session_state.get("quest_notice", []) or []
    if globals().get("is_research_lab_master", False):
        return "경사하강법 마스터가 되었습니다!"
    if any("퀘스트 완료" in item or "퀘스트" in item and "완료" in item for item in notices):
        return "퀘스트 완료! 퀘스트 센터에서 보상을 수령하세요."
    if notices:
        return "연구 기록이 저장되었습니다."
    return "기록되었습니다."


def show_failure_effect(alpha, attempt_id, delay_seconds=0):
    """Full-screen failure effect that describes the observed situation without diagnosing the exact cause."""
    if alpha > 1.0:
        headline = "💣 공이 크게 튕겨 나갔어요!"
        message = "공이 목표 지점을 지나 멀리 벗어나면서 움직임이 점점 커지는 상황이에요. 이동 경로를 살펴보고 설정을 바꿔 다시 관찰해 보세요."
    elif np.isclose(alpha, 1.0, atol=1e-8):
        headline = "💥 공이 양쪽을 오가고 있어요!"
        message = "공이 골짜기 양쪽을 번갈아 이동하고 있어요. 움직임이 줄어들지 않아 목표 지점에 머무르지 못한 상황이에요."
    elif 0.5 < alpha < 1.0:
        headline = "🔁 목표를 지나쳐 움직이고 있어요!"
        message = "공이 목표 지점을 넘어 반대편으로 이동하고 있어요. 좌우로 오가는 움직임이 나타나며, 이번 실험에서는 착지 조건을 만족하지 못했어요."
    elif 0 < alpha < 0.5:
        headline = "🐢 아직 목표에 도착하지 못했어요!"
        message = "공이 한쪽 방향에서 목표에 조금씩 가까워지고 있어요. 이번 실험에서는 정해진 이동이 끝날 때까지 착지 조건을 만족하지 못했어요."
    else:
        headline = "🧭 이번 실험은 착지하지 못했어요"
        message = "공이 이번 실험의 목표 조건에 도착하지 못했어요. 이동 경로와 실험 설정을 바꿔 다시 관찰해 보세요."

    particles = [
        ("💣","-42vw","-28vh"),("💥","38vw","-30vh"),("☄️","-43vw","18vh"),
        ("🧨","42vw","20vh"),("⚡","-18vw","34vh"),("🔥","20vw","-37vh"),
        ("🚨","-30vw","-8vh"),("🪨","14vw","-33vh"),("🌋","-30vw","25vh"),
        ("💣","-18vw","-22vh"),("💥","20vw","-12vh"),("🧨","-8vw","28vh"),
        ("☠️","8vw","-30vh"),("⚠️","32vw","12vh"),("🔥","-38vw","-20vh"),
        ("💥","4vw","38vh"),
    ]
    p_html = "".join(
        f'<span class="gd-particle" style="--dx:{dx};--dy:{dy};--particle-delay:{i*.045:.2f}s">{emoji}</span>'
        for i, (emoji, dx, dy) in enumerate(particles)
    )
    st.markdown(
        f"""<div class="gd-overlay gd-failure" id="failure-{attempt_id}" style="--effect-delay:{delay_seconds:.2f}s;opacity:0">
        <div class="gd-card">{p_html}
        <div class="gd-kicker">⚠️ GRADIENT DESCENT EXPERIMENT ⚠️</div>
        <div class="gd-emoji gd-bomb-chaos">💣 💥 🧨 🔥</div>
        <div class="gd-head">{html.escape(headline)}</div>
        <div class="gd-copy">{html.escape(message)}</div>
        <div class="gd-copy" style="margin-top:12px;font-weight:800">실험 상황 · 학습률 α={alpha:g}</div>
        </div></div>""",
        unsafe_allow_html=True,
    )
    install_effect_dismissal(f"failure-{attempt_id}", delay_seconds, 2.0, get_experiment_notification())



def show_success_effect(attempt_id, delay_seconds=0):
    bits = [
        ("🎆","left","-5vw","-35vh"),("🎇","left","18vw","-28vh"),
        ("🎉","left","32vw","20vh"),("✨","left","15vw","34vh"),
        ("🎊","left","38vw","-5vh"),("⭐","left","28vw","-18vh"),
        ("🎆","right","5vw","-35vh"),("🎇","right","-18vw","-28vh"),
        ("🎉","right","-32vw","20vh"),("✨","right","-15vw","34vh"),
        ("🎊","right","-38vw","-5vh"),("⭐","right","-28vw","-18vh"),
        ("🪅","left","24vw","8vh"),("🥳","right","-24vw","8vh"),
        ("💫","left","-38vw","-5vh"),("🌟","right","36vw","-22vh"),
        ("🎈","left","-12vw","-38vh"),("🎊","right","12vw","36vh"),
        ("🎇","left","-40vw","26vh"),("✨","right","40vw","-2vh"),
    ]
    p_html = "".join(
        f'<span class="gd-success-particle {side}" style="--dx:{dx};--dy:{dy};animation-delay:{i*.035:.2f}s">{emoji}</span>'
        for i,(emoji,side,dx,dy) in enumerate(bits)
    )
    st.markdown(
        f"""<div class="gd-overlay gd-success" id="success-{attempt_id}" style="--effect-delay:{delay_seconds:.2f}s;opacity:0">
        <div class="gd-card gd-success-card">{p_html}
        <div class="gd-kicker">🎉 LANDING SUCCESS! 🎉</div>
        <div class="gd-emoji">🏆 ⚽ 🥳</div>
        <div class="gd-head">🎉 한 번에 최솟값 도착! 성공!</div>
        <div class="gd-copy">목표 지점에 정확히 도착했어요!<br>α=0.5이면 x_next = x − 0.5·2x = 0이므로 한 번의 업데이트로 최솟값에 도착합니다.<br>성공 점수 100점!</div>
        </div></div>""",
        unsafe_allow_html=True,
    )
    install_effect_dismissal(f"success-{attempt_id}", delay_seconds, 2.0, get_experiment_notification())


def show_turtle_effect(attempt_id, delay_seconds=0):
    turtle_bits = [
        ("🐢", "-42vw", "-22vh"), ("🐢", "38vw", "-28vh"),
        ("🐌", "-40vw", "24vh"), ("💨", "40vw", "18vh"),
        ("🌿", "-24vw", "-34vh"), ("✨", "27vw", "32vh"),
        ("🐢", "0vw", "-38vh"), ("💚", "-5vw", "35vh"),
        ("🐌", "43vw", "-4vh"), ("💨", "-43vw", "-4vh"),
    ]
    p_html = "".join(
        f'<span class="gd-turtle-particle" style="--dx:{dx};--dy:{dy};--particle-delay:{i*.08:.2f}s">{emoji}</span>'
        for i, (emoji, dx, dy) in enumerate(turtle_bits)
    )
    st.markdown(
        f"""<div class="gd-overlay gd-turtle" id="turtle-{attempt_id}" style="--effect-delay:{delay_seconds:.2f}s;opacity:0">
        <div class="gd-card gd-turtle-card">{p_html}
        <div class="gd-kicker">🐢 TURTLE MODE ACTIVATED 🐢</div>
        <div class="gd-emoji">🐢 🐌 💨</div>
        <div class="gd-head">느림보 경사하강법 출동!</div>
        <div class="gd-copy">목표까지 천천히... 한 걸음씩...<br>거북이 군단 출동! 느려도 꾸준히 수렴하는 중! 🌿</div>
        </div></div>""",
        unsafe_allow_html=True,
    )
    install_effect_dismissal(f"turtle-{attempt_id}", delay_seconds, 2.0, get_experiment_notification())


def install_effect_dismissal(effect_id, delay_seconds=0, duration_seconds=2.8, toast_text="기록되었습니다."):
    """Dismiss the full-screen effect, then show a separate white Streamlit-style toast at top-left."""
    components.html(
        f"""
        <script>
        (() => {{
          const effectId = {json.dumps(effect_id)};
          const toastText = {json.dumps(toast_text, ensure_ascii=False)};
          const delayMs = Math.max(0, {float(delay_seconds)} * 1000);
          const durationMs = Math.max(500, {float(duration_seconds)} * 1000);
          let observer = null, attached = false, finished = false, autoTimer = null;

          function showNotification() {{
            if (finished) return;
            finished = true;
            try {{
              const doc = window.parent.document;
              const old = doc.getElementById('gd-native-notification');
              if (old) old.remove();
              const toast = doc.createElement('div');
              toast.id = 'gd-native-notification';
              toast.setAttribute('role', 'status');
              toast.style.cssText = [
                'position:fixed','top:18px','left:18px','right:auto','z-index:2147483647',
                'width:max-content','max-width:min(440px,calc(100vw - 36px))',
                'box-sizing:border-box','display:flex','align-items:flex-start','gap:12px',
                'padding:15px 18px','background:#FFFFFF','color:#202124',
                'border:1px solid #DADCE0','border-radius:12px',
                'box-shadow:0 4px 18px rgba(60,64,67,.24)',
                'font:500 15px/1.45 system-ui,-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif',
                'white-space:normal','overflow-wrap:anywhere','pointer-events:none',
                'opacity:0','transform:translateY(-8px)',
                'transition:opacity .18s ease,transform .18s ease'
              ].join(';');
              const icon = doc.createElement('span');
              icon.textContent = '✓';
              icon.style.cssText = 'flex:0 0 24px;width:24px;height:24px;display:flex;align-items:center;justify-content:center;border-radius:50%;background:#E6F4EA;color:#137333;font-weight:800;font-size:15px';
              const message = doc.createElement('span');
              message.textContent = toastText;
              message.style.cssText = 'color:#202124;font-weight:600';
              const close = doc.createElement('button');
              close.type = 'button';
              close.textContent = '×';
              close.setAttribute('aria-label','알림 닫기');
              close.style.cssText = 'flex:0 0 auto;border:0;background:transparent;color:#5F6368;font-size:22px;line-height:1;padding:0 0 0 4px;cursor:pointer';
              close.addEventListener('click', () => toast.remove());
              toast.append(icon,message,close);
              doc.body.appendChild(toast);
              requestAnimationFrame(() => {{
                toast.style.opacity = '1';
                toast.style.transform = 'translateY(0)';
              }});
              window.setTimeout(() => {{
                if (!toast.isConnected) return;
                toast.style.opacity = '0';
                toast.style.transform = 'translateY(-8px)';
                window.setTimeout(() => toast.remove(), 220);
              }}, 5200);
            }} catch (e) {{
              // Do not let notification failure interrupt the experiment UI.
            }}
          }}

          function attach() {{
            let doc;
            try {{ doc = window.parent.document; }} catch (e) {{ return false; }}
            const el = doc.getElementById(effectId);
            if (!el) return false;
            if (el.dataset.gdDismissReady === '1') return true;
            el.dataset.gdDismissReady = '1';
            attached = true;
            el.style.pointerEvents = 'auto';
            const removeEffect = (ev) => {{
              if (finished) return;
              if (ev) {{
                ev.preventDefault();
                ev.stopPropagation();
                ev.stopImmediatePropagation();
              }}
              if (autoTimer) window.clearTimeout(autoTimer);
              const current = doc.getElementById(effectId);
              if (current) current.remove();
              if (observer) observer.disconnect();
              showNotification();
            }};
            el.addEventListener('pointerdown', removeEffect, true);
            el.addEventListener('click', removeEffect, true);
            // Fallback: the effect auto-closes after its full animation if not tapped.
            autoTimer = window.setTimeout(() => {{
              const current = doc.getElementById(effectId);
              if (current) current.remove();
              if (observer) observer.disconnect();
              showNotification();
            }}, delayMs + durationMs + 100);
            return true;
          }}

          try {{
            if (attach()) return;
            const doc = window.parent.document;
            observer = new MutationObserver(() => {{
              if (attach() && observer) observer.disconnect();
            }});
            observer.observe(doc.body, {{childList:true, subtree:true}});
            window.setTimeout(() => {{
              if (observer) observer.disconnect();
              // If Streamlit changes the DOM before attaching, still report the saved result.
              if (!attached) showNotification();
            }}, delayMs + durationMs + 4000);
          }} catch (e) {{
            window.setTimeout(showNotification, delayMs + durationMs);
          }}
        }})();
        </script>
        """,
        height=0,
    )



# --------------------------- First-time research briefing ---------------------------
if not st.session_state.intro_popup_shown:
    @st.dialog("🔬 연구소 미션 브리핑", width="large")
    def show_intro_briefing():
        st.markdown("""
        <div style="text-align:center;padding:12px 6px 18px">
          <div style="font-size:clamp(44px,8vw,76px)">⚽ 🔎 📈</div>
          <h2 style="color:#17395E!important">한 번의 이동으로 최솟값에 도착하는 학습률을 찾아라!</h2>
          <p style="font-size:17px;color:#40536B!important">학습률을 바꿔가며 공의 움직임을 관찰하고, 가장 알맞은 값을 직접 찾아보세요.</p>
          <hr style="border:0;border-top:1px solid #D7E0EA;margin:20px 0">
          <div style="font-size:clamp(34px,6vw,54px)">🧩 🧪 🏆</div>
          <h3 style="color:#17395E!important">여러 가지 규칙을 발견하고 경사하강법 마스터가 되어라!</h3>
          <p style="font-size:16px;color:#40536B!important">여섯 개의 퀘스트를 완료하고, 실험에서 발견한 규칙을 모아 연구소를 정복하세요.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("미션 시작하기 🚀", type="primary", use_container_width=True):
            st.session_state.intro_popup_shown = True
            st.rerun()
    show_intro_briefing()

# --------------------------- Sidebar ---------------------------
st.sidebar.header("🎮 연구소 컨트롤러")
nickname_input = st.sidebar.text_input("연구원 닉네임", value="익명", max_chars=20)
st.sidebar.metric("📍 시작 위치", "8.0 (고정)")
def close_quest_on_parameter_change():
    st.session_state.quests_visible = False

alpha_input = st.sidebar.slider(
    "학습률 α", 0.05, 1.50, 0.80, 0.05, format="%.2f",
    key="alpha_control", on_change=close_quest_on_parameter_change
)
epochs_input = st.sidebar.slider(
    "최대 이동 횟수", 1, 10, 5, 1,
    key="epochs_control", on_change=close_quest_on_parameter_change
)
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
    {"id":"first_success","title":"첫 번째 착륙","desc":"공을 목표 지점에 성공적으로 착륙시켜라.","reward":1000},
    {"id":"slow","title":"거북이의 비밀","desc":"학습률을 바꾸어 느린 수렴 현상을 발견하라.","reward":1400},
    {"id":"one_step","title":"원샷 착륙","desc":"한 번의 실제 이동으로 목표에 도착하라.","reward":1700},
    {"id":"zigzag","title":"지그재그 탐정","desc":"목표를 넘나들며 수렴하는 조건을 발견하라.","reward":1800},
    {"id":"pingpong","title":"무한 핑퐁 실험","desc":"공이 같은 거리로 계속 튕기는 현상을 재현하라.","reward":1900},
    {"id":"diverge","title":"폭주 경보","desc":"공이 점점 더 멀리 발산하는 조건을 발견하라.","reward":2200},
]
QUEST_XP_TOTAL = sum(quest["reward"] for quest in QUESTS)
TOTAL_REWARD_XP = QUEST_XP_TOTAL
assert TOTAL_REWARD_XP == TARGET_XP, (
    f"퀘스트 보상 합계 오류: {TOTAL_REWARD_XP:,} XP (목표 {TARGET_XP:,} XP)"
)

# Recalculate displayed XP from quest rewards only, including sessions created by older versions.
# Research-rule rewards no longer count toward the total.
if "quests_claimed" in st.session_state:
    st.session_state.xp = sum(
        quest["reward"] for quest in QUESTS
        if quest["id"] in st.session_state.quests_claimed
    )

# Always-visible progress HUD for normal play; researcher mode shows answer sheets instead.
if not st.session_state.researcher_mode:
    completed_quest_count = sum(q["id"] in st.session_state.quests_done for q in QUESTS)
    discovered_rule_count = sum(rid in st.session_state.unlocked_rules for rid in RULES)
    st.markdown(f"""
    <div class="progress-hud">
      <h3>🏆 연구소 진행 현황</h3>
      <p><b>🎯 목표 경험치: {TARGET_XP:,} XP</b> · 6개 퀘스트 보상을 모두 수령하면 총 {TOTAL_REWARD_XP:,} XP를 획득합니다.</p>
    </div>
    """, unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    p1.metric("✨ 퀘스트 경험치", f"{st.session_state.xp:,} / {TARGET_XP:,} XP")
    p2.metric("🏁 완료 퀘스트", f"{completed_quest_count} / {len(QUESTS)}")
    p3.metric("🔓 발견한 규칙", f"{discovered_rule_count} / {len(RULES)}")


# --------------------------- Run and record ---------------------------
if run_button:
    st.session_state.quests_visible = False
    st.session_state.research_visible = False
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
    # scroll-to-graph
    components.html("""
    <script>
    function scrollToGraph() {
      try {
        const doc = window.parent.document;
        const headings = Array.from(doc.querySelectorAll('h1,h2,h3'));
        const graph = headings.find(el => (el.innerText || '').includes('공의 이동 경로'));
        if (graph) graph.scrollIntoView({behavior:'smooth', block:'start'});
      } catch (e) {}
    }
    setTimeout(scrollToGraph, 250);
    setTimeout(scrollToGraph, 900);
    </script>
    """, height=0)
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
    # 퀘스트 완료 알림은 전체 화면 이펙트가 닫힌 뒤 한 번만 표시합니다.

# --------------------------- Quest and codex popups ---------------------------
@st.dialog("🧪 퀘스트 센터", width="large")
def show_quest_dialog():
    with st.container(border=True):
        st.markdown('<div class="panel-heading"><h2>🧪 QUEST CENTER · 퀘스트 센터</h2><p>미션을 완료하고 보상을 수령하세요.</p></div>', unsafe_allow_html=True)
        st.info(f"🎯 목표 경험치: {TARGET_XP:,} XP — 6개 퀘스트 보상을 모두 수령하면 목표 경험치에 도달합니다.")
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
                            if (all(qid in st.session_state.quests_done for qid in REQUIRED_QUEST_IDS)
                                    and all(qid in st.session_state.quests_claimed for qid in REQUIRED_QUEST_IDS)):
                                st.session_state.master_popup_pending = True
                                st.session_state.master_celebration_pending = True
                                st.session_state.quests_visible = False
                            st.rerun()
                    else:
                        st.caption("미완료 — 실험 조건을 바꾸어 도전하세요.")

            if st.session_state.quest_firework:
                components.html("""
                <div style="position:fixed;inset:0;width:100vw;height:100vh;z-index:2147483646;pointer-events:none;background:rgba(4,20,38,.22);">
                  <canvas id="quest-fireworks" style="width:100vw;height:100vh;display:block;"></canvas>
                  <div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:#C7FCFF;font:900 clamp(26px,5vw,52px) system-ui;text-shadow:0 0 18px #16DDF5;">🎆 REWARD CLAIMED! 🎆</div>
                </div>
                <script>
                (() => {
                  const c=document.getElementById('quest-fireworks'),x=c.getContext('2d'),d=window.devicePixelRatio||1,w=window.innerWidth,h=window.innerHeight;
                  c.width=w*d;c.height=h*d;x.scale(d,d);
                  const colors=['#4CF4FF','#FFD166','#FF6B9A','#B8A0FF','#FFFFFF'];
                  let rockets=[],sparks=[],frame=0;
                  function launch(side){
                    const y=h*(.45+Math.random()*.35), targetX=side==='left'?w*(.25+Math.random()*.25):w*(.5+Math.random()*.25);
                    rockets.push({x:side==='left'?-12:w+12,y,tx:targetX,ty:h*(.12+Math.random()*.35),side,col:colors[Math.floor(Math.random()*colors.length)]});
                  }
                  function burst(r){
                    for(let i=0;i<44;i++){const a=Math.PI*2*i/44,v=1.5+Math.random()*3.6;sparks.push({x:r.x,y:r.y,vx:Math.cos(a)*v,vy:Math.sin(a)*v,life:65,col:colors[Math.floor(Math.random()*colors.length)]});}
                  }
                  function draw(){
                    x.clearRect(0,0,w,h);
                    if(frame%18===0){launch('left');launch('right');}
                    rockets=rockets.filter(r=>{
                      const dx=r.tx-r.x,dy=r.ty-r.y,dist=Math.hypot(dx,dy);
                      if(dist<13){burst(r);return false;}
                      r.x+=dx*.075;r.y+=dy*.075;x.fillStyle=r.col;x.beginPath();x.arc(r.x,r.y,3.2,0,7);x.fill();
                      x.globalAlpha=.35;x.beginPath();x.moveTo(r.x,r.y);x.lineTo(r.x-(r.side==='left'?-16:16),r.y+8);x.strokeStyle=r.col;x.stroke();x.globalAlpha=1;return true;
                    });
                    sparks=sparks.filter(p=>{p.x+=p.vx;p.y+=p.vy;p.vy+=.035;p.vx*=.985;p.life--;if(p.life>0){x.globalAlpha=Math.min(1,p.life/18);x.fillStyle=p.col;x.beginPath();x.arc(p.x,p.y,2.2,0,7);x.fill();}return p.life>0;});
                    x.globalAlpha=1;if(++frame<180)requestAnimationFrame(draw);
                  }
                  draw();
                })();
                </script>
                """, height=0)
                components.html("""
                <script>
                (() => {
                  try {
                    const doc = window.parent.document;
                    const old = doc.getElementById('gd-reward-notification');
                    if (old) old.remove();
                    const toast = doc.createElement('div');
                    toast.id = 'gd-reward-notification';
                    toast.setAttribute('role','status');
                    toast.style.cssText = 'position:fixed;top:18px;left:18px;z-index:2147483647;max-width:min(440px,calc(100vw - 36px));box-sizing:border-box;padding:15px 18px;background:#fff;color:#202124;border:1px solid #DADCE0;border-radius:12px;box-shadow:0 4px 18px rgba(60,64,67,.24);font:600 15px/1.45 system-ui,sans-serif;opacity:0;transform:translateY(-8px);transition:opacity .18s ease,transform .18s ease';
                    toast.textContent = '🎁 퀘스트 보상을 수령했습니다.';
                    doc.body.appendChild(toast);
                    requestAnimationFrame(() => { toast.style.opacity='1'; toast.style.transform='translateY(0)'; });
                    setTimeout(() => { if(toast.isConnected){toast.style.opacity='0';toast.style.transform='translateY(-8px)';setTimeout(()=>toast.remove(),220);} }, 5200);
                  } catch(e) {}
                })();
                </script>
                """, height=0)
                st.session_state.quest_firework = False


@st.dialog("🏆 경사하강법 마스터 달성!", width="large")
def show_master_celebration():
    st.markdown("""
    <style>@keyframes masterRevealTitle{0%,62%{opacity:0;transform:translateY(16px) scale(.78);filter:blur(8px)}78%{opacity:1;transform:translateY(-3px) scale(1.06);filter:blur(0)}100%{opacity:1;transform:translateY(0) scale(1);filter:blur(0)}} .master-reveal-title{animation:masterRevealTitle 3.2s cubic-bezier(.2,.8,.2,1) both!important}</style>
    <div style="text-align:center;padding:10px 4px 18px">
      <div style="font-size:clamp(52px,10vw,88px);line-height:1.1">🏆</div>
      <h2 class="master-reveal-title" style="color:#17395E!important;margin:10px 0;opacity:0">경사하강법 마스터가 되었습니다!</h2>
      <p style="font-size:17px;color:#40536B!important">모든 퀘스트 보상을 수령했어요. 실험으로 경사하강법의 움직임을 직접 밝혀냈습니다.</p>
      <div style="display:flex;justify-content:center;gap:12px;flex-wrap:wrap;font-size:30px;margin:18px 0">📈　⚽　🎓　✨</div>
      <div style="display:inline-block;background:#E2F1E8;color:#216343;padding:10px 16px;border-radius:12px;font-weight:850">총 보상 10,000 XP · 퀘스트 6/6</div>
    </div>
    """, unsafe_allow_html=True)
    components.html("""
    <script>
    (()=>{try{
      const doc=window.parent.document;const old=doc.getElementById('master-celebration-fireworks');if(old)old.remove();
      const wrap=doc.createElement('div');wrap.id='master-celebration-fireworks';wrap.style.cssText='position:fixed;inset:0;z-index:2147483000;pointer-events:none;overflow:hidden';
      const c=doc.createElement('canvas');c.style.cssText='width:100%;height:100%;display:block';wrap.appendChild(c);doc.body.appendChild(wrap);
      const x=c.getContext('2d'),d=window.devicePixelRatio||1;let w=window.innerWidth,h=window.innerHeight;
      function resize(){w=window.innerWidth;h=window.innerHeight;c.width=w*d;c.height=h*d;x.setTransform(d,0,0,d,0,0)}resize();
      const colors=['#F4C95D','#78A9D8','#E58BA8','#8BC7A3','#FFFFFF'];let rockets=[],sparks=[],frame=0;
      function launch(side){rockets.push({x:side==='left'?-8:w+8,y:h*(.55+Math.random()*.35),tx:w*(.18+Math.random()*.64),ty:h*(.12+Math.random()*.38),col:colors[Math.floor(Math.random()*colors.length)]})}
      function burst(r){for(let i=0;i<92;i++){const a=2*Math.PI*i/52,v=1.1+Math.random()*3.4;sparks.push({x:r.x,y:r.y,vx:Math.cos(a)*v,vy:Math.sin(a)*v,life:70,col:r.col})}}
      function draw(){x.clearRect(0,0,w,h);if(frame%7===0){launch('left');launch('right')}
        rockets=rockets.filter(r=>{const dx=r.tx-r.x,dy=r.ty-r.y;if(Math.hypot(dx,dy)<12){burst(r);return false}r.x+=dx*.07;r.y+=dy*.07;x.fillStyle=r.col;x.beginPath();x.arc(r.x,r.y,3,0,7);x.fill();return true});
        sparks=sparks.filter(p=>{p.x+=p.vx;p.y+=p.vy;p.vy+=.025;p.life--;if(p.life>0){x.globalAlpha=Math.min(1,p.life/20);x.fillStyle=p.col;x.beginPath();x.arc(p.x,p.y,2,0,7);x.fill()}return p.life>0});x.globalAlpha=1;
        if(++frame<360)requestAnimationFrame(draw);else setTimeout(()=>wrap.remove(),1500)
      }draw();
    }catch(e){}})();
    </script>
    """, height=0)

if st.session_state.quests_visible:
    show_quest_dialog()
if st.session_state.master_popup_pending and is_research_lab_master:
    st.session_state.master_popup_pending = False
    show_master_celebration()

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
              const maxAbsX = Math.max(12, ...points.map(p => Math.abs(p.x))) * 1.12;
              const xMin = -maxAbsX, xMax = maxAbsX;
              const maxY = Math.max(150, ...points.map(p => p.y)) * 1.12;
              const yMin = -maxY * 0.035, yMax = maxY;
              const xStep = Math.max(1, Math.ceil((xMax-xMin)/8/5)*5);
              const yStep = Math.max(10, Math.ceil((yMax-yMin)/6/10)*10);
              let frame=0, lastTick=0, finished=false;
              const frameMs={frame_duration};
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
                for(let y=0;y<=yMax;y+=yStep) {{
                  const yy=py(y,d); ctx.beginPath();ctx.moveTo(d.left,yy);ctx.lineTo(w-d.right,yy);ctx.stroke();
                  ctx.fillStyle='#68758a';ctx.fillText(String(y),d.left-22,yy);
                }}
                for(let x=Math.ceil(xMin/xStep)*xStep;x<=xMax;x+=xStep) {{
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
            # The canvas starts at frame 0 immediately, then advances once per frame_duration.
            # Wait for the final ball transition to finish, plus a safety buffer, before any full-screen effect.
            animation_duration = max(
                0.0,
                ((len(xs_anim) - 1) * frame_duration + transition_duration) / 1000.0 + 0.35,
            )
            if result["alpha"] < 0.5:
                # Always show turtle effect for alpha < 0.5, regardless of result status.
                show_turtle_effect(result["attempt_id"], delay_seconds=animation_duration)
                st.session_state.effect_shown_for_attempt = result["attempt_id"]
            elif code == "success":
                # α=0.5 one-step landing is an ordinary success; mastery is only awarded by all quest claims.
                show_success_effect(result["attempt_id"], delay_seconds=animation_duration)
                st.session_state.effect_shown_for_attempt = result["attempt_id"]
            else:
                show_failure_effect(result["alpha"], result["attempt_id"], delay_seconds=animation_duration)
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
st.caption("경사하강법 시뮬레이터 · 실험하고 발견하고 해금하세요!")
