import streamlit as st
from transformers import pipeline
import re
import tempfile
import whisper
import torch
import matplotlib.pyplot as plt
import os

# 한글 폰트 설정 (Windows용 기본 설정)
def set_korean_font():
    try:
        plt.rcParams['font.family'] = 'Malgun Gothic'
    except:
        pass
set_korean_font()

# 모델 로딩 (캐싱)
@st.cache_resource(show_spinner=False)
def load_emotion_model():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
emotion_analyzer = load_emotion_model()

@st.cache_resource(show_spinner=False)
def load_whisper_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return whisper.load_model("small", device=device)
whisper_model = load_whisper_model()

# 텍스트 전처리
def clean_text(text: str) -> str:
    text = text.strip().replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^가-힣a-zA-Z0-9.,!? ]", "", text)
    return text

# 자살 위험 판단
def detect_suicide_risk(emotion_scores: dict, conversation_length: int) -> str:
    sadness = emotion_scores.get('sadness', 0)
    fear = emotion_scores.get('fear', 0)
    
    if conversation_length < 50:
        return "정보 부족"

    if sadness > 0.45 or (sadness > 0.3 and fear > 0.3):
        return "⚠️ 위험 신호 감지"
    elif sadness > 0.25 or fear > 0.2:
        return "❗️ 관심 필요"
    else:
        return "✅ 위험 신호 없음"

# 조직 적응력 세부 항목
def detailed_organization_evaluation(emotion_scores: dict) -> dict:
    discipline_score = emotion_scores.get('neutral', 0) + (1 - emotion_scores.get('anger', 0))
    loyalty_score = emotion_scores.get('joy', 0) + emotion_scores.get('love', 0)
    stress_resilience_score = (1 - emotion_scores.get('sadness', 0)) + (1 - emotion_scores.get('fear', 0))

    discipline = min(1, max(0, discipline_score / 2))
    loyalty = min(1, max(0, loyalty_score / 2))
    stress_resilience = min(1, max(0, stress_resilience_score / 2))

    return {
        "규율성": round(discipline, 2),
        "충성심": round(loyalty, 2),
        "스트레스 저항력": round(stress_resilience, 2)
    }

# 심층 심리 및 성격 보고서 생성 함수
def generate_detailed_personality_report(emotion_scores: dict, participants: list, conversation_length: int) -> str:
    participants_str = ", ".join(participants) if participants else "분석 대상자"
    
    if conversation_length < 50:
        return f"""
### ⚠️ 심층 분석 제한
대화 내용이 너무 짧아(글자수: {conversation_length}자), 심도 있는 심리 분석에 제한이 있습니다.
제공된 정보만으로는 신뢰성 있는 분석을 하기 어려우므로, 추가적인 대화 내용을 확보해주시면 더 정확한 분석이 가능합니다.
분석이 가능한 부문에 대해서는 아래와 같이 요약하여 제시합니다.
"""

    joy = emotion_scores.get('joy', 0)
    sadness = emotion_scores.get('sadness', 0)
    anger = emotion_scores.get('anger', 0)
    fear = emotion_scores.get('fear', 0)
    love = emotion_scores.get('love', 0)
    surprise = emotion_scores.get('surprise', 0)
    disgust = emotion_scores.get('disgust', 0)
    neutral = emotion_scores.get('neutral', 0)
    
    dominant_emotion = max(emotion_scores, key=emotion_scores.get)
    emotion_korean_map = {
        'joy': '기쁨', 'sadness': '슬픔', 'anger': '분노', 'fear': '두려움',
        'love': '사랑', 'surprise': '놀람', 'disgust': '혐오', 'neutral': '중립'
    }
    
    conversation_type = "일상적인 대화"
    if dominant_emotion in ['anger', 'sadness', 'fear']:
        conversation_type = "갈등 또는 어려움에 대한 대화"
    elif dominant_emotion in ['joy', 'love']:
        conversation_type = "긍정적이고 화목한 대화"

    report_text = f"""
### 🗣️ 대화 참여자
- 분석 대상: {participants_str}
- 대화 내용: 대화에서 전반적으로 **'{emotion_korean_map.get(dominant_emotion)}'** 감정이 가장 두드러지며, 이는 **{conversation_type}**였음을 시사합니다.

### 📝 심리 및 무의식 분석
- **심리 상태**:
    - **긍정 정서**: 대화에서 기쁨({joy:.2f})과 사랑({love:.2f}) 같은 긍정적 감정이 두드러져, 전반적으로 {'활발하고 긍정적인 성향' if (joy+love) > 0.5 else '내성적이거나 감정 표현에 신중한 경향'}을 보입니다.
    - **부정 정서**: 슬픔({sadness:.2f})과 두려움({fear:.2f})이 다소 {'높아' if sadness > 0.3 or fear > 0.2 else '낮아'}, {'감정적으로 불안정하거나 스트레스 상황에 놓여 있을 가능성' if sadness > 0.3 or fear > 0.2 else '비교적 안정적인 심리 상태'}로 추정됩니다.
    - **분노/혐오**: 분노({anger:.2f})나 혐오({disgust:.2f})는 {'낮은 편' if anger < 0.3 and disgust < 0.3 else '주의가 필요한 수준'}으로, {'평소 감정 조절이 잘 되고' if anger < 0.3 and disgust < 0.3 else '내적 갈등이나 불만을 내포하고 있을 수 있습니다.'}를 시사합니다.
- **무의식적 경향성**:
    - **관계성**: 대화의 긍정 정서가 높다면, 타인과의 상호작용에서 긍정적 관계를 형성하려는 무의식적 욕구가 강합니다. 반대로 두려움이 높다면 관계에서 오는 불안감이나 회피 경향이 무의식적으로 작용할 수 있습니다.
    - **자아 방어**: 분노나 혐오가 표출될 경우, 이는 외부 환경으로부터 자신을 보호하려는 무의식적 방어 기제로 해석될 수 있습니다. 감정의 과도한 억제는 스트레스의 무의식적 축적을 의미하기도 합니다.

"""
    return report_text[:4000]

# 최종 보고서 생성
def generate_final_report(results: dict, participants: list, cleaned_text: str) -> str:
    emotion_scores = results["감정 비율"]
    org_eval = results["조직 적응력 세분화"]
    suicide_risk_status = results["자살 위험 여부"]

    # 군생활 적응도 결론
    adaptation_score = round((org_eval['규율성'] + org_eval['충성심'] + org_eval['스트레스 저항력']) / 3, 2)
    
    if adaptation_score > 0.8:
        adaptation_conclusion = "매우 긍정적: 조직 적응력이 매우 높아 군생활에 큰 어려움이 없을 것으로 예상됩니다."
    elif adaptation_score > 0.6:
        adaptation_conclusion = "긍정적: 조직 생활에 잘 적응할 것으로 보이나, 지속적인 관심이 필요합니다."
    elif adaptation_score > 0.4:
        adaptation_conclusion = "보통: 심리적 불안정 요소가 일부 보이며, 상황에 따라 적응에 어려움을 겪을 수 있습니다."
    else:
        adaptation_conclusion = "주의 필요: 심리적 지원과 세심한 관심이 요구되며, 조직 적응에 어려움이 있을 수 있습니다."

    suicide_risk_reason = ""
    if suicide_risk_status == "⚠️ 위험 신호 감지":
        suicide_risk_reason = "대화에서 감지된 '슬픔'과 '두려움' 감정 점수가 매우 높아, 심리적 불안정성이 상당한 수준임을 시사합니다."
    elif suicide_risk_status == "❗️ 관심 필요":
        suicide_risk_reason = "대화에서 감지된 '슬픔' 또는 '두려움' 감정 점수가 일정 수준 이상으로 나타나, 심리적 어려움이 있을 수 있음을 시사합니다."
    else:
        suicide_risk_reason = "대화 내용에 기반하여 특별한 위험 신호는 감지되지 않았습니다."

    report = f"""
# 📄 대화 내용 분석 리포트

---

### 1. 자살 위험도 평가
- **판정 결과**: **{suicide_risk_status}**
- **근거**: {suicide_risk_reason} 단순 AI 분석이므로 전문가의 추가 상담이 반드시 필요합니다.

---

### 2. 조직 적응 능력 평가
- **규율성**: **{org_eval['규율성']:.2f}** (점수가 높을수록 규칙을 잘 준수하고 차분한 성향을 의미)
- **충성심**: **{org_eval['충성심']:.2f}** (점수가 높을수록 소속감과 긍정적인 유대감을 의미)
- **스트레스 저항력**: **{org_eval['스트레스 저항력']:.2f}** (점수가 높을수록 어려움에 대한 심리적 회복력이 높음을 의미)
- **결론**: 평가 점수를 종합했을 때, 군생활에 대한 심리적 적응도는 **'{adaptation_conclusion}'**으로 판단됩니다.

---

{generate_detailed_personality_report(emotion_scores, participants, len(cleaned_text))}

---

### 📜 분석 관련 근거
- **법률적 근거**: 본 분석은 비전문가용 참고 자료로, **정식 진단 및 법적 효력을 가지지 않습니다**. 개인정보보호법에 의거, 대상자의 동의 없이 무단으로 정보를 수집하거나 활용하는 것은 법적 제재의 대상이 될 수 있습니다. 모든 활용은 개인정보 보호 원칙을 철저히 준수해야 합니다.
- **학술적 근거**: 본 분석은 **폴 에크만(Paul Ekman)의 보편적 기본 감정 이론**과 같은 심리학적 모델에 기반한 자연어 처리(NLP) 기술을 활용합니다. AI 모델(DistilRoBERTa 기반)이 텍스트의 문맥과 단어 패턴을 분석하여 감정을 식별하고, 이 데이터를 바탕으로 조직 행동 및 심리 이론에 따라 해석한 내용입니다.

---

🔍 **본 보고서는 입력된 텍스트에 기반한 AI 분석 결과이며, 참고용으로만 사용하십시오.**
"""
    return report

# 참여자 이름 추출
def extract_person_names(text: str) -> list:
    pattern = re.compile(r"([가-힣a-zA-Z0-9_]{1,20}):")
    matches = pattern.findall(text)
    seen = set()
    participants = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            participants.append(m)
    return participants[:3]

# 감정 분석
def analyze_texts(texts: list) -> dict:
    combined_text = " ".join(texts)
    cleaned = clean_text(combined_text)
    
    if len(cleaned) < 10:
        return {
            "감정 비율": {},
            "지배 감정": "정보 부족",
            "자살 위험 여부": "정보 부족",
            "조직 적응력 세분화": {"규율성": 0, "충성심": 0, "스트레스 저항력": 0}
        }

    emotion_results = emotion_analyzer(cleaned, truncation=True, max_length=512)[0]
    emotion_scores = {r['label']: r['score'] for r in emotion_results}
    dominant_emotion = max(emotion_scores, key=emotion_scores.get)
    suicide_risk = detect_suicide_risk(emotion_scores, len(cleaned))
    org_eval = detailed_organization_evaluation(emotion_scores)
    
    return {
        "감정 비율": emotion_scores,
        "지배 감정": dominant_emotion,
        "자살 위험 여부": suicide_risk,
        "조직 적응력 세분화": org_eval,
        "cleaned_text": cleaned
    }

# Whisper STT
def transcribe_audio(file_buffer) -> str:
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tmp_file.write(file_buffer.read())
            tmp_path = tmp_file.name
        
        result = whisper_model.transcribe(tmp_path, fp16=False)
        return result["text"]
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

# 인증
def verify_access_code():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    def check_code():
        if st.session_state.access_code_input == "airforce2158":
            st.session_state.authenticated = True
            st.session_state.auth_fail = False
        else:
            st.session_state.authenticated = False
            st.session_state.auth_fail = True

    if not st.session_state.authenticated:
        st.title("🔐 인증 필요")
        st.text_input(
            "접근 코드 입력",
            type="password",
            key="access_code_input",
            on_change=check_code
        )

        if st.session_state.get("auth_fail"):
            st.error("인증 실패")

        st.stop()
        return False

    return True

# Streamlit 앱 메인
def main():
    if not verify_access_code():
        return
    
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;700&display=swap');
        body {
            font-family: 'Noto Sans KR', sans-serif;
            background-color: #f7f9fc;
            color: #222222;
            margin: 0;
            padding: 0;
            font-size: 0.95rem;
        }
        .stApp {
            max-width: 760px;
            margin: 20px auto 40px auto;
            padding: 20px 25px 40px 25px;
            background: #ffffff;
            border-radius: 20px;
            box-shadow: 0 4px 30px rgba(0,0,0,0.1);
            line-height: 1.6;
        }
        h1 {
            font-size: 2.8rem;
            font-weight: 700;
            color: #1f3a93;
            text-align: center;
            margin-bottom: 25px;
            letter-spacing: 1.2px;
        }
        h2 {
            font-size: 1.8rem;
            color: #34495e;
            font-weight: 600;
            margin-top: 30px;
            margin-bottom: 15px;
        }
        h3 {
            font-size: 1.3rem;
            color: #4a4a4a;
            font-weight: 500;
            margin-top: 25px;
            margin-bottom: 10px;
        }
        .subtitle {
            font-size: 1.1rem;
            color: #7f8c8d;
            text-align: center;
            margin-bottom: 30px;
        }
        .stTabs [data-baseweb="tab-list"] {
            justify-content: center;
            gap: 12px;
            margin-bottom: 20px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #dfe6e9;
            border-radius: 15px;
            padding: 10px 30px;
            font-weight: 600;
            color: #2c3e50;
            transition: background-color 0.3s ease;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #b0c4de;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: #2980b9;
            color: white;
            box-shadow: 0 3px 12px rgba(41, 128, 185, 0.6);
        }
        .stFileUploader > label,
        .stTextArea > label {
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 10px;
            display: block;
        }
        .stTextArea textarea {
            font-size: 0.95rem !important;
            min-height: 180px !important;
            line-height: 1.5 !important;
            padding: 12px !important;
            border-radius: 12px !important;
            border: 1.8px solid #bdc3c7 !important;
            resize: vertical !important;
            font-family: 'Noto Sans KR', sans-serif !important;
        }
        .stButton > button {
            background-color: #27ae60;
            color: #fff;
            font-weight: 700;
            padding: 14px 30px;
            border-radius: 30px;
            border: none;
            font-size: 1.1rem;
            cursor: pointer;
            transition: background-color 0.3s ease;
            width: 100%;
            max-width: 280px;
            margin: 10px auto 25px auto;
            display: block;
            box-shadow: 0 5px 15px rgba(39, 174, 96, 0.4);
        }
        .stButton > button:hover {
            background-color: #1e8449;
            box-shadow: 0 7px 20px rgba(30, 132, 73, 0.6);
            transform: translateY(-2px);
        }
        .report {
            background-color: #fdfdfd;
            border: 1.5px solid #d0d7de;
            border-radius: 18px;
            padding: 25px 30px;
            font-size: 0.95rem;
            color: #333333;
            line-height: 1.7;
            white-space: pre-wrap;
            box-shadow: inset 0 0 15px rgba(0,0,0,0.03);
            margin-top: 15px;
            overflow-x: auto;
            font-family: 'Noto Sans KR', sans-serif;
        }
        @media (max-width: 480px) {
            .stApp {
                margin: 10px 10px 30px 10px;
                padding: 15px 15px 30px 15px;
            }
            h1 { font-size: 2rem; }
            h2 { font-size: 1.3rem; }
            h3 { font-size: 1.1rem; }
            .stButton > button {
                font-size: 1rem;
                max-width: 100%;
                padding: 12px 20px;
            }
            .stTextArea textarea {
                min-height: 150px !important;
                font-size: 0.9rem !important;
            }
            .report {
                font-size: 0.9rem;
            }
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("🔍 심리 및 조직 적응 분석 AI 리포트")

    tab1, tab2, tab3, tab4 = st.tabs(["  1. 음성파일(STT) → 분석", "2. 텍스트 파일 → 분석", "3. 복사붙여넣기 대화분석", "💡 질문 예시"])

    with tab1:
        audio_file = st.file_uploader("🔊 음성 파일 업로드 (mp3, wav, m4a)", type=["mp3", "wav", "m4a"], key="audio_uploader")
        if audio_file:
            st.audio(audio_file)
            if st.button("📝 음성 → 텍스트 변환 및 분석"):
                with st.spinner("음성 인식 중... (파일 크기에 따라 수 분이 소요될 수 있습니다)"):
                    transcript = transcribe_audio(audio_file)
                st.text_area("🎤 변환된 텍스트", transcript, height=200)
                participants = extract_person_names(transcript)
                with st.spinner("감정 분석 및 보고서 생성 중..."):
                    results = analyze_texts([transcript])
                    if 'cleaned_text' in results:
                        report = generate_final_report(results, participants, results['cleaned_text'])
                    else:
                        report = "대화 내용이 너무 짧아 분석이 불가능합니다."
                st.markdown("### 📄 분석 결과 보고서")
                st.markdown(f'<div class="report">{report}</div>', unsafe_allow_html=True)

    with tab2:
        text_file = st.file_uploader("📄 텍스트 파일 업로드 (.txt)", type=["txt"], key="textfile_uploader")
        if text_file:
            try:
                content = text_file.read().decode('utf-8')
            except UnicodeDecodeError:
                text_file.seek(0)
                content = text_file.read().decode('cp949', errors='ignore')
            
            st.text_area("📜 텍스트 내용", content, height=200)
            participants = extract_person_names(content)
            if st.button("🔍 텍스트 분석 시작"):
                with st.spinner("감정 분석 중..."):
                    results = analyze_texts([content])
                    if 'cleaned_text' in results:
                        report = generate_final_report(results, participants, results['cleaned_text'])
                    else:
                        report = "대화 내용이 너무 짧아 분석이 불가능합니다."
                st.markdown("### 📄 분석 결과 보고서")
                st.markdown(f'<div class="report">{report}</div>', unsafe_allow_html=True)

    with tab3:
        input_text = st.text_area("💬 대화 내용 복사-붙여넣기", height=300)
        
        if st.button("분석 시작", key="paste_analysis"):
            if not input_text.strip():
                st.warning("텍스트를 입력해주세요.")
            else:
                participants = extract_person_names(input_text)
                with st.spinner("감정 분석 중..."):
                    results = analyze_texts([input_text])
                    if 'cleaned_text' in results:
                        report = generate_final_report(results, participants, results['cleaned_text'])
                    else:
                        report = "대화 내용이 너무 짧아 분석이 불가능합니다."
                st.markdown("### 📄 분석 결과 보고서")
                st.markdown(f'<div class="report">{report}</div>', unsafe_allow_html=True)
                
    with tab4:
        st.header("💡 심리 및 조직 적응 분석을 위한 질문 예시")
        st.markdown("""
        이 질문들은 대상자의 심리 상태, 가치관, 조직 적응력을 자연스럽게 파악하는 데 도움이 될 수 있는 예시입니다. 상황과 대상에 맞게 변형하여 활용하세요.
        """)

        st.subheader("1. 아이스브레이킹 및 일상 질문 (Rapport 형성)")
        st.info("""
        - 요즘 가장 즐거운 일이나 관심사가 있나요?
        - 주말이나 쉬는 날에는 보통 무엇을 하며 시간을 보내나요?
        - 최근에 재미있게 본 영화나 드라마가 있다면 어떤 점이 좋았나요?
        """)

        st.subheader("2. 성격 및 가치관 파악 질문")
        st.info("""
        - 스스로 생각하기에 자신의 가장 큰 장점과 단점은 무엇이라고 생각하나요?
        - 살면서 가장 중요하게 생각하는 가치(예: 정직, 성실, 성장, 안정 등)는 무엇인가요?
        - 어떤 사람과 함께 일할 때 가장 편안하고, 또 어떤 사람과 일하는 것이 힘든가요?
        - 예상치 못한 문제가 발생했을 때, 보통 어떻게 대처하는 편인가요?
        """)
        
        st.subheader("3. 스트레스 및 조직 적응력 관련 질문")
        st.info("""
        - 스트레스를 받을 때 주로 어떤 방식으로 해소하나요?
        - 단체 생활이나 조직 문화에서 가장 중요하다고 생각하는 점은 무엇인가요?
        - 선임이나 동료와 의견 충돌이 있었던 경험이 있나요? 있었다면 어떻게 해결했나요?
        - 힘들거나 어려운 일이 있을 때 주변에 도움을 요청하는 편인가요, 아니면 혼자 해결하려고 하나요?
        """)

        st.subheader("4. 대인관계 및 사회성 질문")
        st.info("""
        - 새로운 환경이나 낯선 사람들과 만나는 것에 대해 어떻게 느끼나요?
        - 주변 친구나 동료들은 자신을 어떤 사람이라고 평가하는 것 같나요?
        - 다른 사람의 부탁을 거절해야 할 때, 솔직하게 이야기하는 편인가요?
        """)

if __name__ == "__main__":
    main()