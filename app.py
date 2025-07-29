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
def detect_suicide_risk_ml(emotion_scores: dict) -> bool:
    sadness = emotion_scores.get('sadness', 0)
    fear = emotion_scores.get('fear', 0)
    return sadness > 0.3 or fear > 0.2

# 조직 적응력 세부 항목
def detailed_organization_evaluation(emotion_scores: dict) -> dict:
    discipline = emotion_scores.get('fear', 0)
    loyalty = emotion_scores.get('joy', 0)
    stress_resilience = 1 - emotion_scores.get('sadness', 0)
    return {
        "규율성": round(discipline, 2),
        "충성심": round(loyalty, 2),
        "스트레스 저항력": round(stress_resilience, 2)
    }

# 심층 심리 및 성격 보고서 생성 함수
def generate_detailed_personality_report(emotion_scores: dict, participants: list) -> str:
    participants_str = ", ".join(participants) if participants else "분석 대상자"
    
    joy = emotion_scores.get('joy', 0)
    sadness = emotion_scores.get('sadness', 0)
    anger = emotion_scores.get('anger', 0)
    fear = emotion_scores.get('fear', 0)
    love = emotion_scores.get('love', 0)
    surprise = emotion_scores.get('surprise', 0)
    disgust = emotion_scores.get('disgust', 0)
    neutral = emotion_scores.get('neutral', 0)

    report_text = f"""
# 심층 심리 및 무의식, 성격 분석 보고서

분석 대상: {participants_str}

---

본 보고서는 대화 내용을 기반으로 AI 감정 분석 모델의 결과를 활용하여, 대상자의 심리 상태, 무의식적 경향, 그리고 성격의 장단점을 종합적으로 해석한 내용입니다.

---

## 1. 심리 상태 분석

- 기쁨과 사랑과 같은 긍정 정서의 정도는 각각 {joy:.2f}, {love:.2f}로 나타났으며, 이는 대상자가 대체로 {'긍정적이고 친화적인 성향을 보임' if (joy+love) > 0.5 else '내향적이거나 조심스러운 성향을 가질 수 있음'}을 의미합니다.

- 슬픔({sadness:.2f})과 두려움({fear:.2f})이 다소 {'높은 편' if sadness > 0.3 or fear > 0.2 else '낮은 편'}으로, 감정적으로 {'불안정하거나 스트레스가 존재할 수 있음' if sadness > 0.3 or fear > 0.2 else '비교적 안정적인 상태임'}을 시사합니다.

- 분노({anger:.2f})와 혐오({disgust:.2f})의 정도는 {'낮아 평온한 상태' if anger < 0.3 and disgust < 0.3 else '다소 긴장이나 갈등의 가능성'}을 내포합니다.

- 놀람({surprise:.2f})과 중립({neutral:.2f}) 감정은 상황에 대한 적응력과 균형을 반영합니다.

---

## 2. 무의식적 경향성

대화 분석 결과, 다음과 같은 무의식적 경향이 감지됩니다.

- 두려움 및 슬픔 감정의 상대적 증가는 대인관계에서 불안감이나 회피 경향, 자기보호적인 심리 상태를 내포할 수 있습니다.

- 긍정 정서가 낮은 경우, 내적 갈등이나 자기 인식의 불안정함이 무의식적으로 작용할 가능성이 있습니다.

- 분노 및 혐오 감정이 높을 경우, 무의식적으로 스트레스가 누적되거나 주변 환경과의 갈등 가능성이 존재합니다.

---

## 3. 성격의 장점

- 긍정 정서가 적절히 발현되는 경우, 사회적 유대감이 높고 조직 내 협력 및 친화력이 뛰어납니다.

- 스트레스 저항력이 높다면 위기 상황에서도 침착함을 유지할 수 있어 안정적인 역할 수행이 가능합니다.

- 높은 충성심과 규율성은 조직 내 신뢰 형성에 긍정적 영향을 미칩니다.

---

## 4. 성격의 단점 및 개선 과제

- 슬픔과 두려움이 과도하게 높은 경우, 우울감과 불안 장애로 발전할 위험이 있어 심리적 지원이 필요합니다.

- 분노나 혐오가 높은 경우, 갈등 관리 및 감정 조절 훈련이 권장됩니다.

- 내성적이고 조심스러운 성향은 대인관계 확장에 제약이 될 수 있으므로 점진적 사회성 훈련이 도움이 됩니다.

---

## 결론

{participants_str}은(는) 위와 같은 심리·무의식적 특성과 성격적 장단점을 가지고 있으며, 본 보고서는 객관적 AI 분석에 기반한 해석임을 참고하시기 바랍니다.

심리적 안정과 조직 적응력 강화를 위한 지속적인 관심과 지원이 권장됩니다.
"""

    return report_text[:4000]

# 보고서 생성
def generate_report(results: dict, participants: list) -> str:
    dominant = results["지배 감정"]
    emotion_scores = results["감정 비율"]
    org_eval = results["조직 적응력 세분화"]
    suicide_risk = results["자살 위험 여부"]
    personality = results["조직 생활 평가"]

    adaptation_score = round((org_eval['규율성'] + org_eval['충성심'] + org_eval['스트레스 저항력']) / 3, 2)

    emotion_korean_map = {
        'joy': '기쁨',
        'sadness': '슬픔',
        'anger': '분노',
        'fear': '두려움',
        'love': '사랑',
        'surprise': '놀람',
        'disgust': '혐오',
        'neutral': '중립'
    }

    def emotion_explanation(emotion):
        if emotion in ['joy', 'love']:
            return "긍정적 정서 상태를 의미합니다."
        elif emotion in ['anger', 'fear', 'sadness', 'disgust']:
            return "부정적 정서 상태 또는 스트레스 상태를 의미할 수 있습니다."
        else:
            return "중립적인 감정입니다."

    if participants:
        participants_str = ", ".join(participants)
    else:
        participants_str = "알 수 없음"

    report = f"""
# 🧠 두 사의 심리·조직 적응 반석 보고서 (Two Deserts Psychological and Organizational Adaptability Cornerstone Report)

## 분석 대상: {participants_str}  (Analysis Subject(s))

## 1. 주요 지배 감정: **{emotion_korean_map.get(dominant.lower(), dominant)} ({dominant})** (Main Dominant Emotion)

- 감정 분포:
"""
    for emotion, score in emotion_scores.items():
        report += f"- {emotion_korean_map.get(emotion.lower(), emotion)} ({emotion}): {score:.3f} ({emotion_explanation(emotion.lower())})\n"

    report += f"""

## 2. 조직 적응력 평가 (Organizational Adaptability Evaluation)

- 규율성: {org_eval['규율성']} (Discipline)
- 충성심: {org_eval['충성심']} (Loyalty)
- 스트레스 저항력: {org_eval['스트레스 저항력']} (Stress Resilience)

## 3. 자살 위험 여부: {"⚠️ 위험 검지됨" if suicide_risk else "✅ 위험 신호 없음"} (Suicide Risk: {"⚠️ Risk Detected" if suicide_risk else "✅ No Risk Signal"})

## 4. 조직 생활 종합 평가 (Overall Organizational Life Evaluation)

{personality}

## 5. 군생화 적응도 추정 점수: **{adaptation_score} / 1.0** (Estimated Military Life Adaptability Score)

---

"""

    detailed_personality_report = generate_detailed_personality_report(emotion_scores, participants)
    report += detailed_personality_report + "\n\n"

    report += """---
🔍 본 보고서는 입력된 텍스트에 기반한 AI 분석 결과이며, 참고용으로 사용하세요. (This report is an AI analysis result based on the input text and should be used for reference.)
"""
    return report[:4000]

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
    emotion_results = emotion_analyzer(cleaned, truncation=True, max_length=512)[0]
    emotion_scores = {r['label']: r['score'] for r in emotion_results}
    dominant_emotion = max(emotion_scores, key=emotion_scores.get)
    suicide_risk = detect_suicide_risk_ml(emotion_scores)
    org_eval = detailed_organization_evaluation(emotion_scores)

    if dominant_emotion in ['joy', 'love']:
        personality = "긍정적이고 조직 생활에 잘 적응할 가능성이 높음"
    elif dominant_emotion in ['anger', 'fear']:
        personality = "조직 내 갈등이나 불안 요소가 존재할 수 있음"
    elif dominant_emotion == 'sadness':
        personality = "우울 성향이 있어 관심이 필요할 수 있음"
    else:
        personality = "평균적인 정서 상태로 보임"

    return {
        "감정 비율": emotion_scores,
        "지배 감정": dominant_emotion,
        "조직 생활 평가": personality,
        "자살 위험 여부": suicide_risk,
        "조직 적응력 세분화": org_eval
    }

# Whisper STT
def transcribe_audio(file_buffer) -> str:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(file_buffer.read())
        tmp_path = tmp_file.name
    result = whisper_model.transcribe(tmp_path, fp16=False)
    os.remove(tmp_path)
    return result["text"]

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
            font-size: 1.05rem !important;
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
            font-size: 1.15rem;
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
            font-size: 1rem;
            color: #333333;
            line-height: 1.7;
            white-space: pre-wrap;
            box-shadow: inset 0 0 15px rgba(0,0,0,0.03);
            margin-top: 15px;
            overflow-x: auto;
            font-family: 'Noto Sans KR', sans-serif;
        }
        /* 모바일 대응 */
        @media (max-width: 480px) {
            .stApp {
                margin: 10px 10px 30px 10px;
                padding: 15px 15px 30px 15px;
            }
            h1 {
                font-size: 2rem;
            }
            h2 {
                font-size: 1.3rem;
            }
            h3 {
                font-size: 1.1rem;
            }
            .stButton > button {
                font-size: 1rem;
                max-width: 100%;
                padding: 12px 20px;
            }
            .stTextArea textarea {
                min-height: 150px !important;
                font-size: 1rem !important;
            }
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("🔍 심리 및 조직 적응 분석 AI 리포트")

    tabs = st.tabs(["1. 음성파일(STT) → 분석", "2. 텍스트 파일 → 분석", "3. 복사붙여넣기 대화분석"])

    with tabs[0]:
        audio_file = st.file_uploader("🔊 음성 파일 업로드 (mp3, wav, m4a)", type=["mp3", "wav", "m4a"], key="audio_uploader")
        if audio_file:
            st.audio(audio_file)
            if st.button("📝 음성 → 텍스트 변환 및 분석"):
                with st.spinner("음성 인식 중..."):
                    transcript = transcribe_audio(audio_file)
                st.text_area("🎤 변환된 텍스트", transcript, height=200)
                participants = extract_person_names(transcript)
                with st.spinner("감정 분석 및 보고서 생성 중..."):
                    results = analyze_texts([transcript])
                    report = generate_report(results, participants)
                st.markdown("### 📄 분석 결과 보고서")
                st.markdown(f'<div class="report">{report}</div>', unsafe_allow_html=True)

    with tabs[1]:
        text_file = st.file_uploader("📄 텍스트 파일 업로드 (.txt)", type=["txt"], key="textfile_uploader")
        if text_file:
            content = text_file.read().decode('utf-8')
            st.text_area("📜 텍스트 내용", content, height=200)
            participants = extract_person_names(content)
            if st.button("🔍 텍스트 분석 시작"):
                with st.spinner("감정 분석 중..."):
                    results = analyze_texts([content])
                    report = generate_report(results, participants)
                st.markdown("### 📄 분석 결과 보고서")
                st.markdown(f'<div class="report">{report}</div>', unsafe_allow_html=True)

    with tabs[2]:
        input_text = st.text_area("💬 대화 내용 복사-붙여넣기", height=300)
        participants = extract_person_names(input_text)
        if st.button("분석 시작", key="paste_analysis"):
            if not input_text.strip():
                st.warning("텍스트를 입력해주세요.")
            else:
                with st.spinner("감정 분석 중..."):
                    results = analyze_texts([input_text])
                    report = generate_report(results, participants)
                st.markdown("### 📄 분석 결과 보고서")
                st.markdown(f'<div class="report">{report}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
