import streamlit as st
from transformers import pipeline
import re
import tempfile
import whisper
import torch
import matplotlib.pyplot as plt

# 한글 폰트 설정 (Korean font setting)
def set_korean_font():
    try:
        plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows용 (For Windows)
    except:
        pass
set_korean_font()

# 모델 로딩 (캐싱) (Model loading (caching))
@st.cache_resource(show_spinner=False)
def load_emotion_model():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
emotion_analyzer = load_emotion_model()

@st.cache_resource(show_spinner=False)
def load_whisper_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return whisper.load_model("small", device=device)
whisper_model = load_whisper_model()

# 텍스트 전처리 (Text preprocessing)
def clean_text(text: str) -> str:
    text = text.strip().replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^가-힣a-zA-Z0-9.,!? ]", "", text)
    return text

# 자살 위험 판단 (Suicide risk detection)
def detect_suicide_risk_ml(emotion_scores: dict) -> bool:
    sadness = emotion_scores.get('sadness', 0)
    fear = emotion_scores.get('fear', 0)
    return sadness > 0.3 or fear > 0.2

# 조직 적응력 세부 항목 (Detailed organizational adaptability items)
def detailed_organization_evaluation(emotion_scores: dict) -> dict:
    discipline = emotion_scores.get('fear', 0)
    loyalty = emotion_scores.get('joy', 0)
    stress_resilience = 1 - emotion_scores.get('sadness', 0)
    return {
        "규율성": round(discipline, 2),
        "충성심": round(loyalty, 2),
        "스트레스 저항력": round(stress_resilience, 2)
    }

# 보고서 생성 (Report generation)
def generate_report(results: dict) -> str:
    dominant = results["지배 감정"]
    emotion_scores = results["감정 비율"]
    org_eval = results["조직 적응력 세분화"]
    suicide_risk = results["자살 위험 여부"]
    personality = results["조직 생활 평가"]

    adaptation_score = round((org_eval['규율성'] + org_eval['충성심'] + org_eval['스트레스 저항력']) / 3, 2)

    # 감정 레이블 한글 번역 맵 (Korean translation map for emotion labels)
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
            return "긍정적 정서 상태를 의미합니다." # Indicates a positive emotional state.
        elif emotion in ['anger', 'fear', 'sadness', 'disgust']:
            return "부정적 정서 상태 또는 스트레스 상태를 의미할 수 있습니다." # May indicate a negative emotional state or stress.
        else:
            return "중립적인 감정입니다." # It's a neutral emotion.

    report = f"""
# 🧠 두 사막의 심리·조직 적응 반석 보고서 (Two Deserts Psychological and Organizational Adaptability Cornerstone Report)

## 1. 주요 지배 감정: **{emotion_korean_map.get(dominant.lower(), dominant)} ({dominant})** (Main Dominant Emotion)

- 감정 분포: (Emotion Distribution)
"""
    for emotion, score in emotion_scores.items():
        # 영어 레이블과 함께 한글 번역 추가 (Add Korean translation along with English label)
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

🔍 본 보고서는 입력된 텍스트에 기반한 AI 분석 결과이며, 참고용으로 사용하세요. (This report is an AI analysis result based on the input text and should be used for reference.)
"""
    return report[:4000]

# 감정 분석 (Emotion analysis)
def analyze_texts(texts: list) -> dict:
    combined_text = " ".join(texts)
    cleaned = clean_text(combined_text)
    emotion_results = emotion_analyzer(cleaned, truncation=True, max_length=512)[0]
    emotion_scores = {r['label']: r['score'] for r in emotion_results}
    dominant_emotion = max(emotion_scores, key=emotion_scores.get)
    suicide_risk = detect_suicide_risk_ml(emotion_scores)
    org_eval = detailed_organization_evaluation(emotion_scores)

    if dominant_emotion in ['joy', 'love']:
        personality = "긍정적이고 조직 생활에 잘 적응할 가능성이 높음" # Positive and highly likely to adapt well to organizational life
    elif dominant_emotion in ['anger', 'fear']:
        personality = "조직 내 갈등이나 불안 요소가 존재할 수 있음" # Potential for conflict or anxiety within the organization
    elif dominant_emotion == 'sadness':
        personality = "우울 성향이 있어 관심이 필요할 수 있음" # May have a depressive tendency and require attention
    else:
        personality = "평균적인 정서 상태로 보임" # Appears to be in an average emotional state

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
    return result["text"]

# 인증 (Authentication)
def verify_access_code():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    def check_code():
        if st.session_state.access_code_input == "airforce2158":
            st.session_state.authenticated = True
        else:
            st.session_state.authenticated = False
            st.session_state.auth_fail = True

    if not st.session_state.authenticated:
        st.title("🔐 인증 필요")  # Authentication Required
        st.text_input(
            "접근 코드 입력",      # Enter Access Code
            type="password",
            key="access_code_input",
            on_change=check_code
        )

        # 인증 실패 메시지 출력 (엔터 입력 후 실패한 경우)
        if st.session_state.get("auth_fail"):
            st.error("인증 실패")  # Authentication Failed

        st.stop()
        return False

    return True


# 세션 초기화 (Session reset)
def reset_state():
    """
    이전 분석 결과 및 입력 파일 정보를 세션 상태에서 삭제합니다.
    새로운 분석이 시작될 때마다 깨끗한 상태를 보장하여 불필요한 재분석을 방지합니다.
    이는 다른 탭에서 새로운 파일을 업로드하거나 텍스트를 입력할 때도 적용됩니다.
    (Deletes previous analysis results and input file information from the session state.
    Ensures a clean state for each new analysis to prevent unnecessary re-analysis.
    This also applies when uploading new files or entering text in different tabs.)
    """
    for key in ["audio_file", "text_file", "input_text", "result"]:
        if key in st.session_state:
            del st.session_state[key]

# Streamlit 앱 (Streamlit App)
def main():
    if not verify_access_code():
        return

    # Custom CSS for improved aesthetics and mobile responsiveness
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;700&display=swap');
        body {
            font-family: 'Noto Sans KR', sans-serif;
            background-color: #f0f2f6; /* Light gray background */
            color: #333333;
            margin: 0;
            padding: 0;
        }
        .stApp {
            max-width: 800px; /* Max width for desktop */
            margin: 20px auto;
            padding: 20px;
            background-color: #ffffff;
            border-radius: 15px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        }
        h1, h2, h3 {
            color: #2c3e50; /* Dark blue-gray for headers */
            text-align: center;
            margin-bottom: 20px;
        }
        h1 {
            font-size: 2.5em;
            font-weight: 700;
            padding-bottom: 10px;
            border-bottom: 2px solid #3498db; /* Blue underline */
        }
        h2 {
            font-size: 1.8em;
            font-weight: 600;
            color: #34495e;
        }
        h3 {
            font-size: 1.4em;
            font-weight: 500;
            color: #555555;
        }
        .subtitle {
            text-align: center;
            color: #7f8c8d;
            font-size: 1.1em;
            margin-bottom: 30px;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            justify-content: center;
            margin-bottom: 20px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #ecf0f1; /* Light gray for inactive tabs */
            border-radius: 10px;
            padding: 10px 20px;
            font-weight: 600;
            color: #7f8c8d;
            transition: all 0.3s ease;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #dfe6e9;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: #3498db; /* Blue for active tab */
            color: white;
            box-shadow: 0 2px 8px rgba(52, 152, 219, 0.4);
        }
        .stFileUploader, .stTextArea, .stButton {
            margin-bottom: 20px;
        }
        .stFileUploader > label, .stTextArea > label {
            font-weight: 600;
            color: #34495e;
            margin-bottom: 10px;
            display: block;
        }
        .stButton > button {
            background-color: #2ecc71; /* Green for buttons */
            color: white;
            border-radius: 10px;
            padding: 12px 25px;
            font-size: 1.1em;
            font-weight: 600;
            border: none;
            box-shadow: 0 4px 10px rgba(46, 204, 113, 0.3);
            transition: all 0.3s ease;
            width: 100%; /* Full width on mobile */
            max-width: 250px; /* Max width for desktop */
            display: block;
            margin: 0 auto;
        }
        .stButton > button:hover {
            background-color: #27ae60;
            box-shadow: 0 6px 15px rgba(46, 204, 113, 0.4);
            transform: translateY(-2px);
        }
        .report {
            background-color: #f9f9f9;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 25px;
            line-height: 1.8;
            font-size: 1.05em;
            white-space: pre-wrap; /* Preserve whitespace and line breaks */
            word-wrap: break-word; /* Break long words */
            box-shadow: inset 0 1px 5px rgba(0, 0, 0, 0.05);
        }
        .report h1, .report h2, .report h3 {
            text-align: left; /* Align report headers to left */
            color: #2c3e50;
            margin-top: 20px;
            margin-bottom: 15px;
            border-bottom: none;
            padding-bottom: 0;
        }
        .report h1 { font-size: 2em; }
        .report h2 { font-size: 1.5em; }
        .report ul {
            list-style-type: disc;
            margin-left: 20px;
            padding-left: 0;
        }
        .stSpinner > div {
            text-align: center;
            margin-top: 20px;
        }
        /* Mobile responsiveness */
        @media (max-width: 768px) {
            .stApp {
                margin: 10px;
                padding: 15px;
                border-radius: 10px;
            }
            h1 {
                font-size: 2em;
            }
            h2 {
                font-size: 1.5em;
            }
            .subtitle {
                font-size: 1em;
            }
            .stTabs [data-baseweb="tab"] {
                padding: 8px 15px;
                font-size: 0.9em;
            }
            .stButton > button {
                padding: 10px 20px;
                font-size: 1em;
            }
            .report {
                padding: 15px;
                font-size: 0.95em;
            }
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""<h1>🧠 군 조직 성향 및 자살 위험 종합 분석</h1>""", unsafe_allow_html=True) # Military Organizational Tendency and Suicide Risk Comprehensive Analysis
    st.markdown('<p class="subtitle">최신 AI 감정 모델을 활용한 심층 분석 시스템</p>', unsafe_allow_html=True) # In-depth analysis system using the latest AI emotion models
    st.write("---")

   
    tab1, tab2, tab3 = st.tabs(["🎙️ 음성파일(STT)", "📄 텍스트파일", "📝 복사한 대화 분석"])

    with tab1:
        st.header("음성 파일 업로드")
        audio_file = st.file_uploader("파일 선택 (mp3, wav, m4a, flac)", type=["mp3", "wav", "m4a", "flac"])
        
        if audio_file is not None:
            if "audio_transcribed" not in st.session_state or st.session_state.get("last_audio_filename") != audio_file.name:
                reset_state()
                with st.spinner("음성 텍스트 변환 중..."):
                    transcript = transcribe_audio(audio_file)
                st.session_state.audio_transcribed = transcript
                st.session_state.last_audio_filename = audio_file.name

                with st.spinner("분석 중..."):
                    result = analyze_texts([transcript])
                st.session_state.result = result
            else:
                transcript = st.session_state.audio_transcribed
                result = st.session_state.result

            st.subheader("변환된 텍스트")
            st.write(transcript)
            st.markdown("---")
            st.header("📋 분석 결과")
            st.markdown(f'<div class="report">{generate_report(result)}</div>', unsafe_allow_html=True)

    with tab2:
        st.header("텍스트 파일 업로드")
        text_file = st.file_uploader("텍스트 파일 업로드 (.txt)", type=["txt"])
        
        if text_file is not None:
            if "last_text_filename" not in st.session_state or st.session_state.get("last_text_filename") != text_file.name:
                reset_state()
                text_content = text_file.read().decode("utf-8")
                st.session_state.last_text_filename = text_file.name
                with st.spinner("분석 중..."):
                    result = analyze_texts([text_content])
                st.session_state.result = result
            else:
                result = st.session_state.result

            st.markdown("---")
            st.header("📋 분석 결과")
            st.markdown(f'<div class="report">{generate_report(result)}</div>', unsafe_allow_html=True)

    with tab3:
        st.header("텍스트 직접 입력")
        input_text = st.text_area("분석할 대화 내용 입력", height=300, placeholder="여기에 붙여넣기 하세요...")

        if st.button("분석 시작"):
            if input_text.strip() == "":
                st.warning("내용이 비어 있습니다.")
            else:
                if st.session_state.get("last_input_text") != input_text:
                    reset_state()
                    st.session_state.last_input_text = input_text
                    with st.spinner("분석 중..."):
                        result = analyze_texts([input_text])
                    st.session_state.result = result
                else:
                    result = st.session_state.result

                st.markdown("---")
                st.header("📋 분석 결과")
                st.markdown(f'<div class="report">{generate_report(result)}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
