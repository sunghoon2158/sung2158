import streamlit as st
from transformers import pipeline
import re
import tempfile
import whisper
import torch
import matplotlib.pyplot as plt

# 한글 폰트 설정
def set_korean_font():
    try:
        plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows용
    except:
        pass
set_korean_font()

# 모델 캐싱
@st.cache_resource(show_spinner=False)
def load_emotion_model():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
emotion_analyzer = load_emotion_model()

@st.cache_resource(show_spinner=False)
def load_whisper_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return whisper.load_model("small", device=device)
whisper_model = load_whisper_model()

# 텍스트 정리
def clean_text(text: str) -> str:
    text = text.strip().replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^가-힣a-zA-Z0-9.,!? ]", "", text)
    return text

# 자살 위험 탐지
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

# 리포트 생성
def generate_report(results: dict) -> str:
    dominant = results["지배 감정"]
    emotion_scores = results["감정 비율"]
    org_eval = results["조직 적응력 세분화"]
    suicide_risk = results["자살 위험 여부"]
    personality = results["조직 생활 평가"]

    adaptation_score = round((org_eval['규율성'] + org_eval['충성심'] + org_eval['스트레스 저항력']) / 3, 2)

    def emotion_explanation(emotion):
        if emotion in ['joy', 'love']:
            return "긍정적 정서 상태를 의미합니다."
        elif emotion in ['anger', 'fear', 'sadness', 'disgust']:
            return "부정적 정서 상태 또는 스트레스 상태를 의미할 수 있습니다."
        else:
            return "중립적인 감정입니다."

    report = f"""
# 🧠 두 사람의 심리·조직 적응 분석 보고서

## 1. 주요 지배 감정: **{dominant}**

- 감정 분포:
"""
    for emotion, score in emotion_scores.items():
        report += f"- {emotion}: {score:.3f} ({emotion_explanation(emotion.lower())})\n"

    report += f"""

## 2. 조직 적응력 평가

- 규율성: {org_eval['규율성']}
- 충성심: {org_eval['충성심']}
- 스트레스 저항력: {org_eval['스트레스 저항력']}

## 3. 자살 위험 여부: {"⚠️ 위험 감지됨" if suicide_risk else "✅ 위험 신호 없음"}

## 4. 조직 생활 종합 평가

{personality}

## 5. 군생활 적응도 추정 점수: **{adaptation_score} / 1.0**

---

🔍 본 보고서는 입력된 텍스트에 기반한 AI 분석 결과이며, 참고용으로 사용하세요.
"""
    return report[:4000]

# 텍스트 분석
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

# Whisper STT (fp16 옵션 추가)
def transcribe_audio(file_buffer) -> str:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(file_buffer.read())
        tmp_path = tmp_file.name
    result = whisper_model.transcribe(tmp_path, fp16=False)  # 오류 방지 위해 fp16=False
    return result["text"]

# 접근 코드
def verify_access_code():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if not st.session_state.authenticated:
        st.title("🔐 인증 필요")
        access_code = st.text_input("접근 코드 입력", type="password")
        if st.button("확인"):
            if access_code == "airforce2158":
                st.session_state.authenticated = True
                st.success("인증 성공")
                st.query_params = {}
                st.stop()
            else:
                st.error("인증 실패")
        return False
    return True

# Streamlit 앱
def main():
    if not verify_access_code():
        return

    st.markdown("""
    <style>
    /* 전체 컨테이너 */
    .main {
        max-width: 480px;
        margin: auto;
        font-family: 'Malgun Gothic', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        padding: 15px 20px;
        background-color: #f7f9fc;
        color: #1a2935;
    }

    /* 제목 */
    h1 {
        font-size: 1.9rem;
        font-weight: 600;
        color: #336699;
        text-align: center;
        margin-bottom: 8px;
    }

    p.subtitle {
        font-size: 1.1rem;
        color: #607d8b;
        text-align: center;
        margin-top: 0;
        margin-bottom: 20px;
        font-weight: 400;
    }

    /* 탭 버튼 */
    .stTabs [role="tablist"] button {
        font-size: 1.1rem !important;
        padding: 12px 18px !important;
        min-width: 130px;
        border-radius: 12px 12px 0 0 !important;
        margin-right: 8px !important;
        color: #607d8b !important;
        background-color: #e1e8f0 !important;
        border: none !important;
        transition: background-color 0.3s ease, color 0.3s ease;
    }
    .stTabs [role="tablist"] button[aria-selected="true"] {
        background-color: #4a90e2 !important;
        color: white !important;
        font-weight: 600;
        box-shadow: 0 4px 10px rgba(74, 144, 226, 0.4);
    }

    /* 버튼 스타일 */
    div.stButton > button {
        width: 100%;
        background-color: #4a90e2;
        color: white;
        font-size: 1.15rem;
        padding: 14px;
        border-radius: 14px;
        border: none;
        cursor: pointer;
        box-shadow: 0 4px 12px rgba(74, 144, 226, 0.5);
        transition: background-color 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #357abd;
    }

    /* 입력창 */
    textarea, input[type="text"], input[type="password"] {
        font-size: 1.1rem !important;
        padding: 12px !important;
        border-radius: 12px !important;
        border: 1.5px solid #b0bec5 !important;
        width: 100% !important;
        box-sizing: border-box;
        background-color: white;
        color: #1a2935;
        transition: border-color 0.3s ease;
    }
    textarea:focus, input[type="text"]:focus, input[type="password"]:focus {
        border-color: #4a90e2 !important;
        outline: none !important;
    }

    /* 분석 결과 박스 */
    .report {
        background-color: white;
        padding: 18px 22px;
        border-radius: 16px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        font-size: 1rem;
        white-space: pre-wrap;
        color: #34495e;
        margin-top: 15px;
    }

    /* 경고 및 성공 메시지 */
    .stAlert > div {
        border-radius: 12px !important;
        font-weight: 500;
        font-size: 1rem;
    }

    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1>🧠 군 조직 성향 및 자살 위험 종합 분석</h1>", unsafe_allow_html=True)
    st.markdown('<p class="subtitle">최신 AI 감정 모델을 활용한 심층 분석 시스템</p>', unsafe_allow_html=True)
    st.write("---")

    tab1, tab2, tab3 = st.tabs(["🎙️ 음성파일(STT)", "📄 텍스트파일", "📝 복사한 대화 분석"])

    with tab1:
        st.header("음성 파일 업로드")
        audio_file = st.file_uploader("파일 선택 (mp3, wav, m4a, flac)", type=["mp3","wav","m4a","flac"])
        if audio_file is not None:
            with st.spinner("음성 텍스트 변환 중..."):
                transcript = transcribe_audio(audio_file)
            st.subheader("변환된 텍스트")
            st.write(transcript)

            with st.spinner("분석 중..."):
                result = analyze_texts([transcript])
            st.markdown("---")
            st.header("📋 분석 결과")
            st.markdown(f'<div class="report">{generate_report(result)}</div>', unsafe_allow_html=True)

    with tab2:
        st.header("텍스트 파일 업로드")
        text_file = st.file_uploader("텍스트 파일 업로드 (.txt)", type=["txt"])
        if text_file is not None:
            text_content = text_file.read().decode("utf-8")
            with st.spinner("분석 중..."):
                result = analyze_texts([text_content])
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
                with st.spinner("분석 중..."):
                    result = analyze_texts([input_text])
                st.markdown("---")
                st.header("📋 분석 결과")
                st.markdown(f'<div class="report">{generate_report(result)}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
