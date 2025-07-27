import streamlit as st

def main():
    st.title("간단한 이름 환영 앱") # 앱의 제목을 설정합니다.

    st.write("아래에 이름을 입력하고 엔터를 눌러주세요.") # 사용자에게 지시사항을 보여줍니다.

    # 텍스트 입력 위젯을 생성합니다.
    user_name = st.text_input("당신의 이름은 무엇인가요?", "여기에 이름을 입력하세요")

    # 이름이 입력되면 환영 메시지를 표시합니다.
    if user_name and user_name != "여기에 이름을 입력하세요": # 기본 텍스트가 아닌 실제 이름이 입력되었는지 확인
        st.success(f"안녕하세요, {user_name}님! Streamlit 앱에 오신 것을 환영합니다.")
    else:
        st.info("이름을 입력해주세요.")

if __name__ == "__main__":
    main()