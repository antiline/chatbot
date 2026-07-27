from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

from ollama_utils import ollama_session

classifier_store = {}
answer_store = {}

ANSWER_MODEL_NAME = "qwen2.5:7b-64k"

SYSTEM_PROMPT = """
너는 한국어 세무사 AI 챗봇이다.

규칙:
1) 세무(부가세, 종합소득세, 법인세, 원천세, 연말정산, 사업자등록, 세금계산서, 현금영수증 등) 관련 질문에만 답변한다.
2) 세무와 무관한 질문에는 절대 답변하지 않는다.
3) 세무와 무관한 질문이면 아래 문장만 출력한다.
'세무 관련 질문에만 답변할 수 있습니다.'
4) 이모티콘과 불필요한 특수기호를 사용하지 않는다.
5) 확정적인 신고 및 법률 판단은 세무사 또는 전문가 확인이 필요하다고 안내한다.
"""

CLASSIFIER_PROMPT = """
너는 사용자 질문을 아래 3가지 라벨 중 하나로 분류하는 분류기다.

라벨 정의:
- 세무: 현재 질문 자체가 세무/세금 관련 질문
- 비세무: 세무와 직접 관련 없는 질문
- 메타: 이전 대화 내용을 요약/정리/재설명/재구성 요청하는 질문

분류 기준:
- 세무 관련: 부가세, 종합소득세, 법인세, 원천세, 연말정산, 사업자등록, 세금계산서, 현금영수증, 신고, 납부, 환급, 가산세, 공제, 증빙 등 세금/세무 실무와 직접 관련된 질문
- 비세무 관련: 일반 상식, 코딩, 날씨, 번역, 고민상담, 맛집, 건강, 투자 일반론, 요리 등 세무와 직접 관련 없는 질문
- 메타 관련: 현재 문장 자체에 세무 키워드가 없더라도, 이전 대화를 요약/정리/재설명해 달라는 요청이면 메타로 분류한다.

중요 규칙:
- 세무 관련 질문엔 "세무"라고 답한다.
- 비세무 관련 질문엔 "비세무"라고 답한다.
- 메타 관련 질문엔 "메타"라고 답한다.
- 반드시 아래 셋 중 하나만 출력한다.
세무
비세무
메타
- 설명하지 않는다.
- 다른 문장을 절대 출력하지 않는다.
"""

# 답변기 히스토리
def get_answer_session_history(session_id: str):
    if session_id not in answer_store:
        answer_store[session_id] = InMemoryChatMessageHistory()
    return answer_store[session_id]

# 분류기 히스토리
def get_classifier_session_history(session_id: str):
    if session_id not in classifier_store:
        classifier_store[session_id] = InMemoryChatMessageHistory()
    return classifier_store[session_id]

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

classifier_prompt = ChatPromptTemplate.from_messages([
    ("system", CLASSIFIER_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

# 분류기 실행 함수: 히스토리를 참고하되, '세무'와 '메타'일 때만 히스토리에 저장하는 정책 적용
def classify_question_and_save_conditional(classifier_chain, question: str, session_id: str) -> str:
    history = get_classifier_session_history(session_id)

    result = classifier_chain.invoke({
        "question": question,
        "history": history.messages,  # 수동으로 히스토리 주입
    })

    raw_label = (result.content or "").strip()
    print(f"[분류 원본 출력] {raw_label!r}")

    raw_label = (raw_label or "").strip()
    compact = raw_label.replace("\n", "").replace(" ", "")

    # 정확 일치 우선
    if compact in ("세무", "비세무", "메타"):
        label = compact

    # 방어적 처리
    if "비세무" in raw_label:
        label = "비세무"
    elif "메타" in raw_label:
        label = "메타"
    elif "세무" in raw_label:
        label = "세무"
    else:
        label = "비세무"  # 기본값
    
    # 분류기 히스토리 저장 정책: 세무/메타만 저장, 비세무는 저장 안 함
    if label in ("세무", "메타"):
        history.add_user_message(question)
        history.add_ai_message(label)

    return label

# 답변 히스토리에 이전 대화가 있는지 확인하는 함수
def has_answer_history(session_id: str) -> bool:
    history = answer_store.get(session_id)
    if history is None:
        return False

    return len(history.messages) > 0

def main():
    # Ollama 자동 실행 및 상담 종료 시 자동 종료
    with ollama_session():
        # qwen2.5 는 비추론 모델 — reasoning/think 파라미터를 넣지 않는다(넣으면 행).
        llm = ChatOllama(
            model=ANSWER_MODEL_NAME,
            temperature=0,
        )
        # 분류기
        classifier_chain = classifier_prompt | llm
        
        # 답변기
        answer_chain = answer_prompt | llm
        
        answer_with_history = RunnableWithMessageHistory(
            answer_chain,
            get_answer_session_history,
            input_messages_key="question",
            history_messages_key="history",
        )

        print("세무사 AI Chatbot")
        print("종료하려면 '종료' 또는 '끝' 입력")
        print()

        # 분류기와 답변기 세션을 분리하여 관리
        base_session_id = "user-1"
        classifier_session_id = f"classifier:{base_session_id}"
        answer_session_id = f"answer:{base_session_id}"

        while True:
            user_input = input("상담자: ").strip()

            if not user_input:
                print("AI 세무사: 질문을 입력해주세요.\n")
                continue

            if user_input.lower() in ["종료", "끝"]:
                print("상담을 종료합니다.")
                break

            try:
                # 분류(세무, 비세무, 메타)
                label = classify_question_and_save_conditional(
                    classifier_chain,
                    user_input,
                    classifier_session_id
                )
                print(f"[분류결과] {label}")

                # 비세무면 고정 문구 출력
                if label == "비세무":
                    print("AI 세무사: 세무 관련 질문에만 답변할 수 있습니다.")
                    print()
                    continue

                # 메타 요청 처리
                if label == "메타":
                    # 히스토리가 없으면 다음 문구 출력
                    if not has_answer_history(answer_session_id):
                        print("AI 세무사: 요약할 이전 세무 상담 내용이 없습니다.")
                        print()
                        continue

                    response = answer_with_history.invoke(
                        {"question": user_input},
                        config={"configurable": {"session_id": answer_session_id}}
                    )

                    print("AI 세무사:", (response.content or "").strip())
                    print()
                    continue

                # 세무 질문 처리
                if label == "세무":
                    response = answer_with_history.invoke(
                        {"question": user_input},
                        config={"configurable": {"session_id": answer_session_id}}
                    )

                    print("AI 세무사:", (response.content or "").strip())
                    print()
                    continue

                # 혹시 모를 예외 라벨 방어
                print("AI 세무사: 세무 관련 질문에만 답변할 수 있습니다.")
                print()

            # 디버깅을 위해 예외 메시지 출력 추가 (실제 배포 시에는 제거)
            except Exception as e:
                print("AI 세무사: 일시적인 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")
                print()

if __name__ == "__main__":
    main()
