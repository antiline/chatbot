import os
from typing import Any

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory


classifier_store: dict[str, InMemoryChatMessageHistory] = {}
answer_store: dict[str, InMemoryChatMessageHistory] = {}

# qwen2.5 7B (비추론, 한국어 강함) 을 num_ctx 64K 로 구운 파생 태그 qwen2.5:7b-64k 로 통일.
# 7B 는 32GB 에 가볍게 로드돼(콜드로드 짧음) 상시 상주 없이도 견딜 만하다.
CLASSIFIER_MODEL_NAME = "qwen2.5:7b-64k"
ANSWER_MODEL_NAME = "qwen2.5:7b-64k"

# /api를 붙이지 않습니다.
# 예: http://localhost:11434
# 예: https://ollama.example.com
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


SYSTEM_PROMPT = """
너는 한국어 세무사 AI 챗봇이다.

규칙:
1) 세무(부가세, 종합소득세, 법인세, 원천세, 연말정산, 사업자등록,
   세금계산서, 현금영수증 등) 관련 질문에만 답변한다.
2) 세무와 무관한 질문에는 절대 답변하지 않는다.
3) 세무와 무관한 질문이면 아래 문장만 출력한다.
'세무 관련 질문에만 답변할 수 있습니다.'
4) 이모티콘과 불필요한 특수기호를 사용하지 않는다.
5) 확정적인 신고 및 법률 판단은 세무사 또는 전문가 확인이
   필요하다고 안내한다.
"""

CLASSIFIER_PROMPT = """
너는 사용자 질문을 아래 3가지 라벨 중 하나로 분류하는 한국어 분류기다.

라벨 정의:
- 세무: 현재 질문 자체가 세무/세금 관련 질문
- 비세무: 세무와 직접 관련 없는 질문
- 메타: 이전 대화 내용을 요약/정리/재설명/재구성 요청하는 질문

중요 규칙:
- 반드시 "세무", "비세무", "메타" 중 하나만 출력한다.
- 설명하거나 다른 문장을 출력하지 않는다.
"""


def get_answer_session_history(
    session_id: str,
) -> InMemoryChatMessageHistory:
    if session_id not in answer_store:
        answer_store[session_id] = InMemoryChatMessageHistory()

    return answer_store[session_id]


def get_classifier_session_history(
    session_id: str,
) -> InMemoryChatMessageHistory:
    if session_id not in classifier_store:
        classifier_store[session_id] = InMemoryChatMessageHistory()

    return classifier_store[session_id]


def normalize_label(raw_label: str) -> str:
    raw_label = (raw_label or "").strip()
    compact = raw_label.replace("\n", "").replace(" ", "")

    if compact in ("세무", "비세무", "메타"):
        return compact

    if "비세무" in raw_label:
        return "비세무"

    if "메타" in raw_label:
        return "메타"

    if "세무" in raw_label:
        return "세무"

    return "비세무"


answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

classifier_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", CLASSIFIER_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)


# Vercel 안에서 Ollama를 실행하지 않고 외부 주소에 접속합니다.
# ⚠️ qwen2.5 는 비추론(thinking 미지원) 모델이라 reasoning/think 파라미터를
# 절대 넣지 않는다. langchain reasoning=True/False 는 /api/chat 에 think 를
# 보내는데, 비추론 모델에 think 를 주면 ollama 가 응답 없이 멈춘다(행).
classifier_llm = ChatOllama(
    model=CLASSIFIER_MODEL_NAME,
    base_url=OLLAMA_BASE_URL,
    temperature=0.0,
    num_predict=32,
    repeat_penalty=1.3,
    repeat_last_n=256,
)

answer_llm = ChatOllama(
    model=ANSWER_MODEL_NAME,
    base_url=OLLAMA_BASE_URL,
    temperature=0.0,
)

classifier_chain = classifier_prompt | classifier_llm
answer_chain = answer_prompt | answer_llm

answer_with_history = RunnableWithMessageHistory(
    answer_chain,
    get_answer_session_history,
    input_messages_key="question",
    history_messages_key="history",
)


def classify_question(
    question: str,
    session_id: str,
) -> str:
    history = get_classifier_session_history(session_id)

    result = classifier_chain.invoke(
        {
            "question": question,
            "history": history.messages,
        }
    )

    raw_label = (getattr(result, "content", "") or "").strip()
    label = normalize_label(raw_label)

    if label in ("세무", "메타"):
        history.add_user_message(question)
        history.add_ai_message(label)

    return label


def has_answer_history(session_id: str) -> bool:
    history = answer_store.get(session_id)
    return history is not None and len(history.messages) > 0


def process_question(
    question: str,
    base_session_id: str,
) -> dict[str, Any]:
    question = question.strip()
    base_session_id = base_session_id.strip()

    if not question:
        raise ValueError("질문을 입력해주세요.")

    if not base_session_id:
        raise ValueError("session_id가 필요합니다.")

    classifier_session_id = f"classifier:{base_session_id}"
    answer_session_id = f"answer:{base_session_id}"

    label = classify_question(
        question=question,
        session_id=classifier_session_id,
    )

    if label == "비세무":
        return {
            "label": label,
            "answer": "세무 관련 질문에만 답변할 수 있습니다.",
        }

    if label == "메타" and not has_answer_history(answer_session_id):
        return {
            "label": label,
            "answer": "요약할 이전 세무 상담 내용이 없습니다.",
        }

    if label in ("세무", "메타"):
        response = answer_with_history.invoke(
            {"question": question},
            config={
                "configurable": {
                    "session_id": answer_session_id,
                }
            },
        )

        answer = (getattr(response, "content", "") or "").strip()

        return {
            "label": label,
            "answer": answer,
        }

    return {
        "label": "비세무",
        "answer": "세무 관련 질문에만 답변할 수 있습니다.",
    }