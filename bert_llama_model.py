import os
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

from ollama_utils import ollama_session

classifier_store = {}
answer_store = {}

CLASSIFIER_BERT_MODEL_DIR = "./model/kobert_tax_classifier"

ANSWER_MODEL_NAME = "qwen2.5:7b-64k"      # 세무 답변 생성 모델

ID2LABEL = {
    0: "비세무",
    1: "세무",
    2: "메타",
}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

BERT_MAX_LENGTH = 256
BERT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

# 분류기 라벨 정규화 함수 (세무/비세무/메타 중 하나로 변환)
def normalize_label(raw_label: str) -> str:
    """
    분류기 출력이 약간 흔들려도 세무/비세무/메타 중 하나로 정규화
    """
    raw_label = (raw_label or "").strip()
    compact = raw_label.replace("\n", "").replace(" ", "")

    # 정확 일치 우선
    if compact in ("세무", "비세무", "메타"):
        return compact

    # 방어적 처리
    if "비세무" in raw_label:
        return "비세무"
    elif "메타" in raw_label:
        return "메타"
    elif "세무" in raw_label:
        return "세무"

    # 기본값
    return "비세무"

class BertTaxMetaClassifier:
    def __init__(self, model_dir: str, max_length: int = 256):
        self.model_dir = model_dir
        self.max_length = max_length
        self.device = BERT_DEVICE

        # 저장된 토크나이저 로드 시도 -> 실패 시 원본 KoBERT fallback
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_dir,
                trust_remote_code=True
            )
            print(f"[분류기] tokenizer 로드 성공: {model_dir}")
        except Exception as e:
            print(f"[분류기] 로컬 tokenizer 로드 실패: {e}")
            print("[분류기] monologg/kobert tokenizer로 fallback")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "monologg/kobert",
                trust_remote_code=True
            )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            trust_remote_code=True
        )
        self.model.to(self.device)
        self.model.eval()

        print(f"[분류기] BERT model 로드 완료: {model_dir}")
        print(f"[분류기] device: {self.device}")

    @torch.no_grad()
    def predict(self, text: str):
        text = (text or "").strip()

        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0].detach().cpu()

        pred_id = int(torch.argmax(probs).item())
        pred_label = ID2LABEL[pred_id]

        prob_dict = {
            ID2LABEL[i]: float(probs[i].item()) for i in range(len(ID2LABEL))
        }

        return {
            "label": pred_label,
            "confidence": float(probs[pred_id].item()),
            "probabilities": prob_dict,
        }


answer_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

def classify_question_and_save_conditional(
    bert_classifier: BertTaxMetaClassifier,
    question: str,
    session_id: str
) -> str:
    history = get_classifier_session_history(session_id)

    # 현재 질문만 BERT 분류 (텍스트 분류기는 history 직접 입력 안 씀)
    result = bert_classifier.predict(question)

    raw_label = result["label"]
    label = normalize_label(raw_label)

    # 디버깅 출력 (원하면 제거 가능)
    probs = result["probabilities"]
    print(
        f"[분류결과 상세] label={label}, "
        f"confidence={result['confidence']:.4f}, "
        f"probs={{비세무:{probs['비세무']:.4f}, 세무:{probs['세무']:.4f}, 메타:{probs['메타']:.4f}}}"
    )

    # 분류기 히스토리 저장 정책: 세무/메타만 저장
    # (지금 BERT 분류 자체는 히스토리를 안 쓰지만, 추후 규칙 혼합용으로 남겨둠)
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
    with ollama_session():
        bert_classifier = BertTaxMetaClassifier(
            model_dir=CLASSIFIER_BERT_MODEL_DIR,
            max_length=BERT_MAX_LENGTH
        )
        # qwen2.5 는 비추론 모델 — reasoning/think 파라미터를 넣지 않는다(넣으면 행).
        answer_llm = ChatOllama(
            model=ANSWER_MODEL_NAME,
            temperature=0,
        )
        
        # 답변기
        answer_chain = answer_prompt | answer_llm
        
        answer_with_history = RunnableWithMessageHistory(
            answer_chain,
            get_answer_session_history,
            input_messages_key="question",
            history_messages_key="history",
        )

        print("세무사 AI Chatbot (분류기/답변기 2모델 분리)")
        print(f"- 분류기 모델: {CLASSIFIER_BERT_MODEL_DIR}")
        print(f"- 답변기 모델: {ANSWER_MODEL_NAME} (GPU)")
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
                    bert_classifier,
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
