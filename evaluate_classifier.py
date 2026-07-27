import pandas as pd
from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix,
    accuracy_score
)

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# =========================
# 설정
# =========================
CLASSIFIER_MODEL_NAME = "qwen2.5:7b-64k"

CSV_PATH = "valid.csv"   # 평가용 CSV 경로
QUESTION_COL = "text"    # 질문 컬럼명
LABEL_COL = "label"          # 정답 라벨 컬럼명 (0/1/2)

# 라벨 순서 고정 (리포트/혼동행렬 순서)
LABELS = ["비세무", "세무", "메타"]

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

# =========================
# 모델 출력 라벨 정규화
# =========================
def normalize_pred_label(raw_label: str) -> str:
    raw_label = (raw_label or "").strip()
    compact = raw_label.replace("\n", "").replace(" ", "")

    # 정확 일치
    if compact in ("세무", "비세무", "메타"):
        return compact

    # 방어적 처리 (비세무 먼저)
    if "비세무" in raw_label:
        return "비세무"
    elif "메타" in raw_label:
        return "메타"
    elif "세무" in raw_label:
        return "세무"

    # 이상 출력 기본값
    return "비세무"


# =========================
# CSV 정답 라벨 정규화 (0/1/2)
# 0=비세무, 1=세무, 2=메타
# =========================
def normalize_gold_label(label) -> str:
    """
    CSV의 label 컬럼이 숫자(0/1/2)일 때 문자열 라벨로 변환
    """
    # pandas에서 0.0처럼 들어오는 경우도 있어서 float->int 처리
    try:
        x = int(float(label))
    except Exception:
        raise ValueError(f"정답 라벨이 숫자(0/1/2)가 아닙니다: {label!r}")

    mapping = {
        0: "비세무",
        1: "세무",
        2: "메타",
    }
    if x not in mapping:
        raise ValueError(f"허용되지 않은 정답 라벨 값: {x} (허용: 0/1/2)")
    return mapping[x]


# =========================
# 분류기 체인 생성
# =========================
def build_classifier_chain():
    # qwen2.5 는 비추론 모델 — reasoning/think 파라미터를 넣지 않는다(넣으면 행).
    llm = ChatOllama(
        model=CLASSIFIER_MODEL_NAME,
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", CLASSIFIER_PROMPT),
        ("human", "{question}")
    ])

    return prompt | llm


# =========================
# 단건 예측
# =========================
def predict_label(classifier_chain, question: str) -> str:
    result = classifier_chain.invoke({"question": question})
    raw_label = (getattr(result, "content", "") or "").strip()
    pred = normalize_pred_label(raw_label)
    return pred


# =========================
# 평가
# =========================
def evaluate_classifier(csv_path: str):
    df = pd.read_csv(csv_path)

    # 필수 컬럼 체크
    for col in [QUESTION_COL, LABEL_COL]:
        if col not in df.columns:
            raise ValueError(f"CSV에 '{col}' 컬럼이 없습니다. 현재 컬럼: {list(df.columns)}")

    # 결측 제거
    df = df.dropna(subset=[QUESTION_COL, LABEL_COL]).copy()

    # 정답 라벨 변환
    df["gold"] = df[LABEL_COL].apply(normalize_gold_label)

    classifier_chain = build_classifier_chain()

    preds = []
    total = len(df)

    for idx, row in enumerate(df.itertuples(index=False), start=1):
        q = str(getattr(row, QUESTION_COL)).strip()
        gold = getattr(row, "gold")

        pred = predict_label(classifier_chain, q)
        preds.append(pred)

        # 진행 로그
        print(f"[{idx}/{total}] pred={pred}, gold={gold} | q={q[:80]!r}")

    df["pred"] = preds

    y_true = df["gold"].tolist()
    y_pred = df["pred"].tolist()

    # 지표 계산
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)
    f1_micro = f1_score(y_true, y_pred, labels=LABELS, average="micro", zero_division=0)

    print("\n" + "=" * 60)
    print("평가 결과")
    print("=" * 60)
    print(f"Accuracy      : {acc:.4f}")
    print(f"F1 (macro)    : {f1_macro:.4f}")
    print(f"F1 (weighted) : {f1_weighted:.4f}")
    print(f"F1 (micro)    : {f1_micro:.4f}")

    print("\n[분류 리포트]")
    print(classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        target_names=LABELS,
        digits=4,
        zero_division=0
    ))

    print("[혼동행렬] (행=정답, 열=예측)")
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{x}" for x in LABELS],
        columns=[f"pred_{x}" for x in LABELS]
    )
    print(cm_df)

    # 오답 샘플 저장
    wrong_df = df[df["gold"] != df["pred"]].copy()
    if not wrong_df.empty:
        wrong_path = "misclassified_samples.csv"
        wrong_df.to_csv(wrong_path, index=False, encoding="utf-8-sig")
        print(f"\n오답 샘플 저장 완료: {wrong_path} ({len(wrong_df)}건)")
    else:
        print("\n오답 샘플 없음 (전부 정답)")

    # 전체 결과 저장
    out_path = "eval_predictions.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"전체 예측 결과 저장 완료: {out_path}")


if __name__ == "__main__":
    evaluate_classifier(CSV_PATH)