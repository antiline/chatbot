# 🧾 Tax AI Chatbot (LangChain + Ollama)

세무 관련 질문만 답변하는 개인 프로젝트용 AI 챗봇입니다.

- 분류기(세무/비세무/메타)와 답변기를 두는 구조이며, 두 역할 모두
  로컬 Ollama 모델 하나(`qwen3:30b-64k`)를 공유합니다.
- 이 태그는 Hermes GPT fallback 과도 공유하므로 32GB 머신에서 모델이
  한 벌만 로드됩니다.

## 모델
- 베이스: `qwen3:30b` (Qwen3 30B-A3B MoE, 활성 3B라 빠름 / 도구·긴 컨텍스트 지원)
- 파생 태그: `qwen3:30b-64k` = `qwen3:30b` + `num_ctx 65536`
  (서비스 특성상 최소 64K 컨텍스트 필요 — `models/Modelfile.qwen3-64k`)

> qwen3 는 사고과정을 내보내므로 코드에서 `reasoning=True` 로 호출해
> 사고과정을 `content` 밖으로 분리한다(그래야 분류 라벨·답변이 깨끗함).

---

## 준비물
- Python 3.10+
- Ollama 설치 및 실행 중 (https://ollama.com/download)
- 30B 모델 로드에 RAM/통합메모리 여유 필요 (로드 시 ~22GB)

## Ollama 모델 준비
    python scripts/create_ollama_models.py

    → `ollama pull qwen3:30b` 후 `qwen3:30b-64k` 파생 태그를 생성합니다.
      완료되면 `ollama list` 에 `qwen3:30b-64k` 가 보입니다.

## 실행
    python main.py        # 대화형 메뉴 (single / 2모델 / BERT+LLM)
    # 또는 API 서버
    fastapi run api/index.py --host 0.0.0.0 --port 8000
