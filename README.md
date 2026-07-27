# 🧾 Tax AI Chatbot (LangChain + Ollama)

세무 관련 질문만 답변하는 개인 프로젝트용 AI 챗봇입니다.

- 분류기(세무/비세무/메타)와 답변기를 두는 구조이며, 두 역할 모두
  로컬 Ollama 모델 하나(`qwen2.5:7b-64k`)를 공유합니다.
- 32GB 머신에서 모델이 한 벌만 로드되도록 두 역할이 같은 태그를 씁니다.

## 모델
- 베이스: `qwen2.5:7b` (비추론, 한국어 강함 / 32GB 에 가볍게 로드)
- 파생 태그: `qwen2.5:7b-64k` = `qwen2.5:7b` + `num_ctx 65536`
  (서비스 특성상 최소 64K 컨텍스트 필요 — `models/Modelfile.qwen3-64k`)

> ⚠️ qwen2.5 는 비추론 모델이므로 `reasoning`/`think` 파라미터를 넣지 않는다.
> 비추론 모델에 think 를 주면 Ollama 가 응답 없이 멈춘다(행).

---

## 준비물
- Python 3.10+
- Ollama 설치 및 실행 중 (https://ollama.com/download)
- 7B 모델 로드에 RAM/통합메모리 여유 필요 (로드 시 ~8GB)

## Ollama 모델 준비
    python scripts/create_ollama_models.py

    → `ollama pull qwen2.5:7b` 후 `qwen2.5:7b-64k` 파생 태그를 생성합니다.
      완료되면 `ollama list` 에 `qwen2.5:7b-64k` 가 보입니다.

## 실행
    python main.py        # 대화형 메뉴 (single / 2모델 / BERT+LLM)
    # 또는 API 서버
    fastapi run api/index.py --host 0.0.0.0 --port 8000
