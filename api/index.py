import os
import secrets

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from chatbot_service import process_question


app = FastAPI(
    title="세무 AI 챗봇 API",
    version="1.0.0",
)


# 인터넷에 노출(예: Cloudflare Tunnel)할 때는 CHATBOT_TOKEN 을 설정해 보호한다.
# 미설정(로컬 개발)이면 인증을 건너뛴다.
CHATBOT_TOKEN = os.getenv("CHATBOT_TOKEN", "").strip()


def require_token(authorization: str = Header(default="")) -> None:
    if not CHATBOT_TOKEN:
        return  # 로컬 개발: 토큰 미설정 시 열어둠

    scheme, _, credential = authorization.partition(" ")
    if scheme.lower() != "bearer" or not secrets.compare_digest(credential, CHATBOT_TOKEN):
        raise HTTPException(status_code=401, detail="Unauthorized")


allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://localhost:8000",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=2000,
        description="사용자의 질문",
    )
    session_id: str = Field(
        min_length=1,
        max_length=100,
        description="사용자별 대화 세션 ID",
    )


class ChatResponse(BaseModel):
    label: str
    answer: str


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "세무 AI 챗봇 API",
        "docs": "/docs",
    }


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest, _: None = Depends(require_token)) -> ChatResponse:
    try:
        result = process_question(
            question=request.question,
            base_session_id=request.session_id,
        )

        return ChatResponse(**result)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        # 운영 환경에서는 내부 오류 내용을 그대로 사용자에게 보내지 않습니다.
        print(f"Chatbot error: {exc!r}")

        raise HTTPException(
            status_code=503,
            detail="AI 모델 서버에 연결할 수 없거나 응답 처리 중 오류가 발생했습니다.",
        ) from exc