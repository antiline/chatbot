# 맥미니 배포 (SemuFlow 챗봇 백엔드)

호스트 포트를 하나도 열지 않고 Cloudflare Tunnel(`llm.semuflow.com`)로만 노출하는
docker-compose 배포. 무거운 LLM(Ollama)은 호스트에서 재사용한다.

```
docker compose (호스트 포트 발행 0):
  chatbot(FastAPI)  내부 8000만 (ports 발행 안 함)
  cloudflared ──→ chatbot:8000 (내부망) ──→ llm.semuflow.com (아웃바운드 터널)
  chatbot ──→ host.docker.internal:11434 (호스트 Ollama)
```

터널은 **대시보드 관리형** — 커넥터는 `TUNNEL_TOKEN` 하나로 실행하고,
라우팅(ingress)은 Cloudflare 대시보드의 Public Hostname 으로 관리한다.

## 사전 준비 (호스트, 1회)
1. Ollama 실행(brew) + 모델 등록
   ```
   python scripts/create_ollama_models.py   # ollama pull qwen2.5:7b → qwen2.5:7b-64k 생성
   ```
2. Cloudflare Zero Trust → Networks → Tunnels → **Create a tunnel**(Cloudflared)
   - 설치 화면의 **토큰(eyJ…)** 을 `.env` 의 `TUNNEL_TOKEN` 에 넣는다
   - **Public Hostname 추가**: `llm.semuflow.com` → Service **HTTP** `chatbot:8000`
     (semuflow.com 이 같은 계정에 있으므로 DNS CNAME 은 자동 생성됨)

## 비밀/설정 주입 (`.env` 하나)
```
cp .env.example .env
```
- `CHATBOT_TOKEN` : 앱 인증 토큰 (Vercel 의 CHATBOT_TOKEN 과 동일 값)
- `TUNNEL_TOKEN`  : 위 터널 커넥터 토큰
- `.env` 는 `.gitignore` 로 커밋되지 않음

## 실행
```
docker compose up -d --build
```
- `docker-compose.yml`(범용 챗봇) + `docker-compose.override.yml`(이 배포: 포트 제거 + cloudflared) 자동 병합
- 확인: `curl https://llm.semuflow.com/api/health` → `{"status":"ok"}`

## 재부팅 자동복구
- 컨테이너는 `restart: unless-stopped`
- Docker Desktop(macOS)은 GUI 로그인 필요 → **자동 로그인 + "로그인 시 Docker 시작"** 켜둘 것
- Ollama 는 `brew services start ollama`
