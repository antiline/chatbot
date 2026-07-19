# 맥미니 배포 (SemuFlow 챗봇 백엔드)

호스트 포트를 열지 않고 Cloudflare Tunnel(`llm.semuflow.com`)로만 노출하는
docker-compose 배포. 무거운 LLM(Ollama)은 호스트에서 재사용한다.

```
docker compose (호스트 포트 발행 0):
  chatbot(FastAPI)  내부 8000만
  cloudflared ──→ chatbot:8000 (내부망) ──→ llm.semuflow.com (아웃바운드 터널)
  chatbot ──→ host.docker.internal:11434 (호스트 Ollama)
```

## 사전 준비 (호스트, 1회)
1. Ollama 실행 + 모델 등록
   ```
   python scripts/download_models.py
   python scripts/create_ollama_models.py   # bllossom_3b_classifier:q4km, bllossom_8b_tax_answer:q4km
   ```
2. Cloudflare Tunnel 생성 (없으면)
   ```
   cloudflared tunnel create semuflow-chatbot
   ```
   - 생성된 credentials JSON 을 `secrets/tunnel-creds.json` 으로 복사 (git 제외)
   - `llm.semuflow.com` 은 Cloudflare 대시보드에서 CNAME → `<TUNNEL_ID>.cfargotunnel.com` (프록시 ON)
     ※ semuflow.com 이 cloudflared 계정과 다른 계정이면 `route dns` 대신 수동 CNAME

## 비밀/설정 주입
- `cp .env.example .env` 후 `CHATBOT_TOKEN` 채우기 (Vercel 의 CHATBOT_TOKEN 과 동일 값)
- `secrets/tunnel-creds.json` 배치
- 둘 다 `.gitignore` 로 커밋되지 않음

## 실행
```
docker compose up -d --build
```
- `docker-compose.yml`(범용) + `docker-compose.override.yml`(이 배포: 포트 제거 + cloudflared) 자동 병합
- 확인: `curl https://llm.semuflow.com/api/health` → `{"status":"ok"}`

## 재부팅 자동복구
- 컨테이너는 `restart: unless-stopped`
- Docker Desktop(macOS)은 GUI 로그인 필요 → **자동 로그인 + "로그인 시 Docker 시작"** 켜둘 것
- Ollama 는 `brew services start ollama`
