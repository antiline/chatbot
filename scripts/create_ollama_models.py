from __future__ import annotations

import subprocess

# 분류기·답변 모두 이 단일 로컬 태그를 쓴다(Ollama 에 한 벌만 로드).
BASE_MODEL = "qwen3:30b"          # Ollama 레지스트리에서 직접 pull (HF GGUF 불필요)
MODEL_NAME = "qwen3:30b-64k"      # num_ctx 64K 를 구운 파생 태그
MODELFILE = "models/Modelfile.qwen3-64k"


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> None:
    print(f"[1/2] Pulling base model from Ollama registry: {BASE_MODEL}")
    run(["ollama", "pull", BASE_MODEL])

    print(f"[2/2] Creating derived model (num_ctx 64K): {MODEL_NAME}")
    run(["ollama", "create", MODEL_NAME, "-f", MODELFILE])

    print("Done. Check:")
    run(["ollama", "list"])


if __name__ == "__main__":
    main()
