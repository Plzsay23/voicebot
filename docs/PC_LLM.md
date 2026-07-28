# PC(WSL) GPU LLM 서버

파이는 24시간 켜두고, PC 가 켜져 있을 때만 PC 의 GPU 로 답변을 만든다.
PC 가 꺼져 있으면 파이가 자기 CPU 로 같은 모델을 돌린다(자동, 재시작 불필요).

- 모델: `EXAONE-3.5-2.4B-Instruct-Q4_K_M.gguf` — **파이와 같은 파일**
- 엔진: `llama-cpp-python[server]` (llama.cpp CUDA 빌드) — 파이와 같은 엔진
- 인터페이스: OpenAI 호환 `/v1` (파이의 `REMOTE_LLM_URL` 이 그대로 붙는다)

같은 모델·같은 엔진·같은 시스템 프롬프트를 쓰는 이유는 **PC 로 붙었을 때와
로컬로 떨어졌을 때 답변 성격이 달라지지 않게** 하려는 것이다. 대화 history 가
두 백엔드를 오가도 말투가 튀지 않는다. GPU 로 얻는 건 속도뿐이다
(파이 CPU 2.2 tok/s → GPU 수십 tok/s).

## 설치 (한 번만)

```bash
bash scripts/setup_pc_llm.sh
```

하는 일: `.venv-llm` 생성 → llama-cpp-python 을 CUDA(sm_120, RTX 5060)로 빌드
→ gguf 다운로드 → `~/.bashrc` 에 `chat` 등록.

전제조건은 스크립트가 확인한다: WSL 에 `nvcc`(CUDA 툴킷), `g++`, `python3-dev`.
GPU 가 다르면 `CUDA_ARCH=89 bash scripts/setup_pc_llm.sh` (3060=86, 4060/4090=89).

## 사용

```bash
chat
```

서버가 없으면 띄우고(백그라운드) 바로 터미널 대화창이 열린다. 서버는 창을 닫아도
남아 있어서 파이가 계속 쓸 수 있다.

`chat` 은 **볼트 검색 서버(:8081)도 같이 띄운다** — 옵시디언 리서치위키를 제리가
참고하게 하는 것으로, 이것도 "PC 켜져 있을 때만"이라 생명주기를 맞췄다.
자세한 건 [VAULT_RAG.md](VAULT_RAG.md). 끄려면 `PC_LLM_NO_VAULT=1 chat`.

| 명령 | 하는 일 |
|---|---|
| `chat` | 서버 확인/기동 + 대화창 |
| `chat --serve-only` | 서버만 띄운다 |
| `chat --stop` | 서버 종료 |
| `chat --log` | 서버 로그 따라가기 |

환경변수로 바꿀 수 있는 것: `PC_LLM_PORT`(8080), `PC_LLM_CTX`(4096),
`PC_LLM_GPU_LAYERS`(-1 = 전 레이어 GPU).

**GPU 메모리**: 5060 Laptop 은 8GB 인데 브라우저·게임이 이미 6GB 쯤 쓰고 있을 때가
많다. 그러면 서버 기동이 실패한다. 그때는 레이어를 나눠 올린다:

```bash
PC_LLM_GPU_LAYERS=20 chat
```

**게임 중이면 느려진다.** 2026-07-27 실측: 게임이 VRAM 5.7GB·GPU 91% 를 쓰는 동안
6~7 tok/s 까지 떨어졌다. 모델은 1.8GB 로 전부 VRAM 에 올라가 있었으니(스필 아님)
순수한 연산 경합이다. 느리다고 빌드를 의심하기 전에 `nvidia-smi` 로 util 을 보라.

대화창은 `ros_nodes/voice_common.py` 를 그대로 import 한다. 즉 이 창에서 잘 나오면
파이에서도 같은 답이 나온다. 프롬프트를 손볼 때 여기서 먼저 확인하면 된다.

## 파이에서 PC 로 닿게 하기

**여기가 이 구성의 유일한 관문이다.** WSL2 는 기본이 NAT 라서, WSL 안에서
`0.0.0.0:8080` 에 붙여도 파이에서 보이지 않는다. 두 가지 중 하나를 하면 된다.

### 방법 1 — mirrored 네트워킹 (권장, 한 번만)

`C:\Users\yjhan\.wslconfig` 를 이미 만들어 두었다(`networkingMode=mirrored`).
적용하려면 **WSL 을 한 번 내려야 한다** — 열려 있는 WSL 터미널이 다 닫힌다:

```powershell
wsl --shutdown
```

그리고 관리자 PowerShell 에서 인바운드 8080 을 한 번 열어준다
(방화벽 규칙 추가는 관리자 권한이 필요하다):

```powershell
New-NetFirewallRule -DisplayName "voicebot LLM 8080" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8080 -Profile Private
```

`-Profile Private` 이므로 집/연구실 사설망에서만 열린다. 공용 네트워크에서는 닫힌다.

### 방법 2 — portproxy (mirrored 가 안 될 때)

Docker Desktop 등과 충돌해서 NAT 로 되돌렸다면, WSL 의 IP 로 포워딩을 걸어야 한다.
**WSL IP 는 재시작마다 바뀌므로 그때마다 다시 걸어야 한다.** 관리자 PowerShell:

```powershell
netsh interface portproxy add v4tov4 listenport=8080 listenaddress=0.0.0.0 connectport=8080 connectaddress=(wsl hostname -I).Trim()
```

### 확인

`chat` 이 마지막에 파이에 넣을 주소를 찍어준다. 지금 PC 의 LAN 주소는
`192.168.0.41` 이므로, 파이 `~/voicebot/.env` 에:

```
REMOTE_LLM_URL=http://192.168.0.41:8080/v1
```

파이에서 확인:

```bash
curl -s http://192.168.0.41:8080/v1/models
```

모델 목록이 나오면 끝이다. 파이의 dialog_node 는 발화마다(캐시 10초) 이 주소를
확인하고, 응답이 없으면 조용히 로컬로 떨어진다. `[답변완료 ... , remote]` /
`, local]` 로그로 어느 쪽이 답했는지 알 수 있다.

**PC IP 가 바뀌면** 조용히 로컬로 떨어진다. 공유기에서 이 PC 에 IP 를 고정
할당해 두는 게 좋다. Tailscale 을 쓰면 `100.126.123.94` 같은 tailnet 주소를
써도 된다(공유기 밖에서도 붙는다).

## 자동 시작을 원하면

Windows 작업 스케줄러에 로그온 트리거로 이걸 걸면 PC 를 켤 때마다 서버가 뜬다:

```
wsl.exe -e bash -lc "bash /mnt/c/ysj/voicebot/scripts/pc_llm_chat.sh --serve-only"
```

모델이 GPU 에 상주하므로 그만큼 VRAM(약 1.7GB)을 계속 먹는다. 게임을 자주 한다면
자동 시작은 걸지 말고 필요할 때 `chat` 을 치는 편이 낫다.
