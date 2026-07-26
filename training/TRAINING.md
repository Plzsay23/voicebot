# SenseVoice 파인튜닝 실행 안내 (리눅스 GPU 서버)

학습은 학교/연구실 GPU 서버에서, 녹음은 파이에서, 배포는 다시 파이로.
이 문서만 위에서부터 따라 하면 된다. 서버에 sudo 권한이 없어도 된다.

## 이 작업이 필요한 이유 (기준점 측정 결과, 30발화)

| 지표 | 현재 값 |
|---|---|
| CER | 0.096 |
| 완전일치 | 17/30 (56.7%) |
| 웨이크워드 인식 | 28/30 (93.3%) |
| 평균 추론 | 0.71초 |

오류 13개 중 **11개가 웨이크워드 "제리"** 였다. "제야"(7회), "재이", "젤리",
"젤이", "데야" 로 잘못 들었다. 반면 "삼성전자", "비트코인 시세", "무슨 요일이야"
같은 일반 어휘는 거의 다 맞혔다. 즉 모델이 한국어를 못하는 게 아니라 **"제리"라는
고유명사만 모른다.** 파인튜닝으로 고치기에 딱 맞는 문제다.

---

## 0. 준비물

| 항목 | 위치 |
|---|---|
| 녹음 데이터 | 파이 `~/voicebot/training/data/yjhan/` |
| 학습 스크립트 | 이 저장소 `training/` |
| GPU | VRAM 8GB 이상이면 충분 (SenseVoiceSmall 은 234M 파라미터) |
| 디스크 | 약 15GB (torch 3GB + 모델 1GB + 여유) |

---

## 1. 서버에 코드와 데이터 올리기

서버에서:

```bash
git clone https://github.com/Plzsay23/voicebot.git ~/voicebot
cd ~/voicebot
```

데이터는 git 에 없으므로(용량 때문에 gitignore) 따로 옮긴다.
파이에서 직접 보내는 게 가장 간단하다 — **파이에서** 실행:

```bash
scp -r ~/voicebot/training/data/yjhan <서버계정>@<서버주소>:~/voicebot/training/data/
```

서버가 외부에서 안 보이면 파이 → 내 PC → 서버 순으로 두 번 나눠 보내면 된다.

확인 (서버에서):

```bash
wc -l ~/voicebot/training/data/yjhan/manifest.jsonl && ls ~/voicebot/training/data/yjhan/wav | wc -l
```

두 숫자가 같아야 한다.

---

## 2. 학습 환경 구성

GPU 노드에서 실행한다. 로그인 노드에 GPU가 없는 클러스터라면 먼저 잡을
할당받을 것(`srun --gres=gpu:1 --pty bash` 등, 학교마다 다름).

```bash
cd ~/voicebot/training && bash setup_train_env.sh
```

- 파이썬 3.9~3.11 을 자동으로 찾는다. 없으면 `module avail python` 으로 찾아
  `module load` 한 뒤 다시 실행한다.
- 드라이버 버전을 보고 torch 빌드(cu118/cu121/cu128)를 자동 선택한다.
  잘못 골랐으면 `CUDA_TAG=cu121 bash setup_train_env.sh` 로 지정한다.
- 마지막에 **GPU 연산 테스트까지 통과**해야 넘어간다. 여기서 실패하면
  torch 빌드와 GPU가 안 맞는 것이니 CUDA_TAG 를 바꿔서 재설치한다.

---

## 3. 학습 데이터 만들기

```bash
cd ~/voicebot/training
source venv/bin/activate
python prepare_funasr_data.py --data data/yjhan
```

`funasr_data/train.jsonl`, `val.jsonl` 이 생긴다.

### 공개 한국어 데이터 섞기 (발화 300개 이상 모은 뒤에)

내 목소리만으로 학습하면 모델이 원래 알던 한국어를 잊는다(catastrophic
forgetting). 데이터가 충분해지면 이렇게 섞는다:

```bash
python fetch_open_data.py --dataset zeroth --hours 3
python prepare_funasr_data.py --data data/yjhan data/zeroth --repeat 5
```

`--repeat 5` 는 내 목소리 데이터를 5배 복제해 비중을 키운다. 공개 데이터가
훨씬 많으므로 이렇게 균형을 맞춘다.

### 잡음 증강 (선택)

파이4에서 DeepFilterNet 실시간 처리가 불가능하다고 판명됐으므로(RTF 1.0~1.7,
플러그인이 패닉을 내며 마이크가 죽는다), 잡음 대응은 학습 단계에서 한다.
파이에서 선풍기·에어컨·키보드 소리를 몇 분 녹음해 `data/noise/*.wav` 로 두면:

```bash
python prepare_funasr_data.py --data data/yjhan --noise-dir data/noise --noise-copies 2
```

발화마다 SNR 5/10/15/20dB 중 하나로 잡음을 섞은 버전을 2개씩 추가한다.
파이 CPU는 전혀 안 쓴다.

---

## 4. 파인튜닝

```bash
bash finetune_sensevoice.sh
```

기본값은 20 epoch, lr 5e-5, GPU 0번. 바꾸려면:

```bash
EPOCH=30 LR=0.0001 GPUS=0 bash finetune_sensevoice.sh
```

- 첫 실행 때 원본 SenseVoiceSmall(약 1GB)을 받으므로 시간이 걸린다.
- 로그는 `outputs/train.log`, 체크포인트는 `outputs/model.pt.ep*`.
- 세션이 끊길 수 있으면 `tmux new -s train` 안에서 돌릴 것.

> **30발화로 돌릴 때 주의**: 이 단계는 "학습 → export → 파이 배포" 경로가
> 뚫리는지 확인하는 용도다. 나온 성능 수치는 믿지 말 것. 데이터가 적으면
> 학습 데이터만 외워버려서(과적합) 나머지는 오히려 나빠진다. 경로가 확인되면
> 300발화 이상 녹음한 뒤 다시 돌린다.

---

## 5. ONNX 변환 (여기가 제일 위험한 구간)

파이는 파이토치가 아니라 `funasr_onnx` 로 돌아간다. 학습한 모델을 ONNX 로
다시 뽑고 int8 양자화까지 성공해야 배포가 된다. **이게 안 되면 아무리 잘
학습해도 파이에 못 올린다.**

```bash
python export_onnx.py --model-dir outputs --test-wav data/yjhan/wav/0000.wav
```

이 스크립트는 4단계를 거친다:

1. ONNX export
2. `config.yaml` / `am.mvn` / bpe 모델 등 부속 파일 복사 —
   이게 빠지면 파이에서 로드 자체가 실패한다
3. int8 양자화 (기본 `matmul`: MatMul 연산만. 현재 파이에 배포된 것과 같은
   방식이고 Cortex-A72 에서 안정적이다)
4. `funasr_onnx` 로 실제 로드 + 전사 테스트

**4단계까지 통과해야 성공이다.** 여기서 통과하면 파이에서도 뜬다.
실패하면 그 메시지를 그대로 들고 오면 된다.

결과물: `outputs/sensevoice_ko_ft/`

---

## 6. 파이에 배포

서버에서 파이로 직접 못 보내면 PC를 거친다.

```bash
scp -r outputs/sensevoice_ko_ft <파이계정>@<파이주소>:~/voicebot/models/
```

**기존 모델을 덮어쓰지 않는다.** 새 폴더로 두고 환경변수로 갈아 끼우면
언제든 되돌릴 수 있다. 파이에서 `~/voicebot/.env` 에 한 줄 추가:

```
SENSEVOICE_DIR=/home/chatbot/voicebot/models/sensevoice_ko_ft
```

되돌리려면 이 줄을 지우거나 주석 처리하면 끝이다.

---

## 7. 좋아졌는지 확인

파이에서:

```bash
python3 training/eval_stt.py --compare training/baseline.json
```

학습 전 대비 CER / 완전일치 / 웨이크워드 인식률 / 추론시간 변화가 표로 나온다.

**볼 것 두 가지:**

- **웨이크워드 인식률** — 이게 올라가야 목적 달성이다.
- **추론시간** — 0.71초에서 크게 늘면 안 된다. 양자화가 제대로 안 된 것이니
  `--quantize` 방식을 바꿔 다시 export 한다.

주의: 학습에 쓴 데이터로 평가하면 당연히 좋게 나온다(외운 것). 제대로 보려면
**평가용으로 따로 녹음한 발화**로 재야 한다. 300발화 녹음할 때 50개 정도는
학습에 넣지 말고 평가 전용으로 빼둘 것.

---

## 문제가 생기면

| 증상 | 확인할 것 |
|---|---|
| `setup_train_env.sh` 에서 파이썬 못 찾음 | `module avail python` 후 `module load` |
| GPU 연산 테스트 실패 | `CUDA_TAG` 를 바꿔 재설치 (cu118/cu121/cu128) |
| 학습 중 CUDA out of memory | `BATCH=2000 bash finetune_sensevoice.sh` |
| export 후 로드 실패 | 부속 파일 누락. 2단계 출력의 `[!] 못 찾은 파일` 확인 |
| 파이에서 추론이 느려짐 | `--quantize matmul` 로 다시 export |
| 성능이 오히려 나빠짐 | 과적합. 데이터를 늘리거나 EPOCH/LR 을 낮출 것 |
