# STT 파인튜닝 (SenseVoiceSmall)

목표: 웨이크워드 "제리"와 자주 쓰는 명령어를 **내 목소리 + 파이 마이크 환경**에서
확실하게 알아듣게 만든다. 녹음은 파이, 학습은 학교/연구실 GPU 서버, 추론은 파이.

실행 절차는 [TRAINING.md](TRAINING.md) 를 따른다. 이 문서는 데이터가 뭔지와
녹음 방법을 설명한다.

---

## 0. 데이터셋이란 (처음이라면 여기부터)

음성 인식 학습 데이터는 결국 **오디오 파일 + 그 오디오에 뭐라고 말했는지 적은 정답 텍스트**
두 개의 짝일 뿐이다. 이 짝의 목록을 적어둔 파일을 manifest 라고 부른다.

이 프로젝트는 한 줄에 하나씩 JSON 을 쓰는 `manifest.jsonl` 형식을 쓴다:

```json
{"index": 0, "audio_filepath": "wav/0000.wav", "text": "제리", "duration": 0.92, "speaker": "yjhan", "sample_rate": 16000}
{"index": 1, "audio_filepath": "wav/0001.wav", "text": "제리야 오늘 날씨 어때", "duration": 1.84, "speaker": "yjhan", "sample_rate": 16000}
```

오디오 규격은 서비스 환경과 동일하게 맞춘다: **16kHz, 모노, 16bit PCM wav**.
학습 때 본 소리와 실제로 들어올 소리가 다르면 파인튜닝 효과가 사라지기 때문에,
녹음도 **반드시 파이의 ReSpeaker 마이크로** 해야 한다.

디렉터리 구조:

```
training/
  prompts_ko.txt              읽을 문장 목록 (115개, 자유롭게 편집 가능)
  record_dataset.py           녹음 도구 (파이에서 실행)
  fetch_open_data.py          공개 데이터셋 받기 (PC에서 실행)
  data/
    yjhan/                    내가 녹음한 것
      wav/0000.wav ...
      manifest.jsonl
    zeroth/                   공개 데이터
      wav/000000.wav ...
      manifest.jsonl
```

`data/` 는 용량이 크므로 git 에 올리지 않는다(.gitignore 처리됨).

---

## 1. 내 목소리 녹음 (파이에서)

115문장을 3회차까지 녹음해 345발화를 만든다.

```bash
cd ~/voicebot && git pull
python3 training/record_dataset.py --speaker yjhan --passes 3 --fast
```

- 문장이 하나씩 뜬다 → `Enter` 로 녹음 시작 → 읽고 → `Enter` 로 종료 → 자동 저장
- `--fast` 는 재생·확인 단계를 건너뛴다. 대량 녹음할 때 시간이 절반으로 준다.
  레벨 경고가 뜬 파일은 끝에 목록으로 모아서 알려준다.
- `q` 로 언제든 중단해도 되고, 다시 실행하면 **이어서** 진행된다.
- 앞뒤 무음은 자동으로 잘린다.
- 진행 상황만 보려면 `--passes 3 --list`.

**녹음 요령**

- 실제 사용할 때와 같은 위치·같은 목소리 크기로. 너무 또박또박 읽지 말 것.
- **회차마다 톤·속도·마이크 거리를 바꿀 것.** 3번 다 똑같이 읽으면 같은 데이터를
  3장 복사한 것과 다를 게 없어서 학습 효과가 안 난다. 1회차는 평소대로,
  2회차는 조금 빠르게/멀리서, 3회차는 편하게 흘리듯이 — 이런 식으로.
- 평소 쓰는 환경 그대로.

**분량 기준**

| 분량 | 기대 효과 |
|---|---|
| 115발화 (1회차) | 웨이크워드 인식률 개선을 확인할 수 있는 최소선 |
| **345발화 (3회차, 약 45~60분)** | 도메인 특화 효과가 뚜렷해지는 실용 지점 ← **목표** |
| 1000발화 이상 | 더 좋지만 수집 비용 대비 효과는 완만해짐 |

### 평가 전용 데이터 (중요)

학습에 쓴 발화로 성능을 재면 당연히 잘 나온다(외운 것). 진짜 실력을 보려면
**학습에 안 쓴 발화**가 따로 있어야 한다. 화자 이름을 달리해 한 벌 더 받는다:

```bash
python3 training/record_dataset.py --speaker yjhan_eval --passes 1
```

이건 `--fast` 없이 확인하며 천천히 받는 게 좋다. 학습 때
`prepare_funasr_data.py --data data/yjhan` 만 넘기면 `yjhan_eval` 은 자연히 빠진다.

### 잡음 녹음 (증강용)

발화 녹음이 끝나면 배경 소음도 받아둔다. 각 30초씩:

```bash
bash ~/voicebot/scripts/record_noise.sh 선풍기 30
bash ~/voicebot/scripts/record_noise.sh 에어컨 30
bash ~/voicebot/scripts/record_noise.sh 키보드 30
bash ~/voicebot/scripts/record_noise.sh 무음 30
```

녹음 중에는 말하지 말 것. '무음'은 마이크 자체 노이즈를 받는 것이라 꼭 필요하다.

---

## 2. 공개 한국어 데이터 (GPU 서버에서)

내 목소리 데이터만으로 학습하면 모델이 원래 알던 한국어를 잊어버린다
(catastrophic forgetting). 공개 데이터를 3~10배 정도 섞어서 함께 학습한다.

발화 300개 이상 모은 뒤에 하면 된다. 서버에서:

```bash
python training/fetch_open_data.py --list          # 후보 보기
python training/fetch_open_data.py --dataset zeroth --hours 3
```

| 이름 | 규모 | 라이선스 | 성격 |
|---|---|---|---|
| `zeroth` (Bingsu/zeroth-korean) | 51시간 | CC BY 4.0 | 한국어 낭독체. 바로 받아짐. **기본 추천** |
| `fleurs` (google/fleurs ko_kr) | ~10시간 | CC BY 4.0 | 격식체 낭독. 문장이 김 |
| `commonvoice` (CV 17 ko) | 소규모 | CC0 | 화자·마이크 다양. HF 로그인 필요 |
| KsponSpeech (AI Hub) | 1000시간 | 별도 승인 | 한국어 자유대화, 품질 최상. 가입·신청 필요, 수백 GB |

우선 `zeroth 3시간`이면 충분하다. 부족하면 나중에 늘린다.

---

## 3. 학습 실행

학습은 이 PC가 아니라 **학교/연구실 리눅스 GPU 서버**에서 돌린다.
전체 절차(환경 구성 → 학습 → ONNX 변환 → 파이 배포 → 성능 비교)는
[TRAINING.md](TRAINING.md) 에 있다.

| 스크립트 | 실행 위치 | 용도 |
|---|---|---|
| `record_dataset.py` | 파이 | 녹음 |
| `eval_stt.py` | 파이 | 정확도 측정 (학습 전/후 비교) |
| `fetch_open_data.py` | 서버 | 공개 한국어 데이터 받기 |
| `prepare_funasr_data.py` | 서버 | 학습 포맷 변환 + 잡음 증강 |
| `setup_train_env.sh` | 서버 | venv + torch + funasr 설치 |
| `setup_train_env_conda.sh` | 서버 | 같은 것의 conda 판 (환경 `voicebot-train`) |
| `finetune_sensevoice.sh` | 서버 | 파인튜닝 |
| `export_onnx.py` | 서버 | ONNX int8 변환 + 로드 검증 |
