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
  prompts_domain.txt          손으로 쓴 도메인 문장 뱅크 (웨이크워드/명령/유사음)
  build_prompts.py            대본 생성 (PC/서버에서 실행)
  prompts_ko.txt              ← 자동 생성된 학습용 대본
  prompts_eval.txt            ← 자동 생성된 평가용 홀드아웃 (학습과 안 겹침)
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

## 1. 녹음 대본 만들기 (PC/서버에서)

문장을 손으로 "다양하게" 고르면 반드시 발음이 편중된다. 대신 공개 한국어
코퍼스에서 **음소 커버리지를 최대화하는 문장을 탐욕적으로 골라낸다**(greedy set
cover). TTS/ASR 대본 설계의 표준 기법이다. 커버리지 단위는 한글을 자모로 쪼갠

| 단위 | 뜻 | 왜 |
|---|---|---|
| CV | 초성+중성 | 음절 시작 소리 |
| VC | 중성+종성 | 받침 |
| BD | 앞음절 종성 + 다음음절 초성 | 한국어 음운변동(비음화·유음화·경음화)이 일어나는 자리 |

도메인 문장(`prompts_domain.txt`)은 무조건 전부 넣고, **그것들이 못 덮은 구멍만**
코퍼스 문장으로 메운다. 그래서 같은 분량이라도 손으로 쓴 대본보다 촘촘하다.

```bash
python3 -m venv .venv-tools && .venv-tools/bin/pip install "datasets>=2.19,<4"
.venv-tools/bin/python training/build_prompts.py --count 500 --source zeroth
```

- `--dry-run` 을 붙이면 커버리지 리포트만 보고 파일은 안 쓴다.
- 네트워크가 없으면 `--count 0` — 도메인 뱅크만으로 대본이 나온다.
- `fleurs`/`commonvoice` 는 스트리밍이어도 오디오 tar 를 통째로 받아 아주 느리다.
  `zeroth`(parquet, CC BY 4.0) 하나로 충분하다.
- **문장을 늘리고 싶으면 `prompts_ko.txt` 가 아니라 `prompts_domain.txt` 를 고친다.**
  `prompts_ko.txt` 는 생성물이라 다시 만들면 덮인다.

평가용 홀드아웃(`prompts_eval.txt`)이 자동으로 분리돼 나온다. 지난번엔 이게 없어서
학습에 쓴 문장으로 성능을 재고 "CER 0.008" 같은 무의미한 수치를 얻었다.

## 2. 내 목소리 녹음 (파이에서)

```bash
cd ~/voicebot && git pull
python3 training/record_dataset.py --speaker yjhan2 --passes 2 --fast
```

- 문장이 하나씩 뜬다 → `Enter` 로 녹음 시작 → 읽고 → `Enter` 로 종료 → 자동 저장
- `--fast` 는 재생·확인 단계를 건너뛴다. 대량 녹음할 때 시간이 절반으로 준다.
  레벨 경고가 뜬 파일은 끝에 목록으로 모아서 알려준다.
- `q` 로 언제든 중단해도 되고, 다시 실행하면 **이어서** 진행된다.
  이어녹음 기준은 줄 번호가 아니라 **문장 텍스트**라, 대본을 다시 생성해
  순서가 바뀌어도 이미 녹음한 것을 정확히 알아본다.
- 앞뒤 무음은 자동으로 잘린다.
- 진행 상황만 보려면 `--passes 2 --list`.

**녹음 요령**

- 실제 사용할 때와 같은 위치·같은 목소리 크기로. 너무 또박또박 읽지 말 것.
- **회차마다 톤·속도·마이크 거리를 바꿀 것.** 3번 다 똑같이 읽으면 같은 데이터를
  3장 복사한 것과 다를 게 없어서 학습 효과가 안 난다. 1회차는 평소대로,
  2회차는 조금 빠르게/멀리서, 3회차는 편하게 흘리듯이 — 이런 식으로.
- 평소 쓰는 환경 그대로.

**분량 기준** (대본 500문장 = 학습 425 + 홀드아웃 75 기준)

| 분량 | 기대 효과 |
|---|---|
| 425발화 (1회차, 약 70분) | 최소선 |
| **850발화 (2회차, 약 2시간)** | 도메인 특화가 확실히 걸리는 지점 ← **목표** |
| 3회차 이상 | 같은 문장을 또 읽는 것보다 `--count` 를 키워 문장을 늘리는 게 낫다 |

**여러 날에 나눠 녹음하는 게 오히려 좋다.** 같은 자리에서 몰아 녹음하면 목 상태와
마이크 위치가 고정돼 데이터 다양성이 떨어진다. 이어녹음이 되니 나눠 해도 된다.

### 평가 전용 데이터 (중요)

학습에 쓴 발화로 성능을 재면 당연히 잘 나온다(외운 것). `build_prompts.py` 가
겹치지 않는 홀드아웃 대본을 따로 만들어 두므로 그걸로 한 벌 더 받는다:

```bash
python3 training/record_dataset.py --prompts training/prompts_eval.txt --speaker yjhan2_eval --passes 1
```

이건 `--fast` 없이 확인하며 천천히 받는 게 좋다. 학습 때
`prepare_funasr_data.py --data data/yjhan2` 만 넘기면 `yjhan2_eval` 은 자연히 빠진다.

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

## 3. 공개 한국어 데이터 (GPU 서버에서)

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

## 4. 학습 실행

학습은 이 PC가 아니라 **학교/연구실 리눅스 GPU 서버**에서 돌린다.
전체 절차(환경 구성 → 학습 → ONNX 변환 → 파이 배포 → 성능 비교)는
[TRAINING.md](TRAINING.md) 에 있다.

| 스크립트 | 실행 위치 | 용도 |
|---|---|---|
| `build_prompts.py` | PC/서버 | 녹음 대본 생성 (음소 커버리지 최적화) |
| `record_dataset.py` | 파이 | 녹음 |
| `eval_stt.py` | 파이 | 정확도 측정 (학습 전/후 비교) |
| `fetch_open_data.py` | 서버 | 공개 한국어 데이터 받기 |
| `prepare_funasr_data.py` | 서버 | 학습 포맷 변환 + 잡음 증강 |
| `setup_train_env.sh` | 서버 | venv + torch + funasr 설치 |
| `setup_train_env_conda.sh` | 서버 | 같은 것의 conda 판 (환경 `voicebot-train`) |
| `finetune_sensevoice.sh` | 서버 | 파인튜닝 |
| `export_onnx.py` | 서버 | ONNX int8 변환 + 로드 검증 |
