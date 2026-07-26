# STT 파인튜닝 (SenseVoiceSmall)

목표: 웨이크워드 "제리"와 자주 쓰는 명령어를 **내 목소리 + 파이 마이크 환경**에서
확실하게 알아듣게 만든다. 학습은 이 PC(RTX 5060), 추론은 파이에서 그대로.

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

```bash
cd ~/voicebot && git pull
python3 training/record_dataset.py --speaker yjhan
```

- 문장이 하나씩 뜬다 → `Enter` 로 녹음 시작 → 읽고 → `Enter` 로 종료
- 저장된 소리를 바로 들려준다. 마음에 안 들면 `r` 로 다시.
- `q` 로 언제든 중단해도 되고, 다시 실행하면 **이어서** 진행된다.
- 앞뒤 무음은 자동으로 잘리고, 소리가 너무 작거나 크면 경고가 뜬다.

**녹음 요령**

- 실제 사용할 때와 같은 위치·같은 목소리 크기로. 너무 또박또박 읽지 말 것.
- 같은 문장이라도 톤·속도·거리를 조금씩 바꿔가며 여러 번 넣으면 좋다.
- 평소 쓰는 환경 그대로. 선풍기·에어컨 소리가 나는 상태도 일부 섞으면 강건해진다.

**분량 기준**

| 분량 | 기대 효과 |
|---|---|
| 115문장(1회, 약 10분) | 웨이크워드 인식률 개선 확인 가능한 최소선 |
| 300~400발화(약 40분) | 도메인 어휘 특화 효과가 뚜렷해지는 실용 지점 ← **권장** |
| 1000발화 이상 | 더 좋지만 수집 비용 대비 효과는 완만해짐 |

300~400발화를 만들려면 `prompts_ko.txt` 를 2~3회 반복 녹음하거나(매번 톤을 바꿔서),
자주 쓰는 문장을 직접 추가하면 된다.

---

## 2. 공개 한국어 데이터 (PC에서)

내 목소리 데이터만으로 학습하면 모델이 원래 알던 한국어를 잊어버린다
(catastrophic forgetting). 공개 데이터를 3~10배 정도 섞어서 함께 학습한다.

```bash
pip install "datasets>=2.19" soundfile librosa
py training/fetch_open_data.py --list          # 후보 보기
py training/fetch_open_data.py --dataset zeroth --hours 3
```

| 이름 | 규모 | 라이선스 | 성격 |
|---|---|---|---|
| `zeroth` (Bingsu/zeroth-korean) | 51시간 | CC BY 4.0 | 한국어 낭독체. 바로 받아짐. **기본 추천** |
| `fleurs` (google/fleurs ko_kr) | ~10시간 | CC BY 4.0 | 격식체 낭독. 문장이 김 |
| `commonvoice` (CV 17 ko) | 소규모 | CC0 | 화자·마이크 다양. HF 로그인 필요 |
| KsponSpeech (AI Hub) | 1000시간 | 별도 승인 | 한국어 자유대화, 품질 최상. 가입·신청 필요, 수백 GB |

우선 `zeroth 3시간`이면 충분하다. 부족하면 나중에 늘린다.

---

## 3. 이후 단계 (아직 미구현)

3. 학습 데이터 병합 + train/val 분리 스크립트
4. SenseVoiceSmall 파인튜닝 (FunASR, 이 PC의 CUDA)
5. 파인튜닝 결과 → ONNX export → int8 양자화 → 파이 배포
6. 기존 모델 대비 정확도/속도 A-B 비교

### 미리 알아둘 위험 요소

- **ONNX export 경로 확인 필요.** 파이가 쓰는 건 `funasr_onnx` 런타임이라,
  파인튜닝한 파이토치 모델을 다시 ONNX 로 뽑고 int8 양자화까지 성공해야 배포가 된다.
  이 경로가 막히면 추론 속도(현재 0.4~0.5초)를 지키기 어려울 수 있다.
  **데이터 수집을 많이 하기 전에 4~5단계를 소량 데이터로 먼저 뚫어보는 것을 권장한다.**
- 데이터 포맷은 모델과 무관하게 만들었으므로, 설령 모델을 whisper 계열 등으로
  바꾸더라도 녹음한 데이터는 그대로 재사용할 수 있다.
