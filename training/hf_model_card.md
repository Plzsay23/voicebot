---
language:
- ko
license: other
license_name: funasr-model-open-source-license
license_link: https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE
base_model: FunAudioLLM/SenseVoiceSmall
pipeline_tag: automatic-speech-recognition
tags:
- speech-recognition
- korean
- sensevoice
- onnx
- int8
- raspberry-pi
---

# SenseVoiceSmall 한국어 파인튜닝 (웨이크워드 "제리")

라즈베리파이 4B 음성비서용으로 [SenseVoiceSmall](https://huggingface.co/FunAudioLLM/SenseVoiceSmall)
을 파인튜닝한 뒤 ONNX int8 로 변환한 모델이다.

## 왜 만들었나

원본 SenseVoiceSmall 은 한국어 일반 어휘를 잘 알아듣는데 **웨이크워드 "제리"만
못 알아들었다.** 파인튜닝 전 30발화 측정에서 오류 13개 중 11개가 "제리" 였고,
"제야"(7회), "재이", "젤리", "젤이", "데야" 로 잘못 들었다. 반면 "삼성전자",
"비트코인 시세", "무슨 요일이야" 같은 어휘는 거의 다 맞혔다.

| 지표 | 파인튜닝 전 |
|---|---|
| CER | 0.096 |
| 완전일치 | 17/30 (56.7%) |
| 웨이크워드 인식 | 28/30 (93.3%) |
| 평균 추론(파이4) | 0.71초 |

<!-- 파인튜닝 후 수치는 eval_stt.py --compare 결과로 채울 것 -->

## 학습 데이터

한국어 화자 1명의 자체 녹음 345발화. 16kHz mono, 라즈베리파이에 연결된
ReSpeaker Lite USB 마이크로 녹음했다. 문장 목록은 웨이크워드와 음성비서가
실제로 받는 명령어(날씨·시간·검색·타이머 등)로 구성했다.
학습은 20 epoch(val loss 0.693 → 0.645), 마지막 5개 체크포인트를 평균냈다.

## 한계 — 읽고 쓸 것

- **화자 특화 모델이다.** 한 사람 목소리로 학습해서 다른 화자에겐 원본보다
  나쁠 수 있다. 범용 한국어 STT 를 찾는다면 원본 SenseVoiceSmall 을 쓰는 게 낫다.
- **"제리"라는 특정 웨이크워드에 맞춰져 있다.** 다른 호출어를 쓸 거면 그대로
  가져다 쓸 이유가 없다.
- **공개 한국어 코퍼스를 섞지 않았다.** 이 모델은 위 345발화만으로 학습됐고,
  그래서 원본이 알던 일반 한국어를 일부 잊었을 수 있다(catastrophic forgetting).
  학습 문장에 없는 어휘에서 원본보다 나쁠 가능성을 염두에 두고 쓸 것.

## 쓰는 법

```python
from funasr_onnx import SenseVoiceSmall

model = SenseVoiceSmall("경로/sensevoice_ko_ft", batch_size=1, quantize=True)
print(model(["input.wav"], language="ko", use_itn=True))
# ['<|ko|><|NEUTRAL|><|Speech|><|woitn|>제리']
```

폴더에 `model_quant.onnx` 외에 `config.yaml`, `am.mvn`,
`chn_jpn_yue_eng_ko_spectok.bpe.model` 이 함께 있어야 로드된다.

| 파일 | 크기 |
|---|---|
| `model.onnx` (fp32) | 897MB |
| `model_quant.onnx` (int8, MatMul) | 230MB |

양자화는 MatMul 연산만 int8 로 돌렸다. ARM Cortex-A72 에서는 전체 op 를
양자화하는 것보다 이쪽이 안정적이고 빠르다.

## 라이선스와 출처

원본 [iic/SenseVoiceSmall](https://huggingface.co/FunAudioLLM/SenseVoiceSmall)
의 파생물이며 **FunASR Model Open Source License Agreement** 를 따른다.
전문은 [MODEL_LICENSE](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE) 참고.
SenseVoice 코드 자체는 MIT 다.

학습·변환 스크립트: https://github.com/Plzsay23/voicebot (`training/`)
