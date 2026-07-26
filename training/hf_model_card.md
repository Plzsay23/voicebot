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

# SenseVoiceSmall fine-tuned for Korean (wake word "제리" / Jerry)

A fine-tune of [SenseVoiceSmall](https://huggingface.co/FunAudioLLM/SenseVoiceSmall),
exported to ONNX so it runs on a Raspberry Pi 4B voice assistant. Both the fp32
export and an int8 quantized version are here.

## Why this exists

The base model already handles everyday Korean just fine. The problem was one
word: the wake word **"제리"**. In a 30-utterance baseline, 11 of the 13 errors
were that single word — it kept coming out as "제야" (7 times), "재이", "젤리",
"젤이", "데야". Meanwhile it happily nailed things like "삼성전자",
"비트코인 시세", and "무슨 요일이야".

So this isn't a "make Korean ASR better" model. It's a "make one stubborn wake
word work on my hardware, in my voice" model.

| Baseline (before fine-tuning) | |
|---|---|
| CER | 0.096 |
| Exact match | 17/30 (56.7%) |
| Wake word detected | 28/30 (93.3%) |
| Avg inference on Pi 4 | 0.71 s |

<!-- Fill in post-fine-tune numbers from the 75-sentence holdout. -->

## Training data (round 2)

| Source | Amount |
|---|---|
| Own recordings (1 speaker) | 425 sentences × 2 passes = **850 utterances / 57 min** |
| Noise-augmented copies | 850 (mouse, keyboard, mic self-noise; SNR 5–20 dB) |
| Zeroth-Korean audio | 1,119 utterances / 3 h |
| **Total for training** | 3,119 rows / 299 min (plus 550 held out for validation) |

Recordings are 16 kHz mono, captured through the ReSpeaker Lite USB mic attached
to the Pi. That part matters more than it sounds — training on audio that came
through a different mic than the one you'll actually use throws away most of the
benefit. Each sentence was read twice with deliberately different pace, tone, and
mic distance; the two takes differ by about 25% in length at the median, so the
second pass is real data rather than a copy.

20 epochs, val loss 7.19 → 1.29 (flat after epoch 16). The last 5 checkpoints
were averaged.

**How the sentences were chosen.** Picking sentences by hand always ends up
phonetically lopsided, so instead the script greedily selects sentences that
maximize phoneme coverage — set cover over three units: onset+nucleus,
nucleus+coda, and coda→next-onset (where most Korean phonological alternation
happens). The pool of general Korean sentences comes from the
[Zeroth-Korean](https://huggingface.co/datasets/Bingsu/zeroth-korean)
transcripts (CC BY 4.0). Wake-word phrases, real commands, and near-homophone
negatives ("체리", "저리", "처리", "자리", "소리"…) were written by hand, since
no corpus contains those.

The 75-sentence holdout shares **zero** sentences with the training set. The
previous round had no holdout at all, which made its numbers meaningless.

## Things you should know before using this

- **It's speaker-specific.** One person's voice, one microphone. On other
  speakers it may well be worse than the base model. If you want general-purpose
  Korean ASR, use the original SenseVoiceSmall.
- **It's tuned for the wake word "제리".** Different wake word, no reason to
  reach for this.
- **Public corpus audio is mixed in this time** (3 h of Zeroth), which should
  limit catastrophic forgetting — but only one speaker's data got the noise
  augmentation and the 2× repetition, so expect some drift on general Korean.

## Usage

```python
from funasr_onnx import SenseVoiceSmall

model = SenseVoiceSmall("path/to/sensevoice_ko_ft", batch_size=1, quantize=True)
print(model(["input.wav"], language="ko", use_itn=True))
# ['<|ko|><|NEUTRAL|><|Speech|><|woitn|>제리']
```

The loader reads a *folder*, not a single file. Alongside the ONNX you need
`config.yaml`, `am.mvn`, and `chn_jpn_yue_eng_ko_spectok.bpe.model` — the last
one isn't a training artifact, it ships with the original SenseVoiceSmall.

| File | Size |
|---|---|
| `model.onnx` (fp32) | ~900 MB |
| `model_quant.onnx` (int8, MatMul only) | ~240 MB |

Only MatMul ops are quantized. Quantizing everything makes the file marginally
smaller but tends to run *slower* on ARM Cortex-A72, where the extra
quantize/dequantize traffic isn't worth it. If you're on x86, measure for
yourself — the tradeoff is different there.

## License and credit

Derived from [iic/SenseVoiceSmall](https://huggingface.co/FunAudioLLM/SenseVoiceSmall)
and covered by the **FunASR Model Open Source License Agreement**
([full text](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE)).
The SenseVoice *code* is MIT; the weights are not — worth reading before you
redistribute anything built on this.

Sentence text from Zeroth-Korean (CC BY 4.0). The audio in this training set was
recorded by me; Zeroth's own audio was used as-is for the 3 h mix-in.

Training and export scripts: https://github.com/Plzsay23/voicebot (`training/`)
