#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_onnx.py — 파인튜닝한 SenseVoice 체크포인트를 파이에서 쓸 수 있는
ONNX(int8) 로 변환하고, 실제로 로드·전사가 되는지까지 검증한다.

파이의 stt_node 는 funasr_onnx.SenseVoiceSmall 로 모델 '폴더'를 읽는다.
그래서 model_quant.onnx 하나만으론 안 되고 config.yaml / am.mvn / bpe 모델이
같은 폴더에 함께 있어야 한다. 이 스크립트가 그 폴더를 통째로 만들어 준다.

사용:
    python export_onnx.py --model-dir outputs
    python export_onnx.py --model-dir outputs --test-wav data/yjhan/wav/0000.wav
    python export_onnx.py --model-dir outputs --quantize full   # 전체 op 양자화

양자화 방식:
    matmul  MatMul 연산만 int8 (기본). 현재 파이에 배포된 모델과 같은 방식.
            ARM Cortex-A72 에는 이쪽이 안정적이고 빠르다.
    full    onnxruntime 기본 동적 양자화(더 많은 op). 파일은 작지만
            ARM 에서 오히려 느려질 수 있으니 반드시 파이에서 실측할 것.
    none    양자화 안 함(fp32). 파이에선 느려서 비권장, 디버깅용.
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

# funasr_onnx 가 모델 폴더에서 찾는 파일들. onnx 본체 외에 이것들이 없으면
# 로드 자체가 실패한다.
AUX_FILES = [
    "config.yaml",
    "am.mvn",
    "chn_jpn_yue_eng_ko_spectok.bpe.model",
    "configuration.json",
]


def find_checkpoint(model_dir: Path) -> Path:
    """학습 산출물 폴더에서 쓸 체크포인트를 고른다 (평균 모델 우선)."""
    for name in ("model.pt.avg5", "model.pt.avg10"):
        p = model_dir / name
        if p.exists():
            return p

    # 평균 모델은 학습이 끝까지 갔을 때만 만들어진다. 없다는 건 학습이
    # 중간에 끊겼다는 뜻이라 성능이 덜 나온다. 진행은 시키되 경고한다.
    fallback = model_dir / "model.pt"
    eps = sorted(model_dir.glob("model.pt.ep*"),
                 key=lambda p: int(p.name.split("ep")[-1]))
    if not fallback.exists() and eps:
        fallback = eps[-1]
    if fallback.exists():
        print("[!] model.pt.avg5 가 없다 — 학습이 끝까지 안 갔을 가능성이 높다.")
        print(f"    {fallback.name} 로 진행하지만 성능은 평균 모델보다 떨어진다.")
        print("    outputs/train.log 끝부분을 확인할 것.")
        return fallback

    raise FileNotFoundError(
        f"{model_dir} 에서 체크포인트를 못 찾았다. 학습이 끝났는지 확인할 것.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True,
                    help="학습 산출물 폴더 (finetune 의 OUT)")
    ap.add_argument("--out", type=Path, default=None,
                    help="내보낼 폴더 (기본: <model-dir>/sensevoice_ko_ft)")
    ap.add_argument("--quantize", choices=["matmul", "full", "none"],
                    default="matmul")
    ap.add_argument("--test-wav", type=Path, default=None,
                    help="변환 후 이 wav 로 전사 검증")
    args = ap.parse_args()

    out_dir = args.out or (args.model_dir / "sensevoice_ko_ft")
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = find_checkpoint(args.model_dir)
    print(f"체크포인트: {ckpt}  ({ckpt.stat().st_size / 1e6:.0f}MB)")

    # ---------------------------------------------------------- 1. ONNX export
    print("\n[1/4] ONNX export 중... (원본 모델을 처음 받는다면 몇 분 걸림)")
    from funasr import AutoModel

    model = AutoModel(model=str(args.model_dir), trust_remote_code=True,
                      device="cpu", disable_update=True)
    model.export(type="onnx", quantize=False)

    # export 결과는 모델 폴더 안에 model.onnx 로 떨어진다.
    produced = list(args.model_dir.rglob("model.onnx"))
    if not produced:
        print("[x] model.onnx 가 생성되지 않았다. export 로그를 확인할 것.")
        return 1
    src_onnx = produced[0]
    src_root = src_onnx.parent
    print(f"    생성: {src_onnx}  ({src_onnx.stat().st_size / 1e6:.0f}MB)")

    fp32 = out_dir / "model.onnx"
    if src_onnx.resolve() != fp32.resolve():
        shutil.copy2(src_onnx, fp32)
        # torch 는 가중치를 model.onnx.data 로 따로 빼놓는다. 이걸 같이 안
        # 옮기면 onnx.load 가 "should be stored in ..., but it is not regular
        # file" 로 죽는다.
        for ext in list(src_root.glob(src_onnx.name + ".data")) + \
                   list(src_root.glob("*.onnx_data")):
            shutil.copy2(ext, out_dir / ext.name)
            print(f"    외부 가중치: {ext.name} ({ext.stat().st_size / 1e6:.0f}MB)")

    # 파이의 funasr_onnx 는 모델 파일 하나만 보고, 양자화도 합쳐진 쪽이 안전하다.
    # 원본이 1GB 정도라 2GB protobuf 한계 안에 들어간다.
    externals = list(out_dir.glob("*.onnx.data")) + list(out_dir.glob("*.onnx_data"))
    if externals:
        print("    외부 가중치를 model.onnx 하나로 합치는 중...")
        import onnx
        from onnx.external_data_helper import convert_model_from_external_data
        try:
            m = onnx.load(str(fp32))            # 외부 데이터까지 메모리로 읽는다
            convert_model_from_external_data(m)
            onnx.save(m, str(fp32))
        except Exception as e:
            print(f"[x] 합치기 실패: {e}")
            print("    모델이 2GB 를 넘으면 단일 파일로 못 만든다.")
            return 1
        for p in externals:
            p.unlink()
        print(f"    -> {fp32.name} ({fp32.stat().st_size / 1e6:.0f}MB)")

    # ---------------------------------------------------------- 2. 부속 파일
    print("\n[2/4] 부속 파일 복사")
    missing = []
    for name in AUX_FILES:
        found = next((p for p in [src_root / name, args.model_dir / name]
                      if p.exists()), None)
        if found is None:
            found = next(iter(args.model_dir.rglob(name)), None)
        if found is None:
            missing.append(name)
            continue
        shutil.copy2(found, out_dir / name)
        print(f"    {name}")
    if missing:
        print(f"    [!] 못 찾은 파일: {', '.join(missing)}")
        print("        원본 SenseVoiceSmall 폴더에서 직접 복사해야 할 수 있다.")

    # ---------------------------------------------------------- 3. 양자화
    if args.quantize == "none":
        print("\n[3/4] 양자화 건너뜀 (fp32)")
        final = fp32
    else:
        print(f"\n[3/4] int8 양자화 ({args.quantize})")
        from onnxruntime.quantization import QuantType, quantize_dynamic

        final = out_dir / "model_quant.onnx"
        kwargs = dict(model_input=str(fp32), model_output=str(final),
                      weight_type=QuantType.QInt8)
        if args.quantize == "matmul":
            # ARM 에서는 MatMul 만 int8 로 돌리는 게 안정적이고 빠르다.
            kwargs["op_types_to_quantize"] = ["MatMul"]
        quantize_dynamic(**kwargs)
        print(f"    {fp32.stat().st_size / 1e6:.0f}MB "
              f"-> {final.stat().st_size / 1e6:.0f}MB")

    # ---------------------------------------------------------- 4. 검증
    print("\n[4/4] funasr_onnx 로 로드 검증")
    try:
        from funasr_onnx import SenseVoiceSmall
    except ImportError:
        print("    [!] funasr_onnx 미설치 — 로드 검증을 건너뛴다.")
        print("        pip install funasr-onnx 후 다시 실행하면 검증까지 된다.")
        print(f"\n완료: {out_dir}")
        return 0

    try:
        t0 = time.time()
        m = SenseVoiceSmall(str(out_dir), batch_size=1,
                            quantize=(args.quantize != "none"))
        print(f"    로드 성공 ({time.time() - t0:.1f}s)")
    except Exception as e:
        print(f"    [x] 로드 실패: {e}")
        print("        파이에 올려도 똑같이 실패한다. 위 부속 파일부터 확인할 것.")
        return 1

    if args.test_wav and args.test_wav.exists():
        t0 = time.time()
        res = m([str(args.test_wav)], language="ko", use_itn=True)
        raw = res[0] if res else ""
        if isinstance(raw, dict):
            raw = raw.get("text", "")
        print(f"    전사 테스트 ({time.time() - t0:.2f}s): {raw}")
    else:
        print("    (--test-wav 를 주면 전사까지 확인한다)")

    print(f"\n완료: {out_dir}")
    print("\n파이로 보내기:")
    print(f"    scp -r {out_dir} chatbot-pi:~/voicebot/models/sensevoice_ko_ft")
    return 0


if __name__ == "__main__":
    sys.exit(main())
