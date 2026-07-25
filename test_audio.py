import subprocess
from pathlib import Path

AUDIO_PATH = Path("test_deepfilter.wav")


def record_audio(seconds: int = 5):
    cmd = [
        "timeout",
        "--signal=INT",
        f"{seconds}s",
        "parecord",
        "--rate=16000",
        "--channels=1",
        "--format=s16le",
        "--file-format=wav",
        str(AUDIO_PATH),
    ]

    print(f"[REC] recording for {seconds} seconds...")

    result = subprocess.run(cmd)

    # timeout은 시간이 끝나면 보통 124를 반환함
    # 0 또는 124면 녹음 성공으로 처리
    if result.returncode not in (0, 124):
        raise RuntimeError(f"Recording failed. returncode={result.returncode}")

    if not AUDIO_PATH.exists() or AUDIO_PATH.stat().st_size == 0:
        raise RuntimeError("Recording file was not created or is empty.")

    print(f"[REC] saved: {AUDIO_PATH} ({AUDIO_PATH.stat().st_size} bytes)")


def play_audio():
    if not AUDIO_PATH.exists():
        raise FileNotFoundError(AUDIO_PATH)

    print("[PLAY] playing...")
    subprocess.run(["paplay", str(AUDIO_PATH)], check=True)


if __name__ == "__main__":
    record_audio(5)
    play_audio()