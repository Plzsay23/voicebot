#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mic_node: ReSpeaker 마이크를 parecord로 상시 캡처하고, silero-VAD(신경망)로
발화 구간을 잘라 wav로 저장한 뒤 경로를 /voice/utterance_path 로 발행한다.
TTS 재생 중(/voice/speaking=True)에는 캡처를 무시해 에코를 막는다.

silero-VAD는 노이즈와 음성을 잘 구분하므로 노이즈 환경에서도 오작동이 적다.
"""

import os
import wave
import threading
import subprocess
from collections import deque
from pathlib import Path

import numpy as np
import onnxruntime as ort
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool

SR = int(os.getenv("SAMPLE_RATE", "16000"))
FRAME_SAMPLES = 512          # silero v5: 16k에서 512샘플(32ms) 고정
FRAME_BYTES = FRAME_SAMPLES * 2
FRAME_MS = 32
CONTEXT_SAMPLES = 64         # silero v5: 직전 프레임 끝 64샘플을 앞에 붙여 576샘플로 넣어야 함
                             # (빼먹으면 모델이 무조건 0에 가까운 확률만 뱉는다)

SILERO_PATH = os.getenv(
    "SILERO_VAD_PATH",
    str(Path(__file__).resolve().parent.parent / "models" / "silero" / "silero_vad.onnx"),
)
VAD_THRESHOLD = float(os.getenv("SILERO_THRESHOLD", "0.5"))
VAD_MIN_SPEECH_MS = int(os.getenv("VAD_MIN_SPEECH_MS", "300"))
VAD_HANG_MS = int(os.getenv("VAD_HANG_MS", "500"))       # 끝 판정 무음 지속
VAD_MAX_UTT_MS = int(os.getenv("VAD_MAX_UTT_MS", "12000"))
VAD_PREROLL_MS = int(os.getenv("VAD_PREROLL_MS", "250"))  # 발화 앞 보존
UTT_DIR = Path("/tmp/voice_utt")


class SileroVAD:
    def __init__(self, path):
        so = ort.SessionOptions()
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.sr = np.array(SR, dtype=np.int64)
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)

    def prob(self, pcm_float: np.ndarray) -> float:
        frame = pcm_float.astype(np.float32)
        x = np.concatenate([self.context, frame]).reshape(1, -1)
        out, self.state = self.sess.run(
            None, {"input": x, "state": self.state, "sr": self.sr}
        )
        self.context = frame[-CONTEXT_SAMPLES:]
        return float(out[0, 0])


class MicNode(Node):
    def __init__(self):
        super().__init__("mic_node")
        self.pub = self.create_publisher(String, "/voice/utterance_path", 10)
        self.create_subscription(Bool, "/voice/speaking", self.on_speaking, 10)
        self.speaking = False
        self.utt_idx = 0
        UTT_DIR.mkdir(parents=True, exist_ok=True)

        self.vad = SileroVAD(SILERO_PATH)
        self.get_logger().info(f"mic_node 시작: parecord + silero-VAD (thr={VAD_THRESHOLD})")

        self.proc = subprocess.Popen(
            ["parecord", f"--rate={SR}", "--channels=1", "--format=s16le", "--raw"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        self.thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.thread.start()

    def on_speaking(self, msg: Bool):
        self.speaking = msg.data

    def capture_loop(self):
        buf = b""
        in_speech = False
        voiced = []
        hang = 0
        preroll = deque(maxlen=max(1, VAD_PREROLL_MS // FRAME_MS))
        hang_frames = max(1, VAD_HANG_MS // FRAME_MS)
        min_frames = max(1, VAD_MIN_SPEECH_MS // FRAME_MS)
        max_frames = max(1, VAD_MAX_UTT_MS // FRAME_MS)

        while rclpy.ok():
            chunk = self.proc.stdout.read(FRAME_BYTES)
            if not chunk:
                break
            buf += chunk
            while len(buf) >= FRAME_BYTES:
                frame = buf[:FRAME_BYTES]
                buf = buf[FRAME_BYTES:]

                if self.speaking:
                    # TTS 재생 중: 캡처 무시 + 상태 리셋
                    in_speech = False
                    voiced = []
                    hang = 0
                    preroll.clear()
                    self.vad.reset()
                    continue

                pcm = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
                p = self.vad.prob(pcm)

                if p >= VAD_THRESHOLD:
                    if not in_speech:
                        in_speech = True
                        voiced = list(preroll)  # 프리롤 포함
                    voiced.append(frame)
                    hang = 0
                    if len(voiced) >= max_frames:
                        self.flush_utterance(voiced)
                        in_speech = False
                        voiced = []
                        self.vad.reset()
                else:
                    if in_speech:
                        voiced.append(frame)
                        hang += 1
                        if hang >= hang_frames:
                            speech_frames = len(voiced) - hang
                            if speech_frames >= min_frames:
                                self.flush_utterance(voiced)
                            in_speech = False
                            voiced = []
                            hang = 0
                            self.vad.reset()
                    else:
                        preroll.append(frame)

    def flush_utterance(self, frames):
        data = b"".join(frames)
        self.utt_idx += 1
        path = UTT_DIR / f"utt_{self.utt_idx:06d}.wav"
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes(data)
        dur = len(data) / 2 / SR
        self.get_logger().info(f"발화 감지 {dur:.1f}s -> {path.name}")
        self.pub.publish(String(data=str(path)))

    def destroy_node(self):
        try:
            self.proc.terminate()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = MicNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
