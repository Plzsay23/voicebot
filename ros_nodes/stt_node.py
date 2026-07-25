#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stt_node: /voice/utterance_path 로 들어온 발화 wav를 SenseVoice(int8)로 전사해
/voice/transcript 로 발행한다. 빈 결과(nospeech)는 버린다.
"""

import os
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import voice_common as vc


class SttNode(Node):
    def __init__(self):
        super().__init__("stt_node")
        self.get_logger().info("SenseVoice(int8) 로딩 중...")
        t0 = time.time()
        self.model = vc.load_stt()
        # 워밍업(무음)으로 첫 추론 지연 숨김
        try:
            import wave
            wu = Path("/tmp/_stt_warmup.wav")
            with wave.open(str(wu), "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(vc.SAMPLE_RATE)
                w.writeframes(b"\x00\x00" * vc.SAMPLE_RATE)
            vc.transcribe(self.model, str(wu))
            wu.unlink(missing_ok=True)
        except Exception as e:
            self.get_logger().warn(f"워밍업 건너뜀: {e}")
        self.get_logger().info(f"STT 준비 완료 ({time.time()-t0:.1f}s)")

        self.pub = self.create_publisher(String, "/voice/transcript", 10)
        self.create_subscription(String, "/voice/utterance_path", self.on_utt, 10)

    def on_utt(self, msg: String):
        path = msg.data
        if not os.path.exists(path):
            return
        t0 = time.time()
        try:
            text = vc.transcribe(self.model, path)
        except Exception as e:
            self.get_logger().error(f"전사 실패: {e}")
            text = ""
        finally:
            try:
                os.remove(path)
            except Exception:
                pass
        if not text:
            return
        self.get_logger().info(f"[STT {time.time()-t0:.1f}s] {text}")
        self.pub.publish(String(data=text))


def main():
    rclpy.init()
    node = SttNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
