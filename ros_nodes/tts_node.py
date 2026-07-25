#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tts_node: /voice/response 텍스트를 edge-tts로 합성해 기본 스피커로 재생한다.
재생 동안 /voice/speaking=True 를 발행해 mic_node가 캡처를 멈추게 하고(에코 방지),
끝나면 짧은 여유 뒤 False 를 발행한다.

재생은 별도 스레드에서 하여, speaking=True 메시지가 재생 시작 전에 실제로
전달되도록 한다(콜백을 블록하지 않음).
"""

import time
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool

import voice_common as vc

TAIL_GUARD_S = 0.6  # 재생 후 마이크 재개까지 여유(잔향/에코 방지)


class TtsNode(Node):
    def __init__(self):
        super().__init__("tts_node")
        self.speaking_pub = self.create_publisher(Bool, "/voice/speaking", 10)
        self.create_subscription(String, "/voice/response", self.on_response, 10)
        self.lock = threading.Lock()
        self.busy = False
        self.get_logger().info("tts_node 시작 (edge-tts)")

    def on_response(self, msg: String):
        text = msg.data.strip()
        if not text:
            return
        with self.lock:
            if self.busy:
                return
            self.busy = True
        threading.Thread(target=self.speak, args=(text,), daemon=True).start()

    def speak(self, text):
        try:
            self.speaking_pub.publish(Bool(data=True))
            time.sleep(0.05)
            self.get_logger().info(f"[재생] {text}")
            vc.synthesize_and_play(text)
        except Exception as e:
            self.get_logger().error(f"TTS 실패: {e}")
        finally:
            time.sleep(TAIL_GUARD_S)
            self.speaking_pub.publish(Bool(data=False))
            self.busy = False


def main():
    rclpy.init()
    node = TtsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
