#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dialog_node: /voice/transcript 를 받아 웨이크워드("제리")가 감지된 경우에만
필요시 웹검색 후 EXAONE로 답변을 생성해 /voice/response 로 발행한다.
웨이크워드가 없으면 무시(상시 청취지만 이름 부를 때만 반응).
"""

import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool

import voice_common as vc


class DialogNode(Node):
    def __init__(self):
        super().__init__("dialog_node")
        self.get_logger().info(f"EXAONE 로딩 중... ({vc.LOCAL_LLM_MODEL_PATH.name})")
        t0 = time.time()
        self.llm = vc.load_llm()
        self.get_logger().info(f"LLM 준비 완료 ({time.time()-t0:.1f}s)")

        self.history = []
        self.busy = False
        self.pub = self.create_publisher(String, "/voice/response", 10)
        self.create_subscription(String, "/voice/transcript", self.on_text, 10)
        self.create_subscription(Bool, "/voice/speaking", self.on_speaking, 10)
        self.speaking = False
        self.get_logger().info(f'대기 중: "{vc.BOT_NAME}" 라고 부르면 응답합니다.')

    def on_speaking(self, msg: Bool):
        self.speaking = msg.data

    def on_text(self, msg: String):
        text = msg.data.strip()
        if not text or self.busy or self.speaking:
            return

        # 종료어
        if vc.normalize_text(text) in ("종료", "그만", "끝", "종료해", "잘자"):
            self.get_logger().info("종료어 감지(무시 가능)")

        if not vc.has_wake_name(text):
            self.get_logger().info(f"[무시] 웨이크워드 없음: {text}")
            return

        user_text = vc.remove_wake_name(text)
        if not user_text:
            user_text = "응, 왜 불렀어?"

        self.busy = True
        try:
            context = None
            if vc.needs_search(user_text):
                self.get_logger().info(f"[검색] {user_text}")
                context = vc.web_search(user_text) or None

            self.get_logger().info(f"[생성] {user_text}")
            t0 = time.time()
            answer = vc.ask_llm(self.llm, self.history, user_text, context)
            self.get_logger().info(f"[답변 {time.time()-t0:.1f}s] {answer}")
            self.pub.publish(String(data=answer))
        except Exception as e:
            self.get_logger().error(f"생성 실패: {e}")
        finally:
            self.busy = False


def main():
    rclpy.init()
    node = DialogNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
