#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dialog_node: /voice/transcript 를 받아 웨이크워드("제리")가 감지된 경우에만
필요시 웹검색 후 EXAONE로 답변을 생성해 /voice/response 로 발행한다.
웨이크워드가 없으면 무시(상시 청취지만 이름 부를 때만 반응).

생성은 반드시 워커 스레드에서 돈다. 콜백 안에서 돌리면 rclpy 단일 스레드
실행기가 그 20~30초 동안 막혀서, 그동안 들어온 transcript 가 버려지지 않고
큐에 쌓였다가 답변이 끝나는 순간 줄줄이 처리됐다(self.busy 는 재진입이
불가능하니 무용지물, self.speaking 도 갱신이 밀려 항상 낡은 값이었다).
"""

import os
import time
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool

import voice_common as vc

# 생성이 끝난 뒤 tts_node 가 speaking 을 올려 마이크 게이트를 넘겨받을 때까지
# busy 를 유지한다. tts 가 죽어도 영원히 잠기지는 않도록 상한을 둔다.
BUSY_HANDOFF_TIMEOUT_S = float(os.getenv("BUSY_HANDOFF_TIMEOUT_S", "3.0"))


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
        # 한 응답의 마지막 문장까지 보냈음을 알린다. tts_node 는 이 신호를 받고
        # 큐가 다 비워진 뒤에야 마이크를 다시 연다(문장 사이에 마이크가 열려
        # 자기 목소리를 주워담는 것을 막는다).
        self.done_pub = self.create_publisher(Bool, "/voice/response_done", 10)
        # 응답을 받아들인 순간부터 재생이 끝날 때까지 mic_node 가 아예 캡처를
        # 하지 않게 하는 신호. speaking 만으로는 생성 중(재생 전) 구간이 뚫린다.
        self.busy_pub = self.create_publisher(Bool, "/voice/busy", 10)
        self.create_subscription(String, "/voice/transcript", self.on_text, 10)
        self.create_subscription(Bool, "/voice/speaking", self.on_speaking, 10)
        self.speaking = False
        self.saw_speaking = False
        self.get_logger().info(f'대기 중: "{vc.BOT_NAME}" 라고 부르면 응답합니다.')

    def on_speaking(self, msg: Bool):
        self.speaking = msg.data
        if msg.data:
            self.saw_speaking = True

    def on_text(self, msg: String):
        text = msg.data.strip()
        if not text:
            return
        # 생성 중이거나 재생 중에 도착한 것은 버린다. 답변이 끝난 뒤에 처리하면
        # 사용자가 이미 지나간 말에 뒤늦게 대답하는 꼴이 된다.
        if self.busy or self.speaking:
            self.get_logger().info(f"[무시] 응답 처리 중: {text}")
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
        self.saw_speaking = False
        self.busy_pub.publish(Bool(data=True))
        threading.Thread(target=self._respond, args=(user_text,), daemon=True).start()

    def _respond(self, user_text: str):
        try:
            context = None
            if vc.needs_search(user_text):
                self.get_logger().info(f"[검색] {user_text}")
                context = vc.web_search(user_text) or None

            self.get_logger().info(f"[생성] {user_text}")
            t0 = time.time()
            n = 0
            for sentence in vc.ask_llm_stream(
                self.llm, self.history, user_text, context
            ):
                n += 1
                dt = time.time() - t0
                if n == 1:
                    self.get_logger().info(f"[첫문장 {dt:.1f}s] {sentence}")
                else:
                    self.get_logger().info(f"[문장{n} {dt:.1f}s] {sentence}")
                self.pub.publish(String(data=sentence))
            self.get_logger().info(f"[답변완료 {time.time()-t0:.1f}s, {n}문장]")
        except Exception as e:
            self.get_logger().error(f"생성 실패: {e}")
        finally:
            self.done_pub.publish(Bool(data=True))
            # tts_node 가 speaking 을 올려 게이트를 넘겨받은 뒤에 busy 를 내린다.
            # 먼저 내리면 그 틈에 마이크가 열려 자기 목소리를 주워담는다.
            deadline = time.time() + BUSY_HANDOFF_TIMEOUT_S
            while not self.saw_speaking and time.time() < deadline:
                time.sleep(0.05)
            self.busy = False
            self.busy_pub.publish(Bool(data=False))


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
