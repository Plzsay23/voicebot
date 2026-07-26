#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tts_node: /voice/response 로 들어오는 문장을 edge-tts로 합성해 순서대로 재생한다.

dialog_node 가 답변을 문장 단위로 흘려보내므로(파이4에서 LLM 이 느려 전체 답변을
기다리면 20초 넘게 조용하다), 이 노드는 문장을 큐에 쌓아 순서대로 처리한다.
합성과 재생을 별도 스레드로 나눠, 앞 문장을 재생하는 동안 뒤 문장을 미리
합성해 둔다. 그래야 문장 사이가 벌어지지 않는다.

/voice/speaking 은 첫 문장 재생 직전에 True 가 되고, dialog_node 가
/voice/response_done 을 보낸 뒤 큐가 완전히 빌 때까지 True 로 유지된다.
문장마다 껐다 켜면 그 틈에 mic_node 가 자기 목소리를 주워담는다.
"""

import queue
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool

import voice_common as vc

TAIL_GUARD_S = 0.6  # 재생 후 마이크 재개까지 여유(잔향/에코 방지)
WAV_SLOTS = 4       # 돌려쓰는 임시 wav 개수


class TtsNode(Node):
    def __init__(self):
        super().__init__("tts_node")
        self.speaking_pub = self.create_publisher(Bool, "/voice/speaking", 10)
        self.create_subscription(String, "/voice/response", self.on_response, 10)
        self.create_subscription(Bool, "/voice/response_done", self.on_done, 10)

        self.text_q = queue.Queue()   # 합성 대기 문장
        self.wav_q = queue.Queue()    # 재생 대기 wav
        self.done = threading.Event()  # 이번 응답의 문장이 다 들어왔다
        self.synth_busy = False
        self.speaking = False
        self.slot = 0

        self.playing = False
        threading.Thread(target=self._synth_loop, daemon=True).start()
        threading.Thread(target=self._play_loop, daemon=True).start()
        threading.Thread(target=self._release_loop, daemon=True).start()
        self.get_logger().info("tts_node 시작 (edge-tts, 문장 스트리밍)")

    # ---------- 구독 ----------
    def on_response(self, msg: String):
        text = msg.data.strip()
        if not text:
            return
        self.done.clear()
        self.text_q.put(text)

    def on_done(self, _msg: Bool):
        self.done.set()

    # ---------- 합성 ----------
    def _synth_loop(self):
        while True:
            text = self.text_q.get()
            self.synth_busy = True
            try:
                self.slot = (self.slot + 1) % WAV_SLOTS
                path = vc.BASE_DIR / f"tts_{self.slot}.wav"
                t0 = time.time()
                vc.synthesize(text, path)
                self.get_logger().info(f"[합성 {time.time()-t0:.1f}s] {text}")
                self.wav_q.put((path, text))
            except Exception as e:
                self.get_logger().error(f"TTS 합성 실패: {e}")
            finally:
                self.synth_busy = False
                self.text_q.task_done()

    # ---------- 재생 ----------
    def _play_loop(self):
        while True:
            path, text = self.wav_q.get()
            self.playing = True
            try:
                if not self.speaking:
                    self.speaking = True
                    self.speaking_pub.publish(Bool(data=True))
                    time.sleep(0.05)
                self.get_logger().info(f"[재생] {text}")
                vc.play_wav(path)
            except Exception as e:
                self.get_logger().error(f"TTS 재생 실패: {e}")
            finally:
                self.playing = False
                self.wav_q.task_done()

    def _idle(self):
        return (
            self.done.is_set()
            and not self.playing
            and not self.synth_busy
            and self.text_q.empty()
            and self.wav_q.empty()
        )

    def _release_loop(self):
        """이번 응답을 끝까지 다 말했으면 마이크를 다시 연다.

        재생 스레드에서 바로 풀지 않고 따로 감시하는 이유: 마지막 문장 재생이
        response_done 보다 먼저 끝나는 경우가 있고, 그때 재생 쪽에서만 판단하면
        아무도 speaking 을 내리지 않아 마이크가 영영 닫힌 채로 남는다.
        """
        while True:
            time.sleep(0.1)
            if not self.speaking or not self._idle():
                continue
            time.sleep(TAIL_GUARD_S)
            if not self._idle():
                continue  # 대기하는 사이 다음 문장이 들어왔다
            self.speaking = False
            self.speaking_pub.publish(Bool(data=False))


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
