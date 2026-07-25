#!/usr/bin/env bash
# 상시 청취 ROS 2 음성비서 런처.
# ROS 2 Humble 환경을 소싱하고, voicebot venv 파이썬이 rclpy를 import 할 수 있게
# ROS site-packages를 PYTHONPATH에 추가한 뒤 4개 노드를 실행한다.
set -e

source /opt/ros/humble/setup.bash
export PYTHONPATH="/opt/ros/humble/lib/python3.10/site-packages:${PYTHONPATH}"
export PYTHONUTF8=1
export PYTHONUNBUFFERED=1

cd "$(dirname "$0")"
VENV=~/voicebot/.venv/bin/python

MIC_DELAY="${MIC_DELAY:-50}"   # stt/dialog 모델 로딩을 기다렸다가 마이크 시작

pids=()
cleanup() {
    echo "[launcher] 종료 중..."
    for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done
    pkill -f parecord 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[launcher] stt_node 시작..."
$VENV stt_node.py & pids+=($!)
echo "[launcher] dialog_node 시작..."
$VENV dialog_node.py & pids+=($!)
echo "[launcher] tts_node 시작..."
$VENV tts_node.py & pids+=($!)

echo "[launcher] 모델 로딩 대기 ${MIC_DELAY}s 후 mic_node 시작..."
sleep "$MIC_DELAY"
echo "[launcher] mic_node 시작 (이제 '제리'라고 부르세요)"
$VENV mic_node.py & pids+=($!)

wait
