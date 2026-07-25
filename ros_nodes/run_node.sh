#!/usr/bin/env bash
# 단일 ROS 노드 실행 헬퍼 (ROS 소싱 + venv 브릿지)
# 사용: bash run_node.sh mic_node.py
source /opt/ros/humble/setup.bash
export PYTHONPATH="/opt/ros/humble/lib/python3.10/site-packages:${PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTHONUTF8=1
cd "$(dirname "$0")"
exec ~/voicebot/.venv/bin/python "$1"
