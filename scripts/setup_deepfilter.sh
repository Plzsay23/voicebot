#!/usr/bin/env bash
# setup_deepfilter.sh — DeepFilterNet(LADSPA)을 PipeWire 필터체인으로 걸어
# 잡음 제거된 가상 마이크 "DeepFilterMic" 을 만든다. 라즈베리파이에서 실행.
#
# 예전에 시도했던 Python 판 DeepFilterNet은 파이4에서 실시간 처리가 안 됐지만
# (RTF > 1), 이 LADSPA 판은 Rust로 컴파일돼 있어 CPU 부담이 훨씬 적다.
# PipeWire 레벨에 걸리므로 parecord/mic_node 등 기존 코드는 수정할 필요가 없다.
#
# 사용:
#   bash ~/voicebot/scripts/setup_deepfilter.sh            # 마이크 자동 감지
#   MIC_SOURCE=alsa_input.xxx bash ~/.../setup_deepfilter.sh  # 직접 지정
#
# 되돌리기:
#   bash ~/voicebot/scripts/setup_deepfilter.sh --uninstall

set -euo pipefail

VERSION="0.5.6"
LADSPA_DIR="$HOME/.ladspa"
CONF_DIR="$HOME/.config/pipewire"
CONF="$CONF_DIR/filter-chain.conf"
SERVICE_DIR="$HOME/.config/systemd/user"
SERVICE="$SERVICE_DIR/pipewire-filter-chain.service"
ATTEN="${ATTEN:-80}"   # 감쇠 한계(dB). 낮출수록 원음 보존, 높일수록 강하게 제거

info() { printf '\033[1;32m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[!]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[x]\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- uninstall
if [[ "${1:-}" == "--uninstall" ]]; then
  info "DeepFilter 필터체인 제거 중..."
  systemctl --user disable --now pipewire-filter-chain.service 2>/dev/null || true
  rm -f "$SERVICE" "$CONF"
  systemctl --user daemon-reload
  info "기본 마이크를 원본으로 되돌립니다."
  ORIG=$(pactl list sources short | awk '$2 !~ /\.monitor$/ && $2 ~ /^alsa_input/ {print $2; exit}')
  [[ -n "$ORIG" ]] && pactl set-default-source "$ORIG"
  info "완료. (플러그인 파일 $LADSPA_DIR 은 남겨둠)"
  exit 0
fi

# ---------------------------------------------------------------- 0. 사전 점검
ARCH=$(uname -m)
[[ "$ARCH" == "aarch64" ]] || die "이 스크립트는 aarch64용이다 (현재: $ARCH)"
command -v pactl >/dev/null || die "pactl 이 없다. PipeWire/PulseAudio 확인 필요."

info "PipeWire: $(pipewire --version 2>/dev/null | head -1 || echo '확인 불가')"

# 필요한 패키지 중 없는 것만 설치 (sudo 비밀번호를 물어볼 수 있음)
MISSING=()
dpkg -s pipewire-pulse >/dev/null 2>&1 || MISSING+=(pipewire-pulse)
dpkg -s wireplumber   >/dev/null 2>&1 || MISSING+=(wireplumber)
dpkg -s ladspa-sdk    >/dev/null 2>&1 || MISSING+=(ladspa-sdk)
if (( ${#MISSING[@]} )); then
  info "설치 필요: ${MISSING[*]}"
  sudo apt update && sudo apt install -y "${MISSING[@]}"
else
  info "필요한 패키지가 모두 설치되어 있다."
fi

# ---------------------------------------------------------------- 1. 플러그인
SO="$LADSPA_DIR/libdeep_filter_ladspa-$VERSION-aarch64-unknown-linux-gnu.so"
mkdir -p "$LADSPA_DIR"
if [[ -f "$SO" ]]; then
  info "플러그인 이미 있음: $(du -h "$SO" | cut -f1)"
else
  info "플러그인 다운로드 중 (약 50MB)..."
  URL="https://github.com/Rikorose/DeepFilterNet/releases/download/v$VERSION/$(basename "$SO")"
  wget -q --show-progress -O "$SO" "$URL" || die "다운로드 실패: $URL"
  chmod +x "$SO"
fi

grep -q 'LADSPA_PATH' "$HOME/.bashrc" 2>/dev/null || \
  echo "export LADSPA_PATH=\$HOME/.ladspa" >> "$HOME/.bashrc"
export LADSPA_PATH="$LADSPA_DIR"

if command -v listplugins >/dev/null; then
  listplugins 2>/dev/null | grep -qi deep \
    && info "플러그인 인식 확인됨" \
    || warn "listplugins 가 플러그인을 못 찾음 (계속 진행하되 실패하면 여길 의심할 것)"
fi

# ---------------------------------------------------------------- 2. 마이크 감지
if [[ -n "${MIC_SOURCE:-}" ]]; then
  MIC="$MIC_SOURCE"
else
  # .monitor(출력 되돌림)를 제외한 실제 입력 중, ReSpeaker/seeed 를 우선 선택
  MIC=$(pactl list sources short \
        | awk '$2 !~ /\.monitor$/ && $2 ~ /^alsa_input/ {print $2}' \
        | grep -i -m1 -E 'respeaker|seeed' || true)
  [[ -n "$MIC" ]] || MIC=$(pactl list sources short \
        | awk '$2 !~ /\.monitor$/ && $2 ~ /^alsa_input/ {print $2; exit}')
fi
[[ -n "$MIC" ]] || die "입력 마이크를 못 찾음. 'pactl list sources short' 확인 후 MIC_SOURCE= 로 지정."
[[ "$MIC" == *DeepFilterMic* ]] && die "감지된 게 DeepFilterMic 자신이다. MIC_SOURCE= 로 원본 마이크를 지정할 것."
info "원본 마이크: $MIC"

# ---------------------------------------------------------------- 3. 설정 파일
mkdir -p "$CONF_DIR"
[[ -f "$CONF" ]] && cp "$CONF" "$CONF.bak.$(date +%s)" && warn "기존 설정을 백업했다."

cat > "$CONF" <<EOF
# 자동 생성됨: scripts/setup_deepfilter.sh
context.properties = {
    log.level = 2
}

context.spa-libs = {
    audio.convert.* = audioconvert/libspa-audioconvert
    support.*       = support/libspa-support
}

context.modules = [
    { name = libpipewire-module-rt
        args = { nice.level = -11 }
        flags = [ ifexists nofail ]
    }
    { name = libpipewire-module-protocol-native }
    { name = libpipewire-module-client-node }
    { name = libpipewire-module-adapter }
    { name = libpipewire-module-link-factory }

    { name = libpipewire-module-filter-chain
        args = {
            node.description = "DeepFilter Noise Canceling Source"
            media.name       = "DeepFilter Noise Canceling Source"
            filter.graph = {
                nodes = [
                    {
                        type   = ladspa
                        name   = deepfilter
                        plugin = $SO
                        label  = deep_filter_mono
                        control = {
                            "Attenuation Limit (dB)" = $ATTEN
                        }
                    }
                ]
            }
            audio.position = [ MONO ]
            capture.props = {
                node.name         = "deepfilter_capture"
                node.passive      = true
                audio.rate        = 48000
                stream.dont-remix = true
                node.target       = "$MIC"
            }
            playback.props = {
                node.name   = "DeepFilterMic"
                media.class = "Audio/Source"
                audio.rate  = 48000
            }
        }
    }
]
EOF
info "설정 생성: $CONF"

# ---------------------------------------------------------------- 4. 서비스
mkdir -p "$SERVICE_DIR"
cat > "$SERVICE" <<EOF
[Unit]
Description=PipeWire Filter Chain for DeepFilter
After=pipewire.service pipewire-pulse.service wireplumber.service
Requires=pipewire.service

[Service]
Type=simple
ExecStart=/usr/bin/pipewire -c %h/.config/pipewire/filter-chain.conf
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
EOF

info "서비스 등록 및 재시작..."
systemctl --user daemon-reload
systemctl --user enable pipewire-filter-chain.service >/dev/null
systemctl --user restart pipewire.service;       sleep 2
systemctl --user start   wireplumber.service;    sleep 2
systemctl --user restart pipewire-pulse.service; sleep 2
systemctl --user restart pipewire-filter-chain.service
sleep 3

# ---------------------------------------------------------------- 5. 확인
if pactl list sources short | grep -q DeepFilterMic; then
  info "성공: DeepFilterMic 생성됨"
  pactl list sources short | sed 's/^/    /'
  pactl set-default-source DeepFilterMic
  info "기본 마이크를 DeepFilterMic 으로 설정했다."
  echo
  echo "  다음: 잡음 제거 효과 비교 -> bash ~/voicebot/scripts/compare_deepfilter.sh"
else
  warn "DeepFilterMic 이 안 보인다. 아래 로그를 확인할 것:"
  echo "    systemctl --user status pipewire-filter-chain.service"
  echo "    journalctl --user -u pipewire-filter-chain.service -n 50"
  exit 1
fi
