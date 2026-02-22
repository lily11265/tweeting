# ============================================================
# audio/playback.py — 오디오 재생 추상화 레이어
# ============================================================
"""
크로스 플랫폼 오디오 재생 모듈.

우선순위:
  1. sounddevice (pip install sounddevice) — macOS / Linux / Windows
  2. winsound (Windows 내장) — 폴백

외부에서 HAS_PLAYBACK 플래그를 확인하여 재생 가능 여부를 판단합니다.
"""

import os
import wave
import time
import tempfile
import threading

# numpy
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# scipy (속도 변경용 리샘플링)
try:
    from scipy.signal import resample as scipy_resample
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# --- 재생 백엔드 선택 ---
_BACKEND = None  # "sounddevice" | "winsound" | None

try:
    import sounddevice as sd
    import soundfile as sf
    _BACKEND = "sounddevice"
except ImportError:
    pass

if _BACKEND is None:
    try:
        import winsound
        _BACKEND = "winsound"
    except ImportError:
        pass

HAS_PLAYBACK = _BACKEND is not None


def prepare_playback_wav(data, sr, t0, t1, speed=1.0, volume=1.0):
    """
    재생용 임시 WAV 파일을 생성합니다.

    Args:
        data: numpy float64 배열 (전체 음원)
        sr: 샘플 레이트
        t0: 시작 시간 (초)
        t1: 종료 시간 (초)
        speed: 재생 속도 (1.0 = 원속)
        volume: 볼륨 (0.0 ~ 1.0)

    Returns:
        (tmp_path, actual_duration) — 임시 WAV경로, 실제 재생 길이(초)
    """
    if not HAS_NUMPY:
        raise RuntimeError("numpy가 필요합니다")

    i0 = max(0, int(t0 * sr))
    i1 = min(len(data), int(t1 * sr))
    segment = data[i0:i1].copy()

    if len(segment) < 64:
        return None, 0.0

    # 속도 변경: 리샘플링
    if abs(speed - 1.0) > 0.01 and HAS_SCIPY:
        new_len = int(len(segment) / speed)
        if new_len < 64:
            new_len = 64
        segment = scipy_resample(segment, new_len)

    # float64 → int16 PCM (볼륨 적용)
    max_val = np.max(np.abs(segment))
    if max_val > 0:
        segment = segment / max_val  # 정규화
    pcm = (segment * 32767 * volume).astype(np.int16)

    # 임시 WAV 파일 생성
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    try:
        with wave.open(tmp_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())
    finally:
        os.close(tmp_fd)

    actual_duration = (t1 - t0) / speed
    return tmp_path, actual_duration


def play_wav_async(wav_path, stop_event, duration, on_done=None):
    """
    WAV 파일을 비동기 재생합니다 (백그라운드 스레드).

    Args:
        wav_path: WAV 파일 경로
        stop_event: threading.Event — set 시 재생 중단
        duration: 예상 재생 시간 (초)
        on_done: 재생 완료 콜백 (에러 메시지 또는 None)
    """
    if not HAS_PLAYBACK:
        if on_done:
            on_done("오디오 재생 백엔드를 사용할 수 없습니다.\n"
                    "pip install sounddevice soundfile")
        return

    def _worker():
        error = None
        try:
            if _BACKEND == "sounddevice":
                _play_sounddevice(wav_path, stop_event, duration)
            else:
                _play_winsound(wav_path, stop_event, duration)
        except Exception as e:
            error = str(e)
        finally:
            # 임시 파일 정리
            try:
                os.unlink(wav_path)
            except Exception:
                pass
            if on_done:
                on_done(error)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return thread


def stop_playback():
    """현재 재생을 중지합니다."""
    if _BACKEND == "sounddevice":
        try:
            sd.stop()
        except Exception:
            pass
    elif _BACKEND == "winsound":
        try:
            winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:
            pass


# ---- 내부 백엔드 구현 ----

def _play_sounddevice(wav_path, stop_event, duration):
    """sounddevice 백엔드로 WAV 재생."""
    data, samplerate = sf.read(wav_path, dtype="float32")
    sd.play(data, samplerate)

    end_wall = time.time() + duration + 0.3
    while time.time() < end_wall and not stop_event.is_set():
        time.sleep(0.05)

    if stop_event.is_set():
        sd.stop()


def _play_winsound(wav_path, stop_event, duration):
    """winsound 백엔드로 WAV 재생 (Windows 전용)."""
    winsound.PlaySound(wav_path,
                       winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
    end_wall = time.time() + duration + 0.3
    while time.time() < end_wall and not stop_event.is_set():
        time.sleep(0.05)

    if stop_event.is_set():
        try:
            winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:
            pass
