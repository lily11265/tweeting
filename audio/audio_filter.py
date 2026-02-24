# ============================================================
# audio/audio_filter.py — 오디오 주파수 필터링 모듈
# 밴드패스 필터 및 STFT 기반 시간-주파수 폴리곤 마스킹
# ============================================================

import os
import wave
import tempfile
import numpy as np

try:
    from scipy.signal import butter, sosfilt, stft, istft
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from scipy.signal import resample as scipy_resample
    HAS_RESAMPLE = True
except ImportError:
    HAS_RESAMPLE = False


def bandpass_filter(data: np.ndarray, sr: int,
                    f_low: float, f_high: float,
                    order: int = 5) -> np.ndarray:
    """
    Butterworth 밴드패스 필터로 특정 주파수 대역만 통과.

    Args:
        data: 1D float 오디오 배열
        sr: 샘플 레이트
        f_low: 하한 주파수 (Hz)
        f_high: 상한 주파수 (Hz)
        order: 필터 차수 (기본 5)

    Returns:
        필터링된 오디오 배열
    """
    if not HAS_SCIPY:
        raise RuntimeError("scipy가 필요합니다 (pip install scipy)")

    nyq = sr / 2.0
    low = max(f_low, 1.0) / nyq
    high = min(f_high, nyq - 1) / nyq

    if low >= high:
        return data.copy()

    low = max(low, 0.001)
    high = min(high, 0.999)

    sos = butter(order, [low, high], btype='band', output='sos')
    return sosfilt(sos, data).astype(data.dtype)


def polygon_mask_filter(data: np.ndarray, sr: int,
                        polygon_points: list,
                        nperseg: int = 1024) -> np.ndarray:
    """
    STFT 기반 시간-주파수 폴리곤 마스킹 필터.
    폴리곤 내부의 시간-주파수 영역만 통과시킨다.

    Args:
        data: 1D float 오디오 배열
        sr: 샘플 레이트
        polygon_points: [(time_sec, freq_hz), ...] 폴리곤 꼭짓점
        nperseg: STFT 윈도우 크기

    Returns:
        필터링된 오디오 배열
    """
    if not HAS_SCIPY:
        raise RuntimeError("scipy가 필요합니다 (pip install scipy)")

    if len(polygon_points) < 3:
        return data.copy()

    noverlap = nperseg * 3 // 4

    # STFT 계산
    freqs, times, Zxx = stft(data, fs=sr, nperseg=nperseg, noverlap=noverlap)

    # 폴리곤 마스크 생성 (시간-주파수 그리드에 대해)
    mask = _create_polygon_mask(times, freqs, polygon_points)

    # 마스크 적용 (부드러운 경계를 위해 가우시안 블러 적용)
    try:
        from scipy.ndimage import gaussian_filter
        mask = gaussian_filter(mask.astype(float), sigma=1.5)
    except ImportError:
        mask = mask.astype(float)

    Zxx_filtered = Zxx * mask

    # iSTFT로 복원
    _, reconstructed = istft(Zxx_filtered, fs=sr, nperseg=nperseg, noverlap=noverlap)

    # 원본 길이에 맞춤
    if len(reconstructed) > len(data):
        reconstructed = reconstructed[:len(data)]
    elif len(reconstructed) < len(data):
        reconstructed = np.pad(reconstructed, (0, len(data) - len(reconstructed)))

    return reconstructed.astype(data.dtype)


def _create_polygon_mask(times: np.ndarray, freqs: np.ndarray,
                         polygon_points: list) -> np.ndarray:
    """
    시간-주파수 그리드에 대한 폴리곤 내부 마스크 생성.
    Ray-casting 알고리즘 사용.

    Args:
        times: STFT 시간 축 배열
        freqs: STFT 주파수 축 배열
        polygon_points: [(time_sec, freq_hz), ...]

    Returns:
        (n_freqs, n_times) 불리언 마스크
    """
    n_freqs = len(freqs)
    n_times = len(times)
    mask = np.zeros((n_freqs, n_times), dtype=bool)

    # 폴리곤 좌표 배열
    poly = np.array(polygon_points)  # (N, 2) — (time, freq)
    n = len(poly)

    for fi in range(n_freqs):
        for ti in range(n_times):
            t = times[ti]
            f = freqs[fi]
            if _point_in_polygon(t, f, poly, n):
                mask[fi, ti] = True

    return mask


def _point_in_polygon(x: float, y: float,
                      poly: np.ndarray, n: int) -> bool:
    """Ray-casting 알고리즘으로 점이 폴리곤 내부에 있는지 판정."""
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def prepare_filtered_wav(data: np.ndarray, sr: int,
                         t0: float, t1: float,
                         f_low: float, f_high: float,
                         speed: float = 1.0,
                         volume: float = 1.0):
    """
    시간+주파수 범위로 필터링된 WAV 파일 생성.

    Args:
        data: 전체 오디오 데이터
        sr: 샘플 레이트
        t0, t1: 시간 범위 (초)
        f_low, f_high: 주파수 범위 (Hz)
        speed: 재생 속도
        volume: 볼륨 (0~1)

    Returns:
        (tmp_path, actual_duration) or (None, 0.0)
    """
    # 시간 범위 추출
    i0 = max(0, int(t0 * sr))
    i1 = min(len(data), int(t1 * sr))
    segment = data[i0:i1].copy()

    if len(segment) < 64:
        return None, 0.0

    # 밴드패스 필터 적용
    nyq = sr / 2.0
    if f_low > 1.0 or f_high < nyq - 1:
        segment = bandpass_filter(segment, sr, f_low, f_high)

    # 속도 변경
    if abs(speed - 1.0) > 0.01 and HAS_RESAMPLE:
        new_len = int(len(segment) / speed)
        if new_len < 64:
            new_len = 64
        segment = scipy_resample(segment, new_len)

    # float → int16 PCM
    max_val = np.max(np.abs(segment))
    if max_val > 0:
        segment = segment / max_val
    pcm = (segment * 32767 * volume).astype(np.int16)

    # 임시 WAV 파일
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    try:
        with wave.open(tmp_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())
    finally:
        os.close(tmp_fd)

    actual_duration = (t1 - t0) / speed
    return tmp_path, actual_duration


def prepare_polygon_wav(data: np.ndarray, sr: int,
                        polygon_points: list,
                        speed: float = 1.0,
                        volume: float = 1.0):
    """
    폴리곤 영역으로 필터링된 WAV 파일 생성.

    Args:
        data: 전체 오디오 데이터
        sr: 샘플 레이트
        polygon_points: [(time_sec, freq_hz), ...] 폴리곤 꼭짓점
        speed: 재생 속도
        volume: 볼륨 (0~1)

    Returns:
        (tmp_path, actual_duration) or (None, 0.0)
    """
    if len(polygon_points) < 3:
        return None, 0.0

    # 폴리곤의 시간 범위 결정
    times = [p[0] for p in polygon_points]
    t0 = max(0, min(times))
    t1 = min(len(data) / sr, max(times))

    if t1 - t0 < 0.01:
        return None, 0.0

    # 시간 범위 추출
    i0 = max(0, int(t0 * sr))
    i1 = min(len(data), int(t1 * sr))
    segment = data[i0:i1].copy()

    if len(segment) < 64:
        return None, 0.0

    # 폴리곤 좌표를 세그먼트 기준으로 오프셋
    offset_points = [(t - t0, f) for t, f in polygon_points]

    # 폴리곤 마스킹 적용
    segment = polygon_mask_filter(segment, sr, offset_points)

    # 속도 변경
    if abs(speed - 1.0) > 0.01 and HAS_RESAMPLE:
        new_len = int(len(segment) / speed)
        if new_len < 64:
            new_len = 64
        segment = scipy_resample(segment, new_len)

    # float → int16 PCM
    max_val = np.max(np.abs(segment))
    if max_val > 0:
        segment = segment / max_val
    pcm = (segment * 32767 * volume).astype(np.int16)

    # 임시 WAV 파일
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    try:
        with wave.open(tmp_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())
    finally:
        os.close(tmp_fd)

    actual_duration = (t1 - t0) / speed
    return tmp_path, actual_duration
