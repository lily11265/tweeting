# ============================================================
# audio/sanitizer.py — WAV 전처리 및 포맷 변환
# ============================================================

import os
import struct
import shutil
import subprocess
import wave
from pathlib import Path

# pydub (MP3→WAV 변환용)
try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False

# numpy / scipy (리샘플링용)
try:
    import numpy as np
    from scipy.signal import resample_poly as _resample_poly
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# 파일 선택 다이얼로그용 필터
AUDIO_FILETYPES = [
    ("음원 파일", "*.wav *.mp3"),
    ("WAV 파일", "*.wav"),
    ("MP3 파일", "*.mp3"),
]


def convert_mp3_to_wav(mp3_path, wav_path=None):
    """
    MP3 파일을 WAV로 변환.
    wav_path가 None이면 같은 폴더에 .wav 확장자로 저장.
    반환: 변환된 WAV 파일 경로
    """
    mp3_path = Path(mp3_path)
    if wav_path is None:
        wav_path = mp3_path.with_suffix(".wav")
    else:
        wav_path = Path(wav_path)

    if HAS_PYDUB:
        audio = AudioSegment.from_mp3(str(mp3_path))
        audio.export(str(wav_path), format="wav")
    else:
        # pydub 없으면 ffmpeg 직접 호출
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(mp3_path), str(wav_path)],
            capture_output=True, check=True
        )
    return str(wav_path)


def sanitize_wav(wav_path, output_path=None, target_sr=48000, target_bits=16):
    """
    WAV 파일을 tuneR::readWave()가 확실히 읽을 수 있는 형식으로 전처리.

    처리 내용:
      1) fmt 청크 정규화 (SM4 등 cbSize 문제 → "missing value where TRUE/FALSE needed" 해결)
      2) 24/32-bit → 16-bit 변환 (writeWave/resamp 크래시 방지)
      3) 스테레오 → 모노 변환
      4) 고해상도(>48kHz) 다운샘플링
      5) 비표준 청크(junk, wamd, _PMX 등) 제거

    반환: (출력 경로, 변환 로그 문자열)
    """
    wav_path = Path(wav_path)
    if output_path is None:
        output_path = wav_path

    log_lines = []

    try:
        with open(wav_path, "rb") as f:
            data = f.read()
    except Exception as e:
        return str(wav_path), f"[sanitize] 파일 읽기 실패: {e}"

    if len(data) < 44:
        return str(wav_path), f"[sanitize] 파일 너무 작음: {len(data)} bytes"

    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return str(wav_path), "[sanitize] WAV 형식 아님 (RIFF/WAVE 헤더 없음)"

    # === 1. 청크 파싱 ===
    pos = 12
    fmt_offset = None
    fmt_size = 0
    data_offset = None
    data_size = 0
    extra_chunks = []

    while pos < len(data) - 8:
        chunk_id = data[pos:pos + 4]
        chunk_sz = struct.unpack_from("<I", data, pos + 4)[0]

        if chunk_id == b"fmt ":
            fmt_offset = pos + 8
            fmt_size = chunk_sz
        elif chunk_id == b"data":
            data_offset = pos + 8
            data_size = chunk_sz
        else:
            extra_chunks.append(chunk_id.decode("ascii", errors="?"))

        pos += 8 + chunk_sz
        if pos % 2 != 0:
            pos += 1

    if fmt_offset is None:
        return str(wav_path), "[sanitize] fmt 청크 없음"
    if data_offset is None:
        return str(wav_path), "[sanitize] data 청크 없음"

    # fmt 필드 읽기
    fmt_data = data[fmt_offset:fmt_offset + min(fmt_size, 40)]
    audio_format = struct.unpack_from("<H", fmt_data, 0)[0]
    n_channels = struct.unpack_from("<H", fmt_data, 2)[0]
    sample_rate = struct.unpack_from("<I", fmt_data, 4)[0]
    bits_per_sample = struct.unpack_from("<H", fmt_data, 14)[0]

    log_lines.append(f"[sanitize] 원본: {sample_rate}Hz, {n_channels}ch, "
                     f"{bits_per_sample}bit, fmt={fmt_size}B, "
                     f"format={audio_format}, extra_chunks={extra_chunks}")

    # === 2. 변환 필요 여부 판단 ===
    needs_conversion = False
    reasons = []

    if fmt_size != 16 and audio_format == 1:
        needs_conversion = True
        reasons.append(f"fmt 크기 {fmt_size}→16 (SM4/비표준)")

    if bits_per_sample != target_bits:
        needs_conversion = True
        reasons.append(f"{bits_per_sample}bit→{target_bits}bit")

    if n_channels > 1:
        needs_conversion = True
        reasons.append(f"{n_channels}ch→mono")

    if sample_rate > target_sr:
        needs_conversion = True
        reasons.append(f"{sample_rate}Hz→{target_sr}Hz")

    if extra_chunks:
        needs_conversion = True
        reasons.append(f"비표준 청크 제거: {extra_chunks}")

    if not needs_conversion:
        log_lines.append("[sanitize] 변환 불필요 (이미 표준 형식)")
        return str(wav_path), "\n".join(log_lines)

    log_lines.append(f"[sanitize] 변환 필요: {', '.join(reasons)}")

    # === 3. PCM 샘플 읽기 ===
    raw_audio = data[data_offset:data_offset + data_size]
    block_align = n_channels * (bits_per_sample // 8)
    n_frames = len(raw_audio) // block_align

    if n_frames == 0:
        return str(wav_path), "\n".join(log_lines + ["[sanitize] 오디오 데이터 없음"])

    if HAS_SCIPY:
        if bits_per_sample == 16:
            samples = np.frombuffer(raw_audio[:n_frames * block_align], dtype=np.int16)
            samples = samples.reshape(-1, n_channels).astype(np.float64)
            max_val = 32767.0
        elif bits_per_sample == 24:
            byte_count = n_frames * n_channels * 3
            raw = np.frombuffer(raw_audio[:byte_count], dtype=np.uint8).reshape(-1, 3)
            flat = (raw[:, 0].astype(np.int32) |
                    (raw[:, 1].astype(np.int32) << 8) |
                    (raw[:, 2].astype(np.int32) << 16))
            flat[flat >= 0x800000] -= 0x1000000
            samples = flat.astype(np.float64).reshape(-1, n_channels)
            max_val = 8388607.0
        elif bits_per_sample == 32:
            if audio_format == 1:
                samples = np.frombuffer(raw_audio[:n_frames * block_align], dtype=np.int32)
                samples = samples.reshape(-1, n_channels).astype(np.float64)
                max_val = 2147483647.0
            elif audio_format == 3:
                samples = np.frombuffer(raw_audio[:n_frames * block_align], dtype=np.float32)
                samples = samples.reshape(-1, n_channels).astype(np.float64)
                max_val = 1.0
            else:
                log_lines.append(f"[sanitize] 미지원 32-bit format={audio_format}")
                return str(wav_path), "\n".join(log_lines)
        else:
            log_lines.append(f"[sanitize] 미지원 비트 깊이: {bits_per_sample}")
            return str(wav_path), "\n".join(log_lines)

        samples = samples / max_val

        if n_channels > 1:
            samples = samples.mean(axis=1)
        else:
            samples = samples.flatten()

        out_sr = min(sample_rate, target_sr)
        if sample_rate > target_sr:
            from math import gcd
            g = gcd(sample_rate, target_sr)
            up = target_sr // g
            down = sample_rate // g
            samples = _resample_poly(samples, up, down)
            out_sr = target_sr

        target_max = 2 ** (target_bits - 1) - 1
        samples = np.clip(samples * target_max, -target_max, target_max)
        pcm_data = samples.astype(np.int16).tobytes()

    else:
        # scipy 없으면 struct로 기본 변환 (리샘플링은 생략)
        pcm_samples = []
        bytes_per_sample = bits_per_sample // 8

        for i in range(n_frames):
            frame_offset = i * block_align
            channel_vals = []
            for ch in range(n_channels):
                ch_offset = frame_offset + ch * bytes_per_sample
                if bits_per_sample == 16:
                    val = struct.unpack_from("<h", raw_audio, ch_offset)[0]
                    channel_vals.append(val / 32767.0)
                elif bits_per_sample == 24:
                    b = raw_audio[ch_offset:ch_offset + 3]
                    val = b[0] | (b[1] << 8) | (b[2] << 16)
                    if val >= 0x800000:
                        val -= 0x1000000
                    channel_vals.append(val / 8388607.0)
                elif bits_per_sample == 32:
                    val = struct.unpack_from("<i", raw_audio, ch_offset)[0]
                    channel_vals.append(val / 2147483647.0)

            mono_val = sum(channel_vals) / len(channel_vals)
            int_val = max(-32767, min(32767, int(round(mono_val * 32767))))
            pcm_samples.append(struct.pack("<h", int_val))

        pcm_data = b"".join(pcm_samples)
        out_sr = sample_rate  # scipy 없으면 리샘플링 생략

    # === 4. 깨끗한 WAV 작성 ===
    out_channels = 1
    out_bits = target_bits
    out_byte_rate = out_sr * out_channels * (out_bits // 8)
    out_block_align = out_channels * (out_bits // 8)
    pcm_len = len(pcm_data)

    header = bytearray(44)
    header[0:4] = b"RIFF"
    struct.pack_into("<I", header, 4, 36 + pcm_len)
    header[8:12] = b"WAVE"
    header[12:16] = b"fmt "
    struct.pack_into("<I", header, 16, 16)
    struct.pack_into("<H", header, 20, 1)
    struct.pack_into("<H", header, 22, out_channels)
    struct.pack_into("<I", header, 24, out_sr)
    struct.pack_into("<I", header, 28, out_byte_rate)
    struct.pack_into("<H", header, 32, out_block_align)
    struct.pack_into("<H", header, 34, out_bits)
    header[36:40] = b"data"
    struct.pack_into("<I", header, 40, pcm_len)

    output_path = Path(output_path)
    with open(output_path, "wb") as f:
        f.write(header)
        f.write(pcm_data)

    out_duration = (pcm_len // (out_bits // 8)) / out_sr
    log_lines.append(f"[sanitize] ✅ 변환 완료: {out_sr}Hz, {out_channels}ch, "
                     f"{out_bits}bit, {out_duration:.1f}초, "
                     f"{os.path.getsize(output_path):,}B → {output_path.name}")

    return str(output_path), "\n".join(log_lines)


def ensure_wav(file_path, temp_dir):
    """
    파일이 MP3이면 WAV로 변환, WAV이면 그대로 반환.
    모든 WAV는 tuneR 호환을 위해 sanitize (16-bit mono ≤48kHz).
    변환된 파일은 temp_dir에 저장 (원본 보호).
    반환: (wav경로, 로그문자열)
    """
    file_path = Path(file_path)
    logs = []

    if file_path.suffix.lower() == ".mp3":
        wav_name = file_path.stem + ".wav"
        wav_path = Path(temp_dir) / wav_name
        convert_mp3_to_wav(file_path, wav_path)
        logs.append(f"[ensure_wav] MP3→WAV 변환: {file_path.name}")
        result_path, san_log = sanitize_wav(wav_path)
        logs.append(san_log)
        return result_path, "\n".join(logs)

    # WAV 파일 → temp_dir에 복사 후 sanitize (원본 보호)
    sanitized_path = Path(temp_dir) / file_path.name
    if str(sanitized_path.resolve()) != str(file_path.resolve()):
        try:
            shutil.copy2(file_path, sanitized_path)
        except Exception as e:
            logs.append(f"[ensure_wav] ⚠ 복사 실패: {e}")
            return str(file_path), "\n".join(logs)

    result_path, san_log = sanitize_wav(sanitized_path)
    logs.append(san_log)
    return result_path, "\n".join(logs)
