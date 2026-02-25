# ============================================================
# 조류 음성 탐지 R 분석 스크립트 (종합 판별 엔진 v3)
# Python GUI에서 subprocess로 호출됨
# 사용법: Rscript analysis_composite.R <config.json>
#
# === 판별 파이프라인 ===
#   1단계: corMatch로 후보 구간 검출 (넓은 그물)
#   2단계: 각 후보 구간에 대해 종합 평가
#          - 스펙트로그램 상관 (corMatch 원점수)
#          - MFCC 코사인 유사도
#          - 주파수 궤적 DTW
#          - 진폭 포락선 DTW
#          - 주파수 대역 에너지 집중도
#   3단계: 가중합 → 종합 점수로 최종 판정
#
# === 추가 필요 패키지 ===
#   dtw, proxy (DTW 계산용)
# ============================================================

# --- 상수 ---
NYQUIST_SAFETY_FACTOR <- 0.95
MAX_TEMPLATE_DURATION <- 5.0
SPECTROGRAM_MAIN_W <- 1200
SPECTROGRAM_MAIN_H <- 600
SPECTROGRAM_SP_W <- 800
SPECTROGRAM_SP_H <- 500

# ★ 최대 샘플레이트 (조류 울음은 ~12kHz 이하이므로 48kHz면 충분)
# 96kHz 등 고해상도 음원은 이 값으로 다운샘플링하여 오버플로 방지
MAX_SAMPLE_RATE <- 48000

# 종합 점수 기본 가중치 (config에서 종별 오버라이드 가능)
DEFAULT_WEIGHTS <- list(
  cor_score      = 0.18, # 스펙트로그램 상관
  mfcc_score     = 0.18, # MFCC DTW 유사도
  dtw_freq       = 0.13, # 주파수 궤적 DTW
  dtw_env        = 0.08, # 진폭 포락선 DTW
  band_energy    = 0.13, # 주파수 대역 에너지 집중도
  harmonic_ratio = 0.18, # C1: 조화 비율 (새소리 주기성)
  snr            = 0.12 # C1.5: 신호 대 잡음비
)

# 1단계 후보 검출용 넓은 cutoff (본래 cutoff의 이 비율)
# 예: cutoff=0.4이면, 1단계에서는 0.4*0.5=0.2 이상을 후보로 수집
CANDIDATE_CUTOFF_RATIO <- 0.5

# DTW 정규화 거리 → 유사도 변환 파라미터
# similarity = exp(-alpha * normalized_distance)
DTW_ALPHA <- 2.0

# ============================================================
# 로깅 함수
# ============================================================
log_debug <- function(...) {
  msg <- paste0("[DEBUG ", format(Sys.time(), "%H:%M:%S"), "] ", ...)
  cat(msg, "\n")
  flush.console()
}
log_info <- function(...) {
  msg <- paste0("[INFO] ", ...)
  cat(msg, "\n")
  flush.console()
}
log_error <- function(...) {
  msg <- paste0("[ERROR] ", ...)
  cat(msg, "\n", file = stderr())
  cat(msg, "\n")
  flush.console()
}

# ============================================================
# 유틸리티 함수
# ============================================================
safe_dev_off <- function() {
  tryCatch(
    {
      if (dev.cur() > 1) dev.off()
    },
    error = function(e) invisible(NULL)
  )
}

# ============================================================
# ★ safe_resamp: 오버플로 방지 리샘플링
# seewave::resamp()는 내부적으로 nrow(wave) * g 를 계산하는데
# 두 값이 모두 integer이면 R 32-bit 정수 오버플로 발생
# (예: 60초@44100Hz → 48000Hz: 2,645,760 * 48,000 ≈ 127억 > 2^31)
# → g를 numeric(double)으로 강제 변환하여 double 연산으로 승격
# ============================================================
safe_resamp <- function(wav, f, g, output = "Wave") {
  resamp(wav, f = as.numeric(f), g = as.numeric(g), output = output)
}

extract_scores <- function(sc_raw) {
  if (is.data.frame(sc_raw)) {
    vals <- as.numeric(sc_raw$score)
  } else {
    vals <- as.numeric(sc_raw)
  }
  vals[!is.na(vals)]
}

normalize_freq_range <- function(f_low, f_high, samp_rate) {
  nyquist_khz <- samp_rate / 2000
  if (f_high > nyquist_khz) {
    f_low <- f_low / 1000
    f_high <- f_high / 1000
  }
  if (f_high > nyquist_khz) f_high <- nyquist_khz * NYQUIST_SAFETY_FACTOR
  if (f_low < 0) f_low <- 0
  if (f_low >= f_high) f_low <- 0
  list(f_low = f_low, f_high = f_high)
}

# ============================================================
# ★ FIX 0: readWave 안전 래퍼
# 일부 WAV 파일(24-bit, 특수 포맷 등)은 readWave()가 내부적으로
# "missing value where TRUE/FALSE needed" 에러를 발생시킴
# → readWave 실패 시 WaveMC로 읽어서 수동 변환
# ============================================================
safe_readWave <- function(filepath) {
  # 1차 시도: 일반 readWave
  w <- tryCatch(
    {
      readWave(filepath)
    },
    error = function(e) {
      log_info(sprintf("  ★ readWave 실패 → WaveMC 폴백 시도: %s", e$message))
      NULL
    }
  )

  if (!is.null(w)) {
    return(w)
  }

  # 2차 시도: WaveMC로 읽기 (멀티채널/특수 포맷 지원)
  wmc <- tryCatch(
    {
      readWave(filepath, toWaveMC = TRUE)
    },
    error = function(e) {
      log_error(sprintf("  WaveMC 로드도 실패: %s", e$message))
      NULL
    }
  )

  if (is.null(wmc)) stop(sprintf("WAV 파일을 읽을 수 없습니다: %s", filepath))

  # WaveMC → Wave 변환 (첫 채널 또는 채널 평균)
  n_channels <- ncol(wmc)
  log_info(sprintf(
    "  ★ WaveMC 로드 성공: %d채널, %d Hz, %d bit",
    n_channels, wmc@samp.rate, wmc@bit
  ))

  if (n_channels >= 2) {
    # 스테레오 → 모노 (두 채널 평균)
    mono_data <- as.integer(round(
      (as.numeric(wmc[, 1]) + as.numeric(wmc[, 2])) / 2
    ))
    log_info("  ★ WaveMC 스테레오 → 모노 변환")
  } else {
    mono_data <- as.integer(wmc[, 1])
  }

  Wave(
    left = mono_data, samp.rate = wmc@samp.rate,
    bit = wmc@bit, pcm = TRUE
  )
}

# ============================================================
# ★ FIX 1: 스테레오 → 모노 변환 + 비트 깊이 정규화
# monitoR::corMatch는 내부적으로 left 채널만 사용하는데
# 스테레오 writeWave → readWave 과정에서 문제 발생 가능
# 24-bit, 32-bit 등은 writeWave/resamp에서 오류 발생 → 16-bit로 변환
# ============================================================
ensure_mono <- function(wav) {
  if (isTRUE(wav@stereo)) {
    log_info("  ★ 스테레오 → 모노 변환 (두 채널 평균)")
    left <- as.numeric(wav@left)
    right <- as.numeric(wav@right)
    mono_data <- as.integer(round((left + right) / 2))
    wav <- Wave(
      left = mono_data, samp.rate = wav@samp.rate,
      bit = wav@bit, pcm = TRUE
    )
  }

  # 24-bit/32-bit → 16-bit 변환 (tuneR의 writeWave/resamp 안정성)
  if (wav@bit != 16) {
    log_info(sprintf("  ★ %d-bit → 16-bit 변환", wav@bit))
    samples <- as.numeric(wav@left)
    old_max <- 2^(wav@bit - 1) - 1
    new_max <- 32767

    # 정규화 후 16-bit 스케일링
    if (old_max > 0) {
      samples <- samples / old_max * new_max
    }
    samples <- as.integer(round(pmin(pmax(samples, -new_max), new_max)))
    wav <- Wave(
      left = samples, samp.rate = wav@samp.rate,
      bit = 16L, pcm = TRUE
    )
  }

  wav
}

# ============================================================
# ★ FIX 2: 진폭 정규화
# 야외 녹음은 종종 피크가 2~5%에 불과 → 수치 정밀도 문제
# ============================================================
normalize_amplitude <- function(wav) {
  samples <- as.numeric(wav@left)
  current_peak <- max(abs(samples))
  max_val <- 2^(wav@bit - 1) - 1 # 16bit → 32767
  current_ratio <- current_peak / max_val

  if (current_ratio < 0.1) {
    target <- 0.8
    gain <- (target * max_val) / current_peak
    log_info(sprintf(
      "  ★ 진폭 정규화: 피크 %.1f%% → %.0f%% (gain=%.1fx)",
      current_ratio * 100, target * 100, gain
    ))
    samples <- as.integer(round(samples * gain))
    samples[samples > max_val] <- max_val
    samples[samples < -max_val] <- -max_val
    wav <- Wave(
      left = samples, samp.rate = wav@samp.rate,
      bit = wav@bit, pcm = TRUE
    )
  } else {
    log_debug(sprintf("  진폭 OK: 피크 %.1f%%", current_ratio * 100))
  }
  wav
}

# ============================================================
# ★ FIX 3: 주파수 범위 자동 감지/보정
# 사용자 설정 대역에 에너지가 <1%이면 실제 에너지 대역으로 교체
# ============================================================
auto_detect_freq_range <- function(wav, t_start = NULL, t_end = NULL) {
  sr <- wav@samp.rate
  samples <- wav@left

  if (!is.null(t_start) && !is.null(t_end)) {
    s1 <- max(1, round(t_start * sr))
    s2 <- min(length(samples), round(t_end * sr))
    if (s2 > s1) samples <- samples[s1:s2]
  }

  temp_wav <- Wave(
    left = as.integer(samples), samp.rate = sr,
    bit = wav@bit, pcm = TRUE
  )
  spec <- tryCatch(meanspec(temp_wav, f = sr, plot = FALSE), error = function(e) NULL)

  if (is.null(spec) || nrow(spec) < 10) {
    return(list(f_low = 0.5, f_high = sr / 2000 * 0.8, peak = 1.0))
  }

  freqs <- spec[, 1] # kHz
  energy <- spec[, 2]^2
  total <- sum(energy)
  if (total == 0) {
    return(list(f_low = 0.5, f_high = sr / 2000 * 0.8, peak = 1.0))
  }

  cum <- cumsum(energy) / total
  f_5 <- freqs[min(which(cum >= 0.05))]
  f_95 <- freqs[min(which(cum >= 0.95))]
  peak <- freqs[which.max(energy)]

  margin <- (f_95 - f_5) * 0.3
  list(f_low = max(0, f_5 - margin), f_high = f_95 + margin, peak = peak)
}

# ★ 검출 구간별 실제 주파수 바운딩 박스 계산
# 피크 주파수 기반: 스펙트로그램에서 가장 강한 주파수를 찾고,
# 피크 에너지의 10% 이상인 대역으로 바운딩 박스를 산출한다.
detect_freq_bounds <- function(segment, ref_f_low, ref_f_high) {
  tryCatch(
    {
      sr <- segment@samp.rate

      # 2D 스펙트로그램으로 시간-주파수 에너지 분석
      sp <- tryCatch(
        spectro(segment, f = sr, plot = FALSE, ovlp = 50,
                wl = min(1024, 2^floor(log2(length(segment@left) / 2)))),
        error = function(e) NULL
      )

      if (is.null(sp)) {
        # spectro 실패 시 meanspec 폴백
        spec <- meanspec(segment, f = sr, plot = FALSE)
        if (is.null(spec) || nrow(spec) < 5) {
          return(list(det_f_low = ref_f_low, det_f_high = ref_f_high))
        }
        freqs <- spec[, 1]  # kHz
        energy <- spec[, 2]^2
      } else {
        freqs <- sp$freq   # kHz
        amp <- sp$amp      # matrix [freq × time]
        # 각 프레임에서 피크 주파수 → 중앙값 (로버스트한 피크 추정)
        energy <- rowMeans(amp^2)  # 평균 에너지 스펙트럼
      }

      # 템플릿 주파수 범위 내로 제한
      in_band <- freqs >= ref_f_low & freqs <= ref_f_high
      if (sum(in_band) < 3) {
        return(list(det_f_low = ref_f_low, det_f_high = ref_f_high))
      }

      band_freqs <- freqs[in_band]
      band_energy <- energy[in_band]

      if (max(band_energy) <= 0) {
        return(list(det_f_low = ref_f_low, det_f_high = ref_f_high))
      }

      # ★ 피크 주파수 찾기
      peak_idx <- which.max(band_energy)
      peak_energy <- band_energy[peak_idx]
      peak_freq <- band_freqs[peak_idx]

      # ★ 피크에서 아래/위로 확장: 에너지가 피크의 10% 이상인 구간
      threshold <- peak_energy * 0.10
      n <- length(band_freqs)

      # 아래쪽 확장
      low_idx <- peak_idx
      for (i in seq(peak_idx - 1, 1, by = -1)) {
        if (band_energy[i] >= threshold) {
          low_idx <- i
        } else {
          break  # 연속 구간만 포함
        }
      }

      # 위쪽 확장
      high_idx <- peak_idx
      for (i in seq(peak_idx + 1, n, by = 1)) {
        if (band_energy[i] >= threshold) {
          high_idx <- i
        } else {
          break
        }
      }

      det_low <- band_freqs[low_idx]
      det_high <- band_freqs[high_idx]

      # 10% 마진 추가
      margin <- (det_high - det_low) * 0.10
      det_low <- max(ref_f_low, det_low - margin)
      det_high <- min(ref_f_high, det_high + margin)

      # 최소 폭 보장 (0.2 kHz)
      if (det_high - det_low < 0.2) {
        mid <- (det_low + det_high) / 2
        det_low <- max(ref_f_low, mid - 0.1)
        det_high <- min(ref_f_high, mid + 0.1)
      }

      list(det_f_low = round(det_low, 3), det_f_high = round(det_high, 3))
    },
    error = function(e) {
      list(det_f_low = ref_f_low, det_f_high = ref_f_high)
    }
  )
}

# ★ STFT 마스킹: 검출된 바운딩 박스 영역의 주파수 대역을 제거
# 2개 이상의 소리가 겹칠 때, 먼저 검출된 소리를 제거하여
# 숨겨진 소리를 재검출할 수 있도록 한다.
stft_mask_audio <- function(wav, bboxes, expansion = 1.5) {
  sr <- wav@samp.rate
  samples <- as.numeric(wav@left)
  n <- length(samples)

  if (n < 256 || length(bboxes) == 0) return(wav)

  # STFT 파라미터
  wl <- 1024L
  hop <- wl %/% 4L  # 75% 오버랩
  n_frames <- max(1L, (n - wl) %/% hop + 1L)

  # Hanning 윈도우
  window_fn <- 0.5 * (1 - cos(2 * pi * (0:(wl - 1)) / wl))

  # 주파수 bin → kHz 매핑 (양수 주파수만, wl/2+1개)
  n_bins <- wl %/% 2L + 1L
  freq_khz <- (0:(n_bins - 1L)) * sr / wl / 1000  # kHz

  # 프레임 중심 시간 (초)
  frame_times <- ((seq_len(n_frames) - 1L) * hop + wl / 2) / sr

  # STFT 순방향: 복소수 행렬 (wl x n_frames)
  stft <- matrix(complex(real = 0, imaginary = 0), nrow = wl, ncol = n_frames)
  for (i in seq_len(n_frames)) {
    s1 <- (i - 1L) * hop + 1L
    s2 <- s1 + wl - 1L
    if (s2 > n) {
      frame <- c(samples[s1:n], rep(0, s2 - n))
    } else {
      frame <- samples[s1:s2]
    }
    stft[, i] <- fft(frame * window_fn)
  }

  # 바운딩 박스 마스킹 (50% 확장)
  for (bbox in bboxes) {
    t_center <- (bbox$t_start + bbox$t_end) / 2
    t_half <- (bbox$t_end - bbox$t_start) / 2 * expansion
    f_center <- (bbox$f_low + bbox$f_high) / 2
    f_half <- (bbox$f_high - bbox$f_low) / 2 * expansion

    t_lo <- t_center - t_half
    t_hi <- t_center + t_half
    f_lo <- max(0, f_center - f_half)
    f_hi <- f_center + f_half

    # 해당 시간/주파수 bin 마스킹
    t_mask <- frame_times >= t_lo & frame_times <= t_hi
    f_mask <- freq_khz >= f_lo & freq_khz <= f_hi

    if (any(t_mask) && any(f_mask)) {
      f_indices <- which(f_mask)
      t_indices <- which(t_mask)

      # 양수 주파수 마스킹
      for (fi in f_indices) {
        stft[fi, t_indices] <- 0 + 0i
        # 대칭 (음수 주파수) 마스킹
        mirror_fi <- wl - fi + 2L
        if (mirror_fi >= 1L && mirror_fi <= wl && mirror_fi != fi) {
          stft[mirror_fi, t_indices] <- 0 + 0i
        }
      }
    }
  }

  # ISTFT: Overlap-Add 복원
  output <- numeric(n)
  win_sum <- numeric(n)

  for (i in seq_len(n_frames)) {
    s1 <- (i - 1L) * hop + 1L
    s2 <- min(s1 + wl - 1L, n)
    frame_len <- s2 - s1 + 1L

    reconstructed <- Re(fft(stft[, i], inverse = TRUE)) / wl
    output[s1:s2] <- output[s1:s2] + reconstructed[1:frame_len] * window_fn[1:frame_len]
    win_sum[s1:s2] <- win_sum[s1:s2] + window_fn[1:frame_len]^2
  }

  # 윈도우 합 정규화
  nonzero <- win_sum > 1e-10
  output[nonzero] <- output[nonzero] / win_sum[nonzero]

  # 클리핑 방지 정규화
  peak <- max(abs(output))
  orig_peak <- max(abs(samples))
  if (peak > 0 && orig_peak > 0) {
    output <- output * (orig_peak / peak)
  }

  # 정수 변환 (원본 bit depth 유지)
  max_val <- 2^(wav@bit - 1) - 1
  output <- as.integer(round(pmin(pmax(output, -max_val), max_val)))

  Wave(left = output, samp.rate = sr, bit = wav@bit, pcm = TRUE)
}


validate_and_fix_freq_range <- function(wav, f_low, f_high, t_start = NULL, t_end = NULL) {
  sr <- wav@samp.rate
  nyquist_khz <- sr / 2000

  # Hz → kHz 변환
  if (f_high > nyquist_khz) {
    f_low <- f_low / 1000
    f_high <- f_high / 1000
  }
  if (f_high > nyquist_khz) f_high <- nyquist_khz * NYQUIST_SAFETY_FACTOR
  if (f_low < 0) f_low <- 0
  if (f_low >= f_high) f_low <- 0

  # 에너지 검증
  spec <- tryCatch(
    {
      samples <- wav@left
      if (!is.null(t_start) && !is.null(t_end)) {
        s1 <- max(1, round(t_start * sr))
        s2 <- min(length(samples), round(t_end * sr))
        if (s2 > s1) samples <- samples[s1:s2]
      }
      tw <- Wave(left = as.integer(samples), samp.rate = sr, bit = wav@bit, pcm = TRUE)
      meanspec(tw, f = sr, plot = FALSE)
    },
    error = function(e) NULL
  )

  if (!is.null(spec) && nrow(spec) > 0) {
    freqs <- spec[, 1]
    energy <- spec[, 2]^2
    total <- sum(energy)
    if (total > 0) {
      user_band <- freqs >= f_low & freqs <= f_high
      user_ratio <- sum(energy[user_band]) / total

      log_debug(sprintf(
        "  사용자 대역 [%.2f-%.2f kHz] 에너지: %.4f%%",
        f_low, f_high, user_ratio * 100
      ))

      if (user_ratio < 0.01) {
        auto <- auto_detect_freq_range(wav, t_start, t_end)

        # ★ FIX: 자동보정된 대역이 새소리로 타당한지 검증
        # 조류 울음 최소 기준: 피크 주파수 1kHz 이상 (저음 새 포함)
        # 피크가 1kHz 미만이면 배경 소음(바람, 진동 등) 일 가능성이 높음
        BIRD_MIN_PEAK_KHZ <- 0.5

        if (auto$peak < BIRD_MIN_PEAK_KHZ) {
          log_info(sprintf(
            "  ⚠ 주파수 자동보정 감지: [%.2f-%.2f] → [%.3f-%.3f] kHz (피크: %.3f kHz)",
            f_low, f_high, auto$f_low, auto$f_high, auto$peak
          ))
          log_info(sprintf(
            "     이유: 설정 대역 에너지 %.4f%% (기준 <1%%)", user_ratio * 100
          ))
          log_info(paste0(
            "  ★★ 경고: 자동감지 피크(", round(auto$peak, 3), " kHz)가 ",
            BIRD_MIN_PEAK_KHZ, " kHz 미만 → 배경 소음 대역으로 추정됩니다."
          ))
          log_info("     이 음원에 해당 종의 울음이 포함되어 있는지 확인하세요.")
          log_info("     자동보정을 적용하되 자동 튜닝 결과는 신뢰도가 낮을 수 있습니다.")
          # 보정은 적용하되, 최소 bird 주파수(0.5kHz)로 하한을 고정
          f_low <- max(auto$f_low, 0.5)
          f_high <- max(auto$f_high, BIRD_MIN_PEAK_KHZ * 1.5)
        } else {
          log_info(sprintf(
            "  ★★ 주파수 자동보정! [%.2f-%.2f] → [%.3f-%.3f] kHz (피크: %.3f kHz)",
            f_low, f_high, auto$f_low, auto$f_high, auto$peak
          ))
          log_info(sprintf(
            "     이유: 설정 대역에 에너지 %.4f%%뿐 (기준 <1%%)", user_ratio * 100
          ))
          f_low <- auto$f_low
          f_high <- auto$f_high
        }
      } else if (user_ratio < 0.05) {
        # 1~5% 에너지: 자동보정은 하지 않으나 경고 표시
        log_info(sprintf(
          "  ⚠ 경고: 사용자 대역 [%.2f-%.2f kHz] 에너지 %.1f%% (낮음)",
          f_low, f_high, user_ratio * 100
        ))
        log_info("     대역 설정이 실제 울음 주파수와 다를 수 있습니다. 확인하세요.")
      }
    }
  }

  if (f_high > nyquist_khz) f_high <- nyquist_khz * NYQUIST_SAFETY_FACTOR
  list(f_low = f_low, f_high = f_high)
}

validate_species_config <- function(sp) {
  errors <- character()
  if (is.null(sp$name) || nchar(sp$name) == 0) errors <- c(errors, "종 이름이 비어 있음")
  if (is.null(sp$wav_path) || !file.exists(sp$wav_path)) errors <- c(errors, paste0("파일 없음: ", sp$wav_path))
  if (!is.null(sp$t_start) && !is.null(sp$t_end) && sp$t_start >= sp$t_end) errors <- c(errors, "t_start >= t_end")
  if (!is.null(sp$f_low) && !is.null(sp$f_high) && sp$f_low >= sp$f_high) errors <- c(errors, "f_low >= f_high")
  if (!is.null(sp$cutoff) && (sp$cutoff < 0 || sp$cutoff > 1)) errors <- c(errors, "cutoff 범위 이탈")
  errors
}

print_section <- function(title, subtitle = NULL) {
  sep <- strrep("=", 60)
  cat(paste0("\n", sep, "\n"))
  cat(sprintf("  %s\n", title))
  if (!is.null(subtitle)) for (s in subtitle) cat(sprintf("  %s\n", s))
  cat(paste0(sep, "\n"))
}

# ============================================================
# 종합 판별 핵심 함수들
# ============================================================

#' WAV에서 구간 추출 (초 단위)
#' @param wav Wave 객체
#' @param t_start 시작 시간 (초)
#' @param t_end 종료 시간 (초)
#' @return Wave 객체
extract_segment <- function(wav, t_start, t_end) {
  if (is.na(t_start) || is.na(t_end) || is.null(t_start) || is.null(t_end)) {
    return(NULL)
  }
  sr <- wav@samp.rate
  s1 <- max(1, round(t_start * sr))
  s2 <- min(length(wav@left), round(t_end * sr))
  if (is.na(s1) || is.na(s2) || s2 <= s1) {
    return(NULL)
  }

  seg_data <- wav@left[s1:s2]
  Wave(left = seg_data, samp.rate = sr, bit = wav@bit, pcm = TRUE)
}

#' 스펙트로그램 상관 유사도 (per-segment corMatch 대용)
#' corMatch의 핵심 원리(스펙트로그램 간 Pearson 상관)를
#' 임의 두 세그먼트에 대해 직접 계산한다.
#' auto-tune에서 cor_score의 변별력을 정확히 측정하기 위해 사용.
#' @param wav_template 템플릿 Wave
#' @param wav_segment 후보 구간 Wave
#' @return -1~1 상관계수 (corMatch와 동일 스케일)
compute_spectrogram_cor <- function(wav_template, wav_segment) {
  tryCatch(
    {
      sr_t <- wav_template@samp.rate
      sr_s <- wav_segment@samp.rate

      # 스펙트로그램 파라미터 (corMatch 내부 기본값과 유사하게)
      wl <- 512
      ovlp <- 50

      spec_t <- seewave::spectro(wav_template,
        f = sr_t, wl = wl,
        ovlp = ovlp, plot = FALSE
      )$amp
      spec_s <- seewave::spectro(wav_segment,
        f = sr_s, wl = wl,
        ovlp = ovlp, plot = FALSE
      )$amp

      if (is.null(spec_t) || is.null(spec_s)) {
        return(0)
      }
      if (length(spec_t) < 4 || length(spec_s) < 4) {
        return(0)
      }

      # 크기 맞추기: 작은 쪽에 맞춤 (행=주파수빈, 열=시간프레임)
      n_freq <- min(nrow(spec_t), nrow(spec_s))
      n_time <- min(ncol(spec_t), ncol(spec_s))
      spec_t <- spec_t[1:n_freq, 1:n_time]
      spec_s <- spec_s[1:n_freq, 1:n_time]

      # Pearson 상관 (corMatch와 동일 원리)
      cor(as.vector(spec_t), as.vector(spec_s))
    },
    error = function(e) {
      log_debug(sprintf("    스펙트로그램 상관 계산 오류: %s", e$message))
      0
    }
  )
}

#' MFCC 코사인 유사도 계산
#' @param wav_template 템플릿 Wave
#' @param wav_segment 후보 구간 Wave
#' @param numcep MFCC 계수 수
#' @return 0~1 유사도
compute_mfcc_similarity <- function(wav_template, wav_segment, numcep = 13) {
  tryCatch(
    {
      sr_t <- wav_template@samp.rate
      sr_s <- wav_segment@samp.rate

      mfcc_t <- tuneR::melfcc(wav_template,
        sr = sr_t, numcep = numcep,
        wintime = 0.025, hoptime = 0.010
      )
      mfcc_s <- tuneR::melfcc(wav_segment,
        sr = sr_s, numcep = numcep,
        wintime = 0.025, hoptime = 0.010
      )

      if (is.null(mfcc_t) || is.null(mfcc_s)) {
        return(0)
      }
      if (nrow(mfcc_t) == 0 || nrow(mfcc_s) == 0) {
        return(0)
      }

      # 프레임별 평균 MFCC 벡터로 요약 (전체 특성)
      mean_t <- colMeans(mfcc_t, na.rm = TRUE)
      mean_s <- colMeans(mfcc_s, na.rm = TRUE)

      # NaN/NA 보호: melfcc가 NaN을 반환하면 유사도 계산 불가
      if (any(is.na(mean_t)) || any(is.na(mean_s))) {
        return(0)
      }

      # 코사인 유사도
      dot_prod <- sum(mean_t * mean_s)
      norm_t <- sqrt(sum(mean_t^2))
      norm_s <- sqrt(sum(mean_s^2))

      if (is.na(norm_t) || is.na(norm_s) || norm_t == 0 || norm_s == 0) {
        return(0)
      }

      sim <- dot_prod / (norm_t * norm_s)
      if (is.na(sim)) return(0)
      # -1~1을 0~1로 스케일링
      max(0, (sim + 1) / 2)
    },
    error = function(e) {
      log_info(sprintf("    ⚠ MFCC 코사인 유사도 계산 오류: %s", e$message))
      0
    }
  )
}

#' MFCC 시퀀스 DTW 유사도 (프레임 단위 비교 — 시간 변형 대응)
#' @return 0~1 유사도
compute_mfcc_dtw_similarity <- function(wav_template, wav_segment, numcep = 13) {
  tryCatch(
    {
      sr_t <- wav_template@samp.rate
      sr_s <- wav_segment@samp.rate

      mfcc_t <- tuneR::melfcc(wav_template,
        sr = sr_t, numcep = numcep,
        wintime = 0.025, hoptime = 0.010
      )
      mfcc_s <- tuneR::melfcc(wav_segment,
        sr = sr_s, numcep = numcep,
        wintime = 0.025, hoptime = 0.010
      )

      if (is.null(mfcc_t) || is.null(mfcc_s) ||
        nrow(mfcc_t) < 3 || nrow(mfcc_s) < 3) {
        log_info(sprintf(
          "    ⚠ MFCC-DTW: 프레임 부족 (t=%s, s=%s)",
          if (is.null(mfcc_t)) "NULL" else nrow(mfcc_t),
          if (is.null(mfcc_s)) "NULL" else nrow(mfcc_s)
        ))
        return(0)
      }

      # ★ 코사인 거리 행렬 계산 (스케일 무관, 0~2 범위)
      # Euclidean은 차원 수에 비례하여 거리가 커지지만
      # 코사인 거리는 차원에 무관하게 0(동일)~2(반대) 범위
      norm_t <- sqrt(rowSums(mfcc_t^2))
      norm_s <- sqrt(rowSums(mfcc_s^2))
      # 0-벡터 보호
      norm_t[norm_t < 1e-10] <- 1
      norm_s[norm_s < 1e-10] <- 1
      mfcc_t_unit <- mfcc_t / norm_t
      mfcc_s_unit <- mfcc_s / norm_s

      # 코사인 거리 행렬: 1 - cosine_similarity (0~2)
      cos_sim_matrix <- tcrossprod(mfcc_t_unit, mfcc_s_unit)
      cos_dist_matrix <- 1 - cos_sim_matrix

      # ★ NaN/Inf 보호: 수치 불안정으로 생긴 NaN을 1.0(무관)으로 대체
      cos_dist_matrix[is.nan(cos_dist_matrix)] <- 1.0
      cos_dist_matrix[is.infinite(cos_dist_matrix)] <- 2.0
      # 음수 거리 방지 (부동소수점 오차)
      cos_dist_matrix[cos_dist_matrix < 0] <- 0.0

      # DTW — 제약 없이 실행 (프레임 길이 차이가 클 수 있으므로)
      alignment <- tryCatch(
        dtw::dtw(cos_dist_matrix,
          step.pattern = dtw::symmetric2,
          window.type = "none"
        ),
        error = function(e) {
          log_debug(sprintf("    MFCC-DTW fallback 시도: %s", e$message))
          # 최후 수단: asymmetric step pattern
          tryCatch(
            dtw::dtw(cos_dist_matrix,
              step.pattern = dtw::asymmetric,
              window.type = "none"
            ),
            error = function(e2) NULL
          )
        }
      )

      if (is.null(alignment)) {
        log_info(sprintf(
          "    ⚠ MFCC-DTW: DTW 정렬 실패 (행렬 %dx%d)",
          nrow(cos_dist_matrix), ncol(cos_dist_matrix)
        ))
        return(0)
      }

      nd <- alignment$normalizedDistance

      # 코사인 거리의 normalizedDistance: 동일=0, 무관=~1, 반대=~2
      # exp(-DTW_ALPHA * nd)로 매핑: nd=0→1.0, nd=0.3→0.55, nd=1→0.14
      score <- exp(-DTW_ALPHA * nd)

      log_debug(sprintf(
        "    MFCC-DTW: frames_t=%d, frames_s=%d, cosDist=%.3f, score=%.4f",
        nrow(mfcc_t), nrow(mfcc_s), nd, score
      ))

      score
    },
    error = function(e) {
      log_info(sprintf("    ⚠ MFCC-DTW 계산 오류: %s", e$message))
      0
    }
  )
}

#' 주파수 궤적(Dominant Frequency Contour) DTW 유사도
#' dfreq()로 추출한 주파수 컨투어를 [0,1]로 정규화한 뒤 DTW 비교.
#' 정규화를 통해 절대 주파수(kHz)에 무관하게 "모양"만 비교한다.
#' @return 0~1 유사도
compute_freq_contour_dtw <- function(wav_template, wav_segment, f_low, f_high) {
  tryCatch(
    {
      sr <- wav_template@samp.rate

      # bandpass 범위 설정 (kHz → Hz)
      bp_low <- f_low * 1000
      bp_high <- f_high * 1000

      # dominant frequency 추출
      df_t <- seewave::dfreq(wav_template,
        f = sr, bandpass = c(bp_low, bp_high),
        ovlp = 50, threshold = 5, plot = FALSE
      )
      df_s <- seewave::dfreq(wav_segment,
        f = sr, bandpass = c(bp_low, bp_high),
        ovlp = 50, threshold = 5, plot = FALSE
      )

      # 유효한 주파수 값만 추출 (0 또는 NA 제거)
      freq_t <- df_t[, 2]
      freq_s <- df_s[, 2]
      freq_t[freq_t == 0] <- NA
      freq_s[freq_s == 0] <- NA

      # 유효 프레임 비율 체크: 대부분 NA이면 신뢰할 수 없음
      valid_ratio_t <- sum(!is.na(freq_t)) / length(freq_t)
      valid_ratio_s <- sum(!is.na(freq_s)) / length(freq_s)
      if (valid_ratio_t < 0.3 || valid_ratio_s < 0.3) {
        return(0) # 유효 프레임이 30% 미만이면 포기
      }

      # NA 보간 (선형)
      freq_t <- approx(seq_along(freq_t), freq_t, seq_along(freq_t), rule = 2)$y
      freq_s <- approx(seq_along(freq_s), freq_s, seq_along(freq_s), rule = 2)$y

      if (length(freq_t) < 3 || length(freq_s) < 3) {
        return(0)
      }
      if (all(is.na(freq_t)) || all(is.na(freq_s))) {
        return(0)
      }

      # ★ [0,1] 정규화: 종의 주파수 범위 기준으로 스케일링
      #    dfreq는 kHz 단위를 반환하므로 f_low/f_high도 kHz
      freq_range <- f_high - f_low
      if (freq_range < 0.001) freq_range <- 0.001 # 0 나눗셈 방지
      freq_t <- (freq_t - f_low) / freq_range
      freq_s <- (freq_s - f_low) / freq_range
      # 범위 클램핑 (보간으로 약간 벗어날 수 있음)
      freq_t <- pmax(0, pmin(1, freq_t))
      freq_s <- pmax(0, pmin(1, freq_s))

      # Sakoe-Chiba: 길이 차이 + 여유분 (의미 있는 제약)
      len_diff <- abs(length(freq_t) - length(freq_s))
      max_len <- max(length(freq_t), length(freq_s))
      sc_window <- as.integer(len_diff + max_len * 0.15)
      alignment <- tryCatch(
        dtw::dtw(freq_t, freq_s,
          step.pattern = dtw::symmetric2,
          window.type = "sakoechiba",
          window.size = sc_window
        ),
        error = function(e) {
          dtw::dtw(freq_t, freq_s,
            step.pattern = dtw::symmetric2,
            window.type = "none"
          )
        }
      )
      exp(-DTW_ALPHA * alignment$normalizedDistance)
    },
    error = function(e) {
      log_debug(sprintf("    주파수궤적 DTW 오류: %s", e$message))
      0
    }
  )
}

#' 진폭 포락선(Amplitude Envelope) DTW 유사도
#' 에너지 정규화(L2 norm)를 사용하여 스파이크에 견고하게 비교.
#' 양쪽 모두 동일한 포인트 수로 리샘플링하여 시간 분해능 일치.
#' @return 0~1 유사도
compute_envelope_dtw <- function(wav_template, wav_segment) {
  tryCatch(
    {
      sr <- wav_template@samp.rate

      # 진폭 포락선 추출 (힐버트 변환)
      env_t <- env(wav_template, f = sr, envt = "hil", plot = FALSE)
      env_s <- env(wav_segment, f = sr, envt = "hil", plot = FALSE)

      if (length(env_t) < 3 || length(env_s) < 3) {
        return(0)
      }

      env_t <- as.numeric(env_t)
      env_s <- as.numeric(env_s)

      # ★ 양쪽 모두 동일한 포인트 수로 리샘플링 (시간 분해능 통일)
      target_points <- 200
      env_t <- approx(seq_along(env_t), env_t, n = target_points)$y
      env_s <- approx(seq_along(env_s), env_s, n = target_points)$y

      # ★ 에너지 정규화 (L2 norm): 스파이크에 견고, 전체 에너지 분포 보존
      l2_t <- sqrt(sum(env_t^2))
      l2_s <- sqrt(sum(env_s^2))
      if (l2_t > 0) env_t <- env_t / l2_t
      if (l2_s > 0) env_s <- env_s / l2_s

      # Sakoe-Chiba: 동일 길이이므로 15% 여유 (시간 왜곡 허용 범위)
      sc_window <- as.integer(target_points * 0.15)
      alignment <- tryCatch(
        dtw::dtw(env_t, env_s,
          step.pattern = dtw::symmetric2,
          window.type = "sakoechiba",
          window.size = sc_window
        ),
        error = function(e) {
          dtw::dtw(env_t, env_s,
            step.pattern = dtw::symmetric2,
            window.type = "none"
          )
        }
      )
      exp(-DTW_ALPHA * alignment$normalizedDistance)
    },
    error = function(e) {
      log_debug(sprintf("    포락선 DTW 오류: %s", e$message))
      0
    }
  )
}

#' 주파수 대역 에너지 집중도
#' 해당 종의 울음 주파수 대역에 전체 에너지 대비 몇 %가 집중되어 있는가
#' 주파수 대역 에너지 집중도
#' 종의 주파수 대역에 에너지가 집중될수록 높은 값을 반환.
#' 전체 스펙트럼 대비 비율은 대역 너비에 의존하므로,
#' 대역 내 평균 에너지 밀도 vs 대역 외 평균 에너지 밀도의 비율로 계산.
#' @return 0~1 에너지 집중도 (높을수록 대역에 에너지 집중)
compute_band_energy_ratio <- function(wav_segment, f_low, f_high) {
  tryCatch(
    {
      sr <- wav_segment@samp.rate
      spec <- meanspec(wav_segment, f = sr, plot = FALSE)

      if (is.null(spec) || nrow(spec) == 0) {
        return(0)
      }

      freqs <- spec[, 1] # kHz
      power <- spec[, 2]^2

      # 대역 내/외 인덱스
      in_band <- freqs >= f_low & freqs <= f_high
      n_in <- sum(in_band)
      n_out <- sum(!in_band)

      if (n_in == 0 || n_out == 0) {
        return(0)
      }

      # 평균 에너지 밀도 (bin 당)
      mean_in <- sum(power[in_band]) / n_in
      mean_out <- sum(power[!in_band]) / n_out

      if (mean_in + mean_out < 1e-10) {
        return(0)
      }

      # 비율 → 시그모이드 정규화 (0~1)
      # ratio=1 → 대역 내외 동일 → 0.5
      # ratio=10 → 대역에 10배 집중 → ~0.9
      # ratio=0.1 → 대역에 1/10 → ~0.1
      ratio_db <- 10 * log10(mean_in / max(mean_out, 1e-10))
      1 / (1 + exp(-0.3 * (ratio_db - 0)))
    },
    error = function(e) {
      log_debug(sprintf("    대역에너지 계산 오류: %s", e$message))
      0
    }
  )
}

#' 종합 점수 계산
#' @param scores_list 각 판별 요소의 점수가 담긴 named list
#' @param weights 가중치 named list
#' @return 0~1 종합 점수
compute_composite_score <- function(scores_list, weights) {
  total_weight <- 0
  weighted_sum <- 0

  for (nm in names(weights)) {
    if (!is.null(scores_list[[nm]]) && !is.na(scores_list[[nm]])) {
      weighted_sum <- weighted_sum + weights[[nm]] * scores_list[[nm]]
      total_weight <- total_weight + weights[[nm]]
    }
  }

  if (total_weight == 0) {
    return(0)
  }
  weighted_sum / total_weight # 누락된 점수가 있으면 나머지에서 재정규화
}

# ============================================================
# C1: Harmonic Ratio (조화 비율) 함수
# ============================================================
#' 시간 도메인 자기상관 기반 조화 비율 (HNR)
#' 새소리는 성도(syrinx)의 주기적 진동으로 높은 자기상관을 보이고
#' 바람/소음 등 비주기 신호는 낮은 자기상관을 보인다.
#' @param wav_segment 분석할 Wave 객체
#' @param f_low 종 주파수 하한 (kHz)
#' @param f_high 종 주파수 상한 (kHz)
#' @return 0~1 조화 비율 (높을수록 주기적/조화적)
compute_harmonic_ratio <- function(wav_segment, f_low, f_high) {
  tryCatch(
    {
      sr <- wav_segment@samp.rate

      # 1) 종의 주파수 대역으로 밴드패스 필터링
      bp_from <- f_low * 1000 # kHz → Hz
      bp_to <- min(f_high * 1000, sr / 2 - 1) # 나이퀴스트 이하로 제한
      if (bp_from >= bp_to || bp_from < 1) {
        return(0)
      }
      filtered <- seewave::ffilter(wav_segment,
        f = sr,
        from = bp_from, to = bp_to,
        output = "Wave"
      )

      # 2) 필터링된 시간 도메인 신호 추출
      sig <- filtered@left
      if (length(sig) < sr * 0.01) { # 최소 10ms
        return(0)
      }

      # 3) 시간 도메인 자기상관 — 기본 주파수 범위의 lag를 탐색
      #    f_high에 해당하는 최소 주기 ~ f_low에 해당하는 최대 주기
      min_lag <- as.integer(sr / (f_high * 1000)) # 최고 주파수의 주기 (샘플)
      max_lag <- as.integer(sr / (f_low * 1000)) # 최저 주파수의 주기 (샘플)
      min_lag <- max(1, min_lag)
      max_lag <- min(max_lag, length(sig) %/% 2)

      if (min_lag >= max_lag) {
        return(0)
      }

      acf_result <- acf(sig, lag.max = max_lag, plot = FALSE)
      acf_vals <- as.numeric(acf_result$acf)

      # 4) 관심 lag 범위(min_lag ~ max_lag)에서 최대 피크 찾기
      #    lag 인덱스는 1-based이므로 lag=k는 acf_vals[k+1]
      search_start <- min_lag + 1 # +1: acf_vals[1]은 lag=0
      search_end <- min(max_lag + 1, length(acf_vals))

      if (search_start >= search_end) {
        return(0)
      }

      search_region <- acf_vals[search_start:search_end]

      # 피크 검출: 양쪽보다 큰 지점
      if (length(search_region) < 3) {
        return(max(0, max(search_region)))
      }
      peaks <- which(diff(sign(diff(search_region))) == -2) + 1
      if (length(peaks) == 0) {
        # 피크 없으면 구간 내 최댓값 사용
        return(max(0, max(search_region)))
      }

      # 5) 최대 ACF 피크 값 = 조화 비율
      peak_vals <- search_region[peaks]
      max(0, max(peak_vals))
    },
    error = function(e) {
      log_debug(sprintf("    HR 계산 오류: %s", e$message))
      0
    }
  )
}
# ============================================================
# C1.5: SNR (Signal-to-Noise Ratio) 추정 함수
# ============================================================
#' 대역 내 에너지와 대역 외 에너지의 비율로 SNR을 추정한다.
#' 새소리 주파수 대역에 에너지가 집중될수록 SNR이 높고,
#' 배경 소음이 많으면 대역 외에도 에너지가 분산되어 SNR이 낮아진다.
#' @param wav_segment Wave 객체
#' @param f_low 종 주파수 하한 (kHz)
#' @param f_high 종 주파수 상한 (kHz)
#' @return 0~1 정규화된 SNR (높을수록 깨끗한 신호)
compute_snr_ratio <- function(wav_segment, f_low, f_high) {
  tryCatch(
    {
      sr <- wav_segment@samp.rate
      spec <- meanspec(wav_segment, f = sr, plot = FALSE)
      if (is.null(spec) || nrow(spec) < 10) {
        return(0)
      }

      freqs <- spec[, 1] # kHz
      power <- spec[, 2]^2 # 파워 스펙트럼

      # 대역 내 에너지
      in_band <- freqs >= f_low & freqs <= f_high
      signal_power <- sum(power[in_band])

      # 대역 외 에너지 (전체 - 대역 내)
      noise_power <- sum(power[!in_band])

      if (signal_power + noise_power < 1e-10) {
        return(0)
      }

      # 대역 내 에너지 밀도 vs 대역 외 에너지 밀도 비율
      # (bin 수로 정규화하여 대역 너비에 무관하게)
      n_in <- sum(in_band)
      n_out <- sum(!in_band)
      if (n_in == 0 || n_out == 0) {
        return(0)
      }

      mean_signal <- signal_power / n_in
      mean_noise <- noise_power / n_out

      if (mean_noise < 1e-10) {
        return(1.0) # 소음 없음
      }
      snr_db <- 10 * log10(mean_signal / mean_noise)
      # 시그모이드: midpoint=-3dB (야외 녹음에서 현실적), k=0.25
      # -3dB → 0.5, 7dB → ~0.9, -13dB → ~0.1
      1 / (1 + exp(-0.25 * (snr_db - (-3))))
    },
    error = function(e) {
      log_debug(sprintf("    SNR 계산 오류: %s", e$message))
      0
    }
  )
}
# ============================================================
#' 근접 검출 병합: min_gap(초) 이내의 검출 중 최고 점수만 유지
nms_detections <- function(df, min_gap = 0.5) {
  # NA 제거 (time 또는 composite가 NA인 행)
  df <- df[!is.na(df$time) & !is.na(df$composite), , drop = FALSE]

  if (nrow(df) <= 1) {
    return(df)
  }

  # 종합 점수 내림차순 정렬
  df <- df[order(-df$composite), ]

  keep <- logical(nrow(df))
  keep[1] <- TRUE # 최고 점수는 항상 유지

  for (i in 2:nrow(df)) {
    kept_times <- df$time[keep]
    if (all(abs(df$time[i] - kept_times) >= min_gap)) {
      keep[i] <- TRUE
    }
  }

  result <- df[keep, ]
  result[order(result$time), ] # 시간순 재정렬
}

# ============================================================
# ★ 후보 간 지표 정규화 + 복합점수 재산출
# 각 지표를 후보 풀 내에서 min-max 스케일링하여
# 분산 없는(비구별) 지표의 상수 기여를 제거함
# → 가중치가 큰 구별 지표가 복합점수를 실제로 주도
# ============================================================
normalize_candidate_scores <- function(df, weights) {
  if (is.null(df) || nrow(df) < 3) return(df)

  metric_cols <- c("cor_score", "mfcc_score", "dtw_freq", "dtw_env",
                    "band_energy", "harmonic_ratio", "snr")
  available <- intersect(metric_cols, names(df))

  # 정규화 전 범위 로깅
  range_strs <- vapply(available, function(mn) {
    vals <- df[[mn]]
    sprintf("%s=[%.3f,%.3f]", mn, min(vals, na.rm = TRUE), max(vals, na.rm = TRUE))
  }, character(1))
  log_info(sprintf("  ★ 정규화 전 범위: %s", paste(range_strs, collapse = ", ")))

  # 지표별 min-max 정규화
  norm_data <- list()
  for (mn in available) {
    vals <- df[[mn]]
    vmin <- min(vals, na.rm = TRUE)
    vmax <- max(vals, na.rm = TRUE)
    vrange <- vmax - vmin

    if (vrange > 1e-6) {
      norm_data[[mn]] <- (vals - vmin) / vrange
    } else {
      # 분산 없음 → 0.5 (판별 기여 없음)
      norm_data[[mn]] <- rep(0.5, length(vals))
    }
  }

  # 정규화된 점수로 복합점수 재계산
  for (j in seq_len(nrow(df))) {
    scores_list <- lapply(available, function(mn) norm_data[[mn]][j])
    names(scores_list) <- available
    df$composite[j] <- round(compute_composite_score(scores_list, weights), 4)
  }

  log_info(sprintf("  ★ 정규화 후 종합점수: %.3f ~ %.3f (범위 %.3f)",
    min(df$composite, na.rm = TRUE), max(df$composite, na.rm = TRUE),
    max(df$composite, na.rm = TRUE) - min(df$composite, na.rm = TRUE)))

  df
}

# ============================================================
# ★ 멀티 템플릿 앙상블: 시간 근접 후보를 그룹핑 후 전략별 집계
# 동일 시간대에 여러 템플릿이 검출한 경우 최적 점수를 도출
# ============================================================

#' 멀티 템플릿 앙상블
#' @param df data.frame (species, time, composite, template_label, 7개 지표...)
#' @param time_gap 동일 검출로 간주하는 시간 차이 (초, 기본 0.5)
#' @param strategy "max" | "mean" | "weighted_max"
#' @return 앙상블 후 data.frame (클러스터당 1행)
ensemble_multi_template <- function(df, time_gap = 0.5, strategy = "max") {
  if (nrow(df) <= 1) return(df)

  # 시간순 정렬
  df <- df[order(df$time), ]

  # Greedy 클러스터링: 시간 차이 > time_gap이면 새 클러스터
  cluster_id <- integer(nrow(df))
  cluster_id[1] <- 1
  for (i in 2:nrow(df)) {
    if (df$time[i] - df$time[i - 1] > time_gap) {
      cluster_id[i] <- cluster_id[i - 1] + 1
    } else {
      cluster_id[i] <- cluster_id[i - 1]
    }
  }
  df$cluster <- cluster_id

  metric_cols <- c("cor_score", "mfcc_score", "dtw_freq", "dtw_env",
                   "band_energy", "harmonic_ratio", "snr", "composite")
  metric_cols <- intersect(metric_cols, names(df))

  # 클러스터별 집계
  result_rows <- lapply(unique(cluster_id), function(cid) {
    members <- df[df$cluster == cid, , drop = FALSE]

    if (nrow(members) == 1) {
      # 단일 멤버 → 그대로 반환
      row <- members
      row$n_templates_matched <- 1L
      row$cluster <- NULL
      return(row)
    }

    if (strategy == "max") {
      # 최고 composite 점수 행 선택
      best_idx <- which.max(members$composite)
      row <- members[best_idx, , drop = FALSE]
      row$n_templates_matched <- nrow(members)

    } else if (strategy == "mean") {
      # 모든 지표의 평균
      row <- members[1, , drop = FALSE]
      for (col in metric_cols) {
        row[[col]] <- mean(members[[col]], na.rm = TRUE)
      }
      row$time <- members$time[which.max(members$composite)]
      row$template_label <- paste(unique(members$template_label), collapse = "+")
      row$n_templates_matched <- nrow(members)

    } else if (strategy == "weighted_max") {
      # cor_score를 가중치로 한 가중평균
      w <- pmax(members$cor_score, 0.01)
      w <- w / sum(w)
      row <- members[1, , drop = FALSE]
      for (col in metric_cols) {
        row[[col]] <- round(sum(members[[col]] * w, na.rm = TRUE), 4)
      }
      row$time <- members$time[which.max(members$composite)]
      row$template_label <- paste(unique(members$template_label), collapse = "+")
      row$n_templates_matched <- nrow(members)

    } else {
      # 알 수 없는 전략 → max 폴백
      best_idx <- which.max(members$composite)
      row <- members[best_idx, , drop = FALSE]
      row$n_templates_matched <- nrow(members)
    }

    row$cluster <- NULL
    row
  })

  do.call(rbind, result_rows)
}

# ============================================================
# ★ 후보 위치 보정: 대역 에너지 포락선 기반 피크 재정렬
# corMatch는 부분 매칭으로 울음의 시작/끝에 피크가 생기기 쉬움
# → 대역 에너지가 최대인 실제 울음 중심으로 이동
# ============================================================

#' 전체 음원의 대역 에너지 포락선을 한 번만 계산 (O(N))
#' bandpass filter → 제곱 → 이동 평균 → 시간별 에너지 맵
#' @param wav 전체 음원 Wave 객체
#' @param f_low 주파수 하한 (kHz)
#' @param f_high 주파수 상한 (kHz)
#' @param window_sec 에너지 계산 윈도우 크기 (초)
#' @return list(energy, sr) 또는 NULL
compute_band_energy_envelope <- function(wav, f_low, f_high, window_sec = 0.05) {
  tryCatch(
    {
      sr <- wav@samp.rate

      # 밴드패스 필터링 (전체 음원 1회)
      bp_from <- f_low * 1000 # kHz → Hz
      bp_to <- min(f_high * 1000, sr / 2 - 1)
      if (bp_from >= bp_to || bp_from < 1) return(NULL)

      filtered <- seewave::ffilter(wav,
        f = sr, from = bp_from, to = bp_to,
        output = "Wave"
      )

      # ★ 에너지 집중도(비율) 포락선: 대역 에너지 / 전체 에너지
      # 절대 에너지가 아닌 비율을 사용해야 큰 소음에 끌리지 않고
      # 실제로 대역에 에너지가 집중된 구간(새소리)을 찾음
      sig_band <- as.numeric(filtered@left)^2
      sig_total <- as.numeric(wav@left)^2

      win_samples <- max(1, round(window_sec * sr))
      kernel <- rep(1 / win_samples, win_samples)
      energy_band <- stats::filter(sig_band, kernel, sides = 2)
      energy_total <- stats::filter(sig_total, kernel, sides = 2)
      energy_band[is.na(energy_band)] <- 0
      energy_total[is.na(energy_total)] <- 0

      ratio <- as.numeric(energy_band) / (as.numeric(energy_total) + 1e-10)

      list(energy = ratio, sr = sr)
    },
    error = function(e) {
      log_debug(sprintf("  에너지 포락선 계산 오류: %s", e$message))
      NULL
    }
  )
}

#' corMatch 피크를 대역 에너지 최대 지점으로 보정
#' @param energy_env compute_band_energy_envelope() 결과
#' @param peak_time corMatch 피크 시간 (초)
#' @param ref_duration 템플릿 길이 (초)
#' @return 보정된 피크 시간 (초)
refine_peak_position <- function(energy_env, peak_time, ref_duration) {
  if (is.null(energy_env)) return(peak_time)

  sr <- energy_env$sr
  n <- length(energy_env$energy)

  # 탐색 범위: corMatch 피크 기준 ±1 템플릿 길이
  search_radius_samples <- round(ref_duration * sr)
  peak_sample <- max(1, round(peak_time * sr))

  s_start <- max(1, peak_sample - search_radius_samples)
  s_end <- min(n, peak_sample + search_radius_samples)

  if (s_end <= s_start) return(peak_time)

  region <- energy_env$energy[s_start:s_end]
  best_offset <- which.max(region) - 1
  refined_time <- (s_start + best_offset) / sr

  refined_time
}

#' 대역 에너지 포락선에서 피크를 탐색하여 후보 시간 목록을 반환
#' corMatch가 놓친 울음 구간을 보충하는 용도
#' @param energy_env compute_band_energy_envelope() 결과
#' @param ref_duration 템플릿 길이 (초) — 최소 피크 간격으로 사용
#' @return 피크 시간 벡터 (초)
find_energy_peaks <- function(energy_env, ref_duration) {
  if (is.null(energy_env)) return(numeric(0))

  sr <- energy_env$sr
  ratio <- energy_env$energy
  n <- length(ratio)

  # 최소 피크 간격 = 템플릿 길이
  gap_samples <- round(ref_duration * sr)

  # 적응형 임계값: 중앙값 + 2×MAD (이상치에 견고)
  med <- median(ratio)
  mad_val <- mad(ratio)
  threshold <- med + 2 * mad_val
  # 최소 기준: 에너지 집중도 1% 이상
  threshold <- max(threshold, 0.01)

  # Greedy 피크 검출: 최고점 선택 → ±gap 제외 → 반복
  peaks <- numeric(0)
  remaining <- seq_len(n)

  while (length(remaining) > 0) {
    best_in_remaining <- which.max(ratio[remaining])
    max_idx <- remaining[best_in_remaining]

    if (ratio[max_idx] < threshold) break

    peaks <- c(peaks, max_idx)

    # ±gap 범위 제외
    exclude_start <- max(1, max_idx - gap_samples)
    exclude_end <- min(n, max_idx + gap_samples)
    remaining <- remaining[remaining < exclude_start | remaining > exclude_end]
  }

  # 시간으로 변환하여 정렬
  sort(peaks / sr)
}

# ============================================================
# ★ 자동 튜닝: 종 음원 자가진단으로 최적 가중치 결정
# ============================================================
#' 종 음원 내에서 자가진단을 수행하여 최적 가중치를 자동으로 결정한다.
#'
#' 원리:
#'   1. 종 음원에서 울음 구간(template)과 비울음 구간(negative)을 식별
#'   2. 종 음원 내 다른 울음 구간을 sliding window로 탐색
#'   3. 각 구간에 대해 5가지 지표를 계산
#'   4. 울음 구간(양성)과 비울음 구간(음성)에서의 점수 차이(변별력)를 측정
#'   5. 변별력이 높은 지표에 더 높은 가중치를 부여
#'
#' @param wav Wave 객체 (종 음원)
#' @param t_start 템플릿 시작 시간
#' @param t_end 템플릿 종료 시간
#' @param f_low 주파수 하한 (kHz)
#' @param f_high 주파수 상한 (kHz)
#' @return list(weights=list(...), diagnostics=list(...))
auto_tune_weights <- function(wav, templates_info) {
  # templates_info: list of list(t_start, t_end, f_low, f_high)
  # 사용자가 스펙트로그램에서 직접 선택한 새소리 구간들
  sr <- wav@samp.rate
  total_duration <- length(wav@left) / sr

  n_templates <- length(templates_info)
  log_info(sprintf("★ 자동 튜닝 시작 (템플릿 %d개, 유사도 기반 분류)", n_templates))
  log_info(sprintf("  종 음원: %.1f초", total_duration))

  # 1) 모든 템플릿 세그먼트 추출
  ref_segments <- list()
  ref_f_ranges <- list()
  valid_templates <- 0

  for (ti in seq_along(templates_info)) {
    tmpl <- templates_info[[ti]]
    t_s <- tmpl$t_start
    t_e <- tmpl$t_end
    f_l <- tmpl$f_low
    f_h <- tmpl$f_high

    log_info(sprintf(
      "  템플릿 #%d: %.2f~%.2f초, %s~%s Hz",
      ti, t_s, t_e, f_l, f_h
    ))

    seg <- extract_segment(wav, t_s, t_e)
    if (!is.null(seg)) {
      valid_templates <- valid_templates + 1
      ref_segments[[valid_templates]] <- seg
      ref_f_ranges[[valid_templates]] <- list(f_low = f_l, f_high = f_h)
    }
  }

  if (valid_templates < 1) {
    log_error("  템플릿 구간 추출 실패")
    return(list(weights = DEFAULT_WEIGHTS, diagnostics = NULL))
  }

  # 대표 주파수 범위 (전체 템플릿의 합집합)
  all_f_low <- min(sapply(ref_f_ranges, function(r) r$f_low))
  all_f_high <- max(sapply(ref_f_ranges, function(r) r$f_high))

  # 대표 윈도우 길이 (템플릿 평균 길이)
  ref_durations <- sapply(templates_info, function(t) t$t_end - t$t_start)
  window_dur <- mean(ref_durations)

  # 2) 슬라이딩 윈도우로 전체 음원 스캔
  step <- window_dur * 0.5 # 50% 중첩
  n_windows <- floor((total_duration - window_dur) / step)

  if (n_windows < 3) {
    log_info("  ⚠ 음원이 너무 짧아 윈도우 분석 불가")
    return(list(weights = DEFAULT_WEIGHTS, diagnostics = list(
      n_positive = 0, n_negative = 0,
      message = "음원이 너무 짧습니다"
    )))
  }

  log_info(sprintf("  슬라이딩 윈도우: %d개 (%.2f초 간격)", n_windows, step))

  # 3) 각 윈도우에 대해 모든 템플릿과의 MFCC 유사도 계산
  #    → 최대 유사도로 양성/음성 분류
  log_info("  유사도 기반 양성/음성 분류 중...")
  window_max_sims <- numeric(n_windows)
  window_starts <- numeric(n_windows)

  for (wi in seq_len(n_windows)) {
    w_start <- (wi - 1) * step
    w_end <- w_start + window_dur
    window_starts[wi] <- w_start

    # 모든 템플릿 구간과 겹치는지 확인
    overlaps_any <- FALSE
    for (ti in seq_along(templates_info)) {
      tmpl <- templates_info[[ti]]
      overlap <- max(0, min(tmpl$t_end, w_end) - max(tmpl$t_start, w_start))
      if (overlap / window_dur > 0.5) {
        overlaps_any <- TRUE
        break
      }
    }

    if (overlaps_any) {
      # 템플릿 자체와 겹치는 윈도우 → 유사도 1.0 (확정 양성)
      window_max_sims[wi] <- 1.0
      next
    }

    seg <- extract_segment(wav, w_start, w_end)
    if (is.null(seg) || length(seg@left) < 100) {
      window_max_sims[wi] <- 0.0
      next
    }

    # 모든 템플릿과의 MFCC 코사인 유사도 중 최댓값
    max_sim <- 0
    for (ri in seq_along(ref_segments)) {
      sim <- tryCatch(
        compute_mfcc_similarity(ref_segments[[ri]], seg),
        error = function(e) 0
      )
      if (sim > max_sim) max_sim <- sim
    }
    window_max_sims[wi] <- max_sim
  }

  # 4) 유사도 기반 양성/음성 분류
  #    템플릿과 겹치는 윈도우(sim=1.0)는 확정 양성
  #    나머지: 템플릿 간 유사도의 중간값을 임계값으로 사용
  template_self_sims <- c()
  if (valid_templates >= 2) {
    # 템플릿끼리의 유사도 계산 (양성 기준선)
    for (i in 1:(valid_templates - 1)) {
      for (j in (i + 1):valid_templates) {
        sim <- tryCatch(
          compute_mfcc_similarity(ref_segments[[i]], ref_segments[[j]]),
          error = function(e) 0.5
        )
        template_self_sims <- c(template_self_sims, sim)
      }
    }
  }

  # 임계값: 템플릿 간 유사도가 있으면 그 최솟값의 90%, 없으면 0.65
  if (length(template_self_sims) > 0) {
    sim_threshold <- min(template_self_sims) * 0.9
    log_info(sprintf(
      "  템플릿 간 유사도: %.3f~%.3f → 양성 임계값: %.3f",
      min(template_self_sims), max(template_self_sims), sim_threshold
    ))
  } else {
    sim_threshold <- 0.65
    log_info(sprintf("  단일 템플릿 → 양성 임계값: %.3f (기본값)", sim_threshold))
  }

  # 비겹침 윈도우의 유사도 분포 로깅
  non_overlap_sims <- window_max_sims[window_max_sims < 1.0]
  if (length(non_overlap_sims) > 0) {
    log_info(sprintf(
      "  비템플릿 윈도우 유사도: min=%.3f, median=%.3f, max=%.3f",
      min(non_overlap_sims), median(non_overlap_sims), max(non_overlap_sims)
    ))
  }

  # 5) 양성/음성 구간에서 6가지 지표 계산
  # 충분한 샘플로 안정적 통계 추정 (기존 20 → 50, 속도와 정밀도 균형)
  positive_scores <- list()
  negative_scores <- list()
  n_pos <- 0
  n_neg <- 0
  max_samples <- 50

  # ★ 계층적 샘플링: 시간 순서대로가 아닌, 유사도 분포에서 균등 추출
  #    양성은 가장 유사한 것부터, 음성은 가장 비유사한 것부터 추출하되
  #    다양한 유사도 수준을 포함하도록 함

  # ★ 비겹침 윈도우만 분류 대상 (sim=1.0은 확정 양성 — 템플릿 자체)
  non_template_indices <- which(window_max_sims < 1.0)

  # 분류 전략 결정: 절대 임계값 vs 상대 분위수
  # 비겹침 윈도우의 중앙값이 임계값 이상이면 절대 임계값으로는 분리 불가
  # → 상대 분위수 기반 분류로 전환
  use_percentile_split <- FALSE
  if (length(non_template_indices) >= 6) {
    nt_sims <- window_max_sims[non_template_indices]
    n_above_threshold <- sum(nt_sims >= sim_threshold)
    n_below_threshold <- sum(nt_sims < sim_threshold)

    # 음성이 전체의 15% 미만이면 절대 임계값이 부적합
    if (n_below_threshold < length(non_template_indices) * 0.15) {
      use_percentile_split <- TRUE
      log_info(sprintf(
        "  ⚠ 절대 임계값(%.3f)으로 분리 불가 (이상:%d, 미만:%d) → 분위수 기반 분류",
        sim_threshold, n_above_threshold, n_below_threshold
      ))
    }
  }

  if (use_percentile_split) {
    # ★ 분위수 기반 분류: 상위 30% = 양성, 하위 30% = 음성
    # 중간 40%는 애매한 구간이므로 제외
    nt_sims <- window_max_sims[non_template_indices]
    q_high <- quantile(nt_sims, 0.70) # 상위 30% 경계
    q_low  <- quantile(nt_sims, 0.30) # 하위 30% 경계

    # 템플릿 겹침(sim=1.0)은 확정 양성에 포함
    pos_indices <- c(
      which(window_max_sims == 1.0),
      non_template_indices[nt_sims >= q_high]
    )
    neg_indices <- non_template_indices[nt_sims <= q_low]

    log_info(sprintf(
      "  분위수 분류: 양성 임계값=%.3f (상위 30%%), 음성 임계값=%.3f (하위 30%%)",
      q_high, q_low
    ))
    log_info(sprintf(
      "  양성 후보=%d, 음성 후보=%d (총 비겹침=%d)",
      length(pos_indices), length(neg_indices), length(non_template_indices)
    ))
  } else {
    # 원래 절대 임계값 기반 분류
    pos_indices <- which(window_max_sims >= sim_threshold)

    # ★ 적응형 음성 임계값: 최소 10개 음성을 확보할 때까지 단계적 완화
    min_neg_required <- 10
    neg_factor <- 0.7 # 시작: sim_threshold * 0.7
    neg_indices <- which(window_max_sims < sim_threshold * neg_factor &
      window_max_sims < 1.0) # sim=1.0 (템플릿 겹침) 제외

    while (length(neg_indices) < min_neg_required && neg_factor < 0.95) {
      neg_factor <- neg_factor + 0.05
      neg_indices <- which(window_max_sims < sim_threshold * neg_factor &
        window_max_sims < 1.0)
    }

    # 여전히 부족하면: 유사도 하위 N개를 강제 음성으로 사용
    if (length(neg_indices) < min_neg_required) {
      if (length(non_template_indices) >= min_neg_required) {
        sorted_idx <- non_template_indices[order(window_max_sims[non_template_indices])]
        neg_indices <- sorted_idx[1:min(max_samples, length(sorted_idx))]
      } else if (length(non_template_indices) > 0) {
        neg_indices <- non_template_indices
      }
      log_info(sprintf(
        "  ⚠ 음성 부족 → 유사도 하위 %d개 강제 사용 (factor=%.2f)",
        length(neg_indices), neg_factor
      ))
    } else {
      log_info(sprintf(
        "  음성 임계값: sim < %.3f (factor=%.2f, %d건)",
        sim_threshold * neg_factor, neg_factor, length(neg_indices)
      ))
    }
  }

  # 유사도 기준 정렬 후 균등 간격 서브샘플링
  if (length(pos_indices) > max_samples) {
    pos_indices <- pos_indices[order(window_max_sims[pos_indices], decreasing = TRUE)]
    step_p <- max(1, length(pos_indices) %/% max_samples)
    pos_indices <- pos_indices[seq(1, length(pos_indices), by = step_p)][1:min(max_samples, length(pos_indices))]
  }
  if (length(neg_indices) > max_samples) {
    neg_indices <- neg_indices[order(window_max_sims[neg_indices])]
    step_n <- max(1, length(neg_indices) %/% max_samples)
    neg_indices <- neg_indices[seq(1, length(neg_indices), by = step_n)][1:min(max_samples, length(neg_indices))]
  }

  # ★ 윈도우 역할 맵: 어떤 인덱스가 양성/음성으로 배정되었는지 추적
  # 이후 루프에서 재분류하지 않고 이 맵을 사용
  window_role <- rep("skip", n_windows)
  window_role[pos_indices] <- "positive"
  window_role[neg_indices] <- "negative"

  sample_indices <- c(pos_indices, neg_indices)

  for (wi in sample_indices) {
    w_start <- window_starts[wi]
    w_end <- w_start + window_dur

    # ★ 사전 배정된 역할 사용 (재분류하지 않음)
    is_positive <- window_role[wi] == "positive"
    is_negative <- window_role[wi] == "negative"

    if (!is_positive && !is_negative) next

    seg <- extract_segment(wav, w_start, w_end)
    if (is.null(seg) || length(seg@left) < 100) next

    # 가장 유사한 템플릿을 레퍼런스로 사용
    best_ref_idx <- 1
    best_sim <- 0
    for (ri in seq_along(ref_segments)) {
      s <- tryCatch(
        compute_mfcc_similarity(ref_segments[[ri]], seg),
        error = function(e) 0
      )
      if (s > best_sim) {
        best_sim <- s
        best_ref_idx <- ri
      }
    }
    ref_seg <- ref_segments[[best_ref_idx]]
    f_low <- ref_f_ranges[[best_ref_idx]]$f_low
    f_high <- ref_f_ranges[[best_ref_idx]]$f_high

    scores <- tryCatch(
      {
        s <- list()
        # ★ cor_score: 스펙트로그램 상관 (분석 모드의 corMatch와 동일 원리)
        s$cor_score <- compute_spectrogram_cor(ref_seg, seg)
        s$mfcc_score <- compute_mfcc_dtw_similarity(ref_seg, seg)
        s$dtw_freq <- compute_freq_contour_dtw(ref_seg, seg, f_low, f_high)
        s$dtw_env <- compute_envelope_dtw(ref_seg, seg)
        s$band_energy <- compute_band_energy_ratio(seg, f_low, f_high)
        s$harmonic_ratio <- compute_harmonic_ratio(seg, f_low, f_high)
        s$snr <- compute_snr_ratio(seg, f_low, f_high)
        s
      },
      error = function(e) NULL
    )

    if (is.null(scores)) next

    if (is_positive) {
      n_pos <- n_pos + 1
      positive_scores[[n_pos]] <- scores
    } else {
      n_neg <- n_neg + 1
      negative_scores[[n_neg]] <- scores
    }
  }

  log_info(sprintf(
    "  수집 완료: 양성 %d건, 음성 %d건 (유사도 임계값 %.3f, 후보: 양성=%d, 음성=%d)",
    n_pos, n_neg, sim_threshold, length(pos_indices), length(neg_indices)
  ))

  # ★ 최소 3건씩 필요 (n=1~2에서는 sd, Cohen's d 계산이 불안정)
  if (n_pos < 3 || n_neg < 3) {
    log_info(sprintf(
      "  ⚠ 샘플 부족 (양성=%d, 음성=%d, 최소=3) → 기본 가중치 사용", n_pos, n_neg
    ))
    log_info(sprintf("  [진단] 유사도 임계값: %.3f", sim_threshold))
    if (length(non_overlap_sims) > 0) {
      n_above <- sum(non_overlap_sims >= sim_threshold)
      log_info(sprintf("  [진단] 임계값 이상 윈도우: %d/%d개", n_above, length(non_overlap_sims)))
    }
    return(list(weights = DEFAULT_WEIGHTS, diagnostics = list(
      n_positive = n_pos, n_negative = n_neg,
      message = "샘플 부족으로 자동 튜닝 불가 (최소 양성 3건, 음성 3건 필요)",
      sim_threshold = sim_threshold
    )))
  }

  # 6) 각 지표별 변별력(discriminative power) 계산
  #    LOO 교차검증: 각 샘플을 한 번씩 제외하고 Cohen's d를 계산,
  #    평균을 취해 과적합 방지
  metric_names <- c("cor_score", "mfcc_score", "dtw_freq", "dtw_env", "band_energy", "harmonic_ratio", "snr")
  disc_power <- numeric(length(metric_names))
  names(disc_power) <- metric_names

  pos_means <- numeric(length(metric_names))
  neg_means <- numeric(length(metric_names))
  names(pos_means) <- metric_names
  names(neg_means) <- metric_names

  n_all <- n_pos + n_neg
  use_loo <- (n_pos >= 5 && n_neg >= 5) # LOO는 최소 5개 이상일 때만

  for (mi in seq_along(metric_names)) {
    mn <- metric_names[mi]
    pos_vals <- sapply(positive_scores, function(s) {
      v <- s[[mn]]
      if (is.null(v) || is.na(v) || is.nan(v)) 0 else v
    })
    neg_vals <- sapply(negative_scores, function(s) {
      v <- s[[mn]]
      if (is.null(v) || is.na(v) || is.nan(v)) 0 else v
    })

    pos_mean <- mean(pos_vals, na.rm = TRUE)
    neg_mean <- mean(neg_vals, na.rm = TRUE)
    pos_means[mi] <- pos_mean
    neg_means[mi] <- neg_mean

    if (use_loo) {
      # LOO 교차검증: 각 샘플 제외 후 Cohen's d 계산
      loo_ds <- numeric(n_all)

      for (li in seq_len(n_all)) {
        if (li <= n_pos) {
          # 양성 샘플 하나 제외
          loo_pos <- pos_vals[-li]
          loo_neg <- neg_vals
        } else {
          # 음성 샘플 하나 제외
          loo_pos <- pos_vals
          loo_neg <- neg_vals[-(li - n_pos)]
        }

        pm <- mean(loo_pos, na.rm = TRUE)
        nm <- mean(loo_neg, na.rm = TRUE)
        ps <- sd(loo_pos, na.rm = TRUE)
        ns <- sd(loo_neg, na.rm = TRUE)
        ps_d <- sqrt((ps^2 + ns^2) / 2)
        if (is.na(ps_d) || ps_d < 0.001) ps_d <- 0.001
        loo_ds[li] <- max(0, (pm - nm) / ps_d)
      }

      disc_power[mi] <- mean(loo_ds)
    } else {
      # 샘플 부족 시 기존 방식
      pos_sd <- sd(pos_vals, na.rm = TRUE)
      neg_sd <- sd(neg_vals, na.rm = TRUE)
      pooled_sd <- sqrt((pos_sd^2 + neg_sd^2) / 2)
      if (is.na(pooled_sd) || pooled_sd < 0.001) pooled_sd <- 0.001
      disc_power[mi] <- max(0, (pos_mean - neg_mean) / pooled_sd)
    }
  }

  log_info(sprintf("  지표별 변별력%s:", if (use_loo) " (LOO 교차검증)" else ""))
  for (mn in metric_names) {
    log_info(sprintf(
      "    %s: 양성=%.3f, 음성=%.3f → 변별력=%.3f",
      mn, pos_means[mn], neg_means[mn], disc_power[mn]
    ))
  }

  # ★ 역방향 지표 처리: 양성 평균 ≤ 음성 평균이면 변별력 0
  # (점수가 높을수록 새소리가 아닌 방향 → 복합점수 압축의 원인)
  for (mi in seq_along(metric_names)) {
    if (pos_means[mi] <= neg_means[mi]) {
      if (disc_power[mi] > 0) {
        log_info(sprintf(
          "    ★ %s: 역방향 (양성=%.3f ≤ 음성=%.3f) → 가중치 0",
          metric_names[mi], pos_means[mi], neg_means[mi]
        ))
      }
      disc_power[mi] <- 0
    }
  }

  # 7) 변별력을 가중치로 변환 (정규화)
  total_disc <- sum(disc_power)
  if (total_disc < 0.001) {
    log_info("  ⚠ 모든 지표의 변별력이 0 → 기본 가중치 사용")
    optimal_weights <- DEFAULT_WEIGHTS
  } else {
    # ★ 변별력의 제곱근을 사용: 극단적 차이를 완화하여
    #    한 지표에 가중치가 과도하게 몰리는 것을 방지
    raw_weights <- sqrt(disc_power) / sum(sqrt(disc_power))

    # ★ 최소 가중치 제거: 변별력 없는 지표는 가중치 0
    # (기존 min_w=0.05가 비구별 지표의 상수 기여를 유발하여 점수 압축)
    optimal_weights <- as.list(raw_weights)
    names(optimal_weights) <- metric_names
  }

  log_info("  ★ 자동 튜닝 결과 (최적 가중치):")
  for (mn in metric_names) {
    default_w <- DEFAULT_WEIGHTS[[mn]]
    tuned_w <- optimal_weights[[mn]]
    arrow <- if (tuned_w > default_w + 0.03) "↑" else if (tuned_w < default_w - 0.03) "↓" else "="
    log_info(sprintf("    %s: %.3f (기본 %.3f) %s", mn, tuned_w, default_w, arrow))
  }

  list(
    weights = optimal_weights,
    diagnostics = list(
      n_positive = n_pos,
      n_negative = n_neg,
      n_templates = valid_templates,
      positive_means = as.list(pos_means),
      negative_means = as.list(neg_means),
      discriminative_power = as.list(disc_power),
      sim_threshold = sim_threshold
    )
  )
}


# ============================================================
# 메인 실행 시작
# ============================================================
log_debug("R version: ", R.version.string)
log_debug("Platform: ", .Platform$OS.type, " / ", Sys.info()["sysname"])

# --- Portable R 라이브러리 경로 자동 탐색 ---
# 설치형 배포 시, R-Portable/library 경로를 .libPaths()에 추가하여
# 번들된 패키지를 찾을 수 있게 한다.
r_home <- normalizePath(R.home(), winslash = "/")
portable_lib <- file.path(r_home, "library")
if (dir.exists(portable_lib)) {
  .libPaths(c(portable_lib, .libPaths()))
}
# 환경변수 R_LIBS / R_LIBS_USER 경로도 반영
for (env_var in c("R_LIBS", "R_LIBS_USER", "R_LIBS_SITE")) {
  env_path <- Sys.getenv(env_var, unset = "")
  if (nchar(env_path) > 0 && dir.exists(env_path)) {
    .libPaths(c(env_path, .libPaths()))
  }
}
log_debug("Library paths: ", paste(.libPaths(), collapse = "; "))

# --- 패키지 로드 ---
required_pkgs <- c("seewave", "tuneR", "monitoR", "jsonlite", "dtw")
for (pkg in required_pkgs) {
  log_debug("Loading package: ", pkg)
  tryCatch(
    {
      library(pkg, character.only = TRUE)
      log_debug("  OK - version: ", as.character(packageVersion(pkg)))
    },
    error = function(e) {
      log_error(sprintf("패키지 '%s' 로드 실패: %s", pkg, e$message))
      log_error(sprintf("설치 명령: install.packages('%s')", pkg))
      log_error(sprintf("현재 .libPaths: %s", paste(.libPaths(), collapse = "; ")))
      stop(sprintf("필수 패키지 '%s'를 찾을 수 없습니다.", pkg))
    }
  )
}

# --- 설정 파일 읽기 ---
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("설정 파일 경로가 필요합니다.")
if (!file.exists(args[1])) stop(sprintf("설정 파일이 존재하지 않습니다: %s", args[1]))

config <- tryCatch(
  {
    fromJSON(args[1], simplifyVector = FALSE)
  },
  error = function(e) {
    log_error("설정 파일 읽기 실패: ", e$message)
    stop(e)
  }
)

main_wav_path <- config$main_wav
output_dir <- config$output_dir
species_list <- config$species
run_mode <- if (!is.null(config$mode)) config$mode else "analyze"

# 종합 판별 가중치 (config에서 오버라이드 가능)
global_weights <- if (!is.null(config$weights)) config$weights else DEFAULT_WEIGHTS
# 기존 config에 snr 가중치가 없으면 DEFAULT에서 보충
if (is.null(global_weights$snr)) {
  global_weights$snr <- DEFAULT_WEIGHTS$snr
  # 재정규화 (합=1 유지)
  total_w <- sum(unlist(global_weights))
  if (total_w > 0) global_weights <- lapply(global_weights, function(w) w / total_w)
}

# ★ 주파수 대역 필터링 (2-pass 검출) 옵션
freq_filter_enabled <- isTRUE(config$freq_filter_enabled)
if (freq_filter_enabled) {
  log_info("★ 주파수 대역 필터링 활성화 (2-pass 검출)")
}

# ★ 단계적 평가 (Staged Evaluation) 옵션
staged_eval_enabled <- if (!is.null(config$staged_eval)) config$staged_eval else TRUE
if (isTRUE(staged_eval_enabled)) {
  log_info("★ 단계적 평가 활성화 (3단계 Early Rejection)")
}

if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# ============================================================
# ★ 자동 튜닝 모드 (mode == "auto_tune")
# ============================================================
if (run_mode == "auto_tune") {
  log_info("★★★ 자동 튜닝 모드 실행 (유사도 기반) ★★★")

  if (length(species_list) == 0) stop("종 목록이 비어 있습니다.")

  all_results <- list()

  for (i in seq_along(species_list)) {
    sp <- species_list[[i]]
    sp_name <- sp$name
    log_info(sprintf("\n[%d/%d] %s 자동 튜닝 중...", i, length(species_list), sp_name))

    # 종 음원 로드 + 전처리
    sp_wav <- tryCatch(
      {
        w <- safe_readWave(sp$wav_path)
        w <- ensure_mono(w)
        w <- normalize_amplitude(w)
        if (w@samp.rate > MAX_SAMPLE_RATE) {
          w <- safe_resamp(w, f = w@samp.rate, g = MAX_SAMPLE_RATE, output = "Wave")
        }
        w
      },
      error = function(e) {
        log_error(sprintf("  로드 실패: %s", e$message))
        NULL
      }
    )

    if (is.null(sp_wav)) {
      all_results[[sp_name]] <- list(error = "음원 로드 실패")
      next
    }

    # 템플릿 목록 구성 (새 형식: templates 배열 / 구 형식: 단일 필드)
    templates_info <- list()
    if (!is.null(sp$templates) && length(sp$templates) > 0) {
      for (tmpl in sp$templates) {
        freq <- validate_and_fix_freq_range(
          sp_wav, tmpl$f_low, tmpl$f_high,
          tmpl$t_start, tmpl$t_end
        )
        templates_info[[length(templates_info) + 1]] <- list(
          t_start = tmpl$t_start, t_end = tmpl$t_end,
          f_low = freq$f_low, f_high = freq$f_high
        )
      }
    } else {
      # 하위 호환: 단일 템플릿
      freq <- validate_and_fix_freq_range(
        sp_wav, sp$f_low, sp$f_high,
        sp$t_start, sp$t_end
      )
      templates_info[[1]] <- list(
        t_start = sp$t_start, t_end = sp$t_end,
        f_low = freq$f_low, f_high = freq$f_high
      )
    }

    # 자동 튜닝 실행
    tune_result <- auto_tune_weights(sp_wav, templates_info)

    all_results[[sp_name]] <- list(
      weights = tune_result$weights,
      diagnostics = tune_result$diagnostics
    )
  }

  # JSON으로 결과 저장
  result_path <- file.path(output_dir, "auto_tune_results.json")
  writeLines(toJSON(all_results, auto_unbox = TRUE, pretty = TRUE),
    result_path,
    useBytes = TRUE
  )
  log_info(sprintf("\n★ 자동 튜닝 결과 저장: %s", result_path))

  # 요약 출력
  print_section("자동 튜닝 요약")
  for (sp_name in names(all_results)) {
    res <- all_results[[sp_name]]
    if (!is.null(res$error)) {
      cat(sprintf("  %s: 오류 - %s\n", sp_name, res$error))
      next
    }
    cat(sprintf("  %s:\n", sp_name))
    if (!is.null(res$diagnostics)) {
      cat(sprintf(
        "    템플릿 %d개, 양성 %d건, 음성 %d건 분석\n",
        res$diagnostics$n_templates,
        res$diagnostics$n_positive, res$diagnostics$n_negative
      ))
    }
    w <- res$weights
    cat(sprintf(
      "    cor=%.3f  mfcc=%.3f  freq=%.3f  env=%.3f  band=%.3f  hr=%.3f\n",
      w$cor_score, w$mfcc_score, w$dtw_freq, w$dtw_env, w$band_energy,
      w$harmonic_ratio
    ))
  }

  cat("\n[DONE]\n")
  quit(save = "no", status = 0)
}

# ============================================================
# ★ 스펙트로그램 내보내기 모드 (mode == "spectrogram")
# seewave::spectro() 기반 연구용 고품질 스펙트로그램 PNG 생성
# ============================================================
if (run_mode == "spectrogram") {
  log_info("★★★ 스펙트로그램 내보내기 모드 ★★★")

  wav_path <- config$wav_path
  out_path <- config$output_path
  if (is.null(wav_path) || !file.exists(wav_path)) {
    stop(sprintf("WAV 파일이 존재하지 않습니다: %s", wav_path))
  }
  if (is.null(out_path)) stop("output_path가 지정되지 않았습니다.")

  # --- 파라미터 (기본값 포함) ---
  img_w <- if (!is.null(config$width)) config$width else 1600
  img_h <- if (!is.null(config$height)) config$height else 800
  wl <- if (!is.null(config$wl)) config$wl else 512
  ovlp <- if (!is.null(config$ovlp)) config$ovlp else 75
  col_levels <- if (!is.null(config$collevels)) config$collevels else 30
  pal_name <- if (!is.null(config$palette)) config$palette else "spectro.colors"

  # 시간/주파수 범위 (선택적)
  t_start <- config$t_start # NULL이면 전체
  t_end <- config$t_end
  f_low <- config$f_low # Hz 단위
  f_high <- config$f_high


  # 새 파라미터: dB 범위, DPI, 표시 요소 토글
  dB_min <- if (!is.null(config$dB_min)) config$dB_min else -60
  dB_max <- if (!is.null(config$dB_max)) config$dB_max else 0
  img_res <- if (!is.null(config$res)) config$res else 150
  show_title <- if (!is.null(config$show_title)) config$show_title else TRUE
  show_scale <- if (!is.null(config$show_scale)) config$show_scale else TRUE
  show_osc <- if (!is.null(config$show_osc)) config$show_osc else FALSE
  show_det <- if (!is.null(config$show_detections)) config$show_detections else TRUE
  det_cex <- if (!is.null(config$det_cex)) config$det_cex else 0.7

  # 검출 결과 오버레이 (show_det가 FALSE이면 비활성)
  det_list <- if (isTRUE(show_det)) config$detections else NULL

  # --- WAV 로드 + 전처리 ---
  log_info(sprintf("WAV 로드: %s", wav_path))
  wav <- tryCatch(
    {
      w <- safe_readWave(wav_path)
      w <- ensure_mono(w)
      if (w@samp.rate > MAX_SAMPLE_RATE) {
        log_info(sprintf("  다운샘플링: %d Hz → %d Hz", w@samp.rate, MAX_SAMPLE_RATE))
        w <- safe_resamp(w, f = w@samp.rate, g = MAX_SAMPLE_RATE, output = "Wave")
      }
      w
    },
    error = function(e) {
      log_error("WAV 로드 실패: ", e$message)
      stop(e)
    }
  )

  sr <- wav@samp.rate
  total_dur <- length(wav@left) / sr
  log_info(sprintf("  %d Hz, %.1f초, %d samples", sr, total_dur, length(wav@left)))

  # --- 시간 범위 추출 ---
  if (!is.null(t_start) && !is.null(t_end)) {
    t_start <- max(0, as.numeric(t_start))
    t_end <- min(total_dur, as.numeric(t_end))
    if (t_end > t_start) {
      s1 <- max(1, round(t_start * sr))
      s2 <- min(length(wav@left), round(t_end * sr))
      wav <- Wave(
        left = wav@left[s1:s2], samp.rate = sr,
        bit = wav@bit, pcm = TRUE
      )
      log_info(sprintf("  시간 범위 추출: %.2f ~ %.2f초", t_start, t_end))
    }
  } else {
    t_start <- 0
    t_end <- total_dur
  }

  # --- 주파수 범위 (kHz 변환) ---
  nyquist_khz <- sr / 2000
  if (!is.null(f_low) && !is.null(f_high)) {
    flim_low <- as.numeric(f_low) / 1000 # Hz → kHz
    flim_high <- as.numeric(f_high) / 1000
    if (flim_high > nyquist_khz) flim_high <- nyquist_khz * NYQUIST_SAFETY_FACTOR
    # ★ seewave::spectro()는 flim[1]=0이면 내부 인덱스 오류 발생 → 최소 0.001
    if (flim_low < 0.001) flim_low <- 0.001
    if (flim_low >= flim_high) {
      flim_low <- 0.001
      flim_high <- nyquist_khz * NYQUIST_SAFETY_FACTOR
    }
    flim <- c(flim_low, flim_high)
    log_info(sprintf("  주파수 범위: %.3f ~ %.3f kHz", flim[1], flim[2]))
  } else {
    # flim을 지정하지 않으면 NULL → seewave 기본값 사용 (안전)
    flim <- NULL
  }

  # --- 팔레트 선택 ---
  pal_func <- switch(pal_name,
    "spectro.colors" = spectro.colors,
    "reverse.gray"   = reverse.gray.colors.2,
    "heat"           = heat.colors,
    "terrain"        = terrain.colors,
    spectro.colors # 기본값
  )

  # --- wl 유효성 검증 ---
  # wl은 신호 길이보다 작아야 하며, 짝수여야 한다
  n_samples <- length(wav@left)
  if (wl >= n_samples) {
    wl <- 2^floor(log2(n_samples / 2)) # 신호의 절반 이하인 최대 2의 거듭제곱
    if (wl < 32) wl <- 32
    log_info(sprintf("  ★ wl 자동 조정: %d (신호 길이: %d)", wl, n_samples))
  }
  if (wl %% 2 != 0) wl <- wl - 1 # 짝수 보장

  # --- 스펙트로그램 생성 ---
  log_info(sprintf("스펙트로그램 생성: %dx%d, wl=%d, ovlp=%d%%", img_w, img_h, wl, ovlp))

  # collevels: dB 범위 시퀀스 (사용자 설정 반영)
  col_seq <- seq(dB_min, dB_max, length.out = col_levels + 1)

  # 제목용 주파수 범위 (flim이 NULL이면 전체)
  flim_for_title <- if (is.null(flim)) c(0, nyquist_khz) else flim

  tryCatch(
    {
      png(out_path, width = img_w, height = img_h, res = img_res)

      # 검출 오버레이가 있으면 title에 표시할 검출 수 계산
      n_det <- 0
      if (!is.null(det_list) && length(det_list) > 0) {
        n_det <- length(det_list)
      }

      # 메인 제목
      main_title <- sprintf(
        "Spectrogram — %.1f~%.1fs, %.0f~%.0f Hz (wl=%d)",
        t_start, t_end,
        flim_for_title[1] * 1000, flim_for_title[2] * 1000,
        wl
      )
      if (n_det > 0) {
        main_title <- paste0(main_title, sprintf(" [%d detections]", n_det))
      }

      # seewave::spectro 호출 (flim이 NULL이면 인수 생략 → 전체 범위)
      spectro_args <- list(
        wave = wav, f = sr,
        wl = wl, ovlp = ovlp,
        collevels = col_seq,
        palette = pal_func,
        main = if (isTRUE(show_title)) main_title else "",
        osc = isTRUE(show_osc),
        scale = isTRUE(show_scale),
        cexlab = 0.9, cexaxis = 0.8
      )
      if (!is.null(flim)) {
        spectro_args$flim <- flim
      }
      do.call(spectro, spectro_args)

      # --- 검출 결과 오버레이 ---
      if (n_det > 0) {
        log_info(sprintf("  검출 오버레이: %d건", n_det))
        colors <- c(
          "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4",
          "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F"
        )
        # 종별 색상 매핑
        species_names <- unique(sapply(det_list, function(d) d$species))
        sp_colors <- setNames(
          colors[seq_along(species_names) %% length(colors) + 1],
          species_names
        )

        for (det in det_list) {
          det_time <- as.numeric(det$time) - t_start # 추출된 구간 기준으로 보정
          det_sp <- det$species
          det_score <- as.numeric(det$score)

          # 현재 뷰 범위 내인지 확인
          if (det_time < 0 || det_time > (t_end - t_start)) next

          col <- sp_colors[det_sp]
          abline(v = det_time, col = col, lty = 2, lwd = 1.5)
          text(
            det_time, flim_for_title[2] * 0.95,
            labels = sprintf("%s\n%.0f%%", det_sp, det_score * 100),
            col = col, cex = det_cex, adj = c(0, 1), font = 2
          )
        }
      }

      dev.off()
      log_info(sprintf("★ 스펙트로그램 저장 완료: %s", out_path))
      cat(sprintf("[SPECTROGRAM_PATH] %s\n", out_path))
    },
    error = function(e) {
      safe_dev_off()
      log_error("스펙트로그램 생성 실패: ", e$message)
      stop(e)
    }
  )

  cat("[DONE]\n")
  quit(save = "no", status = 0)
}

# ============================================================
# 일반 분석 모드 (기존 파이프라인)
# ============================================================
if (is.null(main_wav_path) || !file.exists(main_wav_path)) {
  stop(sprintf("전체 음원 파일 없음: %s", main_wav_path))
}
if (length(species_list) == 0) stop("종 목록이 비어 있습니다.")

log_debug("Main WAV: ", main_wav_path)
log_debug("Output dir: ", output_dir)
log_debug("Species count: ", length(species_list))
log_debug("Weights: ", paste(names(global_weights), unlist(global_weights),
  sep = "=", collapse = ", "
))

# --- 임시 파일 관리 ---
temp_files <- character()
on.exit(
  {
    for (f in temp_files) if (file.exists(f)) tryCatch(file.remove(f), error = function(e) NULL)
  },
  add = TRUE
)

# --- 전체 음원 로드 + 전처리 ---
print_section(
  "0단계: 전처리",
  c("스테레오→모노 변환, 진폭 정규화를 수행합니다.")
)

log_info("전체 음원 로드 중...")
main_wav <- tryCatch(
  {
    w <- safe_readWave(main_wav_path)
    log_debug(sprintf(
      "  원본: %d Hz, %s, %d bit, %.1f sec",
      w@samp.rate, if (isTRUE(w@stereo)) "stereo" else "mono",
      w@bit, length(w@left) / w@samp.rate
    ))
    w
  },
  error = function(e) {
    log_error("전체 음원 로드 실패: ", e$message)
    stop(e)
  }
)

# ★ 전처리: 스테레오→모노, 진폭 정규화
main_wav <- ensure_mono(main_wav)
main_wav <- normalize_amplitude(main_wav)

# ★ 고해상도 음원 다운샘플링 (integer overflow 방지)
if (main_wav@samp.rate > MAX_SAMPLE_RATE) {
  log_info(sprintf(
    "  ★ 다운샘플링: %d → %d Hz (조류 분석에 충분한 해상도)",
    main_wav@samp.rate, MAX_SAMPLE_RATE
  ))
  main_wav <- safe_resamp(main_wav,
    f = main_wav@samp.rate,
    g = MAX_SAMPLE_RATE, output = "Wave"
  )
}

main_wav_fp <- file.path(tempdir(), "main.wav")
tryCatch(
  {
    writeWave(main_wav, main_wav_fp)
  },
  error = function(e) {
    log_error(sprintf("writeWave 실패 (%d-bit): %s", main_wav@bit, e$message))
    log_info("  ★ 16-bit 강제 변환 후 재시도...")
    samples <- as.numeric(main_wav@left)
    peak <- max(abs(samples))
    if (peak > 0) samples <- samples / peak * 32767
    samples <- as.integer(round(pmin(pmax(samples, -32767), 32767)))
    main_wav <<- Wave(
      left = samples, samp.rate = main_wav@samp.rate,
      bit = 16L, pcm = TRUE
    )
    writeWave(main_wav, main_wav_fp)
    log_info("  ★ 16-bit 변환 후 writeWave 성공")
  }
)
temp_files <- c(temp_files, main_wav_fp)
main_sr <- main_wav@samp.rate

# 전체 음원 주파수 특성 (진단용)
main_freq <- auto_detect_freq_range(main_wav)
log_info(sprintf(
  "전체 음원 에너지: %.3f~%.3f kHz (피크: %.3f kHz)",
  main_freq$f_low, main_freq$f_high, main_freq$peak
))

# --- 스펙트로그램 저장 ---
tryCatch(
  {
    png(file.path(output_dir, "spectro_main.png"),
      width = SPECTROGRAM_MAIN_W, height = SPECTROGRAM_MAIN_H
    )
    par(mar = c(5, 4, 2, 2))
    spectro(main_wav, main = "전체 음원", fastdisp = TRUE)
    dev.off()
  },
  error = function(e) {
    safe_dev_off()
    log_error("스펙트로그램 실패: ", e$message)
  }
)

# ============================================================
# C5: 종별 멀티 템플릿 생성 + 레퍼런스 음원 보관
# ============================================================
n_species <- length(species_list)
all_templates <- list() # corTemplate 객체들
template_names <- character() # "species__label" 형식
template_wavs <- list() # 레퍼런스 음원 구간
template_freqs <- list() # 주파수 범위 (kHz)
template_to_species <- list() # C5: 템플릿 이름 → 종 이름 매핑
species_names_unique <- character()

for (i in seq_along(species_list)) {
  sp <- species_list[[i]]
  sp_name <- sp$name

  # C5: 하위 호환 — templates 배열 없으면 단일 템플릿으로 자동 생성
  tmpls <- sp$templates
  if (is.null(tmpls)) {
    tmpls <- list(list(
      wav_path = sp$wav_path,
      t_start  = sp$t_start,
      t_end    = sp$t_end,
      f_low    = sp$f_low,
      f_high   = sp$f_high,
      label    = "default"
    ))
  }

  if (!(sp_name %in% species_names_unique)) {
    species_names_unique <- c(species_names_unique, sp_name)
  }

  log_info(sprintf("[%d/%d] %s (%d개 템플릿)...", i, n_species, sp_name, length(tmpls)))

  for (ti in seq_along(tmpls)) {
    tmpl <- tmpls[[ti]]
    tmpl_label <- if (!is.null(tmpl$label) && nchar(tmpl$label) > 0) tmpl$label else paste0("tmpl", ti)
    tmpl_name <- if (length(tmpls) == 1) sp_name else paste0(sp_name, "__", tmpl_label)

    log_info(sprintf("  [%s] 템플릿 생성 중...", tmpl_name))

    # --- 음원 로드 & 전처리 & 리샘플링 ---
    sp_wav <- tryCatch(
      {
        w <- safe_readWave(tmpl$wav_path)
        log_debug(sprintf(
          "  원본: %d Hz, %s, %.1f sec",
          w@samp.rate, if (isTRUE(w@stereo)) "stereo" else "mono",
          length(w@left) / w@samp.rate
        ))
        w <- ensure_mono(w)
        w <- normalize_amplitude(w)
        if (w@samp.rate > MAX_SAMPLE_RATE) {
          log_info(sprintf("  다운샘플링: %d → %d Hz", w@samp.rate, MAX_SAMPLE_RATE))
          w <- safe_resamp(w, f = w@samp.rate, g = MAX_SAMPLE_RATE, output = "Wave")
        }
        if (w@samp.rate != main_sr) {
          log_info(sprintf("  리샘플링: %d → %d Hz", w@samp.rate, main_sr))
          w <- safe_resamp(w, f = w@samp.rate, g = main_sr, output = "Wave")
        }
        w
      },
      error = function(e) {
        log_error(sprintf("  로드 실패: %s", e$message))
        NULL
      }
    )

    if (is.null(sp_wav)) next

    sp_fp <- file.path(tempdir(), paste0("sp_", i, "_", ti, ".wav"))
    tryCatch(
      {
        writeWave(sp_wav, sp_fp)
      },
      error = function(e) {
        log_info(sprintf(
          "  ★ writeWave 실패 (%d-bit) → 16-bit 강제 변환: %s",
          sp_wav@bit, e$message
        ))
        samples <- as.numeric(sp_wav@left)
        peak <- max(abs(samples))
        if (peak > 0) samples <- samples / peak * 32767
        samples <- as.integer(round(pmin(pmax(samples, -32767), 32767)))
        sp_wav <<- Wave(
          left = samples, samp.rate = sp_wav@samp.rate,
          bit = 16L, pcm = TRUE
        )
        writeWave(sp_wav, sp_fp)
      }
    )
    temp_files <- c(temp_files, sp_fp)

    # 종별 스펙트로그램 (첫 번째 템플릿만)
    if (ti == 1) {
      tryCatch(
        {
          png(file.path(output_dir, paste0("spectro_", sp_name, ".png")),
            width = SPECTROGRAM_SP_W, height = SPECTROGRAM_SP_H
          )
          par(mar = c(5, 4, 2, 2))
          spectro(sp_wav, main = sp_name, fastdisp = TRUE)
          dev.off()
        },
        error = function(e) {
          safe_dev_off()
        }
      )
    }

    # --- 템플릿 생성 ---
    tryCatch(
      {
        freq <- validate_and_fix_freq_range(
          sp_wav, tmpl$f_low, tmpl$f_high,
          tmpl$t_start, tmpl$t_end
        )
        log_debug(sprintf("  frq.lim: [%.3f, %.3f] kHz", freq$f_low, freq$f_high))

        # ★ candidate cutoff: findPeaks가 사용할 낮은 임계값
        cand_cutoff <- max(0.05, sp$cutoff * CANDIDATE_CUTOFF_RATIO)

        pdf(NULL)
        tpl <- makeCorTemplate(sp_fp,
          t.lim   = c(tmpl$t_start, tmpl$t_end),
          frq.lim = c(freq$f_low, freq$f_high),
          name    = tmpl_name,
          score.cutoff = cand_cutoff
        )
        dev.off()

        # on/off 포인트 보정
        pts_mat <- tpl@templates[[tmpl_name]]@pts
        if (is.matrix(pts_mat) && "amp" %in% colnames(pts_mat)) {
          if (sum(pts_mat[, "amp"] > 0) == 0) {
            log_info("  ⚠ on 포인트 0개 → 중앙값 보정")
            pts_mat[, "amp"] <- pts_mat[, "amp"] - median(pts_mat[, "amp"])
            tpl@templates[[tmpl_name]]@pts <- pts_mat
          }
        }

        ref_segment <- extract_segment(sp_wav, tmpl$t_start, tmpl$t_end)

        all_templates[[length(all_templates) + 1]] <- tpl
        template_names <- c(template_names, tmpl_name)
        template_wavs[[length(template_wavs) + 1]] <- ref_segment
        template_freqs[[length(template_freqs) + 1]] <- freq
        template_to_species[[tmpl_name]] <- sp_name

        log_info(sprintf("  %s 완료", tmpl_name))
      },
      error = function(e) {
        safe_dev_off()
        log_error(sprintf("  템플릿 생성 실패: %s", e$message))
      }
    )
  }
}

if (length(all_templates) == 0) stop("유효한 템플릿이 없습니다.")
log_info(sprintf("유효 템플릿: %d (종 %d개)", length(all_templates), length(species_names_unique)))

# ============================================================
# 템플릿 합치기
# ============================================================
if (length(all_templates) == 1) {
  tpls <- all_templates[[1]]
} else {
  tpls <- do.call(combineCorTemplates, all_templates)
}

# ============================================================
# 1단계: corMatch (넓은 그물로 후보 검출)
# ============================================================
print_section(
  "1단계: corMatch 후보 검출",
  c(
    "낮은 cutoff로 가능한 후보를 넓게 수집합니다.",
    "2단계에서 종합 판별로 정밀 필터링합니다."
  )
)

log_info(sprintf(
  "corMatch 실행 중... (%d Hz, %.1f sec)",
  main_sr, length(main_wav@left) / main_sr
))

# 1단계 cutoff (makeCorTemplate에서 이미 설정됨)
candidate_cutoffs <- vapply(template_names, function(tn) {
  sp_name <- template_to_species[[tn]]
  sp_conf <- species_list[[match(sp_name, vapply(species_list, `[[`, "", "name"))]]
  max(0.05, sp_conf$cutoff * CANDIDATE_CUTOFF_RATIO)
}, numeric(1))
names(candidate_cutoffs) <- template_names
log_debug("1단계 cutoff: ", paste(names(candidate_cutoffs), round(candidate_cutoffs, 3),
  sep = "=", collapse = ", "
))

scores <- tryCatch(
  {
    s <- corMatch(main_wav_fp, tpls)
    log_debug("corMatch 완료")
    s
  },
  error = function(e) {
    log_error("corMatch 실패: ", e$message)
    stop(e)
  }
)

detects <- tryCatch(
  {
    d <- findPeaks(scores)
    log_debug("findPeaks 완료")
    d
  },
  error = function(e) {
    log_error("findPeaks 실패: ", e$message)
    stop(e)
  }
)

# 후보 수 확인
for (nm in template_names) {
  det <- detects@detections[[nm]]
  n_cand <- if (!is.null(det)) nrow(det) else 0
  cat(sprintf("  [%s] 1단계 후보: %d건 (cutoff=%.3f)\n", nm, n_cand, candidate_cutoffs[nm]))

  # ★ 후보 0건일 때 상관계수 진단
  if (n_cand == 0) {
    sc_clean <- extract_scores(scores@scores[[nm]])
    if (length(sc_clean) > 0) {
      cat(sprintf(
        "    ※ 상관계수 통계: min=%.4f, max=%.4f, mean=%.4f, 상위5%%=%.4f\n",
        min(sc_clean), max(sc_clean), mean(sc_clean),
        quantile(sc_clean, 0.95)
      ))
      if (max(sc_clean) < 0.05) {
        cat("    ※ 최대 상관계수가 매우 낮음 → 템플릿-음원 간 유사성 극히 낮음\n")
        cat("    ※ 가능 원인: 주파수 범위 불일치, 녹음환경 차이, 종 불일치\n")
      }
    }
  }
}

# ============================================================
# 2단계: 종합 판별
# ============================================================
print_section(
  "2단계: 종합 판별 (Composite Scoring)",
  c(
    sprintf(
      "가중치: cor=%.0f%%, mfcc=%.0f%%, dtw_freq=%.0f%%, dtw_env=%.0f%%, band=%.0f%%, hr=%.0f%%, snr=%.0f%%",
      global_weights$cor_score * 100,
      global_weights$mfcc_score * 100,
      global_weights$dtw_freq * 100,
      global_weights$dtw_env * 100,
      global_weights$band_energy * 100,
      global_weights$harmonic_ratio * 100,
      global_weights$snr * 100
    ),
    "각 후보 구간에 대해 6가지 지표를 종합 평가합니다."
  )
)

final_results <- list()

# 종 이름 → species_list 인덱스 빠른 조회
sp_name_to_idx <- setNames(
  seq_along(species_list),
  vapply(species_list, `[[`, "", "name")
)

# C5: 템플릿별 독립 평가 후 종 단위 병합
all_template_results <- list()
template_weights_map <- list()  # ★ 종별 가중치 보관 (정규화 시 사용)

for (t_idx in seq_along(template_names)) {
  tmpl_name <- template_names[t_idx]
  sp_name <- template_to_species[[tmpl_name]]
  ref_wav <- template_wavs[[t_idx]]
  ref_freq <- template_freqs[[t_idx]]
  det <- detects@detections[[tmpl_name]]

  # 종 config 찾기
  sp_conf <- species_list[[sp_name_to_idx[sp_name]]]

  # 종별 가중치 오버라이드 확인
  sp_weights_mode <- if (!is.null(sp_conf$weights_mode)) sp_conf$weights_mode else "manual"
  if (sp_weights_mode == "auto" && !is.null(ref_wav)) {
    # ★ 자동 튜닝 — 첫 번째 템플릿의 wav/freq 사용
    first_tmpl <- if (!is.null(sp_conf$templates)) sp_conf$templates[[1]] else sp_conf
    sp_wav_for_tune <- tryCatch(
      {
        w <- safe_readWave(first_tmpl$wav_path)
        w <- ensure_mono(w)
        w <- normalize_amplitude(w)
        w
      },
      error = function(e) NULL
    )
    if (!is.null(sp_wav_for_tune)) {
      log_info(sprintf("  [%s] 자동 가중치 튜닝 중...", sp_name))
      # templates_info 리스트 구성 (auto_tune_weights 시그니처에 맞춤)
      tune_templates_info <- list(list(
        t_start = first_tmpl$t_start, t_end = first_tmpl$t_end,
        f_low = ref_freq$f_low, f_high = ref_freq$f_high
      ))
      tune_res <- auto_tune_weights(sp_wav_for_tune, tune_templates_info)
      sp_weights <- tune_res$weights
      log_info(sprintf(
        "  [%s] 자동 가중치: cor=%.3f mfcc=%.3f freq=%.3f env=%.3f band=%.3f hr=%.3f snr=%.3f",
        sp_name, sp_weights$cor_score, sp_weights$mfcc_score,
        sp_weights$dtw_freq, sp_weights$dtw_env, sp_weights$band_energy,
        sp_weights$harmonic_ratio, sp_weights$snr
      ))
    } else {
      sp_weights <- if (!is.null(sp_conf$weights)) sp_conf$weights else global_weights
    }
  } else {
    sp_weights <- if (!is.null(sp_conf$weights)) sp_conf$weights else global_weights
  }
  # snr 가중치 보충 (기존 config 호환)
  if (is.null(sp_weights$snr)) {
    sp_weights$snr <- DEFAULT_WEIGHTS$snr
    total_w <- sum(unlist(sp_weights))
    if (total_w > 0) sp_weights <- lapply(sp_weights, function(w) w / total_w)
  }
  final_cutoff <- sp_conf$cutoff
  template_weights_map[[tmpl_name]] <- sp_weights  # ★ 정규화용 가중치 보관

  if (is.null(det) || nrow(det) == 0) {
    cat(sprintf("\n  [%s] 후보 0건 → 건너뜀\n", tmpl_name))
    next
  }

  cat(sprintf(
    "\n  [%s] %d건 후보 종합 평가 중... (최종 cutoff=%.3f)\n",
    tmpl_name, nrow(det), final_cutoff
  ))

  if (is.null(ref_wav)) {
    log_error(sprintf("  %s: 레퍼런스 음원 없음, corMatch 점수만 사용", tmpl_name))
  }

  # 레퍼런스 길이 (초)
  ref_duration <- if (!is.null(ref_wav)) length(ref_wav@left) / ref_wav@samp.rate else 1.0

  # ★ 대역 에너지 포락선 1회 사전 계산 (피크 위치 보정용)
  band_env <- compute_band_energy_envelope(
    main_wav, ref_freq$f_low, ref_freq$f_high,
    window_sec = min(0.05, ref_duration * 0.1)
  )
  if (!is.null(band_env)) {
    log_info(sprintf("  ★ 대역 에너지 포락선 계산 완료 (피크 보정 활성)"))
  }

  # 각 후보 구간 평가
  sp_results <- vector("list", nrow(det))
  n_refined <- 0
  n_stage1_reject <- 0
  n_stage2_reject <- 0

  for (j in seq_len(nrow(det))) {
    peak_time_orig <- det$time[j]
    cor_score <- det$score[j]

    # ★ 피크 위치 보정: 대역 에너지 최대 지점으로 재정렬
    peak_time <- refine_peak_position(band_env, peak_time_orig, ref_duration)
    if (is.na(peak_time)) peak_time <- peak_time_orig
    if (!is.na(peak_time) && !is.na(peak_time_orig) &&
        abs(peak_time - peak_time_orig) > ref_duration * 0.1) {
      n_refined <- n_refined + 1
    }

    # 후보 구간 추출
    margin <- ref_duration * 0.2
    seg_start <- max(0, peak_time - ref_duration / 2 - margin)
    seg_end <- peak_time + ref_duration / 2 + margin
    segment <- extract_segment(main_wav, seg_start, seg_end)


    if (is.null(segment) || length(segment@left) < 100) {
      sp_results[[j]] <- data.frame(
        species = sp_name, time = peak_time,
        cor_score = cor_score, mfcc_score = 0, dtw_freq = 0,
        dtw_env = 0, band_energy = 0, harmonic_ratio = 0, snr = 0,
        composite = cor_score * sp_weights$cor_score,
        template_label = tmpl_name,
        det_f_low = round(ref_freq$f_low, 3),
        det_f_high = round(ref_freq$f_high, 3),
        detection_pass = 1L,
        stringsAsFactors = FALSE
      )
      next
    }

    # --- B5: Staged Evaluation (3단계 계층적 Early Rejection) ---
    individual_scores <- list(cor_score = max(0, cor_score))
    staged_rejected <- FALSE

    # ── Stage 1: cor + band + snr + MFCC (~61% 가중치) ──
    individual_scores$band_energy <- compute_band_energy_ratio(
      segment, ref_freq$f_low, ref_freq$f_high
    )
    individual_scores$snr <- compute_snr_ratio(
      segment, ref_freq$f_low, ref_freq$f_high
    )
    if (!is.null(ref_wav)) {
      individual_scores$mfcc_score <- compute_mfcc_dtw_similarity(ref_wav, segment)
    } else {
      individual_scores$mfcc_score <- 0
    }

    if (isTRUE(staged_eval_enabled)) {
      s1_score <- individual_scores$cor_score * sp_weights$cor_score +
                  individual_scores$band_energy * sp_weights$band_energy +
                  individual_scores$snr * sp_weights$snr +
                  individual_scores$mfcc_score * sp_weights$mfcc_score
      remaining_w1 <- sp_weights$harmonic_ratio + sp_weights$dtw_freq + sp_weights$dtw_env
      if (s1_score + remaining_w1 < final_cutoff) {
        n_stage1_reject <- n_stage1_reject + 1
        individual_scores$harmonic_ratio <- 0
        individual_scores$dtw_freq <- 0
        individual_scores$dtw_env <- 0
        staged_rejected <- TRUE
      }
    }

    # ── Stage 2: + harmonic_ratio (~79% 가중치) ──
    if (!staged_rejected) {
      individual_scores$harmonic_ratio <- compute_harmonic_ratio(
        segment, ref_freq$f_low, ref_freq$f_high
      )

      if (isTRUE(staged_eval_enabled)) {
        s2_score <- s1_score +
                    individual_scores$harmonic_ratio * sp_weights$harmonic_ratio
        remaining_w2 <- sp_weights$dtw_freq + sp_weights$dtw_env
        if (s2_score + remaining_w2 < final_cutoff) {
          n_stage2_reject <- n_stage2_reject + 1
          individual_scores$dtw_freq <- 0
          individual_scores$dtw_env <- 0
          staged_rejected <- TRUE
        }
      }
    }

    # ── Stage 3: + dtw_freq + dtw_env (100%) ──
    if (!staged_rejected && !is.null(ref_wav)) {
      individual_scores$dtw_freq <- compute_freq_contour_dtw(
        ref_wav, segment, ref_freq$f_low, ref_freq$f_high
      )
      individual_scores$dtw_env <- compute_envelope_dtw(ref_wav, segment)
    } else if (!staged_rejected) {
      individual_scores$dtw_freq <- 0
      individual_scores$dtw_env <- 0
    }

    composite <- compute_composite_score(individual_scores, sp_weights)

    # ★ 검출 구간의 실제 주파수 범위 계산
    seg_freq_bounds <- detect_freq_bounds(segment, ref_freq$f_low, ref_freq$f_high)

    sp_results[[j]] <- data.frame(
      species = sp_name,
      time = peak_time,
      cor_score = round(individual_scores$cor_score, 4),
      mfcc_score = round(individual_scores$mfcc_score, 4),
      dtw_freq = round(individual_scores$dtw_freq, 4),
      dtw_env = round(individual_scores$dtw_env, 4),
      band_energy = round(individual_scores$band_energy, 4),
      harmonic_ratio = round(individual_scores$harmonic_ratio, 4),
      snr = round(individual_scores$snr, 4),
      composite = round(composite, 4),
      template_label = tmpl_name,
      det_f_low = seg_freq_bounds$det_f_low,
      det_f_high = seg_freq_bounds$det_f_high,
      detection_pass = 1L,
      stringsAsFactors = FALSE
    )

    if (j == 1 || j %% 10 == 0 || j == nrow(det)) {
      cat(sprintf(
        "    [%d/%d] t=%.1f, cor=%.3f, mfcc=%.3f, freq=%.3f, env=%.3f, band=%.3f, snr=%.3f → 종합=%.3f\n",
        j, nrow(det), peak_time,
        individual_scores$cor_score,
        individual_scores$mfcc_score,
        individual_scores$dtw_freq,
        individual_scores$dtw_env,
        individual_scores$band_energy,
        individual_scores$snr,
        composite
      ))
    }
  }

  if (n_refined > 0) {
    log_info(sprintf("  ★ 피크 보정: %d/%d건 위치 재조정 (대역 에너지 기반)", n_refined, nrow(det)))
  }

  # ★ 에너지 피크 보충: corMatch가 놓친 울음 구간을 대역 에너지 포락선으로 탐색
  if (!is.null(band_env) && !is.null(ref_wav)) {
    energy_peaks <- find_energy_peaks(band_env, ref_duration)

    # 기존 후보와 겹치지 않는 에너지 피크만 추가
    existing_times <- sapply(sp_results, function(r) {
      if (is.null(r)) NA else r$time[1]
    })
    existing_times <- existing_times[!is.na(existing_times)]
    min_gap_sec <- max(0.5, ref_duration * 0.5)

    n_energy_added <- 0
    for (ep in energy_peaks) {
      if (length(existing_times) > 0 && any(abs(existing_times - ep) < min_gap_sec)) next

      margin <- ref_duration * 0.2
      seg_start <- max(0, ep - ref_duration / 2 - margin)
      seg_end <- ep + ref_duration / 2 + margin
      segment <- extract_segment(main_wav, seg_start, seg_end)
      if (is.null(segment) || length(segment@left) < 100) next

      individual_scores <- list(cor_score = 0)
      individual_scores$band_energy <- compute_band_energy_ratio(
        segment, ref_freq$f_low, ref_freq$f_high
      )
      individual_scores$harmonic_ratio <- compute_harmonic_ratio(
        segment, ref_freq$f_low, ref_freq$f_high
      )
      individual_scores$snr <- compute_snr_ratio(
        segment, ref_freq$f_low, ref_freq$f_high
      )
      individual_scores$mfcc_score <- compute_mfcc_dtw_similarity(ref_wav, segment)
      individual_scores$dtw_freq <- compute_freq_contour_dtw(
        ref_wav, segment, ref_freq$f_low, ref_freq$f_high
      )
      individual_scores$dtw_env <- compute_envelope_dtw(ref_wav, segment)

      composite <- compute_composite_score(individual_scores, sp_weights)

      n_energy_added <- n_energy_added + 1
      # ★ 에너지 피크 구간의 실제 주파수 범위 계산
      ep_freq_bounds <- detect_freq_bounds(segment, ref_freq$f_low, ref_freq$f_high)

      sp_results[[length(sp_results) + 1]] <- data.frame(
        species = sp_name,
        time = ep,
        cor_score = 0,
        mfcc_score = round(individual_scores$mfcc_score, 4),
        dtw_freq = round(individual_scores$dtw_freq, 4),
        dtw_env = round(individual_scores$dtw_env, 4),
        band_energy = round(individual_scores$band_energy, 4),
        harmonic_ratio = round(individual_scores$harmonic_ratio, 4),
        snr = round(individual_scores$snr, 4),
        composite = round(composite, 4),
        template_label = tmpl_name,
        det_f_low = ep_freq_bounds$det_f_low,
        det_f_high = ep_freq_bounds$det_f_high,
        detection_pass = 1L,
        stringsAsFactors = FALSE
      )
      existing_times <- c(existing_times, ep)
    }

    if (n_energy_added > 0) {
      log_info(sprintf("  ★ 에너지 피크 보충: %d건 추가 후보 (corMatch 미검출 구간)", n_energy_added))
    }
  }

  # 결합
  sp_df <- do.call(rbind, sp_results)

  # ★ Staged eval 통계 출력
  if (isTRUE(staged_eval_enabled) && (n_stage1_reject > 0 || n_stage2_reject > 0)) {
    cat(sprintf("  [%s] Staged eval: S1 조기종료 %d건, S2 조기종료 %d건 / 총 %d건 (%.0f%% 절약)\n",
      tmpl_name, n_stage1_reject, n_stage2_reject, nrow(det),
      (n_stage1_reject + n_stage2_reject) / max(1, nrow(det)) * 100))
  }

  all_template_results[[tmpl_name]] <- sp_df
}

# ============================================================
# C5: 종 단위 병합 + NMS
# ============================================================
for (sp_name in species_names_unique) {
  # 해당 종의 모든 템플릿 결과 합치기
  sp_tmpl_names <- template_names[vapply(template_names, function(tn) {
    template_to_species[[tn]] == sp_name
  }, logical(1))]

  sp_dfs <- lapply(sp_tmpl_names, function(tn) all_template_results[[tn]])
  sp_dfs <- sp_dfs[!vapply(sp_dfs, is.null, logical(1))]

  if (length(sp_dfs) == 0) {
    final_results[[sp_name]] <- NULL
    next
  }

  combined <- do.call(rbind, sp_dfs)

  # 종 config 조회 (앙상블 + cutoff에서 사용)
  sp_conf <- species_list[[sp_name_to_idx[sp_name]]]

  # ★ 멀티 템플릿 앙상블 (NMS 전에 적용)
  ensemble_strategy <- if (!is.null(sp_conf$ensemble_strategy))
                         sp_conf$ensemble_strategy else "max"
  nms_gap <- if (!is.null(sp_conf$nms_gap)) sp_conf$nms_gap else 0.5
  if (length(sp_tmpl_names) > 1 && nrow(combined) > 1) {
    n_before_ensemble <- nrow(combined)
    combined <- ensemble_multi_template(combined,
      time_gap = nms_gap, strategy = ensemble_strategy)
    cat(sprintf("  [%s] 앙상블 (%s): %d건 → %d건\n",
      sp_name, ensemble_strategy, n_before_ensemble, nrow(combined)))
  }

  # ★ 후보 간 정규화: 비구별 지표의 상수 기여 제거 → 점수 압축 방지
  sp_w <- template_weights_map[[sp_tmpl_names[1]]]
  if (!is.null(sp_w) && nrow(combined) >= 3) {
    combined <- normalize_candidate_scores(combined, sp_w)
  }

  # 최종 cutoff 적용
  final_cutoff <- sp_conf$cutoff

  sp_df_pass <- combined[combined$composite >= final_cutoff, , drop = FALSE]
  cat(sprintf(
    "  [%s] 종합판별: %d건 → 최종 %d건 (cutoff=%.3f)\n",
    sp_name, nrow(combined), nrow(sp_df_pass), final_cutoff
  ))

  # 진단
  n_valid <- sum(!is.na(combined$composite))
  if (nrow(sp_df_pass) == 0 && n_valid > 0) {
    best_idx <- which.max(combined$composite)
    if (length(best_idx) > 0) {
      best <- combined[best_idx, ]
      cat(sprintf("    ** 검출 없음. 최고 종합점수: %.4f (t=%.1f)\n", best$composite, best$time))
      cat(sprintf(
        "       내역: cor=%.3f, mfcc=%.3f, freq=%.3f, env=%.3f, band=%.3f, hr=%.3f, snr=%.3f\n",
        best$cor_score, best$mfcc_score, best$dtw_freq, best$dtw_env,
        best$band_energy, best$harmonic_ratio,
        if ("snr" %in% names(best)) best$snr else 0
      ))
      if (best$composite > final_cutoff * 0.7) {
        cat(sprintf(
          "    [제안] cutoff를 %.2f로 낮추면 검출될 수 있습니다.\n",
          best$composite * 0.9
        ))
      }
    }
  } else if (n_valid == 0 && nrow(combined) > 0) {
    cat("    ** 모든 후보의 종합점수가 NA입니다.\n")
  }

  # passed 마킹 (NA composite → FALSE)
  if (nrow(combined) > 0) {
    combined$passed <- ifelse(is.na(combined$composite), FALSE,
      combined$composite >= final_cutoff
    )
  }

  # C2 + C5: NMS 병합 (멀티 템플릿 간 중복도 제거)
  n_before_nms <- sum(combined$passed, na.rm = TRUE)
  if (isTRUE(n_before_nms > 1)) {
    sp_df_pass_only <- combined[combined$passed == TRUE, , drop = FALSE]
    nms_gap <- if (!is.null(sp_conf$nms_gap)) sp_conf$nms_gap else 0.5
    sp_df_pass_nms <- nms_detections(sp_df_pass_only, min_gap = nms_gap)
    removed_times <- setdiff(sp_df_pass_only$time, sp_df_pass_nms$time)
    if (length(removed_times) > 0) {
      combined$passed[combined$time %in% removed_times] <- FALSE
      cat(sprintf(
        "  [%s] NMS 병합: %d건 → %d건 (gap=%.1f초)\n",
        sp_name, n_before_nms, nrow(sp_df_pass_nms), nms_gap
      ))
    }
  }

  final_results[[sp_name]] <- combined
}

# ============================================================
# ★ Pass 2: 주파수 대역 필터링 (검출된 소리 제거 후 재검출)
# ============================================================
if (freq_filter_enabled) {
  # 1단계: Pass 1에서 검출된 바운딩 박스 수집
  pass1_bboxes <- list()
  for (sp_name in names(final_results)) {
    df <- final_results[[sp_name]]
    if (is.null(df) || nrow(df) == 0) next
    df_pass <- df[df$passed == TRUE, , drop = FALSE]
    if (nrow(df_pass) == 0) next

    for (i in seq_len(nrow(df_pass))) {
      row <- df_pass[i, ]
      # 검출 시간 범위: peak_time ± ref_duration/2
      # 레퍼런스 길이를 추정 (템플릿별로 다르므로 0.5초로 기본)
      half_dur <- 0.5
      pass1_bboxes[[length(pass1_bboxes) + 1]] <- list(
        t_start = row$time - half_dur,
        t_end = row$time + half_dur,
        f_low = row$det_f_low,   # kHz
        f_high = row$det_f_high  # kHz
      )
    }
  }

  if (length(pass1_bboxes) > 0) {
    cat(sprintf("\n★ Pass 2: %d개 검출 구간 제거 후 재분석 시작\n", length(pass1_bboxes)))

   tryCatch({
    # 2단계: STFT 마스킹 (50% 확장)
    masked_wav <- stft_mask_audio(main_wav, pass1_bboxes, expansion = 1.5)
    cat("  STFT 마스킹 완료\n")

    # 마스킹된 WAV를 임시 파일로 저장 (corMatch 용)
    masked_wav_fp <- tempfile(fileext = ".wav")
    writeWave(masked_wav, masked_wav_fp)
    temp_files <- c(temp_files, masked_wav_fp)

    # 3단계: corMatch 재실행
    cat("  corMatch 재실행 중...\n")
    pass2_scores <- tryCatch(
      corMatch(masked_wav_fp, tpls),
      error = function(e) {
        cat(sprintf("  Pass 2 corMatch 실패: %s\n", e$message))
        NULL
      }
    )

    if (!is.null(pass2_scores)) {
      pass2_detects <- tryCatch(findPeaks(pass2_scores), error = function(e) NULL)

      if (!is.null(pass2_detects)) {
        # score.cutoff는 makeCorTemplate에서 이미 설정됨 → 별도 설정 불필요

        # 4단계: 각 템플릿별 평가 (Pass 1과 동일 로직)
        pass2_template_results <- list()
        for (t_idx in seq_along(template_names)) {
          tmpl_name <- template_names[t_idx]
          sp_name <- template_to_species[[tmpl_name]]
          ref_wav <- template_wavs[[t_idx]]
          ref_freq <- template_freqs[[t_idx]]
          det2 <- pass2_detects@detections[[tmpl_name]]

          sp_conf <- species_list[[sp_name_to_idx[sp_name]]]
          sp_weights <- template_weights_map[[tmpl_name]]
          if (is.null(sp_weights)) sp_weights <- global_weights
          final_cutoff <- sp_conf$cutoff

          if (is.null(det2) || nrow(det2) == 0) next

          ref_duration <- if (!is.null(ref_wav)) length(ref_wav@left) / ref_wav@samp.rate else 1.0
          sp2_results <- list()

          for (j in seq_len(nrow(det2))) {
            peak_time <- det2[j, "time"]
            cor_score <- det2[j, "score"]
            if (is.na(peak_time) || is.na(cor_score)) next

            # Pass 1 검출과 중복되는 시간이면 건너뜀 (0.3초 이내)
            skip <- FALSE
            for (bbox in pass1_bboxes) {
              if (abs(peak_time - (bbox$t_start + bbox$t_end) / 2) < 0.3) {
                skip <- TRUE
                break
              }
            }
            if (skip) next

            margin <- ref_duration * 0.2
            seg_start <- max(0, peak_time - ref_duration / 2 - margin)
            seg_end <- peak_time + ref_duration / 2 + margin
            # ★ Pass 2는 마스킹된 음원에서 세그먼트 추출
            segment <- extract_segment(masked_wav, seg_start, seg_end)
            if (is.null(segment) || length(segment@left) < 100) next

            individual_scores <- list(cor_score = max(0, cor_score))
            individual_scores$band_energy <- compute_band_energy_ratio(segment, ref_freq$f_low, ref_freq$f_high)
            individual_scores$harmonic_ratio <- compute_harmonic_ratio(segment, ref_freq$f_low, ref_freq$f_high)
            individual_scores$snr <- compute_snr_ratio(segment, ref_freq$f_low, ref_freq$f_high)

            if (!is.null(ref_wav)) {
              individual_scores$mfcc_score <- compute_mfcc_dtw_similarity(ref_wav, segment)
              individual_scores$dtw_freq <- compute_freq_contour_dtw(ref_wav, segment, ref_freq$f_low, ref_freq$f_high)
              individual_scores$dtw_env <- compute_envelope_dtw(ref_wav, segment)
            } else {
              individual_scores$mfcc_score <- 0
              individual_scores$dtw_freq <- 0
              individual_scores$dtw_env <- 0
            }

            composite <- compute_composite_score(individual_scores, sp_weights)
            seg_freq_bounds <- detect_freq_bounds(segment, ref_freq$f_low, ref_freq$f_high)

            sp2_results[[length(sp2_results) + 1]] <- data.frame(
              species = sp_name, time = peak_time,
              cor_score = round(individual_scores$cor_score, 4),
              mfcc_score = round(individual_scores$mfcc_score, 4),
              dtw_freq = round(individual_scores$dtw_freq, 4),
              dtw_env = round(individual_scores$dtw_env, 4),
              band_energy = round(individual_scores$band_energy, 4),
              harmonic_ratio = round(individual_scores$harmonic_ratio, 4),
              snr = round(individual_scores$snr, 4),
              composite = round(composite, 4),
              template_label = tmpl_name,
              det_f_low = seg_freq_bounds$det_f_low,
              det_f_high = seg_freq_bounds$det_f_high,
              detection_pass = 2L,
              stringsAsFactors = FALSE
            )
          }

          if (length(sp2_results) > 0) {
            pass2_template_results[[tmpl_name]] <- do.call(rbind, sp2_results)
          }
        }

        # 5단계: Pass 2 결과를 final_results에 병합
        n_pass2_total <- 0
        for (sp_name in species_names_unique) {
          sp_tmpl_names <- template_names[vapply(template_names, function(tn) {
            template_to_species[[tn]] == sp_name
          }, logical(1))]

          sp2_dfs <- lapply(sp_tmpl_names, function(tn) pass2_template_results[[tn]])
          sp2_dfs <- sp2_dfs[!vapply(sp2_dfs, is.null, logical(1))]
          if (length(sp2_dfs) == 0) next

          sp2_combined <- do.call(rbind, sp2_dfs)
          sp_conf <- species_list[[sp_name_to_idx[sp_name]]]
          final_cutoff <- sp_conf$cutoff

          sp2_combined$passed <- sp2_combined$composite >= final_cutoff
          n_pass2 <- sum(sp2_combined$passed, na.rm = TRUE)
          n_pass2_total <- n_pass2_total + n_pass2

          if (n_pass2 > 0) {
            cat(sprintf("  [%s] Pass 2: %d건 추가 검출\n", sp_name, n_pass2))
          }

          # 기존 결과에 병합
          existing <- final_results[[sp_name]]
          if (!is.null(existing)) {
            # 컬럼 정렬: Pass 1/2 결과의 컬럼이 다를 수 있음
            all_cols <- union(names(existing), names(sp2_combined))
            for (col in setdiff(all_cols, names(existing)))    existing[[col]] <- NA
            for (col in setdiff(all_cols, names(sp2_combined))) sp2_combined[[col]] <- NA
            final_results[[sp_name]] <- rbind(existing[all_cols], sp2_combined[all_cols])
          } else {
            final_results[[sp_name]] <- sp2_combined
          }
        }

        cat(sprintf("★ Pass 2 완료: 총 %d건 추가 검출\n\n", n_pass2_total))
      }
    }
   }, error = function(e) {
     cat(sprintf("\n★ Pass 2 오류 (무시하고 계속): %s\n\n", e$message))
   })
  } else {
    cat("\n★ Pass 2 건너뜀: Pass 1에서 검출된 소리가 없습니다.\n\n")
  }
}

# ============================================================
# 결과 저장
# ============================================================
log_info("결과 저장 중...")

# 1) 최종 검출 결과 (통과 건만)
pass_list <- lapply(names(final_results), function(nm) {
  df <- final_results[[nm]]
  if (is.null(df) || nrow(df) == 0) {
    return(NULL)
  }
  df_pass <- df[df$passed == TRUE, , drop = FALSE]
  if (nrow(df_pass) == 0) {
    return(NULL)
  }
  df_pass$time_display <- sprintf(
    "%02d:%04.1f",
    floor(df_pass$time / 60),
    round(df_pass$time %% 60, 1)
  )
  df_pass[, c(
    "species", "template_label", "time_display", "time", "composite",
    "cor_score", "mfcc_score", "dtw_freq", "dtw_env", "band_energy",
    "harmonic_ratio", "det_f_low", "det_f_high", "detection_pass"
  )]
})
pass_list <- pass_list[!vapply(pass_list, is.null, logical(1))]

if (length(pass_list) > 0) {
  results_csv <- do.call(rbind, pass_list)
} else {
  results_csv <- data.frame(
    species = character(), template_label = character(),
    time_display = character(), time = numeric(),
    composite = numeric(), cor_score = numeric(), mfcc_score = numeric(),
    dtw_freq = numeric(), dtw_env = numeric(), band_energy = numeric(),
    harmonic_ratio = numeric(), det_f_low = numeric(), det_f_high = numeric(),
    detection_pass = integer(),
    stringsAsFactors = FALSE
  )
}

# 기존 호환 CSV (species, time_display, time, score)
compat_csv <- results_csv
if (nrow(compat_csv) > 0) {
  compat_csv$score <- compat_csv$composite
  compat_csv <- compat_csv[, c("species", "time_display", "time", "score", "det_f_low", "det_f_high", "detection_pass")]
}
write.csv(compat_csv, file.path(output_dir, "results.csv"),
  row.names = FALSE, fileEncoding = "UTF-8"
)

# 상세 CSV (모든 지표 포함)
write.csv(results_csv, file.path(output_dir, "results_detailed.csv"),
  row.names = FALSE, fileEncoding = "UTF-8"
)

# 전체 후보 진단 CSV (통과/미통과 모두)
all_candidates <- do.call(rbind, lapply(final_results, function(df) {
  if (is.null(df) || nrow(df) == 0) {
    return(NULL)
  }
  df
}))
if (!is.null(all_candidates) && nrow(all_candidates) > 0) {
  all_candidates$time_display <- sprintf(
    "%02d:%04.1f",
    floor(all_candidates$time / 60),
    round(all_candidates$time %% 60, 1)
  )
  write.csv(all_candidates, file.path(output_dir, "candidates_all.csv"),
    row.names = FALSE, fileEncoding = "UTF-8"
  )
}

# C4: JSON 결과 저장
tryCatch(
  {
    json_results <- list()
    for (sp_name in names(final_results)) {
      df <- final_results[[sp_name]]
      if (is.null(df) || nrow(df) == 0) next

      df_pass <- df[df$passed == TRUE, , drop = FALSE]
      if (nrow(df_pass) == 0) next

      detections <- lapply(seq_len(nrow(df_pass)), function(i) {
        row <- df_pass[i, ]
        list(
          time = round(row$time, 3),
          time_display = sprintf("%02d:%04.1f", floor(row$time / 60), row$time %% 60),
          freq_range = list(
            f_low = round(row$det_f_low, 3),
            f_high = round(row$det_f_high, 3)
          ),
          scores = list(
            composite = round(row$composite, 4),
            cor_score = round(row$cor_score, 4),
            mfcc_score = round(row$mfcc_score, 4),
            dtw_freq = round(row$dtw_freq, 4),
            dtw_env = round(row$dtw_env, 4),
            band_energy = round(row$band_energy, 4),
            harmonic_ratio = round(row$harmonic_ratio, 4)
          ),
          detection_pass = if (!is.null(row$detection_pass)) row$detection_pass else 1L
        )
      })

      json_results[[sp_name]] <- list(
        species = sp_name,
        total_candidates = nrow(df),
        n_detections = nrow(df_pass),
        detections = detections,
        cutoff = species_list[[sp_name_to_idx[sp_name]]]$cutoff
      )
    }

    json_output <- list(
      analysis_date = format(Sys.time(), "%Y-%m-%dT%H:%M:%S"),
      source_file = basename(main_wav_path),
      n_species = length(json_results),
      results = json_results
    )

    writeLines(
      toJSON(json_output, auto_unbox = TRUE, pretty = TRUE),
      file.path(output_dir, "results.json")
    )
    log_info(sprintf("  JSON 결과: %s", file.path(output_dir, "results.json")))
  },
  error = function(e) {
    log_error(sprintf("JSON 저장 실패: %s", e$message))
  }
)

# ============================================================
# 종합 점수 시계열 PNG
# ============================================================
tryCatch(
  {
    for (sp_name in names(final_results)) {
      df <- final_results[[sp_name]]
      if (is.null(df) || nrow(df) == 0) next

      # NA 행 제거 (time 또는 composite가 NA인 행은 플롯 불가)
      df <- df[!is.na(df$time) & !is.na(df$composite), , drop = FALSE]
      if (nrow(df) == 0) next

      png_path <- file.path(output_dir, paste0("composite_", sp_name, ".png"))
      png(png_path, width = 1000, height = 500)
      par(mar = c(5, 4, 3, 2))

      cutoff_val <- species_list[[sp_name_to_idx[sp_name]]]$cutoff

      plot(df$time, df$composite,
        type = "h", lwd = 2,
        col = ifelse(df$composite >= cutoff_val, "darkgreen", "gray70"),
        main = sprintf("%s - 종합 판별 점수", sp_name),
        xlab = "시간 (초)", ylab = "종합 점수",
        ylim = c(0, max(1, max(df$composite, na.rm = TRUE) * 1.1))
      )
      abline(h = cutoff_val, col = "red", lty = 2, lwd = 2)
      text(min(df$time, na.rm = TRUE), cutoff_val + 0.03,
        sprintf("cutoff = %.3f", cutoff_val),
        col = "red", adj = 0
      )

      # 개별 점수도 작게 표시
      points(df$time, df$cor_score * 0.8, pch = 1, cex = 0.5, col = "blue")
      points(df$time, df$mfcc_score * 0.8, pch = 2, cex = 0.5, col = "orange")
      legend("topright",
        legend = c("종합점수", "corMatch", "MFCC", "cutoff"),
        col = c("darkgreen", "blue", "orange", "red"),
        lty = c(1, NA, NA, 2), pch = c(NA, 1, 2, NA),
        cex = 0.8, bg = "white"
      )
      dev.off()
    }
  },
  error = function(e) {
    safe_dev_off()
    log_error("점수 그래프 저장 실패: ", e$message)
  }
)

# ============================================================
# 최종 요약
# ============================================================
print_section("최종 요약")

total_detections <- 0
for (sp_name in species_names_unique) {
  df <- final_results[[sp_name]]
  n_total <- if (!is.null(df)) nrow(df) else 0
  n_pass <- if (!is.null(df)) sum(df$passed, na.rm = TRUE) else 0
  total_detections <- total_detections + n_pass
  cat(sprintf("  %s: 후보 %d건 → 최종 %d건 검출\n", sp_name, n_total, n_pass))

  if (isTRUE(n_pass > 0)) {
    df_pass <- df[df$passed == TRUE, ]
    cat(sprintf(
      "    종합점수: %.3f ~ %.3f (평균 %.3f)\n",
      min(df_pass$composite), max(df_pass$composite), mean(df_pass$composite)
    ))
  }
}

cat(sprintf("\n총 검출: %d건\n", total_detections))
cat(sprintf("결과 파일:\n"))
cat(sprintf("  기존 호환:  %s\n", file.path(output_dir, "results.csv")))
cat(sprintf("  상세 결과:  %s\n", file.path(output_dir, "results_detailed.csv")))
cat(sprintf("  전체 후보:  %s\n", file.path(output_dir, "candidates_all.csv")))
cat("[DONE]\n")

