# ============================================================
# Rank-Biserial 하이브리드 방식 프로토타입
# 기존 Cohen's d 기반 auto_tune_weights() 대체용
# ============================================================
#
# 목적: new_analysis.R:1802-1866의 Cohen's d 기반 변별력 계산을
#        Rank-Biserial Correlation 기반으로 교체하는 프로토타입 코드
#
# 사용법: 이 파일의 함수들은 new_analysis.R에 통합하여 사용할 수 있음
#         테스트 후 기존 코드를 교체
# ============================================================

#' Rank-Biserial Correlation 계산 (Mann-Whitney U 기반)
#'
#' Wendt 공식: r_rb = 1 - 2U / (n1 * n2)
#'
#' @param pos_vals 양성 샘플 값 벡터
#' @param neg_vals 음성 샘플 값 벡터
#' @return list(r_rb, U, p_value, n_pos, n_neg)
compute_rank_biserial <- function(pos_vals, neg_vals) {
  n1 <- length(pos_vals)
  n2 <- length(neg_vals)

  if (n1 < 2 || n2 < 2) {
    return(list(r_rb = 0, U = NA, p_value = 1, n_pos = n1, n_neg = n2))
  }

  # Mann-Whitney U 검정 (동순위 보정 자동 적용)
  wt <- tryCatch(
    suppressWarnings(wilcox.test(pos_vals, neg_vals,
      exact = FALSE,           # 동순위 시 근사 사용
      correct = TRUE,          # 연속 보정
      alternative = "greater"  # 양성 > 음성 방향
    )),
    error = function(e) NULL
  )

  if (is.null(wt)) {
    return(list(r_rb = 0, U = NA, p_value = 1, n_pos = n1, n_neg = n2))
  }

  U <- as.numeric(wt$statistic)
  p_value <- wt$p.value

  # Wendt 공식: r_rb = 1 - 2U / (n1 * n2)
  # 주의: R의 wilcox.test는 U = sum(rank(c(x,y))[1:n1]) - n1*(n1+1)/2
  #       이는 "x가 y보다 큰 쌍의 수"이므로 부호가 반대
  #       r_rb = 2*U/(n1*n2) - 1 로 계산해야 양성>음성일 때 양수
  r_rb <- (2 * U) / (n1 * n2) - 1

  list(
    r_rb = r_rb,
    U = U,
    p_value = p_value,
    n_pos = n1,
    n_neg = n2
  )
}


#' MAD 기반 Robust 크기 효과 계산
#'
#' 포화 구간(r_rb ≈ 1.0)에서 지표 간 차등을 위한 보조 지표
#' MAD = Median Absolute Deviation (× 1.4826으로 정규분포 SD와 일치)
#'
#' @param pos_vals 양성 샘플 값 벡터
#' @param neg_vals 음성 샘플 값 벡터
#' @return robust magnitude (0 이상)
compute_robust_magnitude <- function(pos_vals, neg_vals) {
  med_pos <- median(pos_vals, na.rm = TRUE)
  med_neg <- median(neg_vals, na.rm = TRUE)

  mad_pos <- mad(pos_vals, constant = 1.4826, na.rm = TRUE)
  mad_neg <- mad(neg_vals, constant = 1.4826, na.rm = TRUE)

  mad_pooled <- sqrt((mad_pos^2 + mad_neg^2) / 2)

  if (is.na(mad_pooled) || mad_pooled < 0.001) {
    # MAD가 극소 → 중앙값 차이 자체를 직접 사용 (×10 스케일링)
    return(abs(med_pos - med_neg) * 10)
  }

  abs(med_pos - med_neg) / mad_pooled
}


#' 하이브리드 변별력 계산 (Rank-Biserial + MAD 보정)
#'
#' Step 1: Rank-Biserial로 순위 기반 변별력 측정
#' Step 2: 포화 구간(r_rb > 0.95)에서 MAD 기반 크기 정보로 보정
#'
#' @param pos_vals 양성 샘플 값 벡터
#' @param neg_vals 음성 샘플 값 벡터
#' @return disc_power (변별력, 0 이상)
compute_hybrid_disc_power <- function(pos_vals, neg_vals) {
  rb <- compute_rank_biserial(pos_vals, neg_vals)

  # 역방향 (양성 ≤ 음성) → 변별력 0
  if (rb$r_rb <= 0) {
    return(0)
  }

  r_rb <- rb$r_rb

  # 포화 보정: r_rb가 0.95 이상이고 충분한 샘플이 있을 때
  if (r_rb > 0.95 && rb$n_pos >= 5 && rb$n_neg >= 5) {
    magnitude <- compute_robust_magnitude(pos_vals, neg_vals)
    # log1p로 극단적 크기 차이 완화 + r_rb에 곱하기
    disc_power <- r_rb * (1 + log1p(magnitude))
  } else {
    disc_power <- r_rb
  }

  max(0, disc_power)
}


# ============================================================
# 아래는 new_analysis.R:1802-1866을 대체하는 코드 블록
# (기존 for 루프 내부만 교체)
# ============================================================

#' auto_tune_weights()의 변별력 계산 섹션 대체 코드
#'
#' 기존: Cohen's d (+ LOO)
#' 신규: Rank-Biserial Hybrid (+ 포화 보정)
#'
#' @details
#' 이 함수는 new_analysis.R의 auto_tune_weights() 내부
#' 1802-1866 라인을 대체합니다.
#'
#' 통합 방법:
#'   1. 이 파일의 compute_rank_biserial(), compute_robust_magnitude(),
#'      compute_hybrid_disc_power() 함수를 new_analysis.R에 추가
#'   2. 아래의 compute_disc_power_rankbiserial() 로직으로
#'      기존 Cohen's d 루프를 교체
compute_disc_power_rankbiserial <- function(positive_scores, negative_scores,
                                            metric_names, n_pos, n_neg) {
  disc_power <- numeric(length(metric_names))
  names(disc_power) <- metric_names

  pos_means <- numeric(length(metric_names))
  neg_means <- numeric(length(metric_names))
  names(pos_means) <- metric_names
  names(neg_means) <- metric_names

  # 추가 진단 정보
  rb_values <- numeric(length(metric_names))
  p_values <- numeric(length(metric_names))
  names(rb_values) <- metric_names
  names(p_values) <- metric_names

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

    pos_means[mi] <- mean(pos_vals, na.rm = TRUE)
    neg_means[mi] <- mean(neg_vals, na.rm = TRUE)

    # ★ Rank-Biserial 하이브리드 변별력 계산
    rb_result <- compute_rank_biserial(pos_vals, neg_vals)
    rb_values[mi] <- rb_result$r_rb
    p_values[mi] <- rb_result$p_value

    disc_power[mi] <- compute_hybrid_disc_power(pos_vals, neg_vals)
  }

  # 로깅
  cat(sprintf("  지표별 변별력 (Rank-Biserial 하이브리드):\n"))
  for (mn in metric_names) {
    cat(sprintf(
      "    %s: 양성=%.3f, 음성=%.3f → r_rb=%.3f (p=%.4f) → 변별력=%.3f\n",
      mn, pos_means[mn], neg_means[mn], rb_values[mn], p_values[mn], disc_power[mn]
    ))
  }

  # 역방향 지표 처리 (기존 로직 유지)
  for (mi in seq_along(metric_names)) {
    if (pos_means[mi] <= neg_means[mi]) {
      if (disc_power[mi] > 0) {
        cat(sprintf(
          "    ★ %s: 역방향 (양성=%.3f ≤ 음성=%.3f) → 가중치 0\n",
          metric_names[mi], pos_means[mi], neg_means[mi]
        ))
      }
      disc_power[mi] <- 0
    }
  }

  list(
    disc_power = disc_power,
    pos_means = pos_means,
    neg_means = neg_means,
    rb_values = rb_values,
    p_values = p_values
  )
}


# ============================================================
# 비교 테스트: Cohen's d vs Rank-Biserial 시뮬레이션
# ============================================================

#' 두 방식의 비교 시뮬레이션
#'
#' 다양한 분포 시나리오에서 Cohen's d와 Rank-Biserial의
#' 변별력 추정 안정성을 비교
compare_methods_simulation <- function(n_sim = 100) {
  set.seed(42)

  scenarios <- list(
    # 시나리오 1: 정규분포 (Cohen's d에 유리한 조건)
    normal = list(
      name = "정규분포 (이상적 조건)",
      gen_pos = function(n) pmin(1, pmax(0, rnorm(n, 0.7, 0.1))),
      gen_neg = function(n) pmin(1, pmax(0, rnorm(n, 0.3, 0.1)))
    ),
    # 시나리오 2: 편향 분포 (시그모이드 변환 유사)
    skewed = list(
      name = "편향 분포 (시그모이드 유사)",
      gen_pos = function(n) rbeta(n, 8, 2),    # 좌편향
      gen_neg = function(n) rbeta(n, 2, 5)     # 우편향
    ),
    # 시나리오 3: 이상치 포함
    outlier = list(
      name = "이상치 포함 (야외 녹음)",
      gen_pos = function(n) {
        x <- rnorm(n, 0.7, 0.1)
        x[sample(n, max(1, n %/% 5))] <- runif(max(1, n %/% 5), 0, 0.2)  # 20% 이상치
        pmin(1, pmax(0, x))
      },
      gen_neg = function(n) pmin(1, pmax(0, rnorm(n, 0.3, 0.1)))
    ),
    # 시나리오 4: 소표본
    small_n = list(
      name = "소표본 (n_pos=4, n_neg=8)",
      gen_pos = function(n) pmin(1, pmax(0, rnorm(min(n, 4), 0.7, 0.15))),
      gen_neg = function(n) pmin(1, pmax(0, rnorm(min(n, 8), 0.3, 0.12)))
    ),
    # 시나리오 5: 이봉분포 (harmonic_ratio 유사)
    bimodal = list(
      name = "이봉분포 (harmonic_ratio 유사)",
      gen_pos = function(n) {
        k <- sample(c(0, 1), n, replace = TRUE, prob = c(0.2, 0.8))
        ifelse(k == 1, rbeta(n, 15, 3), rbeta(n, 2, 10))
      },
      gen_neg = function(n) {
        k <- sample(c(0, 1), n, replace = TRUE, prob = c(0.7, 0.3))
        ifelse(k == 1, rbeta(n, 15, 3), rbeta(n, 2, 10))
      }
    ),
    # 시나리오 6: 포화 조건 (완전 분리)
    saturated = list(
      name = "완전 분리 (포화 조건)",
      gen_pos = function(n) runif(n, 0.75, 1.0),
      gen_neg = function(n) runif(n, 0.0, 0.25)
    )
  )

  cat("=" |> rep(70) |> paste(collapse = ""), "\n")
  cat("Cohen's d vs Rank-Biserial 비교 시뮬레이션\n")
  cat("=" |> rep(70) |> paste(collapse = ""), "\n\n")

  for (sc_name in names(scenarios)) {
    sc <- scenarios[[sc_name]]
    cat(sprintf("── %s ──\n", sc$name))

    cohens_ds <- numeric(n_sim)
    rank_bis <- numeric(n_sim)
    hybrid_dp <- numeric(n_sim)

    for (i in seq_len(n_sim)) {
      pos <- sc$gen_pos(20)
      neg <- sc$gen_neg(20)

      # Cohen's d (기존 방식)
      pm <- mean(pos); nm <- mean(neg)
      ps <- sd(pos); ns <- sd(neg)
      pooled_sd <- sqrt((ps^2 + ns^2) / 2)
      if (is.na(pooled_sd) || pooled_sd < 0.001) pooled_sd <- 0.001
      cohens_ds[i] <- max(0, (pm - nm) / pooled_sd)

      # Rank-Biserial
      rb <- compute_rank_biserial(pos, neg)
      rank_bis[i] <- max(0, rb$r_rb)

      # 하이브리드
      hybrid_dp[i] <- compute_hybrid_disc_power(pos, neg)
    }

    cat(sprintf("  Cohen's d:   평균=%.3f, SD=%.3f, CV=%.1f%%\n",
                mean(cohens_ds), sd(cohens_ds),
                100 * sd(cohens_ds) / max(mean(cohens_ds), 0.001)))
    cat(sprintf("  Rank-Bis:    평균=%.3f, SD=%.3f, CV=%.1f%%\n",
                mean(rank_bis), sd(rank_bis),
                100 * sd(rank_bis) / max(mean(rank_bis), 0.001)))
    cat(sprintf("  Hybrid:      평균=%.3f, SD=%.3f, CV=%.1f%%\n",
                mean(hybrid_dp), sd(hybrid_dp),
                100 * sd(hybrid_dp) / max(mean(hybrid_dp), 0.001)))
    cat("\n")
  }
}


# ============================================================
# 실행 예시 (주석 해제하여 사용)
# ============================================================
# compare_methods_simulation(n_sim = 200)
