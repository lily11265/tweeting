# 연구: 자동 튜닝 가중치 산출 — Cohen's d vs Rank-Biserial Correlation

## 1. 배경 및 문제 제기

현재 `auto_tune_weights()` 함수(`new_analysis.R:1802-1866`)는 **Cohen's d**를 사용하여
7개 음향 지표의 변별력(discriminative power)을 계산하고, 이를 가중치로 변환한다.

```
d = (μ_positive - μ_negative) / √((σ²_pos + σ²_neg) / 2)
```

**문제**: Cohen's d는 두 집단이 **정규분포**를 따르고 **등분산**일 때 최적으로 작동한다.
그러나 오디오 지표들의 분포는 이 가정을 위반하는 경우가 빈번하다.

---

## 2. 현재 7개 지표의 분포 특성 분석

### 2.1 분포 비정규성의 원인

| 지표 | 범위 | 변환 방식 | 예상 분포 특성 |
|------|------|-----------|---------------|
| `cor_score` | [0, 1] | Pearson correlation → 정규화 | 좌편향(양성 샘플 고값 집중) |
| `mfcc_score` | [0, 1] | Cosine similarity → 선형 스케일링 | 우측 집중, 좌측 꼬리 |
| `dtw_freq` | [0, 1] | `exp(-2 × normalized_dist)` | **강한 좌편향** (지수 변환) |
| `dtw_env` | [0, 1] | `exp(-k × dist)` + Sakoe-Chiba 제약 | **강한 좌편향** (지수 변환) |
| `band_energy` | [0, 1] | dB비율 → **시그모이드** | 양극단 집중 (천장/바닥 효과) |
| `harmonic_ratio` | [0, 1] | ACF 피크값 | **이봉분포** (주기적 vs 비주기적) |
| `snr` | [0, 1] | dB비율 → **시그모이드** (midpoint -3dB) | 양극단 집중 |

### 2.2 비정규성의 구체적 원인

1. **유계(bounded) 분포**: 모든 지표가 [0, 1] 범위로 제한됨 → 정규분포는 이론적으로 불가
2. **비선형 변환**: `exp()`, sigmoid 변환이 분포 형태를 왜곡
3. **이봉(bimodal) 구조**: 양성/음성 분류 자체가 이봉 분포를 유도
4. **바닥 효과(floor effect)**: DTW 계열 지표에서 비매칭 윈도우는 0 근처에 집중
5. **천장 효과(ceiling effect)**: 시그모이드 지표에서 고SNR 구간은 1.0 근처 포화

### 2.3 관련 연구 근거

- IEEE Xplore 논문 ([Seddik et al., 2004](https://ieeexplore.ieee.org/document/1397204/))에서 MFCC의 통계적 분포 속성(왜도, 첨도)을 분석한 결과, MFCC 계수는 비정규 분포를 보임
- warbleR 패키지의 `mfcc_stats()`는 각 MFCC에 대해 왜도(skewness)와 첨도(kurtosis)를 별도 계산하는데, 이는 MFCC가 정규분포를 따르지 않음을 전제한 설계
- Spectral Kurtosis는 "비정상성 또는 비가우시안 거동을 주파수 영역에서 지적하는 통계 도구"로 정의됨 ([MATLAB docs](https://www.mathworks.com/help/signal/ref/spectralkurtosis.html))
- DTW 거리값은 **양의 편향**(zero-bounded, right-tailed)을 보이는 것이 일반적

---

## 3. Cohen's d의 한계 (현재 구현의 취약점)

### 3.1 이론적 한계

1. **평균과 표준편차에 의존**: 비정규 분포에서 평균은 대표값이 아닐 수 있음
2. **이상치 취약성**: 단일 이상치가 전체 d값을 크게 왜곡 가능
   > "A single outlier could completely distort your Cohen's d value" — [Akinshin, 2023](https://aakinshin.net/posts/cohend-and-outliers/)
3. **소표본 편향**: n < 30에서 Cohen's d는 과대추정 경향 (Hedges' g 보정 필요)
4. **등분산 가정 위반**: pooled SD 계산이 부적절해짐

### 3.2 현재 코드에서의 구체적 문제

```r
# new_analysis.R:1852-1854
ps_d <- sqrt((ps^2 + ns^2) / 2)          # pooled SD
if (is.na(ps_d) || ps_d < 0.001) ps_d <- 0.001  # floor 보정
loo_ds[li] <- max(0, (pm - nm) / ps_d)
```

**문제 1**: `ps_d < 0.001` floor 보정은 분산이 극도로 작을 때 d값을 인위적으로 폭등시킴
- 예: 양성 샘플이 모두 0.95±0.0005이고 음성이 0.90±0.0005이면 → d = (0.05/0.001) = 50
- 실질적 차이가 작음에도 불구하고 과대평가됨

**문제 2**: LOO 교차검증 시 n=5에서 한 샘플 제거 → n=4, 이때 SD 추정이 극히 불안정

**문제 3**: `max(0, ...)` 처리로 음수 d를 모두 0으로 절단 → 정보 손실

**문제 4**: 시그모이드 변환된 지표(band_energy, snr)에서 양극단에 값이 집중되면
pooled SD가 인위적으로 작아져 d가 폭등

---

## 4. Rank-Biserial Correlation 분석

### 4.1 정의 및 수학적 배경

Rank-Biserial Correlation은 Mann-Whitney U 검정의 효과 크기 지표로,
**두 집단 간 순위 우위(dominance)의 정도**를 측정한다.

**Wendt 공식** (U 통계량 기반):
```
r_rb = 1 - (2U) / (n₁ × n₂)
```

**Kerby 단순 차이 공식**:
```
r_rb = f - u
여기서 f = 유리한 쌍의 비율, u = 불리한 쌍의 비율
```

**해석**: 양성 집단에서 무작위로 1개, 음성 집단에서 1개를 뽑았을 때,
양성 > 음성일 확률과 양성 < 음성일 확률의 차이.

- r_rb = +1.0: 모든 양성 값이 모든 음성 값보다 큼 (완전 분리)
- r_rb = 0.0: 두 집단 완전 겹침
- r_rb = -1.0: 모든 음성 값이 양성보다 큼 (역방향)

### 4.2 관련 동치 지표

| 지표 | 공식 | 범위 | 관계 |
|------|------|------|------|
| Rank-Biserial (r_rb) | `1 - 2U/(n₁n₂)` | [-1, 1] | 기본 |
| Cliff's Delta (δ) | 동일 | [-1, 1] | r_rb ≡ δ |
| Vargha-Delaney A | `(r_rb + 1) / 2` | [0, 1] | 우위 확률 |
| Common Language ES | `U / (n₁n₂)` | [0, 1] | A와 동일 |

### 4.3 해석 기준 (Vargha & Delaney, 2000)

| 효과 크기 | |r_rb| | VD-A |
|-----------|--------|------|
| 무시 가능 | < 0.11 | < 0.56 |
| 소(Small) | ≥ 0.11 | ≥ 0.56 |
| 중(Medium) | ≥ 0.28 | ≥ 0.64 |
| 대(Large) | ≥ 0.43 | ≥ 0.71 |

---

## 5. Cohen's d vs Rank-Biserial: 체계적 비교

### 5.1 가정 비교

| 속성 | Cohen's d | Rank-Biserial |
|------|-----------|---------------|
| 분포 가정 | 정규분포 필요 | **분포 무관** |
| 이상치 민감도 | **높음** (평균/SD 기반) | **낮음** (순위 기반) |
| 등분산 가정 | pooled SD에 필요 | 불필요 |
| 소표본 성능 | 편향 큼 (n<30) | 상대적 안정 |
| 유계 데이터 [0,1] | 부적합 | **적합** |
| 해석 용이성 | 표준편차 단위 | 우위 확률 |

### 5.2 오디오 지표 맥락에서의 장단점

#### Rank-Biserial의 장점

1. **분포 무관(distribution-free)**
   - 시그모이드, 지수 변환된 지표에 적합
   - 이봉분포, 편향 분포에 강건
2. **이상치 강건성**
   - 잡음 오염 윈도우(이상치)가 있어도 순위만 바뀔 뿐 결과 왜곡이 적음
3. **유계 데이터 적합**
   - [0, 1] 범위의 점수에 자연스럽게 적용
4. **소표본 안정성**
   - 현재 코드의 최소 조건(양성 3, 음성 3)에서도 작동
   - Cohen's d보다 분산이 작음
5. **직관적 해석**
   - "양성 샘플이 음성보다 높은 비율" → 변별력의 직관적 의미

#### Rank-Biserial의 한계 (★ 핵심)

1. **포화(saturation) 문제** ⚠️
   - 두 집단이 완전 분리되면 r_rb = 1.0으로 포화
   - **여러 지표가 동시에 r_rb = 1.0이면 차이를 구분할 수 없음**
   - 예: cor_score와 mfcc_score 모두 완전 분리 → 둘 다 가중치 동일
   - Cohen's d는 분리 정도에 따라 d=2.0, d=5.0 등 차이 표현 가능

2. **크기 정보 손실**
   - 순위만 사용하므로 값의 절대적 차이를 반영하지 못함
   - 양성=0.51, 음성=0.49 와 양성=0.99, 음성=0.01의 순위 관계가 동일

3. **동순위(ties) 처리**
   - 오디오 지표가 0이나 1로 클리핑되면 동순위 발생
   - 보정 필요 (mid-rank 방법 등)

---

## 6. 포화 문제에 대한 해결책 분석

### 6.1 방법 A: 순수 Rank-Biserial + 포화 보정

```
1. 기본: r_rb = 1 - 2U/(n₁n₂)
2. 포화 시(r_rb ≈ 1.0): 집단 간 중앙값 차이로 보정
   adjusted = r_rb × (1 + α × |median_pos - median_neg|)
```

**장점**: 분포 강건성 유지
**단점**: 보정 파라미터 α 결정 필요, 복잡도 증가

### 6.2 방법 B: 하이브리드 접근 (★ 추천)

```
1차: Rank-Biserial로 변별력 유무 판단
2차: 변별력이 있는 지표 중에서 robust Cohen's d로 세분화

robust_d = (median_pos - median_neg) / MAD_pooled
```

여기서 MAD = Median Absolute Deviation (= median(|x - median(x)|) × 1.4826)

**장점**:
- 1차 스크리닝에서 비정규성 대응
- 2차에서 MAD 기반으로 이상치 강건하게 크기 차이 반영
- MAD는 정규분포에서 SD와 일치 (보정계수 1.4826)

### 6.3 방법 C: Vargha-Delaney A 기반

```
A = U / (n₁ × n₂)    # = (r_rb + 1) / 2

가중치 = f(A - 0.5)   # A=0.5 → 변별력 없음, A=1.0 → 완전 변별
```

**장점**: [0, 1] 범위로 가중치 변환이 직관적
**단점**: 포화 문제 동일 (A=1.0에서 포화)

### 6.4 방법 D: γₚ (Akinshin의 비모수 Cohen's d)

```
γₚ = (powered_median_pos - powered_median_neg) / MAD_pooled
```

Harrell-Davis powered median과 MAD를 사용하는 비모수적 Cohen's d 대안.

> "For normal distributions, γₚ works similar to Cohen's d. In the case of non-normal distribution, it provides a robust and stable alternative." — [Akinshin](https://aakinshin.net/posts/nonparametric-effect-size/)

**장점**: Cohen's d와 동일한 척도, 비모수적 강건성
**단점**: R 구현이 복잡, Harrell-Davis estimator 의존

---

## 7. 실제 시나리오별 영향 분석

### 시나리오 1: 깨끗한 녹음 (고 SNR)

```
양성 샘플: cor=0.85, mfcc=0.90, dtw=0.80, snr=0.95
음성 샘플: cor=0.30, mfcc=0.35, dtw=0.20, snr=0.40
→ 완전 분리 가능 → Rank-Biserial ≈ 1.0 (모든 지표)
→ Cohen's d는 지표별로 3.0~8.0 범위에서 차이
```

이 경우 Rank-Biserial은 모든 지표에 균등 가중치를 부여하지만,
Cohen's d는 분리 정도에 따른 차등 가중치 가능.

**평가**: Cohen's d가 유리한 유일한 시나리오. 하지만 깨끗한 녹음에서는
어떤 가중치든 잘 작동하므로 실질적 영향은 적음.

### 시나리오 2: 잡음이 많은 녹음 (저 SNR)

```
양성 샘플: cor=0.55±0.25, mfcc=0.60±0.20, dtw=0.45±0.30, snr=0.55±0.15
음성 샘플: cor=0.40±0.20, mfcc=0.35±0.15, dtw=0.30±0.25, snr=0.45±0.20
→ 부분적 겹침 → 이상치 다수
```

이 경우 Cohen's d는 이상치에 의해 SD가 팽창 → d값 불안정.
Rank-Biserial은 순위 기반이므로 안정적.

**평가**: Rank-Biserial이 명확히 우수.

### 시나리오 3: 소수 템플릿 (n_pos = 3~5)

```
양성 샘플 3개, 음성 샘플 15개
→ 양성 SD 추정 극히 불안정
→ Cohen's d의 LOO에서 n=2로 SD 계산
```

**평가**: Rank-Biserial이 명확히 우수. U 통계량은 n₁=3에서도 유효.

### 시나리오 4: 시그모이드 포화 지표

```
band_energy: 양성=[0.92, 0.95, 0.88, 0.97, 0.91], 음성=[0.08, 0.12, 0.15, 0.05, 0.10]
→ 시그모이드 천장/바닥 효과로 분산이 인위적으로 작음
→ Cohen's d 폭등 (d > 10)
```

**평가**: Rank-Biserial이 우수. r_rb = 1.0 (포화)이지만 최소한 왜곡은 없음.

---

## 8. 최종 권고: 하이브리드 방식 (방법 B 변형)

### 8.1 권고 알고리즘

```
Step 1: Mann-Whitney U 검정 수행
        U = wilcox.test(pos_vals, neg_vals)$statistic
        r_rb = 1 - 2*U / (n_pos * n_neg)

Step 2: 방향성 확인
        if (r_rb <= 0) → disc_power = 0  (역방향 또는 무변별)

Step 3: 포화 보정 적용
        if (r_rb > 0.95 && n_pos >= 5 && n_neg >= 5) {
          # 포화 구간에서는 robust 크기 정보로 보정
          mad_pos = mad(pos_vals, constant = 1.4826)
          mad_neg = mad(neg_vals, constant = 1.4826)
          mad_pooled = sqrt((mad_pos^2 + mad_neg^2) / 2)
          if (mad_pooled > 0.001) {
            magnitude = abs(median(pos_vals) - median(neg_vals)) / mad_pooled
          } else {
            magnitude = abs(median(pos_vals) - median(neg_vals)) * 10
          }
          # r_rb에 크기 정보를 곱하여 포화 해소
          disc_power = r_rb * (1 + log1p(magnitude))
        } else {
          disc_power = max(0, r_rb)
        }

Step 4: 가중치 변환 (기존과 동일)
        raw_weights = sqrt(disc_power) / sum(sqrt(disc_power))
```

### 8.2 이 방식의 이점

1. **기본은 Rank-Biserial**: 분포 가정 불필요, 이상치 강건
2. **포화 시에만 MAD 기반 보정**: 완전 분리 시에도 지표 간 차등 가능
3. **MAD 사용**: SD 대신 MAD → 이상치에 강건한 산포 추정
4. **log1p 스케일링**: 극단적 magnitude 차이를 완화
5. **기존 가중치 변환 파이프라인 호환**: sqrt + 정규화 유지

### 8.3 추가 고려사항

- **LOO 교차검증**: Rank-Biserial에서도 적용 가능하나, 소표본에서 U 통계량의
  LOO가 불안정할 수 있음. 대안: Bootstrap 신뢰구간 (R `effectsize` 패키지 지원)
- **동순위 보정**: `wilcox.test()`의 `exact=FALSE` 옵션이 동순위를 자동 처리
- **효과 크기 해석 기준 변경**: Cohen's d 기준(0.2/0.5/0.8)에서
  r_rb 기준(0.11/0.28/0.43)으로 변경 필요

---

## 9. 결론

### 결론: Rank-Biserial 전환을 **권고**한다.

**핵심 근거**:

1. **모든 7개 지표가 [0, 1] 유계 + 비선형 변환** → 정규분포 가정 위반이 구조적
2. **이상치 빈도가 높은 야외 녹음** 환경 → 순위 기반 방법이 강건
3. **소표본 상황** (최소 3+3) → Cohen's d의 편향이 큼
4. **시그모이드/지수 변환 지표** → Cohen's d가 인위적으로 폭등 가능

**포화 문제**는 하이브리드 보정(MAD 기반)으로 해결 가능하며,
실제 야외 녹음 데이터에서 모든 지표가 동시에 완전 분리되는 경우는 드물다.

**구현 복잡도**: 최소 — `wilcox.test()` 내장 함수로 U 통계량 직접 얻을 수 있음.

---

## 10. 참고 문헌 및 출처

- [MetricGate: Rank-Biserial Correlation](https://metricgate.com/docs/rank-biserial-correlation/)
- [Guide to Non-Parametric Effect Sizes (Matthew B. Jané)](https://matthewbjane.quarto.pub/Non-Parametric-Effect-Sizes.html)
- [PMC: Effect sizes for nonparametric tests](https://pmc.ncbi.nlm.nih.gov/articles/PMC12701665/)
- [Akinshin: A single outlier could distort Cohen's d](https://aakinshin.net/posts/cohend-and-outliers/)
- [Akinshin: Nonparametric Cohen's d-consistent effect size](https://aakinshin.net/posts/nonparametric-effect-size/)
- [Akinshin: Customization of nonparametric effect size](https://aakinshin.net/posts/nonparametric-effect-size2/)
- [Wilcox (2018): Robust Nonparametric Effect Size analog of Cohen's d](https://digitalcommons.wayne.edu/cgi/viewcontent.cgi?article=2726&context=jmasm)
- [Algina et al. (2005): Robust alternative to Cohen's d](https://pubmed.ncbi.nlm.nih.gov/16221031/)
- [Seddik et al. (2004): MFCC statistical distribution properties](https://ieeexplore.ieee.org/document/1397204/)
- [MATLAB: Spectral Kurtosis](https://www.mathworks.com/help/signal/ref/spectralkurtosis.html)
- [warbleR: mfcc_stats](https://marce10.github.io/warbleR/reference/mfcc_stats.html)
- [effectsize R package: rank_biserial()](https://easystats.github.io/effectsize/reference/rank_biserial.html)
- [rcompanion R package: wilcoxonRG()](https://search.r-project.org/CRAN/refmans/rcompanion/html/wilcoxonRG.html)
- [Kerby (2014): The Simple Difference Formula](https://journals.sagepub.com/doi/full/10.2466/11.IT.3.1)
- [Garstats: Robust effect sizes for 2 independent groups](https://garstats.wordpress.com/2016/05/02/robust-effect-sizes-for-2-independent-groups/)
- [Vargha & Delaney (2000): Critique and improvement of the CL common language effect size](https://doi.org/10.3102/10769986025002101)
