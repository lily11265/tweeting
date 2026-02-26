# 연구: Tweeting 프로그램의 박쥐/양서파충류 울음소리 적용 가능성 분석

## Context

사용자가 현재 조류 음성 탐지 프로그램(Tweeting)이 **박쥐**, **양서류(개구리/두꺼비)**, **파충류(도마뱀붙이/악어류)**의
울음소리도 구별할 수 있는지 연구를 요청함. 이 문서는 코드 분석 + 각 분류군의 음향 특성 비교를 통해
적용 가능성과 필요한 수정 사항을 정리한 연구 결과임.

---

## 1. 현재 프로그램의 핵심 제약 요소 분석

### 1.1 샘플레이트 제한 (★ 가장 큰 병목)

| 위치 | 제약 | 현재 값 |
|------|------|---------|
| `new_analysis.R:30` | `MAX_SAMPLE_RATE <- 48000` | 48 kHz (Nyquist = 24 kHz) |
| `new_analysis.R:28` | 주석: "조류 울음은 ~12kHz 이하이므로 48kHz면 충분" | 조류 전용 설계 |
| `audio/sanitizer.py:95` | `target_sr=48000` | 48 kHz로 다운샘플링 |
| `new_analysis.R:2099-2101, 2233-2235, 2450-2457, 2559-2561` | 48kHz 초과 시 강제 다운샘플링 | 4곳에서 동일 적용 |
| `ui/spectro_settings_dialog.py:152` | f_low 범위: 0–24,000 Hz | GUI 제한 |
| `ui/spectro_settings_dialog.py:156` | f_high 범위: 100–48,000 Hz | GUI 제한 |

**영향**: 48 kHz 샘플레이트 → Nyquist 주파수 24 kHz → **24 kHz 이상의 초음파 대역을 캡처할 수 없음**

### 1.2 조류 특화 하드코딩

| 위치 | 코드 | 영향 |
|------|------|------|
| `new_analysis.R:527` | `BIRD_MIN_PEAK_KHZ <- 0.5` | 자동 주파수 보정 시 0.5 kHz 미만을 소음으로 간주 |
| `new_analysis.R:525-526` | "조류 울음 최소 기준: 피크 주파수 1kHz 이상" | 1 kHz 미만 피크를 경고 |
| `new_analysis.R:1029` | "새소리는 성도(syrinx)의 주기적 진동" | harmonic_ratio 설계 근거가 조류 |

### 1.3 스펙트로그램/MFCC 파라미터

| 파라미터 | 현재 값 | 비고 |
|----------|---------|------|
| `wl` (윈도우 길이) | 512 samples | 48kHz에서 93.75 Hz/bin 해상도 |
| `ovlp` (오버랩) | 50–75% | |
| `wintime` (MFCC) | 25 ms | |
| `hoptime` (MFCC) | 10 ms | |
| `numcep` (MFCC 계수) | 13 | |
| `DTW_ALPHA` | 2.0 | exp(-2×dist) 매핑 |

---

## 2. 분류군별 음향 특성 비교

### 2.1 조류 (현재 대상) — 기준선

| 특성 | 값 |
|------|-----|
| 주파수 범위 | 0.5–12 kHz (대부분 1–8 kHz) |
| 필요 샘플레이트 | 44.1–48 kHz |
| 신호 특성 | 조화음(harmonic) 풍부, 주파수 변조(FM sweep), 반복 패턴 |
| 지속 시간 | 0.1–5초 (종에 따라 다양) |
| HNR (Harmonic-to-Noise) | 높음 (성도/syrinx의 주기적 진동) |

### 2.2 박쥐 — 반향정위(Echolocation) 호출

| 특성 | 값 |
|------|-----|
| **반향정위 주파수** | **20–200 kHz** (대부분 25–100 kHz) |
| **사회적 호출 주파수** | **10–25 kHz** (일부 가청 범위) |
| 필요 샘플레이트 | **192–500 kHz** (반향정위), **96 kHz** (사회적 호출) |
| 호출 유형 | CF(정주파수), FM sweep(주파수 변조), QCF(준정주파수) |
| 지속 시간 | 1–50 ms (매우 짧음) |
| HNR | 낮음-중간 (초음파 FM sweep은 비조화적) |

#### 적용 가능성 평가

| 항목 | 반향정위 (20-200 kHz) | 사회적 호출 (10-25 kHz) |
|------|---|---|
| 샘플레이트 호환 | ❌ **불가** (48kHz 제한) | ⚠️ 부분적 (24kHz Nyquist로 일부 캡처) |
| corMatch 템플릿 매칭 | ❌ 1-50ms 호출은 너무 짧음 | ⚠️ 가능하나 최적화 필요 |
| MFCC | ❌ 25ms 윈도우 > 호출 길이 | ⚠️ 조정 필요 (wintime 축소) |
| DTW | ❌ 프레임 부족 | ⚠️ 제한적 |
| Harmonic Ratio | ❌ CF형만 일부 적합 | ⚠️ 사회적 호출은 복잡한 변조 |
| Band Energy / SNR | ⚠️ 주파수 대역 변경 필요 | ✅ 원리적으로 동일 |

**결론**: 박쥐 반향정위 → **근본적으로 불가** (샘플레이트 4-10배 부족).
박쥐 사회적 호출 → **제한적으로 가능**하나 주요 수정 필요.

### 2.3 양서류 (개구리/두꺼비)

| 특성 | 값 |
|------|-----|
| **주파수 범위** | **0.3–8 kHz** (대부분 0.5–5 kHz) |
| 필요 샘플레이트 | **44.1 kHz** (충분) |
| 호출 유형 | 펄스 반복(pulse train), 음조(tonal), 하강 스윕(down-sweep) |
| 지속 시간 | 0.05–2초 (개별 음절), 코러스는 수분 |
| HNR | 중간-높음 (음조성 호출은 조화음 풍부) |
| 특이점 | **반복률(repetition rate)**이 종 구별의 핵심 |

#### 적용 가능성 평가

| 항목 | 평가 |
|------|------|
| 샘플레이트 호환 | ✅ **완전 호환** (0.3-8 kHz는 48 kHz SR에 충분) |
| corMatch 템플릿 매칭 | ✅ **잘 작동** (반복적 호출 패턴에 적합) |
| MFCC | ✅ **검증됨** (MFCC 기반 개구리 분류 논문 다수) |
| DTW | ✅ **검증됨** (DTW로 개구리 음절 94.3% 정확도 보고) |
| Harmonic Ratio | ✅ 음조성 호출에 잘 작동 (개구리 호출은 조화음 있음) |
| Band Energy / SNR | ✅ 동일 원리 적용 |
| BIRD_MIN_PEAK_KHZ | ⚠️ 일부 저음 종은 0.3-0.5 kHz → 하한 조정 필요 |

**결론**: 양서류 → **가장 호환성 높음**. 최소한의 수정으로 작동 가능.

### 2.4 파충류 (도마뱀붙이/악어류)

| 특성 | 도마뱀붙이(게코) | 악어류 |
|------|---|---|
| **주파수 범위** | **2–13 kHz** | **0.05–2 kHz** (초저주파 포함) |
| 필요 샘플레이트 | 44.1 kHz (충분) | 44.1 kHz (충분) |
| 호출 유형 | 짖기, 찍찍, 딸깍 | 으르렁, 포효, 초저주파 |
| 지속 시간 | 0.05–0.5초 | 0.5–5초 |
| HNR | 중간 (음조성 호출은 높음) | 낮음 (광대역 소음성) |

#### 적용 가능성 평가

| 항목 | 도마뱀붙이 | 악어류 |
|------|---|---|
| 샘플레이트 호환 | ✅ 호환 | ✅ 호환 |
| corMatch | ✅ 반복 호출에 적합 | ⚠️ 긴 호출, 변이 큼 |
| MFCC | ✅ 적합 | ⚠️ 저주파에서 Mel 스케일 해상도 낮음 |
| DTW | ✅ 적합 | ⚠️ 적합 (지속시간 조정 필요) |
| Harmonic Ratio | ✅ 게코 음조 호출에 적합 | ❌ 비조화적 소음 → 낮은 점수 |
| Band Energy / SNR | ✅ 동일 원리 | ⚠️ 초저주파는 환경 소음과 겹침 |
| BIRD_MIN_PEAK_KHZ | ✅ 2-13 kHz로 문제 없음 | ❌ 0.05 kHz → 하한 대폭 축소 필요 |

**결론**: 도마뱀붙이 → **높은 호환성**. 악어류 → **제한적** (초저주파 + 비조화적).

---

## 3. 종합 호환성 매트릭스

| 분류군 | 샘플레이트 | 주파수 범위 | 템플릿 매칭 | MFCC | DTW | Harmonic | 종합 |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **조류** (현재) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 설계 대상 |
| **양서류 (개구리)** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ **거의 즉시 사용 가능** |
| **도마뱀붙이** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ **높은 호환성** |
| 박쥐 사회적 호출 | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ 주요 수정 필요 |
| **악어류** | ✅ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ❌ | ⚠️ 수정 필요 |
| 박쥐 반향정위 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ **근본적 불가** |

---

## 4. 분류군별 필요 수정 사항

### 4.1 양서류 적용 시 (최소 수정)

1. **`BIRD_MIN_PEAK_KHZ` 조정** (`new_analysis.R:527`)
   - 현재: 0.5 kHz → **0.2 kHz**로 축소 (저음 개구리 종 대응)
   - 또는 분류군 파라미터로 외부화

2. **주석/로그 메시지 일반화** (cosmetic)
   - "새소리", "조류 울음" → "대상 종 호출" 등으로 변경
   - 기능적 영향은 없음

3. **추가 고려**: 개구리 코러스 환경에서 개별 호출 분리
   - 현재 NMS (Non-Maximum Suppression) 로직이 이미 근접 검출 병합 처리
   - `min_gap` 파라미터 조정으로 대응 가능

### 4.2 도마뱀붙이 적용 시 (최소 수정)

- 양서류와 동일한 수정 사항
- 짧은 호출(50ms)에 대한 `wintime` 조정 고려
  - 현재 25ms 윈도우는 50ms 호출에서 2프레임만 생성 → MFCC 정확도 저하 가능
  - `wintime=0.015, hoptime=0.005`로 축소하면 개선

### 4.3 악어류 적용 시 (중간 수정)

1. **`BIRD_MIN_PEAK_KHZ` 대폭 축소**: 0.5 → **0.02 kHz** (20 Hz)
2. **MFCC Mel 스케일**: 0-2 kHz 범위에서 Mel 필터뱅크 해상도 부족
   - 대안: 선형 주파수 필터뱅크 또는 LFCC (Linear Frequency Cepstral Coefficients)
3. **Harmonic Ratio**: 악어류 포효는 비조화적 → 이 지표의 가중치를 0에 가깝게 설정
   - 자동 튜닝(Rank-Biserial)이 이를 자동으로 처리할 가능성 있음
4. **SNR/Band Energy**: 초저주파 대역은 환경 소음(바람, 진동)과 겹침 → 신뢰도 저하

### 4.4 박쥐 반향정위 적용 시 (근본적 수정 필요 — 사실상 재설계)

1. **MAX_SAMPLE_RATE**: 48000 → **384000** (384 kHz)
   - R `seewave` 패키지가 이 샘플레이트를 처리할 수 있는지 검증 필요
   - 메모리/성능 문제: 384kHz × 60초 = 23M 샘플/분
2. **sanitizer.py**: `target_sr=48000` → 다운샘플링 비활성화
3. **GUI 주파수 범위**: f_high 상한 48000 → 192000 Hz
4. **MFCC 파라미터**: `wintime=0.002` (2ms), `hoptime=0.001` (1ms) — 1-5ms 호출에 대응
5. **wl (FFT 윈도우)**: 512 → 64-128 (시간 해상도 우선)
6. **corMatch**: 1-50ms 템플릿은 시간 프레임이 극히 적어 상관 계산 불안정
7. **추가 지표 필요**:
   - 호출 시작/종료 주파수, sweep rate, 대역폭
   - Zero-crossing analysis (Anabat 방식)

---

## 5. 기존 소프트웨어와의 비교

| 소프트웨어 | 대상 | 접근법 | 비교 |
|------------|------|--------|------|
| **BirdNET** | 조류 | CNN + 대규모 학습 데이터 | Tweeting과 다른 접근(딥러닝) |
| **BatDetect2** | 박쥐 | CNN + 스펙트로그램 | 256 kHz SR, 초음파 전용 |
| **Kaleidoscope** | 박쥐/조류 | 클러스터링 + 분류기 | 다중 SR 지원 |
| **SonoBat** | 박쥐 | 호출 파라미터 추출 | Zero-crossing + 스펙트로그램 |
| **RIBBIT** | 개구리 | 반복 간격 기반 | 펄스 반복률로 종 식별 |
| **ARBIMON** | 다목적 | 템플릿 매칭 + ML | **Tweeting과 가장 유사한 접근** |

**핵심 인사이트**: ARBIMON(Automated Remote Biodiversity Monitoring Network)은
템플릿 매칭 + 기계학습으로 개구리/새/곤충을 동시에 감시하는 플랫폼.
Tweeting의 corMatch + 복합점수 접근과 원리적으로 유사하며,
**양서류까지 확장이 실질적으로 가능**함을 시사.

**BattyBirdNET 접근법**: BirdNET 아키텍처를 박쥐에 적용한 사례.
256 kHz로 리샘플링 후 0.5625초 세그먼트를 BirdNET에 "3배 빠른 새 노래"로 입력.
BirdNET 임베딩에서 별도 분류기로 박쥐 종 식별. Tweeting에도 유사한 전략 가능.

---

## 6. 학술적 검증 데이터

### 6.1 MFCC 기반 분류 정확도 (분류군별)

| 분류군 | 방법 | 정확도 | 출처 |
|--------|------|--------|------|
| 개구리 (10종) | MFCC + k-NN | **98.1%** | Frog Sound ID System |
| 개구리 (10종) | 20 cepstral + GMM (64 Gaussians) | **99.1%** (가중 에러 0.9%) | 12초 학습 데이터만 사용 |
| 개구리 (18종) | MSAS 템플릿 매칭 vs DTW | **94.3%** (MSAS) | ScienceDirect |
| 개구리 (32종) | MFCC + DNN/LSTM | 높은 정확도 보고 | MDPI Symmetry |
| 박쥐 (21종) | DTW 커널 + GP | **91.7%** (과), **66%** (종) | bioRxiv |
| 박쥐 (17종, UK) | BatDetect2 CNN | **0.88 mAP** | PLOS CompBio |
| 박쥐 | DWT vs MFCC | **93.3%** (DWT 우세) | Academia.edu |

### 6.2 박쥐 분석 시 스펙트로그램 파라미터 (참고)

| 소프트웨어 | FFT 윈도우 | 오버랩 | 주파수 범위 | SR |
|------------|-----------|--------|------------|-----|
| BatDetect2 | **2.3 ms** | 75% | 5–135 kHz | 256 kHz |
| SonoBat | 가변 | - | Full spectrum | 384 kHz |
| Tweeting (현재) | **10.7 ms** (wl=512@48kHz) | 50-75% | 0–24 kHz | 48 kHz |

### 6.3 박쥐 발성 체계의 이중 구조

박쥐는 두 가지 독립적 발성 메커니즘을 사용 (PLOS Biology, 2022):
- **성대막(vocal membrane)**: 10–95 kHz → 반향정위 + 고주파 사회적 호출
- **가성대(ventricular fold)**: 1–5 kHz → 저주파 사회적/투쟁 호출 (데스메탈 그로울과 동일 원리)
- 총 성역(vocal range): **약 7옥타브** (인간 포함 포유류 중 최대)

1-5 kHz 가성대 호출은 조류/개구리와 완전히 겹치므로 **현재 Tweeting으로 탐지 가능**.

---

## 7. 최종 결론

### 즉시 가능 (코드 수정 최소)
- **개구리/두꺼비**: 주파수 범위(0.3-8 kHz)가 조류와 겹치고, MFCC/DTW 기반 분류가
  학술적으로 검증됨. `BIRD_MIN_PEAK_KHZ` 하한 조정만으로 작동 가능.
- **도마뱀붙이(게코)**: 2-13 kHz 범위, 반복적 호출 패턴 → 템플릿 매칭에 적합.

### 수정 후 가능 (중간 노력)
- **악어류**: 초저주파(50 Hz-2 kHz) 대역, Mel 스케일 한계, 비조화 신호.
  LFCC 대안 + harmonic_ratio 자동 비활성화로 대응 가능.
- **박쥐 사회적 호출**: 10-25 kHz 범위의 일부만 48 kHz SR로 캡처 가능.
  MFCC 윈도우 축소 + 전용 파라미터셋 필요.

### 근본적으로 불가 (재설계 필요)
- **박쥐 반향정위**: 20-200 kHz 초음파 → 384 kHz+ 샘플레이트 필요.
  MFCC 윈도우 2ms, FFT 윈도우 64샘플, Zero-crossing 분석 등
  사실상 별도 프로그램 수준의 수정이 필요.

---

## 7. 참고 문헌

- [Wild Mountain Echoes: Recording Ultrasounds](https://www.wildmountainechoes.com/equipment/options-for-recording-ultrasounds/)
- [Evocative Sound: Recording Bat Echolocation](https://www.evocativesound.com/2021/11/17/how-to-record-ultrasonic-bat-echolocation/)
- [Bat Detector - Wikipedia](https://en.wikipedia.org/wiki/Bat_detector)
- [PMC: Automated Detection of Frog Calls (RIBBIT)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8518090/)
- [MDPI: AI Classification of Frogs Using MFCC](https://www.mdpi.com/2073-8994/11/12/1454)
- [ScienceDirect: Frog Call Recognition Using MSAS vs DTW](https://www.sciencedirect.com/science/article/pii/S0898122112002763)
- [PMC: Vocal Plasticity in Geckos](https://pmc.ncbi.nlm.nih.gov/articles/PMC5454267/)
- [PLOS ONE: Gecko Vocalization Frequency (2.47-4.17 kHz)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0146677)
- [Wiley: Vocalization by Extant Nonavian Reptiles](https://anatomypubs.onlinelibrary.wiley.com/doi/10.1002/ar.24553)
- [Springer: Acoustic and Visual Features for Frog Call Classification](https://link.springer.com/article/10.1007/s11265-019-1445-4)
- [ScienceDirect: Acoustic Classification of Frog Calls Using MFCC/LFCC](https://www.sciencedirect.com/science/article/abs/pii/S0003682X17304024)
- [U.S. FWS: Glossary of Acoustic Bat Survey Terms](https://www.fws.gov/node/268772)
- [PLOS Biology: Bats Expand Vocal Range via Dual Laryngeal Structures](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3001881)
- [BatDetect2 - GitHub](https://github.com/macaodha/batdetect2)
- [BattyBirdNET-Analyzer - GitHub](https://github.com/rdz-oss/BattyBirdNET-Analyzer)
- [Bat Call ID with Gaussian Process + DTW Kernel](https://www.researchgate.net/publication/262765967)
- [Frog Sound ID System with MFCC + k-NN](https://link.springer.com/chapter/10.1007/978-3-642-36642-0_5)
- [ARBIMON: Real-time Bioacoustics Monitoring](https://peerj.com/articles/103/)
- [Pitch/Spectral DTW for Harmonic Avian Vocalizations](https://pmc.ncbi.nlm.nih.gov/articles/PMC3745477/)
- [Neural Network for Bat Echolocation Sound Classification](https://www.mdpi.com/2076-2615/13/16/2560)
