# PCA Hash Deduplication

DINOv2 논문([Oquab et al., 2023](https://arxiv.org/abs/2304.07193))에서 제안한 **PCA Hash Deduplication** 방법으로 이미지 데이터셋의 중복을 탐지하고 제거합니다.

## 파이프라인

### 공통 — 특징 추출

```
data_dir
   │
   ▼
[이미지 경로 수집]  ← 재귀 탐색, 지원 확장자 필터
   │
   ▼
[캐시 확인]  ← ~/.cache/pca_dedup/<md5_key>.npz
   │  HIT ──────────────────────────────────────┐
   │  MISS                                      │
   ▼                                            │
[이미지 로드 & 리사이즈]  ← image_size × image_size, BILINEAR
   │
   ▼
[픽셀 벡터화 & 정규화]  ← float32, /255.0 → [0, 1]
   │
   ▼
[캐시 저장]  → .npz (features + valid_paths)
   │                                            │
   └────────────────────────────────────────────┘
   ▼
features  (N, image_size² × 3)
```

### MD5 정확 중복 탐지

```
paths
   │
   ▼
[파일 MD5 해시]  ← 바이트 단위 비교
   │
   ▼
[동일 해시 그룹화]
   │
   ▼
exact_groups  {md5: [path, ...]}
```

### PCA Hash 유사 중복 탐지

```
features  (N, D)
   │
   ▼
[PCA 학습 & 투영]  ← n_components 차원, random_state=42
   │
   ▼
[부호 이진화]  ← projection >= 0  →  bool hash (N, n_components)
   │
   ▼
[Hamming distance 행렬]  ← XOR 합산, chunk=1000 단위 처리
   │
   ▼
[임계값 필터]  ← distance <= hamming_threshold
   │
   ▼
[Union-Find 그룹화]  ← 전이적 중복 묶음
   │
   ▼
near_groups  [[idx, ...], ...]
```

### 모드별 후처리

| 모드 | 후처리 |
|------|--------|
| `analyze` | 보고서 출력 (파일 변경 없음) |
| `deduplicate` | 그룹당 대표 1장 유지, 나머지 복사(`--output_dir`) 또는 삭제(`--inplace`) |
| `cross` | source에서 ref와 겹치는 이미지 제거 (ref PCA 모델 재사용) |

---

## 설치

```bash
pip install -r requirements.txt
```

## 사용법

### 1. 분석만 (파일 수정 없음)

```bash
python pca_dedup.py analyze --data_dir ./your_images
```

결과를 JSON으로 저장:
```bash
python pca_dedup.py analyze --data_dir ./your_images --report report.json
```

### 2. 중복 제거 → 새 디렉토리로 복사

```bash
python pca_dedup.py deduplicate \
    --data_dir ./your_images \
    --output_dir ./images_clean
```

실제 파일 작업 없이 시뮬레이션:
```bash
python pca_dedup.py deduplicate \
    --data_dir ./your_images \
    --output_dir ./images_clean \
    --dry_run
```

### 3. 인플레이스 삭제 (주의: 원본 삭제)

```bash
python pca_dedup.py deduplicate \
    --data_dir ./your_images \
    --inplace
```

### 4. Train ↔ Test Cross-deduplication

Train set에서 Test set과 겹치는 이미지 제거:
```bash
python pca_dedup.py cross \
    --data_dir ./train \
    --ref_dir ./test
```

---

## 주요 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--n_components` | 32 | PCA 성분 수 (= 해시 비트 수). 클수록 정밀 |
| `--hamming_threshold` | 2 | 유사 중복 판단 Hamming distance 임계값 |
| `--image_size` | 64 | 특징 추출 리사이즈 크기 |
| `--no_exact` | False | MD5 정확한 중복 탐지 건너뜀 |
| `--cache_dir` | `~/.cache/pca_dedup` | 특징 벡터 캐시 디렉토리 |
| `--report` | None | 분석 결과 JSON 저장 경로 |
| `--save_html` | None | 인터랙티브 HTML 시각화 저장 경로 |

### 캐시 동작

특징 추출 결과를 `<cache_dir>/<md5_key>.npz` 에 저장합니다.
캐시 키는 **파일 경로 + mtime + size + image_size** 의 MD5이므로,
파일이 추가·변경되거나 `--image_size` 가 바뀌면 자동으로 재추출합니다.

## 파라미터 튜닝 가이드

- **`--hamming_threshold`**: 값이 클수록 더 많은 쌍을 중복으로 판정 (false positive 증가)
  - `0`: 해시가 완전히 동일한 경우만 (매우 엄격)
  - `2`: 권장 기본값 (약간의 색상/노이즈 차이 허용)
  - `4~8`: 느슨한 기준 (다양한 변형 포함)
- **`--n_components`**: DINOv2 논문은 32비트 권장. 데이터셋이 작으면 자동으로 조정됨
- **`--image_size`**: 64 이상 권장. 높을수록 특징이 풍부하지만 메모리·캐시 크기 증가

---

## 출력 예시

```
============================================================
  PCA Hash Deduplication 분석 결과
============================================================
  전체 이미지 수          : 10,000
  PCA 해시 비트 수        : 32
  Hamming 거리 임계값     : 2

  [정확한 중복 (MD5)]
    중복 그룹 수          : 45
    제거 가능 이미지 수   : 52

  [유사 중복 (PCA Hash)]
    중복 그룹 수          : 128
    제거 가능 이미지 수   : 183

  총 제거 가능 이미지 수  : 235  (2.4%)
  중복 제거 후 예상 수    : 9,765
============================================================
```

## 시각화 결과 보기

아래 링크를 클릭하면 중복 탐지 결과를 인터랙티브하게 확인할 수 있습니다.

**[갤러리 보기 — 산점도 + 중복 그룹별 이미지 썸네일](https://htmlpreview.github.io/?https://raw.githubusercontent.com/yujinkimmn/pca/refs/heads/claude/pca-hash-deduplication-XvFm0/dedup_visualization_gallery.html)**

## 지원 이미지 형식

`.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, `.tiff`, `.tif`, `.webp`
