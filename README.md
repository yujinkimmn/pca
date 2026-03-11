# PCA Hash Deduplication

DINOv2 논문([Oquab et al., 2023](https://arxiv.org/abs/2304.07193))에서 제안한 **PCA Hash Deduplication** 방법으로 이미지 데이터셋의 중복을 탐지하고 제거합니다.

## 알고리즘 (DINOv2 Section 3.1)

1. 이미지를 고정 크기로 리사이즈 후 픽셀 벡터로 변환
2. PCA로 차원 축소 (`n_components` 비트의 해시 공간으로 투영)
3. PCA 투영값의 **부호(sign)** 기반 이진화 → 각 이미지의 binary hash 생성
4. **Hamming distance**로 해시 쌍을 비교하여 near-duplicate 탐지
5. Union-Find로 중복 그룹 형성 후, 그룹당 대표 이미지 1장만 유지

정확한 중복(바이트 동일)은 MD5 해시로 별도 탐지합니다.

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

## 주요 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--n_components` | 32 | PCA 성분 수 (= 해시 비트 수). 클수록 정밀 |
| `--hamming_threshold` | 2 | 유사 중복 판단 Hamming distance 임계값 |
| `--image_size` | 64 | 특징 추출 리사이즈 크기 |
| `--no_exact` | False | MD5 정확한 중복 탐지 건너뜀 |
| `--report` | None | 분석 결과 JSON 저장 경로 |

## 파라미터 튜닝 가이드

- **`--hamming_threshold`**: 값이 클수록 더 많은 쌍을 중복으로 판정 (false positive 증가)
  - `0`: 해시가 완전히 동일한 경우만 (매우 엄격)
  - `2`: 권장 기본값 (약간의 색상/노이즈 차이 허용)
  - `4~8`: 느슨한 기준 (다양한 변형 포함)
- **`--n_components`**: DINOv2 논문은 32비트 권장. 데이터셋이 작으면 자동으로 조정됨
- **`--image_size`**: 64 이상 권장. 높을수록 특징이 풍부하지만 메모리 사용 증가

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

**[시각화 결과 보기 (dedup_visualization.html)](https://htmlpreview.github.io/?https://raw.githubusercontent.com/yujinkimmn/pca/refs/heads/claude/pca-hash-deduplication-XvFm0/dedup_visualization.html)**

## 지원 이미지 형식

`.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, `.tiff`, `.tif`, `.webp`
