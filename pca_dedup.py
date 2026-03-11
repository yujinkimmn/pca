"""
PCA Hash Deduplication
======================
DINOv2 논문(Oquab et al., 2023)에서 제안한 방법으로 이미지 데이터셋의 중복을 탐지하고 제거합니다.

알고리즘:
  1. 이미지를 고정 크기로 리사이즈 후 픽셀 벡터로 변환
  2. PCA로 차원 축소 (n_components 비트의 해시)
  3. PCA 투영값의 부호(sign)를 이진화 → 각 이미지의 binary hash 생성
  4. Hamming distance로 해시 쌍을 비교하여 near-duplicate 탐지
  5. 중복 그룹 중 대표 이미지 1장만 남기고 나머지 제거/이동

참고: DINOv2 paper - "DINOv2: Learning Robust Visual Features without Supervision"
       https://arxiv.org/abs/2304.07193  (Section 3.1 Data Processing)
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from tqdm import tqdm


# ---------------------------------------------------------------------------
# 이미지 로딩 유틸
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".tif", ".webp"}


def collect_image_paths(data_dir) -> list[Path]:
    """디렉토리(들)에서 모든 이미지 경로를 재귀적으로 수집합니다."""
    dirs = [data_dir] if isinstance(data_dir, (str, Path)) else data_dir
    paths = []
    for d in dirs:
        d = Path(d)
        for ext in IMAGE_EXTENSIONS:
            paths.extend(d.rglob(f"*{ext}"))
            paths.extend(d.rglob(f"*{ext.upper()}"))
    return sorted(set(paths))


def load_image_as_vector(path: Path, image_size: int) -> Optional[np.ndarray]:
    """이미지를 읽어 (image_size * image_size * 3,) 형태의 float32 벡터로 반환합니다."""
    try:
        img = Image.open(path).convert("RGB")
        img = img.resize((image_size, image_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32).ravel()
        return arr
    except Exception as e:
        print(f"  [경고] 이미지 로드 실패 ({path}): {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# 정확한 MD5 기반 중복 탐지 (픽셀 완전 동일)
# ---------------------------------------------------------------------------
def md5_exact_duplicates(paths: list[Path]) -> dict[str, list[Path]]:
    """MD5 해시를 이용해 바이트 단위로 완전히 동일한 파일을 그룹화합니다."""
    hash_to_paths: dict[str, list[Path]] = defaultdict(list)
    for p in tqdm(paths, desc="MD5 exact-dup scan", ncols=80):
        try:
            h = hashlib.md5(p.read_bytes()).hexdigest()
            hash_to_paths[h].append(p)
        except Exception as e:
            print(f"  [경고] MD5 계산 실패 ({p}): {e}", file=sys.stderr)
    return {h: ps for h, ps in hash_to_paths.items() if len(ps) > 1}


# ---------------------------------------------------------------------------
# PCA Hash 기반 near-duplicate 탐지
# ---------------------------------------------------------------------------
def extract_features(
    paths: list[Path],
    image_size: int,
) -> tuple[np.ndarray, list[Path]]:
    """
    유효하게 로드된 이미지들의 픽셀 특징 행렬과 대응하는 경로 목록을 반환합니다.
    Returns:
        features: (N, D) float32 배열
        valid_paths: 로드에 성공한 경로 목록 (N개)
    """
    vectors = []
    valid_paths = []
    for p in tqdm(paths, desc="이미지 특징 추출", ncols=80):
        vec = load_image_as_vector(p, image_size)
        if vec is not None:
            vectors.append(vec)
            valid_paths.append(p)
    features = np.stack(vectors, axis=0)  # (N, D)
    return features, valid_paths


def compute_pca_hashes(
    features: np.ndarray,
    n_components: int,
    pca_model: Optional[PCA] = None,
) -> tuple[np.ndarray, PCA]:
    """
    PCA 투영 후 부호(sign) 기반 이진 해시를 계산합니다.

    DINOv2 방법:
      - 특징 벡터에 PCA 적용 (n_components 차원)
      - 각 성분의 부호를 비트로 변환 → binary hash

    Args:
        features: (N, D) 특징 행렬
        n_components: PCA 성분 수 (= 해시 비트 수)
        pca_model: 기존 PCA 모델 (None이면 features로 새로 학습)

    Returns:
        hashes: (N, n_components) bool 배열
        pca_model: 학습된 PCA 모델
    """
    n_components = min(n_components, features.shape[0], features.shape[1])

    if pca_model is None:
        print(f"  PCA 학습 중 (n_components={n_components})...")
        pca_model = PCA(n_components=n_components, random_state=42)
        projections = pca_model.fit_transform(features)
    else:
        projections = pca_model.transform(features)

    # 부호 기반 이진화: 양수 → True(1), 음수/0 → False(0)
    hashes = projections >= 0  # (N, n_components) bool
    return hashes, pca_model


def hamming_distance_matrix(hashes: np.ndarray) -> np.ndarray:
    """
    (N, n_bits) bool 배열에서 모든 쌍의 Hamming distance를 효율적으로 계산합니다.
    Returns: (N, N) uint16 배열
    """
    # hashes를 uint8로 변환 후 XOR 기반 Hamming distance
    # 메모리 효율을 위해 chunk 단위로 처리
    N = hashes.shape[0]
    dist = np.zeros((N, N), dtype=np.uint16)

    h = hashes.astype(np.uint8)
    chunk = 1000
    for i in range(0, N, chunk):
        end_i = min(i + chunk, N)
        for j in range(i, N, chunk):
            end_j = min(j + chunk, N)
            # XOR → 합산 = Hamming distance
            xor = h[i:end_i, np.newaxis, :] ^ h[np.newaxis, j:end_j, :]
            d = xor.sum(axis=2).astype(np.uint16)
            dist[i:end_i, j:end_j] = d
            dist[j:end_j, i:end_i] = d.T
    return dist


def find_duplicate_groups(
    dist_matrix: np.ndarray,
    paths: list[Path],
    threshold: int,
) -> list[list[int]]:
    """
    Hamming distance ≤ threshold 인 이미지들을 Union-Find로 그룹화합니다.
    자기 자신(대각선)은 제외합니다.
    """
    N = len(paths)
    parent = list(range(N))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    # 상삼각 행렬만 순회
    rows, cols = np.where(
        (dist_matrix <= threshold) & (np.arange(N)[:, None] < np.arange(N)[None, :])
    )
    for r, c in zip(rows.tolist(), cols.tolist()):
        union(r, c)

    # 그룹 수집
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(N):
        groups[find(i)].append(i)

    # 2개 이상인 그룹만 반환
    return [g for g in groups.values() if len(g) > 1]


# ---------------------------------------------------------------------------
# 보고서 출력
# ---------------------------------------------------------------------------
def print_report(
    total: int,
    exact_groups: dict[str, list[Path]],
    near_groups: list[list[int]],
    near_paths: list[Path],
    n_components: int,
    threshold: int,
) -> dict:
    """분석 결과를 출력하고 요약 dict를 반환합니다."""
    # exact dup 통계
    exact_dup_count = sum(len(v) - 1 for v in exact_groups.values())
    exact_group_count = len(exact_groups)

    # near dup 통계 (exact dup와 겹칠 수 있음)
    near_dup_images = set()
    for g in near_groups:
        for idx in g[1:]:
            near_dup_images.add(near_paths[idx])
    near_dup_count = len(near_dup_images)

    separator = "=" * 60

    print(f"\n{separator}")
    print("  PCA Hash Deduplication 분석 결과")
    print(separator)
    print(f"  전체 이미지 수          : {total:,}")
    print(f"  PCA 해시 비트 수        : {n_components}")
    print(f"  Hamming 거리 임계값     : {threshold}")
    print()
    print(f"  [정확한 중복 (MD5)]")
    print(f"    중복 그룹 수          : {exact_group_count:,}")
    print(f"    제거 가능 이미지 수   : {exact_dup_count:,}")
    print()
    print(f"  [유사 중복 (PCA Hash)]")
    print(f"    중복 그룹 수          : {len(near_groups):,}")
    print(f"    제거 가능 이미지 수   : {near_dup_count:,}")
    print()

    total_removable = exact_dup_count + near_dup_count
    print(f"  총 제거 가능 이미지 수  : {total_removable:,}  ({total_removable/max(total,1)*100:.1f}%)")
    print(f"  중복 제거 후 예상 수    : {total - total_removable:,}")
    print(separator)

    # 상위 exact 중복 그룹 출력
    if exact_groups:
        print("\n  [정확한 중복 그룹 예시 (상위 5개)]")
        for i, (h, ps) in enumerate(list(exact_groups.items())[:5]):
            print(f"    그룹 {i+1} ({len(ps)}장):")
            for p in ps:
                print(f"      - {p}")

    # 상위 near-dup 그룹 출력
    if near_groups:
        print("\n  [유사 중복 그룹 예시 (상위 5개)]")
        for i, g in enumerate(near_groups[:5]):
            print(f"    그룹 {i+1} ({len(g)}장):")
            for idx in g:
                print(f"      - {near_paths[idx]}")

    print()

    return {
        "total_images": total,
        "n_components": n_components,
        "hamming_threshold": threshold,
        "exact_duplicates": {
            "group_count": exact_group_count,
            "removable_count": exact_dup_count,
        },
        "near_duplicates": {
            "group_count": len(near_groups),
            "removable_count": near_dup_count,
        },
        "total_removable": total_removable,
        "estimated_clean_count": total - total_removable,
    }


# ---------------------------------------------------------------------------
# 중복 제거 실행
# ---------------------------------------------------------------------------
def deduplicate(
    exact_groups: dict[str, list[Path]],
    near_groups: list[list[int]],
    near_paths: list[Path],
    output_dir: Optional[str],
    dry_run: bool,
) -> int:
    """
    중복 이미지를 제거합니다.
    - output_dir이 지정된 경우: 남길 이미지를 output_dir로 복사
    - output_dir이 없는 경우: 중복 이미지를 원본에서 삭제 (dry_run=False일 때만)
    """
    # 제거할 이미지 경로 수집
    to_remove: set[Path] = set()

    for ps in exact_groups.values():
        # 첫 번째 이미지만 유지, 나머지 제거
        for p in ps[1:]:
            to_remove.add(p)

    for g in near_groups:
        for idx in g[1:]:
            to_remove.add(near_paths[idx])

    if not to_remove:
        print("  제거할 중복 이미지가 없습니다.")
        return 0

    if output_dir:
        # 살릴 이미지 = 전체 - 제거 대상
        all_paths = set(near_paths)
        for ps in exact_groups.values():
            all_paths.update(ps)

        keep_paths = all_paths - to_remove
        out = Path(output_dir)
        if not dry_run:
            out.mkdir(parents=True, exist_ok=True)

        print(f"\n  유지할 이미지 {len(keep_paths):,}장을 '{output_dir}'로 복사합니다.")
        for p in tqdm(sorted(keep_paths), desc="복사 중", ncols=80):
            dest = out / p.name
            # 파일명 충돌 방지
            if dest.exists():
                stem = dest.stem
                suffix = dest.suffix
                dest = out / f"{stem}_{p.parent.name}{suffix}"
            if not dry_run:
                shutil.copy2(p, dest)
    else:
        print(f"\n  중복 이미지 {len(to_remove):,}장을 삭제합니다.")
        for p in tqdm(sorted(to_remove), desc="삭제 중", ncols=80):
            if not dry_run:
                p.unlink(missing_ok=True)

    if dry_run:
        print("  [dry-run 모드] 실제 파일 작업은 수행하지 않았습니다.")

    return len(to_remove)


# ---------------------------------------------------------------------------
# 두 데이터셋 간 cross-deduplication
# ---------------------------------------------------------------------------
def cross_deduplicate(
    source_dir: str,
    ref_dir: str,
    n_components: int,
    image_size: int,
    threshold: int,
    dry_run: bool,
):
    """
    source_dir의 이미지 중 ref_dir와 유사한 것들을 탐지합니다.
    (예: train set에서 test set과 겹치는 이미지 제거)
    """
    print(f"\n  [Cross-deduplication]")
    print(f"  Source : {source_dir}")
    print(f"  Ref    : {ref_dir}")

    source_paths = collect_image_paths(source_dir)
    ref_paths = collect_image_paths(ref_dir)
    print(f"  Source 이미지 수: {len(source_paths):,}")
    print(f"  Ref 이미지 수   : {len(ref_paths):,}")

    all_paths_list = source_paths + ref_paths
    features, valid_paths = extract_features(all_paths_list, image_size)
    hashes, _ = compute_pca_hashes(features, n_components)

    n_source = sum(1 for p in valid_paths if p in set(source_paths))
    source_hashes = hashes[:n_source]
    ref_hashes = hashes[n_source:]
    source_valid = valid_paths[:n_source]

    # source vs ref Hamming distance
    s = source_hashes.astype(np.uint8)
    r = ref_hashes.astype(np.uint8)

    to_remove = set()
    chunk = 500
    for i in tqdm(range(0, len(s), chunk), desc="Cross-dup 탐지", ncols=80):
        s_chunk = s[i : i + chunk]  # (chunk, bits)
        # (chunk, n_ref, bits)
        xor = s_chunk[:, np.newaxis, :] ^ r[np.newaxis, :, :]
        dists = xor.sum(axis=2)  # (chunk, n_ref)
        matches = np.any(dists <= threshold, axis=1)
        for local_idx, is_match in enumerate(matches):
            if is_match:
                to_remove.add(source_valid[i + local_idx])

    print(f"\n  Ref와 유사한 Source 이미지 수: {len(to_remove):,} / {len(source_valid):,}")
    if to_remove and not dry_run:
        for p in tqdm(sorted(to_remove), desc="삭제 중", ncols=80):
            p.unlink(missing_ok=True)
    elif dry_run:
        print("  [dry-run] 삭제 대상 예시:")
        for p in sorted(to_remove)[:10]:
            print(f"    - {p}")

    return len(to_remove)


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="DINOv2 스타일 PCA Hash Deduplication",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 분석만 (파일 수정 없음)
  python pca_dedup.py analyze --data_dir ./images

  # 중복 제거 후 clean 디렉토리로 저장
  python pca_dedup.py deduplicate --data_dir ./images --output_dir ./images_clean

  # 인플레이스 삭제 (주의!)
  python pca_dedup.py deduplicate --data_dir ./images --inplace

  # train vs test cross-deduplication
  python pca_dedup.py cross --data_dir ./train --ref_dir ./test
""",
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # --- 공통 인자 ---
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data_dir", required=True, nargs="+",
                        help="이미지 데이터셋 디렉토리 (여러 개 지정 가능)")
    common.add_argument("--n_components", type=int, default=32,
                        help="PCA 성분 수 (해시 비트 수). 클수록 정밀 (기본값: 32)")
    common.add_argument("--hamming_threshold", type=int, default=2,
                        help="유사 중복 판단 Hamming distance 임계값 (기본값: 2)")
    common.add_argument("--image_size", type=int, default=64,
                        help="특징 추출 시 리사이즈할 이미지 크기 (기본값: 64)")
    common.add_argument("--no_exact", action="store_true",
                        help="MD5 정확한 중복 탐지 건너뜀")
    common.add_argument("--report", type=str, default=None,
                        help="분석 결과를 저장할 JSON 파일 경로")

    # --- analyze ---
    p_analyze = sub.add_parser("analyze", parents=[common],
                                help="중복 현황 분석 (파일 수정 없음)")

    # --- deduplicate ---
    p_dedup = sub.add_parser("deduplicate", parents=[common],
                              help="중복 이미지 제거")
    p_dedup.add_argument("--output_dir", type=str, default=None,
                         help="중복 제거된 이미지를 복사할 디렉토리 (미지정 시 인플레이스)")
    p_dedup.add_argument("--inplace", action="store_true",
                         help="원본 디렉토리에서 직접 중복 삭제")
    p_dedup.add_argument("--dry_run", action="store_true",
                         help="실제 파일 작업 없이 시뮬레이션만 수행")

    # --- cross ---
    p_cross = sub.add_parser("cross", parents=[common],
                              help="두 데이터셋 간 cross-deduplication")
    p_cross.add_argument("--ref_dir", required=True,
                         help="참조 데이터셋 디렉토리 (예: test set)")
    p_cross.add_argument("--dry_run", action="store_true",
                         help="실제 파일 작업 없이 시뮬레이션만 수행")

    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    if args.mode == "cross":
        removed = cross_deduplicate(
            source_dir=args.data_dir,
            ref_dir=args.ref_dir,
            n_components=args.n_components,
            image_size=args.image_size,
            threshold=args.hamming_threshold,
            dry_run=getattr(args, "dry_run", True),
        )
        print(f"\n  완료: {removed}장 제거됨  ({time.time()-start_time:.1f}s)")
        return

    # -----------------------------------------------------------------------
    # 1. 이미지 수집
    # -----------------------------------------------------------------------
    dirs_str = ", ".join(args.data_dir)
    print(f"\n이미지 수집 중: {dirs_str}")
    paths = collect_image_paths(args.data_dir)
    print(f"  발견된 이미지 수: {len(paths):,}")
    if not paths:
        print("  [오류] 이미지를 찾을 수 없습니다. --data_dir 경로를 확인하세요.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # 2. 정확한 중복 탐지 (MD5)
    # -----------------------------------------------------------------------
    exact_groups: dict[str, list[Path]] = {}
    if not args.no_exact:
        print("\n[1/3] 정확한 중복 탐지 (MD5)...")
        exact_groups = md5_exact_duplicates(paths)

    # -----------------------------------------------------------------------
    # 3. PCA Hash 기반 유사 중복 탐지
    # -----------------------------------------------------------------------
    print(f"\n[2/3] PCA Hash 기반 유사 중복 탐지...")
    print(f"  이미지 크기: {args.image_size}x{args.image_size}, "
          f"해시 비트: {args.n_components}, "
          f"Hamming 임계값: {args.hamming_threshold}")

    features, valid_paths = extract_features(paths, args.image_size)
    print(f"  유효하게 로드된 이미지: {len(valid_paths):,} / {len(paths):,}")

    hashes, pca_model = compute_pca_hashes(features, args.n_components)
    print(f"  PCA 설명 분산 비율: {pca_model.explained_variance_ratio_.sum()*100:.1f}%")

    print("  Hamming distance 계산 중...")
    dist_matrix = hamming_distance_matrix(hashes)

    near_groups = find_duplicate_groups(dist_matrix, valid_paths, args.hamming_threshold)

    # -----------------------------------------------------------------------
    # 4. 보고서 출력
    # -----------------------------------------------------------------------
    print(f"\n[3/3] 결과 집계...")
    summary = print_report(
        total=len(paths),
        exact_groups=exact_groups,
        near_groups=near_groups,
        near_paths=valid_paths,
        n_components=args.n_components,
        threshold=args.hamming_threshold,
    )

    if args.report:
        # 직렬화 가능하도록 Path → str 변환
        serializable_exact = {
            h: [str(p) for p in ps] for h, ps in exact_groups.items()
        }
        serializable_near = [
            [str(valid_paths[idx]) for idx in g] for g in near_groups
        ]
        report_data = {
            **summary,
            "exact_groups": serializable_exact,
            "near_groups": serializable_near,
        }
        Path(args.report).write_text(json.dumps(report_data, ensure_ascii=False, indent=2))
        print(f"  보고서 저장: {args.report}")

    # -----------------------------------------------------------------------
    # 5. 중복 제거 (deduplicate 모드만)
    # -----------------------------------------------------------------------
    if args.mode == "deduplicate":
        output_dir = getattr(args, "output_dir", None)
        inplace = getattr(args, "inplace", False)
        dry_run = getattr(args, "dry_run", False)

        if output_dir is None and not inplace:
            print(
                "\n  [안내] --output_dir 또는 --inplace 옵션을 지정하면 중복 제거를 수행합니다.\n"
                "         현재는 분석 결과만 출력합니다."
            )
        else:
            confirm = dry_run or inplace  # dry_run은 항상 진행
            if inplace and not dry_run:
                ans = input("\n  [주의] 원본 파일을 직접 삭제합니다. 계속하시겠습니까? (y/N): ")
                confirm = ans.strip().lower() == "y"

            if confirm or output_dir:
                removed = deduplicate(
                    exact_groups=exact_groups,
                    near_groups=near_groups,
                    near_paths=valid_paths,
                    output_dir=output_dir,
                    dry_run=dry_run,
                )
                print(f"\n  제거된 이미지 수: {removed:,}")

    elapsed = time.time() - start_time
    print(f"\n  총 소요 시간: {elapsed:.1f}초\n")


if __name__ == "__main__":
    main()
