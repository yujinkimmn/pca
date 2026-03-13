"""
SSCD Copy Detection Deduplication
==================================
Facebook Research의 SSCD(Self-Supervised Copy Detection) 모델을 사용해
이미지 데이터셋의 유사 복사본을 탐지하고 제거합니다.

알고리즘:
  1. SSCD TorchScript 모델로 이미지당 512차원 임베딩 추출 (L2 정규화)
  2. 코사인 유사도(= dot product) 행렬 계산
  3. 임계값 이상인 쌍을 Union-Find로 그룹화
  4. 중복 그룹 중 대표 이미지 1장만 남기고 나머지 제거/이동

모델:
  sscd_disc_mixup  - DISC 데이터셋으로 학습, 복사 탐지에 최적화 (기본값)
  sscd_disc_large  - 더 큰 모델 (정확도↑, 속도↓)

참고: "An image is worth more than a thousand words: Revisiting the value of
       images and language in self-supervised copy detection" (Pizzi et al., 2022)
       https://arxiv.org/abs/2212.06816
"""

import argparse
import hashlib
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# 지원 모델 목록
# HuggingFace Hub: m3/sscd-copy-detection (TorchScript 파일)
# ---------------------------------------------------------------------------
SSCD_HF_REPO = "m3/sscd-copy-detection"
SSCD_MODELS: dict[str, str] = {
    "sscd_disc_mixup":    "sscd_disc_mixup.torchscript.pt",
    "sscd_disc_blur":     "sscd_disc_blur.torchscript.pt",
    "sscd_disc_advanced": "sscd_disc_advanced.torchscript.pt",
}
DEFAULT_MODEL = "sscd_disc_mixup"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".tif", ".webp"}


# ---------------------------------------------------------------------------
# 이미지 경로 수집
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# SSCD 특징 추출
# ---------------------------------------------------------------------------
def _sscd_cache_key(paths: list[Path], model_name: str) -> str:
    """경로 + mtime + size + 모델명 기반 캐시 키."""
    h = hashlib.md5()
    h.update(f"sscd_{model_name}_v1\n".encode())
    for p in sorted(paths):
        try:
            st = p.stat()
            h.update(f"{p}|{st.st_mtime}|{st.st_size}\n".encode())
        except OSError:
            h.update(f"{p}|missing\n".encode())
    return h.hexdigest()


def _load_feature_cache(cache_dir: Path, key: str) -> tuple[np.ndarray, list[Path]] | None:
    cache_file = cache_dir / f"{key}.npz"
    if not cache_file.exists():
        return None
    data = np.load(cache_file, allow_pickle=False)
    return data["features"], [Path(p) for p in data["valid_paths"]]


def _save_feature_cache(
    cache_dir: Path, key: str, features: np.ndarray, valid_paths: list[Path]
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_dir / f"{key}.npz",
        features=features,
        valid_paths=np.array([str(p) for p in valid_paths]),
    )


def extract_sscd_features(
    paths: list[Path],
    model_name: str = DEFAULT_MODEL,
    cache_dir: Optional[Path] = None,
    batch_size: int = 32,
) -> tuple[np.ndarray, list[Path]]:
    """SSCD 모델로 512차원 L2 정규화 임베딩을 추출합니다.

    모델 파일은 cache_dir(기본: ~/.cache/pca_dedup)에 자동 다운로드됩니다.

    Args:
        paths: 이미지 경로 목록
        model_name: 사용할 SSCD 모델 이름
        cache_dir: 캐시 디렉토리
        batch_size: 배치 크기 (GPU 메모리에 맞게 조절)

    Returns:
        features: (N, 512) float32 배열 (L2 정규화됨)
        valid_paths: 로드에 성공한 경로 목록
    """
    try:
        import torch
        import torchvision.transforms as T
    except ImportError:
        raise ImportError(
            "SSCD 특징 추출에는 'torch'와 'torchvision'이 필요합니다.\n"
            "설치: pip install torch torchvision"
        )
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "SSCD 모델 다운로드에는 'huggingface_hub'이 필요합니다.\n"
            "설치: pip install huggingface_hub"
        )

    if cache_dir is not None:
        key = _sscd_cache_key(paths, model_name)
        cached = _load_feature_cache(cache_dir, key)
        if cached is not None:
            feats, valid = cached
            print(f"  [SSCD 캐시 HIT] {len(valid):,}장 로드 ({cache_dir / (key[:8] + '...')}.npz)")
            return feats, valid

    filename = SSCD_MODELS.get(model_name)
    if filename is None:
        raise ValueError(f"알 수 없는 모델: {model_name}. 사용 가능: {list(SSCD_MODELS)}")

    # HuggingFace Hub에서 TorchScript 모델 다운로드 (자동 캐시: ~/.cache/huggingface/hub/)
    print(f"  [SSCD] HuggingFace Hub에서 모델 로드 중... ({SSCD_HF_REPO}/{filename})")
    model_path = hf_hub_download(repo_id=SSCD_HF_REPO, filename=filename)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  [SSCD] 모델 로드 완료. device={device}")
    model = torch.jit.load(model_path, map_location=device)
    model.eval()

    transform = T.Compose([
        T.Resize(288),
        T.CenterCrop(288),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    all_features: list[np.ndarray] = []
    valid_paths: list[Path] = []

    for i in tqdm(range(0, len(paths), batch_size), desc="SSCD 특징 추출", ncols=80):
        batch = paths[i : i + batch_size]
        tensors = []
        batch_valid: list[Path] = []
        for p in batch:
            try:
                img = Image.open(p).convert("RGB")
                tensors.append(transform(img))
                batch_valid.append(p)
            except Exception as e:
                print(f"  [경고] 이미지 로드 실패 ({p}): {e}", file=sys.stderr)
        if not tensors:
            continue
        with torch.no_grad():
            x = torch.stack(tensors).to(device)
            feat = model(x)                                # (B, 512)
            feat = feat / feat.norm(dim=1, keepdim=True)  # L2 normalize
        all_features.append(feat.cpu().numpy())
        valid_paths.extend(batch_valid)

    if not all_features:
        return np.empty((0, 512), dtype=np.float32), []

    features = np.concatenate(all_features, axis=0).astype(np.float32)

    if cache_dir is not None:
        _save_feature_cache(cache_dir, key, features, valid_paths)
        print(f"  [SSCD 캐시 저장] {cache_dir / (key[:8] + '...')}.npz")

    return features, valid_paths


# ---------------------------------------------------------------------------
# 코사인 유사도 기반 중복 그룹 탐지
# ---------------------------------------------------------------------------
def find_cosine_duplicate_groups(
    features: np.ndarray,
    threshold: float,
) -> list[list[int]]:
    """L2 정규화된 특징 벡터의 코사인 유사도로 중복 그룹을 찾습니다.

    features @ features.T = 코사인 유사도 행렬 (L2 정규화 가정).

    Args:
        features: (N, D) L2 정규화된 특징 행렬
        threshold: 코사인 유사도 임계값 (이 값 이상이면 중복으로 판단, 예: 0.95)

    Returns:
        각 그룹 내 이미지 인덱스 목록의 리스트 (2개 이상 포함된 그룹만)
    """
    N = len(features)
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

    chunk = 500
    for i in tqdm(range(0, N, chunk), desc="코사인 유사도 계산", ncols=80):
        end_i = min(i + chunk, N)
        sim = features[i:end_i] @ features.T  # (chunk, N)
        rows, cols = np.where(
            (sim >= threshold)
            & (np.arange(i, end_i)[:, None] < np.arange(N)[None, :])
        )
        for r_local, c in zip(rows.tolist(), cols.tolist()):
            union(i + r_local, c)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(N):
        groups[find(i)].append(i)

    return [g for g in groups.values() if len(g) > 1]


# ---------------------------------------------------------------------------
# 보고서 출력
# ---------------------------------------------------------------------------
def print_report(
    total: int,
    dup_groups: list[list[int]],
    valid_paths: list[Path],
    model_name: str,
    threshold: float,
) -> dict:
    """분석 결과를 출력하고 요약 dict를 반환합니다."""
    dup_images = set()
    for g in dup_groups:
        for idx in g[1:]:
            dup_images.add(valid_paths[idx])
    removable_count = len(dup_images)

    separator = "=" * 60

    print(f"\n{separator}")
    print("  SSCD Copy Detection 분석 결과")
    print(separator)
    print(f"  전체 이미지 수          : {total:,}")
    print(f"  SSCD 모델               : {model_name}")
    print(f"  코사인 유사도 임계값    : {threshold}")
    print()
    print(f"  [유사 중복 (SSCD Cosine)]")
    print(f"    중복 그룹 수          : {len(dup_groups):,}")
    print(f"    제거 가능 이미지 수   : {removable_count:,}")
    print()
    print(f"  총 제거 가능 이미지 수  : {removable_count:,}  ({removable_count/max(total,1)*100:.1f}%)")
    print(f"  중복 제거 후 예상 수    : {total - removable_count:,}")
    print(separator)

    if dup_groups:
        print("\n  [유사 중복 그룹 예시 (상위 5개)]")
        for i, g in enumerate(dup_groups[:5]):
            print(f"    그룹 {i+1} ({len(g)}장):")
            for idx in g:
                print(f"      - {valid_paths[idx]}")

    print()

    return {
        "total_images": total,
        "model_name": model_name,
        "cosine_threshold": threshold,
        "group_count": len(dup_groups),
        "removable_count": removable_count,
        "estimated_clean_count": total - removable_count,
    }


# ---------------------------------------------------------------------------
# 중복 제거 실행
# ---------------------------------------------------------------------------
def deduplicate(
    dup_groups: list[list[int]],
    valid_paths: list[Path],
    output_dir: Optional[str],
    dry_run: bool,
) -> int:
    """중복 이미지를 제거합니다."""
    to_remove: set[Path] = set()
    for g in dup_groups:
        for idx in g[1:]:
            to_remove.add(valid_paths[idx])

    if not to_remove:
        print("  제거할 중복 이미지가 없습니다.")
        return 0

    if output_dir:
        keep_paths = set(valid_paths) - to_remove
        out = Path(output_dir)
        if not dry_run:
            out.mkdir(parents=True, exist_ok=True)
        print(f"\n  유지할 이미지 {len(keep_paths):,}장을 '{output_dir}'로 복사합니다.")
        for p in tqdm(sorted(keep_paths), desc="복사 중", ncols=80):
            dest = out / p.name
            if dest.exists():
                dest = out / f"{p.stem}_{p.parent.name}{p.suffix}"
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
    threshold: float,
    dry_run: bool,
    model_name: str = DEFAULT_MODEL,
    cache_dir: Optional[Path] = None,
    batch_size: int = 32,
):
    """source_dir의 이미지 중 ref_dir와 유사한 것들을 탐지합니다.
    (예: train set에서 test set과 겹치는 이미지 제거)
    """
    print(f"\n  [SSCD Cross-deduplication]")
    print(f"  Source : {source_dir}")
    print(f"  Ref    : {ref_dir}")
    print(f"  모델   : {model_name}, 임계값: {threshold}")

    source_paths = collect_image_paths(source_dir)
    ref_paths = collect_image_paths(ref_dir)
    print(f"  Source 이미지 수: {len(source_paths):,}")
    print(f"  Ref 이미지 수   : {len(ref_paths):,}")

    all_paths_list = source_paths + ref_paths
    features, valid_paths = extract_sscd_features(
        all_paths_list, model_name=model_name,
        cache_dir=cache_dir, batch_size=batch_size,
    )

    source_set = set(source_paths)
    source_idxs = [i for i, p in enumerate(valid_paths) if p in source_set]
    ref_idxs    = [i for i, p in enumerate(valid_paths) if p not in source_set]
    source_feats = features[source_idxs]
    ref_feats    = features[ref_idxs]
    source_valid = [valid_paths[i] for i in source_idxs]
    ref_valid    = [valid_paths[i] for i in ref_idxs]

    to_remove: set[Path] = set()
    cross_pairs: list[tuple[Path, Path]] = []
    chunk = 500

    for i in tqdm(range(0, len(source_feats), chunk), desc="Cross-dup 탐지 (SSCD)", ncols=80):
        s_chunk = source_feats[i : i + chunk]
        sim = s_chunk @ ref_feats.T  # (chunk, n_ref)
        for local_idx in range(len(s_chunk)):
            row = sim[local_idx]
            if row.max() >= threshold:
                src = source_valid[i + local_idx]
                best_ref = ref_valid[int(row.argmax())]
                to_remove.add(src)
                cross_pairs.append((src, best_ref))

    print(f"\n  Ref와 유사한 Source 이미지 수: {len(to_remove):,} / {len(source_valid):,}")
    if to_remove and not dry_run:
        for p in tqdm(sorted(to_remove), desc="삭제 중", ncols=80):
            p.unlink(missing_ok=True)
    elif dry_run:
        print("  [dry-run] 삭제 대상 예시:")
        for p in sorted(to_remove)[:10]:
            print(f"    - {p}")

    return len(to_remove), cross_pairs, features, valid_paths


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Facebook SSCD 기반 Copy Detection Deduplication",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 분석만 (파일 수정 없음)
  python sscd_dedup.py analyze --data_dir ./images

  # 임계값 조정 (기본 0.95, 낮출수록 더 많이 탐지)
  python sscd_dedup.py analyze --data_dir ./images --threshold 0.90

  # 중복 제거 후 clean 디렉토리로 저장
  python sscd_dedup.py deduplicate --data_dir ./images --output_dir ./images_clean

  # 인플레이스 삭제 (주의!)
  python sscd_dedup.py deduplicate --data_dir ./images --inplace

  # 더 큰 모델 사용
  python sscd_dedup.py analyze --data_dir ./images --model sscd_disc_large

  # train vs test cross-deduplication
  python sscd_dedup.py cross --data_dir ./train --ref_dir ./test

  # HTML 갤러리 생성
  python sscd_dedup.py analyze --data_dir ./images --save_html ./report.html
""",
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # --- 공통 인자 ---
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data_dir", action="append", default=None,
                        help="이미지 데이터셋 디렉토리 (여러 개: --data_dir ./a --data_dir ./b)")
    common.add_argument("--threshold", type=float, default=0.95,
                        help="코사인 유사도 임계값 (기본값: 0.95, 범위: 0~1)")
    common.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        choices=list(SSCD_MODELS),
                        help=f"사용할 SSCD 모델 (기본값: {DEFAULT_MODEL})")
    common.add_argument("--batch_size", type=int, default=32,
                        help="배치 크기 (GPU 메모리에 맞게 조절, 기본값: 32)")
    common.add_argument("--cache_dir", type=str,
                        default=str(Path.home() / ".cache" / "pca_dedup"),
                        help="특징 벡터 캐시 디렉토리 (기본값: ~/.cache/pca_dedup)")
    common.add_argument("--report", type=str, default=None,
                        help="분석 결과를 저장할 JSON 파일 경로")
    common.add_argument("--save_html", type=str, default=None, metavar="PATH",
                        help="HTML 시각화를 저장할 경로 (갤러리 + 인터랙티브 scatter)")
    common.add_argument("--thumb", type=int, default=180,
                        help="갤러리 썸네일 크기 px (기본값: 180)")

    # --- analyze ---
    sub.add_parser("analyze", parents=[common],
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

    # --- serve ---
    p_serve = sub.add_parser("serve",
                              help="갤러리 HTML을 서빙하고 삭제 버튼을 활성화하는 로컬 서버 실행")
    p_serve.add_argument("--html", required=True,
                         help="서빙할 갤러리 HTML 파일 경로")
    p_serve.add_argument("--port", type=int, default=7474,
                         help="서버 포트 (기본값: 7474)")
    p_serve.add_argument("--action", choices=["delete", "move"], default="move",
                         help="버튼 클릭 시 동작: move(기본값) 또는 delete")

    args = parser.parse_args()
    if getattr(args, "data_dir", None) is None and args.mode != "serve":
        parser.error("--data_dir 를 하나 이상 지정해야 합니다.")
    return args


def main():
    args = parse_args()
    start_time = time.time()

    if args.mode == "serve":
        from visualize_dedup import run_server
        run_server(args.html, port=args.port, action=args.action)
        return

    if args.mode == "cross":
        removed, cross_pairs, cross_features, cross_valid = cross_deduplicate(
            source_dir=args.data_dir,
            ref_dir=args.ref_dir,
            threshold=args.threshold,
            dry_run=getattr(args, "dry_run", False),
            model_name=args.model,
            cache_dir=Path(args.cache_dir),
            batch_size=args.batch_size,
        )
        print(f"\n  완료: {removed}장 제거됨  ({time.time()-start_time:.1f}s)")

        if args.report:
            report_data = {
                "mode": "cross",
                "source_dir": str(args.data_dir),
                "ref_dir": args.ref_dir,
                "model_name": args.model,
                "cosine_threshold": args.threshold,
                "matched_count": removed,
                "cross_pairs": [[str(s), str(r)] for s, r in cross_pairs],
            }
            Path(args.report).write_text(
                json.dumps(report_data, ensure_ascii=False, indent=2)
            )
            print(f"  보고서 저장: {args.report}")

        if args.save_html and cross_pairs:
            Path(args.save_html).parent.mkdir(parents=True, exist_ok=True)
            try:
                from visualize_dedup import embed_2d, plot_gallery
                gallery_paths: list[Path] = []
                near_groups_vis: list[list[int]] = []
                seen: dict[Path, int] = {}

                def _idx(p: Path) -> int:
                    if p not in seen:
                        seen[p] = len(gallery_paths)
                        gallery_paths.append(p)
                    return seen[p]

                for src, ref in cross_pairs:
                    near_groups_vis.append([_idx(src), _idx(ref)])

                labels = np.full(len(gallery_paths), -1, dtype=int)
                group_meta: dict[int, dict] = {}
                for gid, grp in enumerate(near_groups_vis):
                    for idx in grp:
                        labels[idx] = gid
                    group_meta[gid] = {"type": "near", "size": len(grp)}

                path_to_idx = {p: i for i, p in enumerate(cross_valid)}
                gallery_features = np.stack([
                    cross_features[path_to_idx[p]] for p in gallery_paths
                ])
                coords = embed_2d(gallery_features, "pca")

                data_dir_label = f"{args.data_dir} ↔ {args.ref_dir}"
                print("\n[HTML 시각화 생성 중...]")
                plot_gallery(
                    gallery_paths, labels, group_meta, args.save_html,
                    data_dir_label, 512, args.threshold,
                    thumb=args.thumb, coords=coords,
                )
            except ImportError as e:
                print(f"  [경고] HTML 생성 실패: {e}")
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
    # 2. SSCD 특징 추출
    # -----------------------------------------------------------------------
    print(f"\n[1/2] SSCD 특징 추출...")
    print(f"  모델: {args.model}, 코사인 임계값: {args.threshold}")

    features, valid_paths = extract_sscd_features(
        paths,
        model_name=args.model,
        cache_dir=Path(args.cache_dir),
        batch_size=args.batch_size,
    )
    print(f"  유효하게 로드된 이미지: {len(valid_paths):,} / {len(paths):,}")

    # -----------------------------------------------------------------------
    # 3. 코사인 유사도 기반 중복 그룹 탐지
    # -----------------------------------------------------------------------
    print(f"\n[2/2] 중복 그룹 탐지...")
    dup_groups = find_cosine_duplicate_groups(features, args.threshold)

    # -----------------------------------------------------------------------
    # 4. 보고서 출력
    # -----------------------------------------------------------------------
    summary = print_report(
        total=len(paths),
        dup_groups=dup_groups,
        valid_paths=valid_paths,
        model_name=args.model,
        threshold=args.threshold,
    )

    if args.report:
        serializable_near = [
            [str(valid_paths[idx]) for idx in g] for g in dup_groups
        ]
        report_data = {**summary, "dup_groups": serializable_near}
        Path(args.report).write_text(json.dumps(report_data, ensure_ascii=False, indent=2))
        print(f"  보고서 저장: {args.report}")

    if args.save_html:
        Path(args.save_html).parent.mkdir(parents=True, exist_ok=True)
        try:
            from visualize_dedup import embed_2d, build_labels, plot_gallery, plot_interactive
            print("\n[HTML 시각화 생성 중...]")
            data_dir_label = ", ".join(str(d) for d in args.data_dir)
            coords = embed_2d(features, "pca")
            labels, group_meta = build_labels(len(valid_paths), {}, dup_groups, valid_paths)
            plot_gallery(
                valid_paths, labels, group_meta, args.save_html,
                data_dir_label, 512, args.threshold,
                thumb=args.thumb, coords=coords,
            )
            plot_interactive(
                coords, labels, group_meta, valid_paths, args.save_html,
                "pca", data_dir_label, 512, args.threshold,
            )
        except ImportError as e:
            print(f"  [경고] HTML 생성 실패 (visualize_dedup.py 필요): {e}")

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
            confirm = dry_run or inplace
            if inplace and not dry_run:
                ans = input("\n  [주의] 원본 파일을 직접 삭제합니다. 계속하시겠습니까? (y/N): ")
                confirm = ans.strip().lower() == "y"

            if confirm or output_dir:
                removed = deduplicate(
                    dup_groups=dup_groups,
                    valid_paths=valid_paths,
                    output_dir=output_dir,
                    dry_run=dry_run,
                )
                print(f"\n  제거된 이미지 수: {removed:,}")

    elapsed = time.time() - start_time
    print(f"\n  총 소요 시간: {elapsed:.1f}초\n")


if __name__ == "__main__":
    main()
