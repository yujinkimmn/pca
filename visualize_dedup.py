"""
PCA Hash Deduplication - 시각화
================================
이미지 특징을 2D로 투영하여 중복 그룹을 시각화합니다.

투영 방법:
  - PCA 2D: 빠름, 선형 구조 파악
  - UMAP 2D: 느리지만 군집 구조를 더 잘 표현

중복 그룹은 같은 색상+마커로 표시하고,
고유 이미지는 회색으로 표시합니다.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 헤드리스 환경 대응
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 한글 폰트 설정 (WenQuanYi Zen Hei 또는 Unifont 사용)
_KO_FONT_CANDIDATES = [
    "WenQuanYi Zen Hei",
    "Unifont",
    "NanumGothic",
    "Malgun Gothic",
    "AppleGothic",
]
for _fname in _KO_FONT_CANDIDATES:
    if any(_fname.lower() in f.name.lower() for f in font_manager.fontManager.ttflist):
        matplotlib.rcParams["font.family"] = _fname
        break
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from tqdm import tqdm

# pca_dedup.py의 유틸 재사용
sys.path.insert(0, str(Path(__file__).parent))
from pca_dedup import (
    collect_image_paths,
    extract_features,
    compute_pca_hashes,
    hamming_distance_matrix,
    find_duplicate_groups,
    md5_exact_duplicates,
)


# ---------------------------------------------------------------------------
# 2D 임베딩
# ---------------------------------------------------------------------------

def embed_2d_pca(features: np.ndarray) -> np.ndarray:
    """PCA로 2D 투영."""
    n_components = min(2, features.shape[0], features.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(features)


def embed_2d_umap(features: np.ndarray) -> np.ndarray:
    """UMAP으로 2D 투영 (군집 구조가 더 명확)."""
    try:
        import umap
    except ImportError:
        print("  [경고] umap-learn이 없어 PCA로 대체합니다. (pip install umap-learn)")
        return embed_2d_pca(features)

    n_neighbors = min(15, features.shape[0] - 1)
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
    return reducer.fit_transform(features)


# ---------------------------------------------------------------------------
# 그룹 레이블 생성
# ---------------------------------------------------------------------------

def build_group_labels(
    n: int,
    exact_groups: dict,
    near_groups: list[list[int]],
    valid_paths: list[Path],
) -> tuple[np.ndarray, dict]:
    """
    각 이미지에 그룹 ID를 부여합니다.
    -1: 고유 이미지 (중복 없음)
    0, 1, 2, ...: 중복 그룹 ID

    Returns:
        labels: (N,) int 배열
        group_info: {group_id: {"type": "exact"|"near", "size": int}}
    """
    labels = np.full(n, -1, dtype=int)
    group_info = {}
    group_id = 0

    path_to_idx = {p: i for i, p in enumerate(valid_paths)}

    # exact 중복
    for ps in exact_groups.values():
        indices = [path_to_idx[p] for p in ps if p in path_to_idx]
        if len(indices) >= 2:
            for idx in indices:
                labels[idx] = group_id
            group_info[group_id] = {"type": "exact", "size": len(indices)}
            group_id += 1

    # near 중복 (아직 라벨 없는 것만)
    for g in near_groups:
        if any(labels[i] == -1 for i in g):
            for i in g:
                if labels[i] == -1:
                    labels[i] = group_id
            group_info[group_id] = {"type": "near", "size": len(g)}
            group_id += 1

    return labels, group_info


# ---------------------------------------------------------------------------
# 메인 시각화
# ---------------------------------------------------------------------------

def visualize(
    data_dir: str,
    output_path: str,
    n_components: int,
    hamming_threshold: int,
    image_size: int,
    method: str,
    show_thumbnails: bool,
    thumbnail_size: int,
):
    # 1. 이미지 수집 및 특징 추출
    print(f"\n이미지 수집 중: {data_dir}")
    paths = collect_image_paths(data_dir)
    print(f"  발견된 이미지 수: {len(paths):,}")
    if not paths:
        print("  [오류] 이미지를 찾을 수 없습니다.")
        sys.exit(1)

    features, valid_paths = extract_features(paths, image_size)
    n = len(valid_paths)
    print(f"  유효 이미지: {n:,}")

    # 2. PCA hash 계산 + 중복 탐지
    print("\nPCA hash 계산 중...")
    hashes, pca_model = compute_pca_hashes(features, n_components)

    print("Hamming distance 계산 중...")
    dist_matrix = hamming_distance_matrix(hashes)
    near_groups = find_duplicate_groups(dist_matrix, valid_paths, hamming_threshold)

    print("MD5 정확한 중복 탐지 중...")
    exact_groups = md5_exact_duplicates(valid_paths)

    # 3. 2D 임베딩
    print(f"\n2D 임베딩 ({method.upper()}) 중...")
    if method == "umap":
        coords = embed_2d_umap(features)
    else:
        coords = embed_2d_pca(features)

    # 4. 그룹 레이블
    labels, group_info = build_group_labels(n, exact_groups, near_groups, valid_paths)

    n_dup_groups = len(group_info)
    n_dup_images = (labels >= 0).sum()
    n_unique = (labels == -1).sum()

    print(f"\n결과 요약:")
    print(f"  고유 이미지: {n_unique:,}")
    print(f"  중복 그룹 수: {n_dup_groups:,}")
    print(f"  중복 관련 이미지: {n_dup_images:,}")

    # 5. 시각화
    _render_plot(
        coords=coords,
        labels=labels,
        group_info=group_info,
        valid_paths=valid_paths,
        features=features,
        output_path=output_path,
        method=method,
        data_dir=data_dir,
        n_components=n_components,
        hamming_threshold=hamming_threshold,
        show_thumbnails=show_thumbnails,
        thumbnail_size=thumbnail_size,
        n_unique=int(n_unique),
        n_dup_groups=n_dup_groups,
        n_dup_images=int(n_dup_images),
    )


def _render_plot(
    coords, labels, group_info, valid_paths, features,
    output_path, method, data_dir,
    n_components, hamming_threshold,
    show_thumbnails, thumbnail_size,
    n_unique, n_dup_groups, n_dup_images,
):
    n_groups = len(group_info)

    # 컬러맵: 중복 그룹에 색상 할당
    cmap = plt.get_cmap("tab20" if n_groups <= 20 else "hsv")
    group_colors = {gid: cmap(i / max(n_groups, 1)) for i, gid in enumerate(group_info)}

    fig_w = 14 if show_thumbnails else 10
    fig, ax = plt.subplots(figsize=(fig_w, 8))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    # --- 고유 이미지 (회색) ---
    unique_mask = labels == -1
    ax.scatter(
        coords[unique_mask, 0], coords[unique_mask, 1],
        c="#555577", s=18, alpha=0.4, linewidths=0,
        zorder=2, label=f"고유 이미지 ({n_unique:,}장)",
    )

    # --- 중복 그룹 ---
    for gid, info in group_info.items():
        mask = labels == gid
        color = group_colors[gid]
        marker = "o" if info["type"] == "near" else "D"
        edge = "white" if info["type"] == "exact" else "none"
        size = 80 if info["type"] == "exact" else 55

        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[color], s=size, alpha=0.9,
            linewidths=0.8 if info["type"] == "exact" else 0,
            edgecolors=edge,
            marker=marker, zorder=4,
        )

        # 같은 그룹 이미지 간 선 연결
        idxs = np.where(mask)[0]
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                ax.plot(
                    [coords[idxs[i], 0], coords[idxs[j], 0]],
                    [coords[idxs[i], 1], coords[idxs[j], 1]],
                    color=color, alpha=0.35, linewidth=0.8, zorder=3,
                )

    # --- 썸네일 오버레이 ---
    if show_thumbnails and n_dup_images <= 60:
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        dup_idxs = np.where(labels >= 0)[0]
        for idx in dup_idxs:
            try:
                img = Image.open(valid_paths[idx]).convert("RGB")
                img = img.resize((thumbnail_size, thumbnail_size), Image.BILINEAR)
                arr = np.asarray(img)
                im = OffsetImage(arr, zoom=1.0)
                ab = AnnotationBbox(
                    im, (coords[idx, 0], coords[idx, 1]),
                    frameon=True,
                    bboxprops=dict(
                        edgecolor=group_colors.get(labels[idx], "white"),
                        linewidth=1.5,
                        boxstyle="round,pad=0.1",
                    ),
                    zorder=5,
                )
                ax.add_artist(ab)
            except Exception:
                pass

    # --- 범례 ---
    legend_handles = [
        mpatches.Patch(color="#555577", alpha=0.6, label=f"고유 이미지 ({n_unique:,}장)"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="white",
                   markersize=8, label=f"정확한 중복 (MD5)", linewidth=0),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
                   markersize=8, label=f"유사 중복 (PCA Hash)", linewidth=0),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left", fontsize=9,
        facecolor="#1a1a2e", edgecolor="#888", labelcolor="white",
    )

    # --- 타이틀 및 레이블 ---
    ax.set_title(
        f"PCA Hash Deduplication — {Path(data_dir).name}\n"
        f"{method.upper()} 2D  |  hash bits={n_components}  |  Hamming ≤ {hamming_threshold}  |  "
        f"중복 그룹 {n_dup_groups}개 ({n_dup_images}장)",
        color="white", fontsize=11, pad=12,
    )
    ax.set_xlabel(f"{method.upper()} dim-1", color="#aaa", fontsize=9)
    ax.set_ylabel(f"{method.upper()} dim-2", color="#aaa", fontsize=9)
    ax.tick_params(colors="#666")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    # --- 통계 텍스트박스 ---
    stats_text = (
        f"전체: {len(valid_paths):,}장\n"
        f"고유: {n_unique:,}장\n"
        f"중복 그룹: {n_dup_groups:,}개\n"
        f"중복 이미지: {n_dup_images:,}장\n"
        f"제거 가능: {max(0, n_dup_images - n_dup_groups):,}장"
    )
    ax.text(
        0.99, 0.01, stats_text,
        transform=ax.transAxes, fontsize=8.5,
        verticalalignment="bottom", horizontalalignment="right",
        color="white",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a2e", edgecolor="#555", alpha=0.85),
    )

    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n시각화 저장 완료: {out.resolve()}")
    plt.close()

    # 서브플롯: Hamming distance 히트맵 (소규모 데이터셋용)
    if len(valid_paths) <= 200:
        _render_heatmap(valid_paths, labels, group_info, group_colors,
                        hamming_threshold, n_components, image_size=64,
                        output_path=str(out.with_name(out.stem + "_heatmap" + out.suffix)))


def _render_heatmap(
    valid_paths, labels, group_info, group_colors,
    hamming_threshold, n_components, image_size,
    output_path,
):
    """이미지 간 Hamming distance 히트맵 (소규모 데이터셋용)."""
    from pca_dedup import extract_features, compute_pca_hashes, hamming_distance_matrix

    # 중복 있는 이미지만
    dup_idxs = np.where(labels >= 0)[0]
    if len(dup_idxs) < 2:
        return

    dup_paths = [valid_paths[i] for i in dup_idxs]
    dup_labels = labels[dup_idxs]

    features, _ = extract_features(dup_paths, image_size)
    hashes, _ = compute_pca_hashes(features, n_components)
    dist = hamming_distance_matrix(hashes)

    n = len(dup_paths)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.4), max(5, n * 0.4)))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    im = ax.imshow(dist, cmap="RdYlGn_r", vmin=0, vmax=n_components)
    plt.colorbar(im, ax=ax, label="Hamming distance")

    # 임계값 이하 셀에 강조선
    for i in range(n):
        for j in range(n):
            if i != j and dist[i, j] <= hamming_threshold:
                rect = plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=False, edgecolor="cyan", linewidth=1.5,
                )
                ax.add_patch(rect)

    short_names = [p.name[:15] for p in dup_paths]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7, color="white")
    ax.set_yticklabels(short_names, fontsize=7, color="white")

    # 그룹별 색상 사이드바
    for i, (idx, lbl) in enumerate(zip(dup_idxs, dup_labels)):
        color = group_colors.get(lbl, "gray")
        ax.add_patch(plt.Rectangle((-1.5, i - 0.5), 0.8, 1, color=color, clip_on=False))

    ax.set_title(
        f"Hamming Distance Heatmap (중복 이미지, Hamming ≤ {hamming_threshold} = cyan 강조)",
        color="white", fontsize=10, pad=10,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"히트맵 저장 완료: {Path(output_path).resolve()}")
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="PCA Hash Deduplication 시각화",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python visualize_dedup.py --data_dir ./images
  python visualize_dedup.py --data_dir ./images --method umap --thumbnails
  python visualize_dedup.py --data_dir ./images --output scatter.png
""",
    )
    parser.add_argument("--data_dir", required=True, help="이미지 디렉토리")
    parser.add_argument("--output", default="dedup_visualization.png", help="출력 이미지 경로")
    parser.add_argument("--method", choices=["pca", "umap"], default="pca",
                        help="2D 임베딩 방법 (기본값: pca)")
    parser.add_argument("--n_components", type=int, default=32, help="PCA 해시 비트 수")
    parser.add_argument("--hamming_threshold", type=int, default=2, help="Hamming 거리 임계값")
    parser.add_argument("--image_size", type=int, default=64, help="리사이즈 크기")
    parser.add_argument("--thumbnails", action="store_true",
                        help="중복 이미지에 썸네일 오버레이 (이미지 수 ≤ 60일 때)")
    parser.add_argument("--thumbnail_size", type=int, default=32, help="썸네일 픽셀 크기")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    visualize(
        data_dir=args.data_dir,
        output_path=args.output,
        n_components=args.n_components,
        hamming_threshold=args.hamming_threshold,
        image_size=args.image_size,
        method=args.method,
        show_thumbnails=args.thumbnails,
        thumbnail_size=args.thumbnail_size,
    )
