"""
PCA Hash Deduplication - 분포 시각화
=====================================
이미지를 2D 공간에 투영하여 중복 그룹의 분포를 클러스터링처럼 표현합니다.

  - 고유 이미지: 회색 점
  - 중복 그룹: 그룹별 색상, 같은 그룹끼리 선으로 연결
  - 데이터셋이 작으면 Hamming distance heatmap도 함께 생성
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager
import numpy as np
from sklearn.decomposition import PCA

# 한글 폰트 자동 탐지
for _f in ["WenQuanYi Zen Hei", "Unifont", "NanumGothic", "Malgun Gothic", "AppleGothic"]:
    if any(_f.lower() in f.name.lower() for f in font_manager.fontManager.ttflist):
        matplotlib.rcParams["font.family"] = _f
        break

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

def embed_2d(features: np.ndarray, method: str) -> np.ndarray:
    if method == "umap":
        try:
            import umap
            n_neighbors = min(15, features.shape[0] - 1)
            return umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42).fit_transform(features)
        except ImportError:
            print("  [경고] umap-learn 없음, PCA로 대체합니다.")
    pca = PCA(n_components=min(2, features.shape[0], features.shape[1]), random_state=42)
    return pca.fit_transform(features)


# ---------------------------------------------------------------------------
# 그룹 레이블 생성
# ---------------------------------------------------------------------------

def build_labels(n, exact_groups, near_groups, valid_paths):
    """
    반환:
      labels[i] = -1  → 고유 이미지
      labels[i] = k   → k번 중복 그룹
      group_meta[k]   = {"type": "exact"|"near", "size": int}
    """
    labels = np.full(n, -1, dtype=int)
    group_meta = {}
    gid = 0
    path_to_idx = {p: i for i, p in enumerate(valid_paths)}

    for ps in exact_groups.values():
        idxs = [path_to_idx[p] for p in ps if p in path_to_idx]
        if len(idxs) >= 2:
            for i in idxs:
                labels[i] = gid
            group_meta[gid] = {"type": "exact", "size": len(idxs)}
            gid += 1

    for g in near_groups:
        if any(labels[i] == -1 for i in g):
            for i in g:
                if labels[i] == -1:
                    labels[i] = gid
            group_meta[gid] = {"type": "near", "size": len(g)}
            gid += 1

    return labels, group_meta


# ---------------------------------------------------------------------------
# Scatter plot
# ---------------------------------------------------------------------------

def plot_scatter(coords, labels, group_meta, output_path, method, data_dir,
                 n_components, threshold):
    n_groups = len(group_meta)
    cmap = plt.get_cmap("tab20" if n_groups <= 20 else "hsv")
    group_colors = {gid: cmap(i / max(n_groups, 1)) for i, gid in enumerate(group_meta)}

    n_unique = int((labels == -1).sum())
    n_dup_images = int((labels >= 0).sum())
    n_removable = max(0, n_dup_images - n_groups)

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    # 고유 이미지
    mask = labels == -1
    ax.scatter(coords[mask, 0], coords[mask, 1],
               c="#6677aa", s=20, alpha=0.45, linewidths=0, zorder=2)

    # 중복 그룹
    for gid, meta in group_meta.items():
        mask = labels == gid
        color = group_colors[gid]
        marker = "D" if meta["type"] == "exact" else "o"
        size = 90 if meta["type"] == "exact" else 65

        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[color], s=size, alpha=0.9, marker=marker,
                   edgecolors="white" if meta["type"] == "exact" else "none",
                   linewidths=0.7, zorder=4)

        # 같은 그룹끼리 선 연결
        idxs = np.where(mask)[0]
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                ax.plot([coords[idxs[i], 0], coords[idxs[j], 0]],
                        [coords[idxs[i], 1], coords[idxs[j], 1]],
                        color=color, alpha=0.4, linewidth=1.0, zorder=3)

    # 범례
    legend_handles = [
        mpatches.Patch(color="#6677aa", alpha=0.6, label=f"고유 이미지  ({n_unique:,}장)"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="white",
                   markersize=8, linewidth=0, label="정확한 중복 (MD5)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
                   markersize=8, linewidth=0, label="유사 중복 (PCA Hash)"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9,
              facecolor="#1a1a2e", edgecolor="#555", labelcolor="white")

    ax.set_title(
        f"PCA Hash Deduplication — {Path(data_dir).name}\n"
        f"{method.upper()} 2D  |  hash bits={n_components}  |  Hamming ≤ {threshold}  |  "
        f"중복 그룹 {n_groups}개 ({n_dup_images}장, 제거 가능 {n_removable}장)",
        color="white", fontsize=11, pad=12,
    )
    ax.set_xlabel(f"{method.upper()} dim-1", color="#aaa", fontsize=9)
    ax.set_ylabel(f"{method.upper()} dim-2", color="#aaa", fontsize=9)
    ax.tick_params(colors="#666")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    stats = (f"전체: {len(labels):,}장\n"
             f"고유: {n_unique:,}장\n"
             f"중복 그룹: {n_groups:,}개\n"
             f"중복 이미지: {n_dup_images:,}장\n"
             f"제거 가능: {n_removable:,}장")
    ax.text(0.99, 0.01, stats, transform=ax.transAxes, fontsize=9,
            va="bottom", ha="right", color="white",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a2e",
                      edgecolor="#555", alpha=0.9))

    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Scatter plot 저장: {out.resolve()}")


# ---------------------------------------------------------------------------
# Heatmap (소규모 데이터셋용, ≤ 300장)
# ---------------------------------------------------------------------------

def plot_heatmap(valid_paths, labels, group_meta, group_colors,
                 n_components, image_size, threshold, output_path):
    dup_idxs = np.where(labels >= 0)[0]
    if len(dup_idxs) < 2:
        return

    dup_paths = [valid_paths[i] for i in dup_idxs]
    features, _ = extract_features(dup_paths, image_size)
    hashes, _ = compute_pca_hashes(features, n_components)
    dist = hamming_distance_matrix(hashes)

    n = len(dup_paths)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.45), max(5, n * 0.45)))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    im = ax.imshow(dist, cmap="RdYlGn_r", vmin=0, vmax=n_components)
    plt.colorbar(im, ax=ax, label="Hamming distance")

    for i in range(n):
        for j in range(n):
            if i != j and dist[i, j] <= threshold:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           fill=False, edgecolor="cyan", linewidth=1.5))

    names = [p.name[:15] for p in dup_paths]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7, color="white")
    ax.set_yticklabels(names, fontsize=7, color="white")

    for i, idx in enumerate(dup_idxs):
        ax.add_patch(plt.Rectangle((-1.5, i - 0.5), 0.8, 1,
                                   color=group_colors.get(labels[idx], "gray"),
                                   clip_on=False))

    ax.set_title(f"Hamming Distance Heatmap  (cyan = Hamming ≤ {threshold})",
                 color="white", fontsize=10, pad=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Heatmap 저장: {Path(output_path).resolve()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PCA Hash Deduplication 분포 시각화")
    parser.add_argument("--data_dir", required=True, help="이미지 디렉토리")
    parser.add_argument("--output", default="dedup_visualization.png", help="출력 파일 경로")
    parser.add_argument("--method", choices=["pca", "umap"], default="pca",
                        help="2D 임베딩 방법 (기본: pca | 대규모엔 umap 권장)")
    parser.add_argument("--n_components", type=int, default=32, help="PCA 해시 비트 수")
    parser.add_argument("--hamming_threshold", type=int, default=2, help="Hamming 거리 임계값")
    parser.add_argument("--image_size", type=int, default=64, help="리사이즈 크기")
    args = parser.parse_args()

    print(f"\n이미지 수집: {args.data_dir}")
    paths = collect_image_paths(args.data_dir)
    print(f"  발견: {len(paths):,}장")
    if not paths:
        print("이미지를 찾을 수 없습니다.")
        sys.exit(1)

    features, valid_paths = extract_features(paths, args.image_size)

    print("PCA hash 계산 중...")
    hashes, _ = compute_pca_hashes(features, args.n_components)
    dist_matrix = hamming_distance_matrix(hashes)
    near_groups = find_duplicate_groups(dist_matrix, valid_paths, args.hamming_threshold)

    print("MD5 중복 탐지 중...")
    exact_groups = md5_exact_duplicates(valid_paths)

    print(f"2D 임베딩 ({args.method.upper()}) 중...")
    coords = embed_2d(features, args.method)

    labels, group_meta = build_labels(len(valid_paths), exact_groups, near_groups, valid_paths)
    n_groups = len(group_meta)
    cmap = plt.get_cmap("tab20" if n_groups <= 20 else "hsv")
    group_colors = {gid: cmap(i / max(n_groups, 1)) for i, gid in enumerate(group_meta)}

    out = Path(args.output)
    plot_scatter(coords, labels, group_meta, str(out),
                 args.method, args.data_dir, args.n_components, args.hamming_threshold)

    if len(valid_paths) <= 300:
        plot_heatmap(valid_paths, labels, group_meta, group_colors,
                     args.n_components, args.image_size, args.hamming_threshold,
                     str(out.with_name(out.stem + "_heatmap" + out.suffix)))

    n_unique = int((labels == -1).sum())
    n_dup = int((labels >= 0).sum())
    print(f"\n결과: 전체 {len(valid_paths):,}장 | 고유 {n_unique:,}장 | "
          f"중복 그룹 {n_groups}개 ({n_dup}장)")


if __name__ == "__main__":
    main()
