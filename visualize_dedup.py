"""
PCA Hash Deduplication - 분포 시각화
=====================================
이미지를 2D 공간에 투영하여 중복 그룹의 분포를 클러스터링처럼 표현합니다.

  - 고유 이미지: 회색 점
  - 중복 그룹: 그룹별 색상 + convex hull 영역 표시
  - 우측 패널: 중복 통계 요약 차트
  - 소규모 데이터셋: Hamming distance heatmap 추가 생성
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["axes.unicode_minus"] = False  # 마이너스 부호 깨짐 방지

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
from scipy.spatial import ConvexHull
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

# ── 색상 팔레트 ───────────────────────────────────────────────
UNIQUE_COLOR   = "#bdc3c7"   # 고유 이미지 (연회색)
BG_COLOR       = "#ffffff"   # 배경
PANEL_COLOR    = "#f8f9fa"   # 사이드 패널 배경
GRID_COLOR     = "#e0e0e0"   # 그리드
TEXT_COLOR     = "#2c3e50"   # 기본 텍스트
EXACT_EDGE     = "#e74c3c"   # 정확한 중복 테두리색


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
# Convex hull helper
# ---------------------------------------------------------------------------

def draw_hull(ax, pts, color, alpha_fill=0.15, alpha_edge=0.6, lw=1.5):
    """3개 이상 점이면 convex hull, 2개면 타원, 1개면 원을 그립니다."""
    if len(pts) >= 3:
        try:
            hull = ConvexHull(pts)
            verts = np.append(hull.vertices, hull.vertices[0])
            # 약간 팽창 (padding)
            cx, cy = pts.mean(axis=0)
            padded = pts[hull.vertices] + (pts[hull.vertices] - [cx, cy]) * 0.25
            padded = np.vstack([padded, padded[0]])
            ax.fill(padded[:, 0], padded[:, 1], color=color, alpha=alpha_fill, zorder=2)
            ax.plot(padded[:, 0], padded[:, 1], color=color, alpha=alpha_edge,
                    linewidth=lw, zorder=3)
            return
        except Exception:
            pass
    if len(pts) == 2:
        cx, cy = pts.mean(axis=0)
        dx, dy = pts[1] - pts[0]
        w = max(np.hypot(dx, dy) * 0.7, 20)
        angle = np.degrees(np.arctan2(dy, dx))
        ell = mpatches.Ellipse((cx, cy), width=w * 2, height=max(w * 0.4, 15),
                               angle=angle, color=color, alpha=alpha_fill, zorder=2)
        ax.add_patch(ell)
        ax.plot(pts[:, 0], pts[:, 1], color=color, alpha=alpha_edge,
                linewidth=lw, linestyle="--", zorder=3)
    else:
        ax.scatter(pts[:, 0], pts[:, 1], s=300, color=color,
                   alpha=alpha_fill * 4, zorder=2, linewidths=0)


# ---------------------------------------------------------------------------
# Scatter plot (메인)
# ---------------------------------------------------------------------------

def plot_scatter(coords, labels, group_meta, valid_paths, output_path,
                 method, data_dir, n_components, threshold):
    n_groups = len(group_meta)
    n_unique = int((labels == -1).sum())
    n_dup_images = int((labels >= 0).sum())
    n_removable = max(0, n_dup_images - n_groups)
    total = len(labels)

    cmap = plt.get_cmap("tab10" if n_groups <= 10 else
                        "tab20" if n_groups <= 20 else "hsv")
    group_colors = {gid: cmap(i / max(n_groups, 1)) for i, gid in enumerate(group_meta)}

    # ── 레이아웃: scatter(좌) + 통계 패널(우) ─────────────────
    fig = plt.figure(figsize=(14, 7), facecolor=BG_COLOR)
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.05)
    ax = fig.add_subplot(gs[0])
    ax_stat = fig.add_subplot(gs[1])

    ax.set_facecolor(PANEL_COLOR)
    ax_stat.set_facecolor(PANEL_COLOR)

    # ── 그리드 ────────────────────────────────────────────────
    ax.grid(True, color=GRID_COLOR, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    # ── 고유 이미지 ───────────────────────────────────────────
    mask_unique = labels == -1
    if mask_unique.any():
        ax.scatter(coords[mask_unique, 0], coords[mask_unique, 1],
                   c=UNIQUE_COLOR, s=25, alpha=0.6, linewidths=0,
                   zorder=4, label=f"고유 이미지 ({n_unique:,}장)")

    # ── 중복 그룹 ─────────────────────────────────────────────
    for gid, meta in group_meta.items():
        mask = labels == gid
        pts = coords[mask]
        color = group_colors[gid]
        is_exact = meta["type"] == "exact"

        # convex hull / 영역
        draw_hull(ax, pts, color)

        # 점
        ax.scatter(pts[:, 0], pts[:, 1],
                   c=[color], s=80, alpha=0.95, zorder=5,
                   marker="D" if is_exact else "o",
                   edgecolors=EXACT_EDGE if is_exact else "white",
                   linewidths=1.2 if is_exact else 0.5)

    # ── 범례 ──────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(facecolor=UNIQUE_COLOR, edgecolor="#999",
                       label=f"고유 이미지 ({n_unique:,}장)"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="#555",
                   markersize=9, linewidth=0,
                   markeredgecolor=EXACT_EDGE, markeredgewidth=1.2,
                   label="정확한 중복 (MD5)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#555",
                   markersize=9, linewidth=0, markeredgewidth=0.5,
                   markeredgecolor="white", label="유사 중복 (PCA Hash)"),
        mpatches.Patch(facecolor="#aaa", alpha=0.25, edgecolor="#888",
                       label="중복 그룹 영역 (convex hull)"),
    ]
    ax.legend(handles=legend_handles, loc="best", fontsize=9,
              facecolor="white", edgecolor="#ccc", framealpha=0.9,
              labelcolor=TEXT_COLOR)

    ax.set_title(f"이미지 분포 — {Path(data_dir).name}",
                 fontsize=13, fontweight="bold", color=TEXT_COLOR, pad=10)
    ax.set_xlabel(f"{method.upper()} dim-1", color=TEXT_COLOR, fontsize=9)
    ax.set_ylabel(f"{method.upper()} dim-2", color=TEXT_COLOR, fontsize=9)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)

    # 파라미터 메모
    ax.text(0.01, 0.01,
            f"hash bits={n_components}  |  Hamming ≤ {threshold}  |  {method.upper()} 2D",
            transform=ax.transAxes, fontsize=7.5, color="#888",
            va="bottom", ha="left")

    # ── 통계 패널 ─────────────────────────────────────────────
    ax_stat.axis("off")
    ax_stat.set_xlim(0, 1)
    ax_stat.set_ylim(0, 1)

    # 제목
    ax_stat.text(0.5, 0.97, "중복 현황 요약", ha="center", va="top",
                 fontsize=11, fontweight="bold", color=TEXT_COLOR,
                 transform=ax_stat.transAxes)

    # 카드 스타일 통계
    cards = [
        ("전체 이미지",    f"{total:,}장",        "#3498db"),
        ("고유 이미지",    f"{n_unique:,}장",     "#2ecc71"),
        ("중복 그룹 수",   f"{n_groups:,}개",     "#e67e22"),
        ("중복 이미지",    f"{n_dup_images:,}장", "#e74c3c"),
        ("제거 가능",      f"{n_removable:,}장",  "#9b59b6"),
        ("중복 비율",
         f"{n_dup_images/max(total,1)*100:.1f}%", "#1abc9c"),
    ]

    card_h = 0.11
    card_gap = 0.025
    start_y = 0.88

    for i, (label, value, color) in enumerate(cards):
        y = start_y - i * (card_h + card_gap)
        # 카드 배경
        rect = FancyBboxPatch((0.05, y - card_h), 0.90, card_h,
                              boxstyle="round,pad=0.01",
                              facecolor=color, alpha=0.12,
                              edgecolor=color, linewidth=1.2,
                              transform=ax_stat.transAxes, clip_on=False)
        ax_stat.add_patch(rect)
        ax_stat.text(0.12, y - card_h / 2, label,
                     ha="left", va="center", fontsize=8.5,
                     color=TEXT_COLOR, transform=ax_stat.transAxes)
        ax_stat.text(0.92, y - card_h / 2, value,
                     ha="right", va="center", fontsize=10,
                     fontweight="bold", color=color,
                     transform=ax_stat.transAxes)

    # 파이 차트 (고유 vs 중복)
    pie_y = start_y - len(cards) * (card_h + card_gap) - 0.04
    ax_pie = fig.add_axes([
        ax_stat.get_position().x0 + 0.02,
        ax.get_position().y0,
        ax_stat.get_position().width - 0.02,
        pie_y,
    ])
    if n_dup_images > 0:
        sizes  = [n_unique, n_removable, n_groups]
        clrs   = ["#bdc3c7", "#e74c3c", "#e67e22"]
        labels_pie = ["고유", "중복\n(제거)", "중복\n(유지)"]
        wedges, _, autotexts = ax_pie.pie(
            sizes, labels=labels_pie, colors=clrs,
            autopct="%1.0f%%", startangle=90,
            pctdistance=0.75,
            textprops={"fontsize": 7.5, "color": TEXT_COLOR},
            wedgeprops={"linewidth": 0.8, "edgecolor": "white"},
        )
        for at in autotexts:
            at.set_fontsize(7)
            at.set_color(TEXT_COLOR)
    else:
        ax_pie.pie([1], colors=["#2ecc71"], labels=["전체 고유"],
                   textprops={"fontsize": 8, "color": TEXT_COLOR})
    ax_pie.set_title("구성 비율", fontsize=8, color=TEXT_COLOR, pad=4)

    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=BG_COLOR)
    plt.close()
    print(f"Scatter plot 저장: {Path(output_path).resolve()}")


# ---------------------------------------------------------------------------
# Heatmap (소규모 데이터셋용, ≤ 300장)
# ---------------------------------------------------------------------------

def plot_heatmap(valid_paths, labels, group_meta, group_colors,
                 n_components, image_size, threshold, output_path):
    dup_idxs = np.where(labels >= 0)[0]
    if len(dup_idxs) < 2:
        return

    # 같은 그룹끼리 인접하도록 정렬
    dup_idxs = sorted(dup_idxs, key=lambda i: (labels[i], i))
    dup_paths = [valid_paths[i] for i in dup_idxs]
    dup_labels = [labels[i] for i in dup_idxs]

    features, _ = extract_features(dup_paths, image_size)
    hashes, _ = compute_pca_hashes(features, n_components)
    dist = hamming_distance_matrix(hashes)

    n = len(dup_paths)
    cell = max(0.5, min(0.9, 9.0 / n))
    fig_size = max(6, n * cell + 2)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.88),
                           facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # 히트맵
    im = ax.imshow(dist, cmap="YlOrRd", vmin=0, vmax=n_components,
                   aspect="auto")
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Hamming distance", fontsize=9, color=TEXT_COLOR)
    cbar.ax.tick_params(labelsize=8, colors=TEXT_COLOR)

    # 셀 안에 숫자 표시 (이미지 수 ≤ 25일 때)
    if n <= 25:
        for i in range(n):
            for j in range(n):
                val = dist[i, j]
                txt_color = "white" if val > n_components * 0.55 else TEXT_COLOR
                ax.text(j, i, str(int(val)), ha="center", va="center",
                        fontsize=max(5, 9 - n // 5), color=txt_color,
                        fontweight="bold" if val <= threshold else "normal")

    # near-dup 강조 (임계값 이하)
    for i in range(n):
        for j in range(n):
            if i != j and dist[i, j] <= threshold:
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=False, edgecolor="#2980b9", linewidth=2.0, zorder=5))

    # 그룹 구분선
    prev_lbl = dup_labels[0]
    for k in range(1, n):
        if dup_labels[k] != prev_lbl:
            ax.axhline(k - 0.5, color="#555", linewidth=1.5, zorder=6)
            ax.axvline(k - 0.5, color="#555", linewidth=1.5, zorder=6)
        prev_lbl = dup_labels[k]

    # 축 레이블
    names = [p.name[:18] for p in dup_paths]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right",
                       fontsize=max(6, 9 - n // 8), color=TEXT_COLOR)
    ax.set_yticklabels(names, fontsize=max(6, 9 - n // 8), color=TEXT_COLOR)

    # 그룹 색상 사이드바
    for k, (idx, lbl) in enumerate(zip(dup_idxs, dup_labels)):
        color = group_colors.get(lbl, "#ccc")
        ax.add_patch(plt.Rectangle((-2.2, k - 0.5), 1.0, 1.0,
                                   color=color, clip_on=False, zorder=7))
        # 그룹 타입 표시
        marker = "D" if group_meta.get(lbl, {}).get("type") == "exact" else "o"
        ax.plot(-1.7, k, marker=marker, color="white",
                markersize=4, clip_on=False, zorder=8)

    # 우측 사이드바 (그룹 ID)
    prev_lbl = None
    group_start = 0
    for k, lbl in enumerate(dup_labels + [None]):
        if lbl != prev_lbl and prev_lbl is not None:
            mid = (group_start + k - 1) / 2
            gtype = group_meta.get(prev_lbl, {}).get("type", "")
            label_txt = f"G{prev_lbl+1}\n({'정확' if gtype=='exact' else '유사'})"
            ax.text(n + 0.3, mid, label_txt, ha="left", va="center",
                    fontsize=max(5, 7 - n // 10), color=group_colors.get(prev_lbl, "#555"),
                    fontweight="bold", clip_on=False)
            group_start = k
        prev_lbl = lbl

    ax.set_title(
        f"Hamming Distance Heatmap  |  파란 테두리 = Hamming ≤ {threshold} (near-dup)  |  "
        f"구분선 = 그룹 경계",
        fontsize=10, color=TEXT_COLOR, pad=12, fontweight="bold")

    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
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
    cmap = plt.get_cmap("tab10" if n_groups <= 10 else
                        "tab20" if n_groups <= 20 else "hsv")
    group_colors = {gid: cmap(i / max(n_groups, 1)) for i, gid in enumerate(group_meta)}

    out = Path(args.output)
    plot_scatter(coords, labels, group_meta, valid_paths, str(out),
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
