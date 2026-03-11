"""
PCA Hash Deduplication - 논문용 시각화
=======================================
학술 논문 게재 기준의 고품질 그래프를 생성합니다.

핵심 시각 원칙:
  - 같은 중복 그룹 = 같은 색 + 같은 마커
  - 그룹 경계: PCA 주축 기반 신뢰 타원 (2-sigma)
  - 그룹 레이블: 중심점에 G1, G2, ... 표기
  - Okabe-Ito colorblind-safe 팔레트
  - 300 DPI + PDF 동시 저장
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "axes.unicode_minus": False,
    "pdf.fonttype":       42,   # TrueType — Illustrator 편집 가능
    "ps.fonttype":        42,
    "font.family":        "sans-serif",
    "font.sans-serif":    ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size":          9,
    "axes.titlesize":     10,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "figure.dpi":         300,
    "savefig.dpi":        300,
})

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from matplotlib import font_manager
import numpy as np
from sklearn.decomposition import PCA

# 한글 폰트 탐지
_KO_FONT = None
for _f in ["WenQuanYi Zen Hei", "NanumGothic", "Malgun Gothic", "AppleGothic"]:
    if any(_f.lower() in f.name.lower() for f in font_manager.fontManager.ttflist):
        _KO_FONT = _f
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

# ── Okabe-Ito colorblind-safe 팔레트 ─────────────────────────────────────────
OKABE_ITO = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442",
    "#0072B2", "#D55E00", "#CC79A7", "#000000",
]
UNIQUE_COLOR = "#BBBBBB"


def _group_color(gid: int, n_groups: int) -> str:
    if n_groups <= len(OKABE_ITO):
        return OKABE_ITO[gid % len(OKABE_ITO)]
    return plt.get_cmap("tab20")(gid / max(n_groups, 1))


# ---------------------------------------------------------------------------
# 2D 임베딩
# ---------------------------------------------------------------------------

def embed_2d(features: np.ndarray, method: str) -> np.ndarray:
    if method == "umap":
        try:
            import umap
            n_neighbors = min(15, features.shape[0] - 1)
            return umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                             random_state=42).fit_transform(features)
        except ImportError:
            print("  [경고] umap-learn 없음, PCA로 대체합니다.")
    pca = PCA(n_components=min(2, features.shape[0], features.shape[1]),
              random_state=42)
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
# 신뢰 타원 — PCA 주축 기반 2-sigma
# ---------------------------------------------------------------------------

def _draw_ellipse(ax, pts, color, n_sigma=2.0):
    """
    pts: (k, 2) array
    - k >= 2: 공분산 행렬로 타원
    - k == 1: 작은 원
    """
    if len(pts) == 1:
        ax.scatter(pts[:, 0], pts[:, 1], s=400, color=color,
                   alpha=0.15, linewidths=0, zorder=2)
        return

    cx, cy = pts.mean(axis=0)
    cov = np.cov(pts.T) if pts.shape[0] > 1 else np.eye(2) * 1e-6
    if cov.ndim == 0:
        cov = np.diag([float(cov), 1e-6])

    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0)
    w, h = 2 * n_sigma * np.sqrt(vals)
    angle = np.degrees(np.arctan2(*vecs[:, -1][::-1]))

    # 채움 (연한 색)
    ax.add_patch(Ellipse(xy=(cx, cy), width=max(w, 1e-3), height=max(h, 1e-3),
                         angle=angle, facecolor=color, alpha=0.12,
                         edgecolor="none", zorder=2))
    # 테두리
    ax.add_patch(Ellipse(xy=(cx, cy), width=max(w, 1e-3), height=max(h, 1e-3),
                         angle=angle, facecolor="none",
                         edgecolor=color, linewidth=1.2,
                         alpha=0.75, zorder=3))


# ---------------------------------------------------------------------------
# Scatter plot  — (a) 분포  (b) 그룹 크기 막대
# ---------------------------------------------------------------------------

def plot_scatter(coords, labels, group_meta, valid_paths, output_path,
                 method, data_dir, n_components, threshold):
    n_groups    = len(group_meta)
    n_unique    = int((labels == -1).sum())
    n_dup_imgs  = int((labels >= 0).sum())
    n_removable = max(0, n_dup_imgs - n_groups)
    total       = len(labels)

    group_colors = {gid: _group_color(gid, n_groups) for gid in group_meta}

    # 논문 2-column 너비 기준: 7.16 × 3.5 inch
    fig, (ax, ax_bar) = plt.subplots(
        1, 2, figsize=(7.16, 3.5),
        gridspec_kw={"width_ratios": [3, 1], "wspace": 0.40},
    )

    # ── (a) Scatter ───────────────────────────────────────────────────────
    ax.set_facecolor("white")
    ax.grid(True, color="#EBEBEB", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    # 고유 이미지 (회색, 작게)
    mask_u = labels == -1
    if mask_u.any():
        ax.scatter(coords[mask_u, 0], coords[mask_u, 1],
                   c=UNIQUE_COLOR, s=18, alpha=0.60,
                   linewidths=0, zorder=3)

    # 중복 그룹 — 같은 그룹 = 같은 색
    for gid, meta in group_meta.items():
        mask  = labels == gid
        pts   = coords[mask]
        color = group_colors[gid]
        is_exact = meta["type"] == "exact"

        # 1. 신뢰 타원 (그룹 영역)
        _draw_ellipse(ax, pts, color)

        # 2. 점 (exact: 삼각형, near: 원)
        ax.scatter(pts[:, 0], pts[:, 1],
                   c=[color], s=55, alpha=0.95, zorder=5,
                   marker="^" if is_exact else "o",
                   edgecolors="white", linewidths=0.6)

        # 3. 그룹 레이블 — 중심점
        cx, cy = pts.mean(axis=0)
        ax.text(cx, cy, f"G{gid + 1}",
                fontsize=7.5, fontweight="bold", color=color,
                ha="center", va="center", zorder=6,
                bbox=dict(facecolor="white", alpha=0.75,
                          edgecolor="none", boxstyle="round,pad=0.2"))

    # 범례
    legend_elems = [
        mpatches.Patch(facecolor=UNIQUE_COLOR, edgecolor="none",
                       label=f"Unique  (n = {n_unique})"),
    ]
    for gid, meta in group_meta.items():
        gtype = "Exact (MD5)" if meta["type"] == "exact" else "Near (PCA hash)"
        marker = "^" if meta["type"] == "exact" else "o"
        legend_elems.append(
            plt.Line2D([0], [0], marker=marker, color="w",
                       markerfacecolor=group_colors[gid],
                       markersize=6, linewidth=0,
                       label=f"G{gid + 1}  {gtype}  (n = {meta['size']})"))

    ax.legend(handles=legend_elems, loc="best",
              frameon=True, framealpha=0.92, edgecolor="#CCCCCC",
              handletextpad=0.5, borderpad=0.7)

    ax.set_xlabel(f"{method.upper()} Component 1")
    ax.set_ylabel(f"{method.upper()} Component 2")
    ax.set_title(f"(a)  Image distribution — {method.upper()} 2D projection",
                 loc="left", pad=7)
    for sp in ax.spines.values():
        sp.set_linewidth(0.7)
        sp.set_color("#AAAAAA")

    # ── (b) Group size bar ────────────────────────────────────────────────
    ax_bar.set_facecolor("white")
    ax_bar.grid(axis="x", color="#EBEBEB", linewidth=0.5, zorder=0)
    ax_bar.set_axisbelow(True)

    if n_groups:
        gids   = list(group_meta.keys())
        sizes  = [group_meta[g]["size"] for g in gids]
        colors = [group_colors[g] for g in gids]
        ypos   = list(range(len(gids)))

        bars = ax_bar.barh(ypos, sizes, color=colors,
                           edgecolor="white", linewidth=0.5, height=0.55)
        for bar, sz in zip(bars, sizes):
            ax_bar.text(bar.get_width() + 0.05,
                        bar.get_y() + bar.get_height() / 2,
                        str(sz), va="center", ha="left", fontsize=7)

        ylabels = [
            f"G{g + 1}  ({'Exact' if group_meta[g]['type'] == 'exact' else 'Near'})"
            for g in gids
        ]
        ax_bar.set_yticks(ypos)
        ax_bar.set_yticklabels(ylabels)
        ax_bar.invert_yaxis()
        ax_bar.set_xlim(0, max(sizes) * 1.30)
        ax_bar.set_xlabel("Number of images")
    else:
        ax_bar.text(0.5, 0.5, "No duplicates\ndetected",
                    ha="center", va="center",
                    transform=ax_bar.transAxes, color="#999999")

    ax_bar.set_title("(b)  Duplicate groups", loc="left", pad=7)
    for sp in ax_bar.spines.values():
        sp.set_linewidth(0.7)
        sp.set_color("#AAAAAA")

    # 하단 파라미터 메모 (figure caption 용)
    fig.text(0.5, -0.02,
             f"Dataset: {Path(data_dir).name}   |   "
             f"hash bits = {n_components},  Hamming ≤ {threshold}   |   "
             f"total = {total},  unique = {n_unique},  "
             f"dup groups = {n_groups},  removable = {n_removable}",
             ha="center", fontsize=7, color="#666666")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Scatter 저장: {out.resolve()}")
    print(f"Scatter (PDF): {out.with_suffix('.pdf').resolve()}")


# ---------------------------------------------------------------------------
# Heatmap — 논문용
# ---------------------------------------------------------------------------

def plot_heatmap(valid_paths, labels, group_meta, group_colors,
                 n_components, image_size, threshold, output_path):
    dup_idxs = sorted(np.where(labels >= 0)[0], key=lambda i: (labels[i], i))
    if len(dup_idxs) < 2:
        return

    dup_paths  = [valid_paths[i] for i in dup_idxs]
    dup_labels = [labels[i] for i in dup_idxs]

    features, _ = extract_features(dup_paths, image_size)
    hashes, _   = compute_pca_hashes(features, n_components)
    dist        = hamming_distance_matrix(hashes)

    n = len(dup_paths)
    cell = max(0.45, min(0.80, 8.0 / n))
    fig_w = max(5, n * cell + 2.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_w * 0.85), facecolor="white")
    ax.set_facecolor("white")

    # 컬러맵: 높은 유사도(낮은 거리) = 진한 파랑
    cmap = matplotlib.colormaps["Blues"].reversed()
    im = ax.imshow(dist, cmap=cmap, vmin=0, vmax=n_components, aspect="auto")

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cbar.set_label("Hamming distance", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    cbar.outline.set_linewidth(0.5)

    # 셀 숫자 (≤ 30 장)
    if n <= 30:
        thresh_mid = n_components * 0.40
        for i in range(n):
            for j in range(n):
                v = dist[i, j]
                c = "white" if v < thresh_mid else "#333333"
                ax.text(j, i, str(int(v)),
                        ha="center", va="center",
                        fontsize=max(5, 8 - n // 6),
                        color=c,
                        fontweight="bold" if (i != j and v <= threshold) else "normal")

    # 임계값 이하 쌍 — 검정 테두리
    for i in range(n):
        for j in range(n):
            if i != j and dist[i, j] <= threshold:
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=False, edgecolor="#111111",
                    linewidth=1.6, zorder=5))

    # 그룹 구분선
    prev = dup_labels[0]
    for k in range(1, n):
        if dup_labels[k] != prev:
            ax.axhline(k - 0.5, color="#444444", linewidth=1.0, zorder=6)
            ax.axvline(k - 0.5, color="#444444", linewidth=1.0, zorder=6)
        prev = dup_labels[k]

    # 축 레이블
    names = [p.stem[:18] for p in dup_paths]
    fs = max(5, 8 - n // 7)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=fs)
    ax.set_yticklabels(names, fontsize=fs)

    # 왼쪽 색상 사이드바 — 그룹 색상 + 마커
    for k, lbl in enumerate(dup_labels):
        color  = group_colors.get(lbl, "#aaa")
        is_ex  = group_meta.get(lbl, {}).get("type") == "exact"
        ax.add_patch(plt.Rectangle((-2.5, k - 0.5), 0.8, 1.0,
                                   color=color, clip_on=False, zorder=7))
        ax.plot(-2.1, k, marker="^" if is_ex else "o",
                color="white", markersize=3.5,
                clip_on=False, zorder=8)

    # 오른쪽 그룹 레이블
    prev_lbl, start = dup_labels[0], 0
    for k, lbl in enumerate(dup_labels + [None]):
        if lbl != prev_lbl and prev_lbl is not None:
            mid  = (start + k - 1) / 2
            gtyp = group_meta.get(prev_lbl, {}).get("type", "")
            ax.text(n + 0.5, mid,
                    f"G{prev_lbl + 1}\n({'E' if gtyp == 'exact' else 'N'})",
                    ha="left", va="center",
                    fontsize=max(5, 7 - n // 10),
                    color=group_colors.get(prev_lbl, "#555"),
                    fontweight="bold", clip_on=False)
            start = k
        prev_lbl = lbl

    ax.set_title(
        f"Hamming Distance Matrix  "
        f"(bold border: Hamming $\\leq$ {threshold};  "
        f"E = Exact/MD5,  N = Near/PCA-hash)",
        fontsize=9, pad=10)

    for sp in ax.spines.values():
        sp.set_linewidth(0.7)
        sp.set_color("#AAAAAA")

    out = Path(output_path)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Heatmap 저장: {out.resolve()}")
    print(f"Heatmap (PDF): {out.with_suffix('.pdf').resolve()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PCA Hash Deduplication — 논문용 시각화")
    parser.add_argument("--data_dir",           required=True)
    parser.add_argument("--output",             default="dedup_visualization.png")
    parser.add_argument("--method",             choices=["pca", "umap"], default="pca")
    parser.add_argument("--n_components",       type=int, default=32)
    parser.add_argument("--hamming_threshold",  type=int, default=2)
    parser.add_argument("--image_size",         type=int, default=64)
    args = parser.parse_args()

    print(f"\n이미지 수집: {args.data_dir}")
    paths = collect_image_paths(args.data_dir)
    print(f"  발견: {len(paths):,}장")
    if not paths:
        sys.exit("이미지를 찾을 수 없습니다.")

    features, valid_paths = extract_features(paths, args.image_size)

    print("PCA hash 계산 중...")
    hashes, _   = compute_pca_hashes(features, args.n_components)
    dist_matrix = hamming_distance_matrix(hashes)
    near_groups = find_duplicate_groups(dist_matrix, valid_paths,
                                        args.hamming_threshold)

    print("MD5 중복 탐지 중...")
    exact_groups = md5_exact_duplicates(valid_paths)

    print(f"2D 임베딩 ({args.method.upper()}) 중...")
    coords = embed_2d(features, args.method)

    labels, group_meta = build_labels(len(valid_paths), exact_groups,
                                      near_groups, valid_paths)
    n_groups     = len(group_meta)
    group_colors = {gid: _group_color(gid, n_groups) for gid in group_meta}

    out = Path(args.output)
    plot_scatter(coords, labels, group_meta, valid_paths, str(out),
                 args.method, args.data_dir,
                 args.n_components, args.hamming_threshold)

    if len(valid_paths) <= 300:
        plot_heatmap(valid_paths, labels, group_meta, group_colors,
                     args.n_components, args.image_size,
                     args.hamming_threshold,
                     str(out.with_name(out.stem + "_heatmap" + out.suffix)))

    n_unique = int((labels == -1).sum())
    n_dup    = int((labels >= 0).sum())
    print(f"\n결과: 전체 {len(valid_paths):,}장 | 고유 {n_unique:,}장 | "
          f"중복 그룹 {n_groups}개 ({n_dup}장)")


if __name__ == "__main__":
    main()
