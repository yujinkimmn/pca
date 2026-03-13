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
import os
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
    r, g, b, _ = plt.get_cmap("tab20")(gid / max(n_groups, 1))
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


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
# 네트워크 클러스터 레이아웃 계산
# ---------------------------------------------------------------------------

def _network_layout(labels: np.ndarray, group_meta: dict) -> np.ndarray:
    """
    중복 관계를 시각적으로 드러내는 좌표를 계산합니다.
    - 중복 그룹: 원형으로 배치된 클러스터
    - 고유 이미지: 바깥 링에 고르게 배치
    """
    n = len(labels)
    coords = np.zeros((n, 2))
    n_groups = len(group_meta)

    # 그룹 클러스터 중심 — 정다각형 꼭짓점에 배치
    cluster_r = 3.5 if n_groups > 1 else 0.0
    group_centers = {}
    for k, gid in enumerate(group_meta.keys()):
        angle = 2 * np.pi * k / max(n_groups, 1) - np.pi / 2
        group_centers[gid] = np.array([cluster_r * np.cos(angle),
                                       cluster_r * np.sin(angle)])

    # 각 그룹 내 노드: 소형 원형 배치
    for gid, meta in group_meta.items():
        idxs = np.where(labels == gid)[0]
        cx, cy = group_centers[gid]
        size = len(idxs)
        inner_r = 0.55 * np.sqrt(size)   # 그룹 크기에 비례한 반지름
        for j, idx in enumerate(idxs):
            a = 2 * np.pi * j / max(size, 1) - np.pi / 2
            coords[idx] = [cx + inner_r * np.cos(a),
                           cy + inner_r * np.sin(a)]

    # 고유 이미지: 바깥 링
    unique_idxs = np.where(labels == -1)[0]
    n_unique = len(unique_idxs)
    outer_r = cluster_r + 2.2
    for j, idx in enumerate(unique_idxs):
        a = 2 * np.pi * j / max(n_unique, 1)
        coords[idx] = [outer_r * np.cos(a), outer_r * np.sin(a)]

    return coords


# ---------------------------------------------------------------------------
# 메인 그래프 — (a) 중복 관계 네트워크  (b) 그룹 크기 막대
# ---------------------------------------------------------------------------

def plot_scatter(coords_pca, labels, group_meta, valid_paths, output_path,
                 method, data_dir, n_components, threshold):
    """
    coords_pca 는 사용하지 않음 (PCA 2D는 중복 이미지가 가까이 모이지 않아
    관계 파악이 어려움). 대신 중복 그룹 구조를 직접 반영한 네트워크 레이아웃 사용.
    """
    n_groups    = len(group_meta)
    n_unique    = int((labels == -1).sum())
    n_dup_imgs  = int((labels >= 0).sum())
    n_removable = max(0, n_dup_imgs - n_groups)
    total       = len(labels)

    group_colors = {gid: _group_color(gid, n_groups) for gid in group_meta}

    # 네트워크 레이아웃 좌표
    coords = _network_layout(labels, group_meta)

    # 논문 2-column 기준: 7.16 × 3.8 inch
    fig, (ax, ax_bar) = plt.subplots(
        1, 2, figsize=(7.16, 3.8),
        gridspec_kw={"width_ratios": [3, 1], "wspace": 0.38},
    )

    # ── (a) 중복 관계 네트워크 ────────────────────────────────────────────
    ax.set_facecolor("white")
    ax.set_aspect("equal")
    ax.axis("off")   # 축 숫자/눈금 제거 — 좌표값이 의미 없으므로

    # 그룹 배경 원 (그룹 영역 강조)
    for gid, meta in group_meta.items():
        idxs = np.where(labels == gid)[0]
        if len(idxs) < 2:
            continue
        cx, cy = coords[idxs].mean(axis=0)
        r = np.linalg.norm(coords[idxs] - [cx, cy], axis=1).max() + 0.45
        circle = plt.Circle((cx, cy), r,
                             color=group_colors[gid], alpha=0.08,
                             linewidth=1.2, linestyle="--",
                             edgecolor=group_colors[gid], fill=True,
                             zorder=1)
        ax.add_patch(circle)

    # 그룹 내 연결선 (엣지)
    for gid, meta in group_meta.items():
        idxs = np.where(labels == gid)[0]
        color = group_colors[gid]
        lw = 2.0 if meta["type"] == "exact" else 1.2
        ls = "-" if meta["type"] == "exact" else "--"
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                x0, y0 = coords[idxs[a]]
                x1, y1 = coords[idxs[b]]
                ax.plot([x0, x1], [y0, y1], color=color,
                        alpha=0.55, linewidth=lw, linestyle=ls, zorder=2)

    # 고유 이미지 노드
    mask_u = labels == -1
    if mask_u.any():
        ax.scatter(coords[mask_u, 0], coords[mask_u, 1],
                   c=UNIQUE_COLOR, s=60, alpha=0.70,
                   edgecolors="#888888", linewidths=0.6,
                   zorder=4, marker="o")
        # 파일명 레이블
        for idx in np.where(mask_u)[0]:
            ax.text(coords[idx, 0], coords[idx, 1] - 0.32,
                    valid_paths[idx].stem[:14],
                    ha="center", va="top", fontsize=5.5,
                    color="#888888", zorder=5)

    # 중복 그룹 노드
    for gid, meta in group_meta.items():
        idxs  = np.where(labels == gid)[0]
        color = group_colors[gid]
        is_exact = meta["type"] == "exact"
        marker = "^" if is_exact else "o"

        ax.scatter(coords[idxs, 0], coords[idxs, 1],
                   c=[color], s=100, alpha=0.95, zorder=5,
                   marker=marker,
                   edgecolors="white", linewidths=1.0)

        # 파일명 레이블
        for idx in idxs:
            ax.text(coords[idx, 0], coords[idx, 1] - 0.32,
                    valid_paths[idx].stem[:14],
                    ha="center", va="top", fontsize=5.5,
                    color=color, zorder=6)

        # 그룹 레이블 (중심)
        cx, cy = coords[idxs].mean(axis=0)
        gtype_abbr = "Exact" if is_exact else "Near"
        ax.text(cx, cy + np.linalg.norm(coords[idxs] - [cx, cy], axis=1).max() + 0.65,
                f"Group {gid + 1}  ({gtype_abbr}, n={meta['size']})",
                ha="center", va="bottom", fontsize=7, fontweight="bold",
                color=color, zorder=7)

    # 범례
    legend_elems = [
        mpatches.Patch(facecolor=UNIQUE_COLOR, edgecolor="#888",
                       label="Unique image (no duplicate found)"),
        plt.Line2D([0], [0], color="#555", linewidth=2.0, linestyle="-",
                   label="Exact duplicate (identical MD5)"),
        plt.Line2D([0], [0], color="#555", linewidth=1.2, linestyle="--",
                   label="Near-duplicate (similar PCA hash)"),
    ]
    ax.legend(handles=legend_elems, loc="lower center",
              bbox_to_anchor=(0.5, -0.05),
              frameon=True, framealpha=0.92, edgecolor="#CCCCCC",
              ncol=3, fontsize=7, handlelength=1.5)

    ax.set_title(
        "(a)  Duplicate relationship graph\n"
        "Connected nodes share the same or similar content",
        loc="left", pad=8, fontsize=9)

    # ── (b) 그룹 크기 막대 ────────────────────────────────────────────────
    ax_bar.set_facecolor("white")
    ax_bar.grid(axis="x", color="#EBEBEB", linewidth=0.5, zorder=0)
    ax_bar.set_axisbelow(True)

    # 요약 통계 텍스트 박스
    summary = (f"Total images:  {total}\n"
               f"Unique:            {n_unique}\n"
               f"Dup. groups:    {n_groups}\n"
               f"Dup. images:    {n_dup_imgs}\n"
               f"Removable:      {n_removable}")
    ax_bar.text(0.05, 0.98, summary, transform=ax_bar.transAxes,
                fontsize=7.5, va="top", ha="left",
                family="monospace",
                bbox=dict(facecolor="#F7F7F7", edgecolor="#DDDDDD",
                          boxstyle="round,pad=0.5", linewidth=0.7))

    if n_groups:
        gids   = list(group_meta.keys())
        sizes  = [group_meta[g]["size"] for g in gids]
        colors = [group_colors[g] for g in gids]
        ypos   = list(range(len(gids)))

        # 요약 박스 아래에서 시작 (axes 좌표 0.40 아래)
        bar_ax_h = 0.45
        bar_ax = ax_bar.inset_axes([0.0, 0.0, 1.0, bar_ax_h])
        bar_ax.set_facecolor("white")
        bar_ax.grid(axis="x", color="#EBEBEB", linewidth=0.5, zorder=0)
        bar_ax.set_axisbelow(True)

        bars = bar_ax.barh(ypos, sizes, color=colors,
                           edgecolor="white", linewidth=0.5, height=0.6)
        for bar, sz in zip(bars, sizes):
            bar_ax.text(bar.get_width() + 0.05,
                        bar.get_y() + bar.get_height() / 2,
                        str(sz), va="center", ha="left", fontsize=7)

        ylabels = [
            f"G{g + 1} ({'E' if group_meta[g]['type'] == 'exact' else 'N'})"
            for g in gids
        ]
        bar_ax.set_yticks(ypos)
        bar_ax.set_yticklabels(ylabels, fontsize=7.5)
        bar_ax.invert_yaxis()
        bar_ax.set_xlim(0, max(sizes) * 1.35)
        bar_ax.set_xlabel("Images in group", fontsize=7.5)
        bar_ax.tick_params(labelsize=7)
        for sp in bar_ax.spines.values():
            sp.set_linewidth(0.5)
            sp.set_color("#CCCCCC")

    ax_bar.axis("off")
    ax_bar.set_title("(b)  Group summary\n(E=Exact, N=Near)",
                     loc="left", pad=8, fontsize=9)

    # 하단 파라미터 (caption용)
    fig.text(0.5, -0.01,
             f"Dataset: {Path(data_dir).name}   |   "
             f"PCA hash bits = {n_components},  Hamming threshold = {threshold}",
             ha="center", fontsize=7, color="#777777")

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
# Interactive HTML (Plotly) — 마우스오버 시 실제 이미지 표시
# ---------------------------------------------------------------------------

def _encode_img_b64(path: Path, thumb: int = 120) -> str:
    """이미지를 base64 PNG 문자열로 인코딩합니다."""
    import base64, io
    from PIL import Image as PILImage
    try:
        img = PILImage.open(path).convert("RGB")
        img.thumbnail((thumb, thumb))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


def plot_interactive(coords, labels, group_meta, valid_paths, output_path,
                     method, data_dir, n_components, threshold):
    """
    Plotly로 인터랙티브 HTML을 생성합니다.
    마커에 마우스를 올리면 실제 이미지 + 파일명 + 그룹 정보가 툴팁에 표시됩니다.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("  [경고] plotly 미설치. `pip install plotly`")
        return

    n_groups     = len(group_meta)
    group_colors = {gid: _group_color(gid, n_groups) for gid in group_meta}

    print("  이미지 base64 인코딩 중...")
    b64_imgs = [_encode_img_b64(p) for p in valid_paths]

    fig = go.Figure()

    # ── 고유 이미지 트레이스 ──────────────────────────────────────────────
    mask_u = labels == -1
    u_idxs = np.where(mask_u)[0]
    if len(u_idxs):
        custom = [
            [valid_paths[i].name, b64_imgs[i]]
            for i in u_idxs
        ]
        fig.add_trace(go.Scatter(
            x=coords[u_idxs, 0], y=coords[u_idxs, 1],
            mode="markers",
            name="Unique",
            marker=dict(color=UNIQUE_COLOR, size=9,
                        symbol="circle",
                        line=dict(width=0.5, color="#888888")),
            customdata=custom,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Unique image<br>"
                "<img src='data:image/png;base64,%{customdata[1]}' "
                "width='120' style='border-radius:4px'>"
                "<extra></extra>"
            ),
        ))

    # ── 중복 그룹 트레이스 ───────────────────────────────────────────────
    for gid, meta in group_meta.items():
        mask  = labels == gid
        idxs  = np.where(mask)[0]
        pts   = coords[idxs]
        color = group_colors[gid]
        is_exact = meta["type"] == "exact"
        gtype_str = "Exact (MD5)" if is_exact else "Near (PCA hash)"
        symbol = "triangle-up" if is_exact else "circle"

        custom = [
            [valid_paths[i].name, b64_imgs[i], f"G{gid+1}", gtype_str]
            for i in idxs
        ]
        fig.add_trace(go.Scatter(
            x=pts[:, 0], y=pts[:, 1],
            mode="markers+text",
            name=f"G{gid+1}  {gtype_str}  (n={meta['size']})",
            marker=dict(color=color, size=13, symbol=symbol,
                        line=dict(width=1.5, color="white")),
            text=[f"G{gid+1}"] * len(idxs),
            textposition="top center",
            textfont=dict(size=9, color=color),
            customdata=custom,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Group %{customdata[2]}  |  %{customdata[3]}<br>"
                "<img src='data:image/png;base64,%{customdata[1]}' "
                "width='120' style='border-radius:4px'>"
                "<extra></extra>"
            ),
        ))

        # 그룹 중심점 — 타원 대신 dashed circle shape (Plotly에선 ellipse가 제한적)
        cx, cy = pts.mean(axis=0)
        # 대략적인 반지름
        if len(pts) >= 2:
            r_x = max(np.std(pts[:, 0]) * 2.5, 50)
            r_y = max(np.std(pts[:, 1]) * 2.5, 50)
        else:
            r_x = r_y = 80
        fig.add_shape(type="circle",
                      x0=cx - r_x, y0=cy - r_y,
                      x1=cx + r_x, y1=cy + r_y,
                      line=dict(color=color, width=1.5, dash="dot"),
                      fillcolor=color, opacity=0.08)

    fig.update_layout(
        title=dict(
            text=(f"Image Distribution — {Path(data_dir).name}  |  "
                  f"{method.upper()} 2D  |  hash bits={n_components}, "
                  f"Hamming ≤ {threshold}"),
            font=dict(size=14),
            x=0.5,
        ),
        xaxis=dict(title=f"{method.upper()} Component 1",
                   gridcolor="#EBEBEB", zeroline=False),
        yaxis=dict(title=f"{method.upper()} Component 2",
                   gridcolor="#EBEBEB", zeroline=False),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#DDDDDD",
            borderwidth=1,
        ),
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="#CCCCCC",
            font_size=12,
        ),
        width=900, height=650,
    )

    out_html = Path(output_path).with_suffix(".html")
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    print(f"Interactive HTML 저장: {out_html.resolve()}")


# ---------------------------------------------------------------------------
# Duplicate Gallery HTML — HF 스타일 이미지 썸네일 그리드
# ---------------------------------------------------------------------------

def plot_gallery(valid_paths, labels, group_meta, output_path,
                 data_dir, n_components, threshold, thumb=180,
                 coords=None):
    """
    중복 그룹별로 실제 이미지 썸네일을 나란히 보여주는 HTML 갤러리 생성.
    HuggingFace image deduplication toolkit의 시각화 스타일.
    """
    try:
        from PIL import Image as PILImage
    except ImportError:
        print("  [경고] Pillow 미설치. 갤러리 생성 불가.")
        return

    import base64, io, json as _json, html as _html

    def encode(path):
        try:
            img = PILImage.open(path).convert("RGB")
            img.thumbnail((thumb, thumb))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=75)
            return base64.b64encode(buf.getvalue()).decode()
        except Exception:
            return ""

    n_groups  = len(group_meta)
    n_unique  = int((labels == -1).sum())
    n_dup_img = int((labels >= 0).sum())
    n_removable = max(0, n_dup_img - n_groups)

    COLORS = OKABE_ITO

    def hex_to_rgba(h, a=0.15):
        h = h.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{a})"

    # ── 그룹 섹션 HTML ────────────────────────────────────────────────────
    group_sections = []
    for gid, meta in group_meta.items():
        idxs     = [i for i, l in enumerate(labels) if l == gid]
        color    = COLORS[gid % len(COLORS)]
        bg       = hex_to_rgba(color, 0.10)
        border   = hex_to_rgba(color, 0.60)
        is_exact = meta["type"] == "exact"
        badge    = ("Exact duplicate" if is_exact else "Near-duplicate")
        badge_bg = "#e74c3c" if is_exact else "#2980b9"

        cards = []
        for rank, idx in enumerate(idxs):
            b64  = encode(valid_paths[idx])
            name = valid_paths[idx].name
            keep = "KEEP" if rank == 0 else "REMOVE"
            keep_color = "#27ae60" if rank == 0 else "#e74c3c"
            path_attr = _html.escape(os.path.abspath(str(valid_paths[idx])), quote=True)
            img_tag = (f'<img src="data:image/jpeg;base64,{b64}" '
                       f'style="width:{thumb}px;height:{thumb}px;'
                       f'object-fit:cover;border-radius:6px;">' if b64 else
                       f'<div style="width:{thumb}px;height:{thumb}px;'
                       f'background:#eee;border-radius:6px;display:flex;'
                       f'align-items:center;justify-content:center;color:#aaa;">no img</div>')
            del_btn = f'<button class="del-btn" data-path="{path_attr}">🗑 삭제</button>'
            cards.append(f"""
            <div class="img-card" style="text-align:center;margin:6px;">
              {img_tag}
              <div style="font-size:11px;color:#555;margin-top:4px;
                          max-width:{thumb}px;word-break:break-all;">{name}</div>
              <div style="font-size:10px;font-weight:bold;color:{keep_color};
                          margin-top:2px;">{keep}</div>
              {del_btn}
            </div>""")

        group_sections.append(f"""
        <div style="background:{bg};border:1.5px solid {border};
                    border-radius:10px;padding:16px 20px;margin-bottom:18px;">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
            <span style="background:{badge_bg};color:white;font-size:12px;
                         font-weight:bold;padding:3px 10px;border-radius:20px;">
              {badge}
            </span>
            <span style="font-size:13px;font-weight:bold;color:#333;">
              Group {gid+1} &nbsp;·&nbsp; {meta['size']} images
            </span>
          </div>
          <div style="display:flex;flex-wrap:wrap;gap:4px;">
            {''.join(cards)}
          </div>
        </div>""")

    # ── 고유 이미지 섹션 ──────────────────────────────────────────────────
    unique_idxs = [i for i, l in enumerate(labels) if l == -1]
    unique_cards = []
    for idx in unique_idxs[:48]:   # 최대 48장만 표시
        b64  = encode(valid_paths[idx])
        name = valid_paths[idx].name
        img_tag = (f'<img src="data:image/jpeg;base64,{b64}" '
                   f'style="width:80px;height:80px;object-fit:cover;'
                   f'border-radius:5px;opacity:0.75;">' if b64 else "")
        unique_cards.append(f"""
        <div style="text-align:center;margin:4px;">
          {img_tag}
          <div style="font-size:9px;color:#888;max-width:80px;
                      word-break:break-all;">{name[:16]}</div>
        </div>""")
    more_txt = (f'<div style="color:#aaa;font-size:12px;padding:10px;">'
                f'+ {len(unique_idxs)-48} more unique images</div>'
                if len(unique_idxs) > 48 else "")

    unique_section = f"""
    <div style="background:#f9f9f9;border:1px solid #ddd;border-radius:10px;
                padding:16px 20px;margin-bottom:18px;">
      <div style="font-size:13px;font-weight:bold;color:#666;margin-bottom:10px;">
        Unique images ({n_unique} total, no duplicates found)
      </div>
      <div style="display:flex;flex-wrap:wrap;gap:4px;">
        {''.join(unique_cards)}{more_txt}
      </div>
    </div>"""

    # ── Canvas 산점도 데이터 (순수 JS, CDN 불필요) ────────────────────────
    scatter_section = ""
    if coords is not None:
        import json
        pts = []
        for i, (path, lbl) in enumerate(zip(valid_paths, labels)):
            if lbl == -1:
                color = "#BBBBBB"
                group_label = "Unique"
            else:
                color = COLORS[lbl % len(COLORS)]
                meta  = group_meta[lbl]
                gtype = "Exact" if meta["type"] == "exact" else "Near"
                group_label = f"Group {lbl+1} ({gtype})"
            b64 = encode(path)
            pts.append({
                "x": float(coords[i, 0]),
                "y": float(coords[i, 1]),
                "color": color,
                "label": group_label,
                "name": path.name,
                "img": b64,
            })
        pts_json = json.dumps(pts)

        scatter_section = f"""
  <h2 style="margin-top:32px;">인터랙티브 산점도 (마우스오버 시 이미지)</h2>
  <div style="background:white;border-radius:12px;padding:20px;
              box-shadow:0 1px 4px rgba(0,0,0,0.08);margin-bottom:24px;
              position:relative;">
    <canvas id="scatter" width="860" height="500"
            style="border:1px solid #eee;border-radius:8px;cursor:crosshair;
                   display:block;margin:0 auto;"></canvas>
    <div id="tooltip" style="position:fixed;display:none;background:white;
         border:1px solid #ddd;border-radius:8px;padding:8px 10px;
         box-shadow:0 2px 8px rgba(0,0,0,0.15);pointer-events:none;z-index:999;
         font-size:12px;max-width:220px;text-align:center;"></div>
  </div>
  <script>
  (function(){{
    const pts = {pts_json};
    const canvas = document.getElementById('scatter');
    const ctx = canvas.getContext('2d');
    const tip = document.getElementById('tooltip');
    const W = canvas.width, H = canvas.height, PAD = 40;

    const xs = pts.map(p=>p.x), ys = pts.map(p=>p.y);
    const xMin=Math.min(...xs), xMax=Math.max(...xs);
    const yMin=Math.min(...ys), yMax=Math.max(...ys);
    const xR=xMax-xMin||1, yR=yMax-yMin||1;

    function tx(x){{ return PAD + (x-xMin)/xR*(W-2*PAD); }}
    function ty(y){{ return H-PAD - (y-yMin)/yR*(H-2*PAD); }}

    function draw(){{
      ctx.clearRect(0,0,W,H);
      ctx.fillStyle='#fafafa'; ctx.fillRect(0,0,W,H);
      // grid
      ctx.strokeStyle='#efefef'; ctx.lineWidth=1;
      for(let i=0;i<=5;i++){{
        let gx=PAD+i*(W-2*PAD)/5, gy=PAD+i*(H-2*PAD)/5;
        ctx.beginPath(); ctx.moveTo(gx,PAD); ctx.lineTo(gx,H-PAD); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(PAD,gy); ctx.lineTo(W-PAD,gy); ctx.stroke();
      }}
      pts.forEach(p=>{{
        ctx.beginPath();
        ctx.arc(tx(p.x), ty(p.y), p.label==='Unique'?5:8, 0, Math.PI*2);
        ctx.fillStyle = p.color;
        ctx.globalAlpha = p.label==='Unique'?0.55:0.9;
        ctx.fill();
        ctx.globalAlpha=1;
        ctx.strokeStyle='white'; ctx.lineWidth=1.5; ctx.stroke();
      }});
    }}
    draw();

    canvas.addEventListener('mousemove', function(e){{
      const rect=canvas.getBoundingClientRect();
      const mx=e.clientX-rect.left, my=e.clientY-rect.top;
      let best=null, bestD=Infinity;
      pts.forEach(p=>{{
        const d=Math.hypot(tx(p.x)-mx, ty(p.y)-my);
        if(d<bestD){{ bestD=d; best=p; }}
      }});
      if(best && bestD<20){{
        let html='<b>'+best.name+'</b><br><span style="color:#888">'+best.label+'</span>';
        if(best.img) html+='<br><img src="data:image/jpeg;base64,'+best.img+
          '" style="width:120px;height:120px;object-fit:cover;border-radius:6px;margin-top:6px;">';
        tip.innerHTML=html;
        tip.style.display='block';
        tip.style.left=(e.clientX+14)+'px';
        tip.style.top=(e.clientY-10)+'px';
      }} else {{
        tip.style.display='none';
      }}
    }});
    canvas.addEventListener('mouseleave',()=>{{ tip.style.display='none'; }});
  }})();
  </script>"""

    # ── 전체 HTML ─────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Image Deduplication Gallery — {Path(data_dir).name}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #f4f5f7; margin: 0; padding: 24px; }}
    .container {{ max-width: 1100px; margin: 0 auto; }}
    .header {{ background: white; border-radius: 12px; padding: 24px 28px;
               margin-bottom: 24px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
    .stat-grid {{ display: flex; gap: 20px; flex-wrap: wrap; margin-top: 14px; }}
    .stat {{ background: #f0f4ff; border-radius: 8px; padding: 10px 18px;
             text-align: center; }}
    .stat-val {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
    .stat-lbl {{ font-size: 11px; color: #888; margin-top: 2px; }}
    h1 {{ margin: 0 0 4px; font-size: 20px; color: #1a1a2e; }}
    h2 {{ font-size: 15px; color: #444; margin: 0 0 14px; font-weight: 600; }}
    .params {{ font-size: 12px; color: #999; margin-top: 6px; }}
    .del-btn {{ background:#e74c3c;color:white;border:none;border-radius:4px;
                padding:3px 10px;font-size:10px;cursor:pointer;margin-top:5px; }}
    .del-btn:hover {{ background:#c0392b; }}
    .del-btn:disabled {{ background:#aaa;cursor:not-allowed; }}
    .card-deleted {{ opacity:0.3;pointer-events:none; }}
    .server-note {{ background:#fff8e1;border:1px solid #ffe082;border-radius:6px;
                    padding:7px 12px;margin-top:12px;font-size:12px;color:#795548; }}
    .server-note code {{ background:#f5f5f5;padding:1px 5px;border-radius:3px;font-size:11px; }}
  </style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>Image Deduplication Gallery</h1>
    <div class="params">Dataset: <b>{Path(data_dir).name}</b> &nbsp;|&nbsp;
      PCA hash bits = {n_components} &nbsp;|&nbsp; Hamming threshold = {threshold}</div>
    <div class="server-note">
      🗑 삭제 버튼을 활성화하려면 아래 명령어로 서버를 실행하세요:<br>
      <code>python pca_dedup.py serve --html &lt;이 파일 경로&gt;</code>
      &nbsp;→ 브라우저에서 <code>http://localhost:7474</code> 를 열면 됩니다.
    </div>
    <div class="stat-grid">
      <div class="stat"><div class="stat-val">{len(valid_paths)}</div>
        <div class="stat-lbl">Total images</div></div>
      <div class="stat"><div class="stat-val">{n_unique}</div>
        <div class="stat-lbl">Unique</div></div>
      <div class="stat"><div class="stat-val">{n_groups}</div>
        <div class="stat-lbl">Duplicate groups</div></div>
      <div class="stat"><div class="stat-val" style="color:#e74c3c">{n_removable}</div>
        <div class="stat-lbl">Removable images</div></div>
    </div>
  </div>

  {scatter_section}

  <h2>Duplicate Groups</h2>
  {''.join(group_sections) if group_sections else
   '<div style="color:#aaa;padding:20px;">No duplicates found.</div>'}

  <h2 style="margin-top:28px;">Unique Images</h2>
  {unique_section}
</div>
<script>
document.querySelectorAll('.del-btn').forEach(function(btn) {{
  btn.addEventListener('click', async function() {{
    var path = btn.dataset.path;
    if (!confirm('이 파일을 삭제하시겠습니까?\\n' + path)) return;
    btn.disabled = true;
    btn.textContent = '...';
    try {{
      var r = await fetch('/api/delete', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{path: path}})
      }});
      var d = await r.json();
      if (r.ok) {{
        btn.closest('.img-card').classList.add('card-deleted');
        btn.textContent = '✓ 삭제됨';
      }} else {{
        btn.textContent = '✗ 오류';
        btn.disabled = false;
        alert('오류: ' + d.error);
      }}
    }} catch(e) {{
      btn.textContent = '✗ 서버 없음';
      btn.disabled = false;
      alert('서버가 실행되어 있지 않습니다.\\n\\n아래 명령어로 먼저 서버를 실행하세요:\\npython pca_dedup.py serve --html <이 파일 경로>\\n\\n그 다음 http://localhost:7474 에서 열어주세요.');
    }}
  }});
}});
</script>
</body>
</html>"""

    out = Path(output_path).with_suffix(".html")
    out.write_text(html, encoding="utf-8")
    print(f"Gallery HTML 저장: {out.resolve()}")


# ---------------------------------------------------------------------------
# 삭제 서버
# ---------------------------------------------------------------------------

def run_server(html_path: str, port: int = 7474, action: str = "move") -> None:
    """
    HTML 갤러리를 서빙하고 파일 삭제 API를 제공하는 로컬 서버.

    GET  /            → HTML 파일 반환
    POST /api/delete  → {"path": "/abs/path/to/file"} → 파일 처리 (action에 따라)

    action="move" : 원본 파일의 부모 디렉토리 아래 removed/ 로 이동 (기본값, 복구 가능)
    action="delete": 즉시 영구 삭제
    처리된 파일은 HTML 옆 <stem>_deleted/ 디렉토리의 deletion_log.jsonl 에 기록됩니다.
    """
    import http.server
    import json as _json
    import datetime
    import shutil as _shutil

    html_path_obj = Path(html_path)
    log_dir = html_path_obj.with_name(html_path_obj.stem + "_deleted")
    log_path = log_dir / "deletion_log.jsonl"
    html_bytes = html_path_obj.read_bytes()

    def _append_log(entry: dict) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(_json.dumps(entry, ensure_ascii=False) + "\n")

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ('/', '/index.html'):
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', len(html_bytes))
                self.end_headers()
                self.wfile.write(html_bytes)
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            if self.path == '/api/delete':
                length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(length)
                try:
                    data = _json.loads(body)
                    target = Path(data['path'])
                    if not target.is_file():
                        self._json(404, {'error': 'File not found'})
                        return
                    ts = datetime.datetime.now().isoformat(timespec='seconds')
                    if action == "move":
                        removed_dir = target.parent / "removed"
                        removed_dir.mkdir(parents=True, exist_ok=True)
                        dest = removed_dir / target.name
                        if dest.exists():
                            dest = removed_dir / f"{target.stem}_{target.stat().st_ino}{target.suffix}"
                        _shutil.move(str(target), dest)
                        print(f"  [이동] {target} → {dest}")
                        _append_log({"timestamp": ts, "action": "move", "original": str(target), "moved_to": str(dest)})
                    else:
                        target.unlink()
                        print(f"  [삭제] {target}")
                        _append_log({"timestamp": ts, "action": "delete", "original": str(target)})
                    self._json(200, {'ok': True})
                except Exception as e:
                    self._json(500, {'error': str(e)})
            else:
                self.send_response(404)
                self.end_headers()

        def _json(self, code: int, data: dict) -> None:
            body = _json.dumps(data).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Content-Length', len(body))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):  # 기본 로그 억제
            pass

    server = http.server.HTTPServer(('localhost', port), _Handler)
    action_desc = f"원본 디렉토리의 removed/ 로 이동" if action == "move" else "즉시 영구 삭제"
    print(f"\n  갤러리 서버 실행 중: http://localhost:{port}")
    print(f"  브라우저에서 위 주소를 열면 삭제 버튼이 활성화됩니다.")
    print(f"  버튼 동작: {action_desc}")
    print(f"  처리 기록: {log_path}")
    print(f"  종료: Ctrl+C\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  서버 종료.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PCA Hash Deduplication — 논문용 시각화")
    parser.add_argument("--data_dir",           required=True, nargs="+",
                        help="이미지 데이터셋 디렉토리 (여러 개 지정 가능)")
    parser.add_argument("--output",             default="dedup_visualization.png")
    parser.add_argument("--method",             choices=["pca", "umap"], default="pca")
    parser.add_argument("--n_components",       type=int, default=32)
    parser.add_argument("--hamming_threshold",  type=int, default=2)
    parser.add_argument("--image_size",         type=int, default=64)
    parser.add_argument("--interactive",        action="store_true",
                        help="Plotly HTML 인터랙티브 뷰어도 생성 (마우스오버 시 이미지 표시)")
    parser.add_argument("--no_gallery",         action="store_true",
                        help="중복 그룹 갤러리 HTML 생성 건너뜀")
    args = parser.parse_args()

    data_dir_label = " + ".join(args.data_dir)
    print(f"\n이미지 수집: {data_dir_label}")
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
                 args.method, data_dir_label,
                 args.n_components, args.hamming_threshold)

    if len(valid_paths) <= 300:
        plot_heatmap(valid_paths, labels, group_meta, group_colors,
                     args.n_components, args.image_size,
                     args.hamming_threshold,
                     str(out.with_name(out.stem + "_heatmap" + out.suffix)))

    if not args.no_gallery:
        print("갤러리 HTML 생성 중...")
        gallery_out = str(out.with_name(out.stem + "_gallery.html"))
        plot_gallery(valid_paths, labels, group_meta, gallery_out,
                     data_dir_label, args.n_components, args.hamming_threshold,
                     coords=coords)

    print("인터랙티브 HTML 생성 중...")
    plot_interactive(coords, labels, group_meta, valid_paths, str(out),
                     args.method, data_dir_label,
                     args.n_components, args.hamming_threshold)

    n_unique = int((labels == -1).sum())
    n_dup    = int((labels >= 0).sum())
    print(f"\n결과: 전체 {len(valid_paths):,}장 | 고유 {n_unique:,}장 | "
          f"중복 그룹 {n_groups}개 ({n_dup}장)")


if __name__ == "__main__":
    main()
