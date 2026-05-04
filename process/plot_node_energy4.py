import os
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

from ase.io import read
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

import umap
from scipy.spatial import ConvexHull


def load_node_feat_mean(path, num_feats=64, only_num=None, node_stride=10):
    """
    读取 extxyz，取每个结构的 MACE_node_feats。
    这里按你的需求：取 node_feats[::node_stride, :] 作为样本（会把“原子节点”当作样本）。
    返回：
        data: (n_samples, num_feats)
    注意：这不是“每个结构一个向量”，而是“每个结构抽样若干原子节点向量”。
    """
    atoms = read(path, format="extxyz", index=":")
    dataff = []
    dataal = []
    numff = 0
    numal = 0
    for atom in atoms:
        num = len(atom)
        if only_num is not None and num != only_num:
            continue  # 修正：不符合的结构直接丢弃，避免全0/空样本混入
        feats = atom.arrays["MACE_node_feats"]  # (num_atoms, num_feats)
        feats = feats[::node_stride, :num_feats]  # 防御：若数组更宽
        if "iter" in atom.info:
            dataal.append(feats)
            numal += 1
        else:
            dataff.append(feats)
            numff += 1

    # if len(data_list) == 0:
    #     raise ValueError(
    #         f"No valid samples after filtering only_num={only_num}: {path}"
    #     )

    dataal = np.vstack(dataal).astype(np.float32)
    dataff = np.vstack(dataff).astype(np.float32)

    print(
        f"[{os.path.basename(path)}] samples: {len(dataff)}/{numff} + {len(dataal)}/{numal}, num_feats: {dataff.shape}"
    )

    return dataff, dataal


def reduce_dim_two_sets(
    dataA,
    dataB,
    method="umap",
    n_components=2,
    random_state=42,
    umap_n_neighbors=10,
    umap_min_dist=1.0,
    tsne_perplexity=30,
    standardize_for_tsne=True,
):
    """
    合并 fit，切回 A/B
    """
    X = np.vstack([dataA, dataB])
    nA = len(dataA)

    if method.lower() == "pca":
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        reducer = PCA(n_components=n_components, random_state=random_state)
        emb = reducer.fit_transform(Xs)

    elif method.lower() == "tsne":
        # 建议也标准化一次（否则不同维度尺度会影响 t-SNE）
        if standardize_for_tsne:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        reducer = TSNE(
            n_components=n_components,
            random_state=random_state,
            perplexity=min(tsne_perplexity, max(5, (len(X) - 1) // 3)),
            init="pca",
            learning_rate="auto",
        )
        emb = reducer.fit_transform(X)

    elif method.lower() == "umap":
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        reducer = umap.UMAP(
            n_components=n_components,
            random_state=random_state,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
        )
        emb = reducer.fit_transform(Xs)

    else:
        raise ValueError("method must be 'pca', 'tsne', or 'umap'")

    return emb[:nA], emb[nA:]


def convex_hull_area(points_2d):
    """
    2D hull area。ConvexHull.volume 在 2D 就是 area。
    """
    if len(points_2d) < 3:
        return 0.0, None
    hull = ConvexHull(points_2d)
    return float(hull.volume), hull


def plot_hulls(ax, embA, embB, labelA="FF", labelB="AL"):
    """
    在同一张图上画两套点 + hull 轮廓，并标注 hull 面积比例
    """
    areaA, hullA = convex_hull_area(embA[:, :2])
    areaB, hullB = convex_hull_area(embB[:, :2])

    # 点：用空心/实心提高重叠可见性
    ax.scatter(embA[:, 0], embA[:, 1], s=8, alpha=0.25, c="C0", label=labelA, zorder=2)
    ax.scatter(
        embB[:, 0],
        embB[:, 1],
        s=18,
        alpha=0.35,
        marker="^",
        facecolors="none",
        edgecolors="C1",
        linewidths=0.6,
        label=labelB,
        zorder=3,
    )

    # hull 轮廓
    if hullA is not None:
        poly = embA[hullA.vertices]
        ax.plot(poly[:, 0], poly[:, 1], c="C0", lw=1.5, zorder=4)
    if hullB is not None:
        poly = embB[hullB.vertices]
        ax.plot(poly[:, 0], poly[:, 1], c="C1", lw=1.5, zorder=5)

    ratio = (areaB / areaA) if areaA > 0 else np.nan
    ax.set_title(f"Hull coverage: {labelB}/{labelA} = {ratio:.2f}")
    ax.legend(frameon=True, loc="best")


def plot_density_ff_overlay_al(ax, embFF, embAL, labelFF="FF density", labelAL="AL"):
    """
    FF 用 hexbin 做密度底图，AL 叠加散点 -> 直观看 AL 是否进入 FF 的低密度区/边缘
    """
    hb = ax.hexbin(embFF[:, 0], embFF[:, 1], gridsize=75, mincnt=1)
    ax.scatter(
        embAL[:, 0],
        embAL[:, 1],
        s=14,
        alpha=0.55,
        marker="^",
        facecolors="none",
        edgecolors="C1",
        linewidths=0.7,
        label=labelAL,
        zorder=3,
    )
    ax.set_title("FF density (hexbin) + AL overlay")
    ax.legend(frameon=True, loc="best")
    return hb  # 用于 colorbar


def nn_distance_stats(dataAL, dataFF):
    """
    计算 AL -> FF 最近邻距离（高维特征空间中更可信）
    同时给出 FF -> FF 的 2nd NN 距离作为基线。
    """
    # 标准化再算距离（强烈建议）
    scaler = StandardScaler()
    X = np.vstack([dataFF, dataAL])
    Xs = scaler.fit_transform(X)
    XFF = Xs[: len(dataFF)]
    XAL = Xs[len(dataFF) :]

    # FF -> FF（排除自身最近邻，用 n_neighbors=2 取第二近）
    nn_ff = NearestNeighbors(n_neighbors=2).fit(XFF)
    d_ff, _ = nn_ff.kneighbors(XFF)
    d_ff = d_ff[:, 1]

    # AL -> FF
    nn = NearestNeighbors(n_neighbors=1).fit(XFF)
    d_al, _ = nn.kneighbors(XAL)
    d_al = d_al[:, 0]

    def summary(x):
        return {
            "min": float(np.min(x)),
            "p5": float(np.percentile(x, 5)),
            "median": float(np.median(x)),
            "p95": float(np.percentile(x, 95)),
            "max": float(np.max(x)),
            "mean": float(np.mean(x)),
        }

    return d_ff, d_al, summary(d_ff), summary(d_al)


def plot_nn_distance_hist(ax, d_ff, d_al):
    ax.hist(d_ff, bins=60, alpha=0.55, label="FF → FF (2nd NN)")
    ax.hist(d_al, bins=60, alpha=0.55, label="AL → FF (1st NN)")
    ax.set_title("Nearest-neighbor distance in (standardized) feature space")
    ax.set_xlabel("distance")
    ax.set_ylabel("count")
    ax.legend(frameon=True, loc="best")


if __name__ == "__main__":
    # ========= 输入 =========
    # pathA = r"/home/giga/BIG/data/AmmoniumNitrate/al/work_an_revpbe_4090_c8352Y_v1/data/ff600_label.xyz"  # 经验力场（FF）
    # pathB = r"/home/giga/BIG/data/AmmoniumNitrate/al/work_an_revpbe_4090_c8352Y_v1/data/v1d10_label.xyz"  # 主动学习（AL）
    path = r"/home/giga/BIG/data/lipfecpc/work_lipfecpc_pbe_4090_c8352Y_v7/data/v7d13_label.xyz"

    num_feats = 32
    only_num = 168  # 过滤原子数一致   100 130 pc216  ec168
    node_stride = 1  # 每个结构抽样原子节点的间隔

    method = "tsne"  # "pca" / "tsne" / "umap"
    n_components = 2

    # ========= 1) 读特征 =========
    dataFF, dataAL = load_node_feat_mean(
        path, num_feats=num_feats, only_num=only_num, node_stride=node_stride
    )

    # ========= 2) 降维（合并fit） =========
    embFF, embAL = reduce_dim_two_sets(
        dataFF, dataAL, method=method, n_components=n_components, random_state=42
    )

    with plt.style.context(["science"]):
        fig, ax = plt.subplots(figsize=(5, 5), dpi=250)

        # ====== 坐标轴范围（只用 B 决定，更稳健） ======
        # x_min, x_max = embB[:, 0].min() - 0.1, embB[:, 0].max() + 0.1
        # y_min, y_max = embB[:, 1].min() - 0.1, embB[:, 1].max() + 0.1
        # ax.set_xlim(x_min, x_max)
        # ax.set_ylim(y_min, y_max)

        # ====== Path B：大规模背景点（弱化） ======
        ax.scatter(
            embAL[:, 0],
            embAL[:, 1],
            s=10,  # 小点
            marker=".",
            # c="C1",
            alpha=0.4,  # 关键：低透明度
            # rasterized=True,  # 论文级优化（PDF/LaTeX）
            label="Exploration data",
            zorder=2,
        )

        # ====== Path A：少量重要点 ======
        ax.scatter(
            embFF[:, 0],
            embFF[:, 1],
            s=10,
            # c="C0",
            edgecolors="k",  # 黑边增强可读性
            linewidths=0.6,
            label="Train data",
            zorder=1,
        )

        # # ====== Path C：最关键点 ======
        # ax.scatter(
        #     embC[:, 0],
        #     embC[:, 1],
        #     s=10,
        #     marker="D",
        #     # c="C2",
        #     edgecolors="k",
        #     linewidths=0.6,
        #     label="Selected data",
        #     zorder=4,
        # )

        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")

        ax.legend(
            loc="upper right",
            # frameon=True,
            # framealpha=0.9,
            # handlelength=1.2,
        )

        plt.tight_layout()
        plt.show()

    print("Congratulations!!")
