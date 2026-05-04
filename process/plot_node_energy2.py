import os
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

from ase.io import read, write
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

import umap  # 保留 UMAP


def load_node_feat_mean(path, num_feats=64,only_num=None):
    """
    读取 extxyz，取每个结构的 MACE_node_feats，按原子平均得到 (n_structures, num_feats)
    """
    atoms = read(path, format="extxyz", index=":")
    # if max_num is None:
    #     max_nums = [len(atom) for atom in atoms]
    #     max_num = max(max_nums)
    #     print(
    #         f"[{os.path.basename(path)}] max_atoms_num : {max_num}, n_structures: {len(atoms)}"
    #     )

    # data = np.zeros((len(atoms), max_num, num_feats))
    data_list = []
    for ii, atom in enumerate(atoms):
        num = len(atom)
        if only_num is not None and only_num != num :
            pass
        else:
            # data[ii, :num, :] = atom.arrays["MACE_node_feats"]
            data_list.append(atom.arrays["MACE_node_feats"].mean(0)) # 压缩到一个样本的特征维度
            # data_list.append(atom.arrays["MACE_node_feats"])
            # data_list.append(atom.arrays["MACE_node_feats"].mean(0))
        # data_list.append(atom.arrays["MACE_node_feats"])

    data = np.vstack(data_list)
    # data = data.mean(1)  # (n_structures, num_feats)
    return data, atoms


def reduce_dim_two_sets(
    dataA,
    dataB,
    dataC=None,
    method="tsne",
    n_components=3,
    random_state=42,
    umap_n_neighbors=15,
    umap_min_dist=1.0,
    tsne_perplexity=30,
):
    """
    核心改动：fit 在合并数据上做，然后再切分回 A/B
    """
    nA = len(dataA)
    nB = len(dataB)

    # 拼接所有有效数据（A+B，若有C则加C）
    if dataC is not None:
        nC = len(dataC)
        X = np.vstack([dataA, dataB, dataC])
    else:
        X = np.vstack([dataA, dataB])

    if method.lower() == "pca":
        # PCA 通常建议先标准化
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        reducer = PCA(n_components=n_components, random_state=random_state)
        emb = reducer.fit_transform(Xs)

    elif method.lower() == "tsne":
        # t-SNE 没有 transform，只能一次性对合并后的 X fit_transform
        perplexity = 5 # min(tsne_perplexity, max(5, (len(X) - 1) // 3))
        print(f"perplexity = {perplexity}")
        reducer = TSNE(
            n_components=n_components,
            random_state=random_state,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
        )
        emb = reducer.fit_transform(X)

    elif method.lower() == "umap":
        # UMAP 通常建议先标准化
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
        raise ValueError(f"Unknown method: {method}. Use 'pca', 'tsne', or 'umap'.")

    embA = emb[:nA]
    embB = emb[nA:nA+nB]

    # 根据dataC是否存在，返回对应结果
    if dataC is not None:
        embC = emb[nA+nB:]
        return embA, embB, embC
    else:
        return embA, embB


def kmeans_pick_closest(embedding, n_clusters=30, random_state=42):
    """
    在 embedding 上做 KMeans，返回每个 cluster 最近中心的点索引（相对 embedding）
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(embedding)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    values = embedding

    closest_pt_idx = []
    for iclust in range(kmeans.n_clusters):
        cluster_pts_indices = np.where(labels == iclust)[0]
        cluster_pts = values[cluster_pts_indices]
        cluster_cen = centers[iclust]
        r_dist = np.linalg.norm(cluster_cen[np.newaxis, :] - cluster_pts, axis=1)
        min_idx = np.argmin(r_dist)
        closest_pt_idx.append(cluster_pts_indices[min_idx])

    return np.array(closest_pt_idx, dtype=int), kmeans


if __name__ == "__main__":
    # ========= 你需要改的输入 =========
    # pathA = r"/home/giga/BIG/data/AmmoniumNitrate/al/work_an_revpbe_4090_c8352Y_v1/data/ff600_label.xyz"
    pathA = r"/home/giga/BIG/data/lipfecpc/work_lipfecpc_pbe_4090_c8352Y_v7/data/v7d13_216_init_label.xyz"
    # pathB = r"/home/giga/BIG/data/AmmoniumNitrate/al/work_an_revpbe_4090_c8352Y_v1/data/v1d10_label.xyz"
    # pathB = "/home/giga/BIG/data/AmmoniumNitrate/al/work_an_revpbe_4090_c8352Y_v2/data/AL_0_label.xyz"
    # pathB = "/home/giga/BIG/data/AmmoniumNitrate/al/work_an_revpbe_4090_c8352Y_v2/iter_0/eval/need_label_label.xyz"
    pathB = "/home/giga/BIG/data/lipfecpc/work_lipfecpc_pbe_4090_c8352Y_v7/data/expolration_216_label.xyz"
    pathC = "/home/giga/BIG/data/lipfecpc/work_lipfecpc_pbe_4090_c8352Y_v7/data/v7d13_216_al_label.xyz"

    num_feats = 32
    # max_num = 108
    only_num = None  # 72 108  84 60

    # 降维方式： "pca" / "tsne" / "umap"
    method = "pca"
    n_components = 2

    # KMeans 设置（可选）
    do_kmeans = False
    n_clusters = 30

    # 是否分别对 A/B 各做一次 KMeans 并各写 xyz
    kmeans_separately = False
    # 如果你想在“合并后的 embedding 上做一次 KMeans”，改成 False
    # （注意：那样选出的结构会混在 A/B 两个集合里）

    # ========= 1) 读两个 path 的特征 =========
    dataA, atomsA = load_node_feat_mean(
        pathA, num_feats=num_feats,  only_num=only_num
    )
    dataB, atomsB = load_node_feat_mean(
        pathB, num_feats=num_feats,  only_num=only_num
    )
    dataC, atomsC = load_node_feat_mean(
        pathC, num_feats=num_feats, only_num=only_num
    )
    print(f"finsh load_node_feat_mean")
    # ========= 2) fit 在合并数据上做降维，然后切回 A/B =========
    embA, embB, embC = reduce_dim_two_sets(
        dataA, dataB,dataC, method=method, n_components=n_components, random_state=42
    )
    print(f"finsh reduce_dim_two_sets")

    # ========= 3) 可选：KMeans + 保存挑出来的结构 =========
    # if do_kmeans:
    #     if kmeans_separately:
    #         idxA, _ = kmeans_pick_closest(embA, n_clusters=n_clusters, random_state=42)
    #         idxB, _ = kmeans_pick_closest(embB, n_clusters=n_clusters, random_state=42)

    #         saveA = f"{os.path.splitext(pathA)[0]}_{method}_k{n_clusters}_final.xyz"
    #         saveB = f"{os.path.splitext(pathB)[0]}_{method}_k{n_clusters}_final.xyz"

    #         atomsA_save = [atomsA[i] for i in idxA]
    #         atomsB_save = [atomsB[i] for i in idxB]

    #         print(f"[SAVE] {saveA}  n={len(atomsA_save)}")
    #         print(f"[SAVE] {saveB}  n={len(atomsB_save)}")
    #         write(saveA, atomsA_save, format="extxyz", append=False)
    #         write(saveB, atomsB_save, format="extxyz", append=False)

    #     else:
    #         emb_all = np.vstack([embA, embB])
    #         idx_all, _ = kmeans_pick_closest(
    #             emb_all, n_clusters=n_clusters, random_state=42
    #         )

    #         # idx_all 是合并后的索引，要映射回 A/B
    #         nA = len(embA)
    #         idxA = idx_all[idx_all < nA]
    #         idxB = idx_all[idx_all >= nA] - nA

    #         saveA = f"{os.path.splitext(pathA)[0]}_{method}_k{n_clusters}_final.xyz"
    #         saveB = f"{os.path.splitext(pathB)[0]}_{method}_k{n_clusters}_final.xyz"

    #         atomsA_save = [atomsA[i] for i in idxA]
    #         atomsB_save = [atomsB[i] for i in idxB]

    #         print(f"[SAVE] {saveA}  n={len(atomsA_save)}")
    #         print(f"[SAVE] {saveB}  n={len(atomsB_save)}")
    #         # write(saveA, atomsA_save, format="extxyz", append=False)
    #         # write(saveB, atomsB_save, format="extxyz", append=False)

    # ========= 4) 画图：同一张图上显示两个 path =========
    # # 2D 画前两维；如果你想 3D，我也可以再给你加
    # with plt.style.context(["science"]):
    #     plt.figure(figsize=(5, 4))

    #     plt.scatter(
    #         embA[:, 0],
    #         embA[:, 1],
    #         s=18,
    #         alpha=0.6,
    #         c="C0",
    #         label="Path A",
    #         zorder=3,
    #     )

    #     plt.scatter(
    #         embB[:, 0],
    #         embB[:, 1],
    #         s=40,
    #         marker="^",
    #         facecolors="none",  # 关键：空心
    #         edgecolors="C1",
    #         linewidths=0.8,
    #         label="Path B",
    #         zorder=2,  # 画在上面
    #     )

    #     plt.title(f"Dimensionality Reduction ({method.upper()})")
    #     plt.xlabel("dim-1")
    #     plt.ylabel("dim-2")
    #     plt.legend(frameon=True)
    #     plt.tight_layout()
    #     plt.show()

    # # 核心绘图代码
    # with plt.style.context(["science"]):
    #     # 创建1行2列的子图，设置画布尺寸（宽度加倍，高度保持）
    #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)

    #     # 计算所有数据的x、y轴极值（保证坐标轴范围一致的核心）
    #     all_x = np.concatenate([embA[:, 0], embB[:, 0]])
    #     all_y = np.concatenate([embA[:, 1], embB[:, 1]])
    #     x_min, x_max = all_x.min() - 0.1, all_x.max() + 0.1  # 留少量边距
    #     y_min, y_max = all_y.min() - 0.1, all_y.max() + 0.1

    #     # ========== 第一张子图 ==========
    #     ax1.scatter(
    #         embA[:, 0],
    #         embA[:, 1],
    #         # s=18,
    #         # alpha=0.6,
    #         # c="C0",
    #         label="Path A",
    #         # zorder=3,
    #     )

    #     # ax1.set_title(f"Subplot 1 ({method.upper()})")  # 子图1标题

    #     ax1.scatter(
    #         embB[:, 0],
    #         embB[:, 1],
    #         # s=40,
    #         # marker="^",
    #         # facecolors="none",
    #         # # edgecolors="C1",
    #         # linewidths=0.8,
    #         label="Path B",
    #         # zorder=2,
    #     )
    #     # ax1.set_title(f"Subplot 2 ({method.upper()})")  # 子图2标题

    #     ax1.scatter(
    #         embC[:, 0],
    #         embC[:, 1],
    #         # s=40,
    #         # marker=".",
    #         # facecolors="none",
    #         # edgecolors="C2",
    #         label="Path C",
    #     )
    #     # ========== 统一设置坐标轴（保证一致性） ==========
    #     # 统一设置x/y轴范围
    #     ax1.set_xlim(x_min, x_max)
    #     ax1.set_ylim(y_min, y_max)
    #     # 统一设置轴标签（只在外侧显示，更美观）
    #     fig.supxlabel("dim-1")
    #     fig.supylabel("dim-2")
    #     # 统一设置图例（只显示一次）
    #     ax1.legend(frameon=True, loc="best")

    # # 调整布局，避免元素重叠
    # plt.tight_layout()
    # plt.show()

    with plt.style.context(["science"]):
        fig, ax = plt.subplots(figsize=(5, 5),dpi=250)

        # ====== 坐标轴范围（只用 B 决定，更稳健） ======
        # x_min, x_max = embB[:, 0].min() - 0.1, embB[:, 0].max() + 0.1
        # y_min, y_max = embB[:, 1].min() - 0.1, embB[:, 1].max() + 0.1
        # ax.set_xlim(x_min, x_max)
        # ax.set_ylim(y_min, y_max)

        # ====== Path B：大规模背景点（弱化） ======
        ax.scatter(
            embB[:, 0],
            embB[:, 1],
            s=10,  # 小点
            marker=".",
            # c="C1",
            alpha=0.4,  # 关键：低透明度
            # rasterized=True,  # 论文级优化（PDF/LaTeX）
            label="Exploration data",
            zorder=1,
        )

        # ====== Path A：少量重要点 ======
        ax.scatter(
            embA[:, 0],
            embA[:, 1],
            s=10,
            # c="C0",
            edgecolors="k",  # 黑边增强可读性
            linewidths=0.6,
            label="Train data",
            zorder=3,
        )

        # ====== Path C：最关键点 ======
        ax.scatter(
            embC[:, 0],
            embC[:, 1],
            s=10,
            marker="D",
            # c="C2",
            edgecolors="k",
            linewidths=0.6,
            label="Selected data",
            zorder=4,
        )

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
