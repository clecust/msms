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


def load_node_feat_mean(path, num_feats=64, max_num=None, only_num=None):
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
        if only_num is not None and only_num != num:
            pass
        else:
            # data[ii, :num, :] = atom.arrays["MACE_node_feats"]
            data_list.append(
                atom.arrays["MACE_node_feats"].mean(0)
            )  # 压缩到一个样本的特征维度
            # data_list.append(atom.arrays["MACE_node_feats"])
            # data_list.append(atom.arrays["MACE_node_feats"].sum(0))
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
        perplexity = 5  # min(tsne_perplexity, max(5, (len(X) - 1) // 3))
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
    embB = emb[nA : nA + nB]

    # 根据dataC是否存在，返回对应结果
    if dataC is not None:
        embC = emb[nA + nB :]
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
    pathA = r"/home/giga/BIG/data/ML_TrainTest_ECEMC/al-new/work_ecemc_pbe_4090_c8352Y_v1/data/v1d13_160_init_label.xyz"
    pathB = "/home/giga/BIG/data/ML_TrainTest_ECEMC/al-new/work_ecemc_pbe_4090_c8352Y_v1/data/expolration_160_label.xyz"
    pathC = "/home/giga/BIG/data/ML_TrainTest_ECEMC/al-new/work_ecemc_pbe_4090_c8352Y_v1/data/v1d13_160_al_label.xyz"

    num_feats = 32
    max_num = None
    only_num = None  # 72 108  84 60

    # 降维方式： "pca" / "tsne" / "umap"
    method = "tsne"
    n_components = 2
    save_dir = r"./embeddings"
    cal = False

    if cal:
        dataA, atomsA = load_node_feat_mean(
            pathA, num_feats=num_feats, max_num=max_num, only_num=only_num
        )
        dataB, atomsB = load_node_feat_mean(
            pathB, num_feats=num_feats, max_num=max_num, only_num=only_num
        )
        dataC, atomsC = load_node_feat_mean(
            pathC, num_feats=num_feats, max_num=max_num, only_num=only_num
        )
        print(f"finsh load_node_feat_mean")
        # ========= 2) fit 在合并数据上做降维，然后切回 A/B =========
        embA, embB, embC = reduce_dim_two_sets(
            dataA,
            dataB,
            dataC,
            method=method,
            n_components=n_components,
            random_state=42,
        )

        print(f"finsh reduce_dim_two_sets")

        # 创建保存目录（不存在则自动创建）
        os.makedirs(save_dir, exist_ok=True)
        # 保存为numpy的npy格式（高效存储数组）
        np.save(os.path.join(save_dir, "embA.npy"), embA)
        np.save(os.path.join(save_dir, "embB.npy"), embB)
        np.save(os.path.join(save_dir, "embC.npy"), embC)
        print(f"嵌入向量已保存到：{save_dir}")
    else:
        # ========= 4) 从本地读取嵌入向量（新增核心代码） =========
        print("loading ...")
        embA = np.load(os.path.join(save_dir, "embA.npy"))
        embB = np.load(os.path.join(save_dir, "embB.npy"))
        embC = np.load(os.path.join(save_dir, "embC.npy"))
    print('plot')


    with plt.style.context(["science"]):
        fig, ax = plt.subplots(figsize=(4.2, 4.0), dpi=250)

        # ====== 每轮 iter 对应的样本数 ======
        iter_totals = [
            (0, 602),
            (1, 802),
            (2, 1002),
            (3, 2002),
            (4, 2002),
            (5, 2002),
            (6, 2002),
            (7, 2002),
            (8, 2002),
            (9, 2002),
            # (10, 1001),
            # (11, 1001),
            # (12, 1001),
            # (13, 1001)
        ]

        # ====== 颜色：按 iter 顺序渐变，更适合表示轮次 ======
        # colors = plt.cm.viridis(np.linspace(0.08, 0.95, len(iter_totals)))
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i) for i in range(len(iter_totals))]

        start = 0
        for (iter_id, total), color in zip(iter_totals, colors):
            end = start + total
            embB_i = embB[start:end]

            ax.scatter(
                embB_i[:, 0],
                embB_i[:, 1],
                s=10,
                marker=".",
                c=[color],
                alpha=0.55,
                linewidths=0.0,
                label=f"iter {iter_id}",
                zorder=1,
            )

            # 2) 计算并绘制均值点
            mean_x = embB_i[:, 0].mean()
            mean_y = embB_i[:, 1].mean()

            ax.scatter(
                mean_x,
                mean_y,
                s=42,
                marker="o",
                c=[color],
                edgecolors="k",
                linewidths=0.8,
                zorder=3,
            )

            # 3) 标注 iter 编号
            ax.text(
                mean_x,
                mean_y,
                f"{iter_id}",
                fontsize=5.5,
                ha="left",
                va="bottom",
                zorder=4,
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    fc="white",
                    ec="none",
                    alpha=0.75,
                ),
            )
            start = end

        # ====== 安全检查：防止 embB 行数和 total 累加不一致 ======
        if start != len(embB):
            print(f"Warning: used rows = {start}, but len(embB) = {len(embB)}")

        ax.set_xlabel("Learned atomic representation 1")
        ax.set_ylabel("Learned atomic representation 2")

        ax.legend(
            loc="best",
            frameon=True,
            framealpha=0.9,
            fontsize=5.2,
            handlelength=1.0,
            ncol=2,
        )

        plt.tight_layout()
        plt.savefig("./plot_embB_by_iter.pdf")
        # plt.show()

    print("Congratulations!!")
# with plt.style.context(["science"]):
#     fig, ax = plt.subplots(figsize=(4, 4), dpi=250)

#     # # ====== Path A：少量重要点 ======
#     # ax.scatter(
#     #     embA[:, 0],
#     #     embA[:, 1],
#     #     s=15,
#     #     c="#00B945",
#     #     marker="^",
#     #     edgecolors="k",
#     #     linewidths=0.6,
#     #     label="Initial",
#     #     zorder=3,
#     # )

#     # ====== Path B：按区间分成四块并分别着色 ======
#     # 0 : 2406      -> 常规
#     # 2406 : 8412   -> 噪声
#     # 8412 : 16420  -> 密度
#     # 16420 : 20424 -> 刚性

#     embB_regular = embB[0:2406]
#     embB_noise = embB[2406:8412]
#     embB_density = embB[8412:16420]
#     embB_rigid = embB[16420:20424]

#     ax.scatter(
#         embB_regular[:, 0],
#         embB_regular[:, 1],
#         s=12,
#         marker=".",
#         c="#0C5DA5",
#         alpha=0.45,
#         linewidths=0.6,
#         label="Exploration-Regular",
#         zorder=1,
#     )

#     ax.scatter(
#         embB_noise[:, 0],
#         embB_noise[:, 1],
#         s=12,
#         marker=".",
#         c="#00B945",
#         alpha=0.45,
#         linewidths=0.6,
#         label="Exploration-Noise",
#         zorder=1,
#     )

#     ax.scatter(
#         embB_density[:, 0],
#         embB_density[:, 1],
#         s=12,
#         marker=".",
#         c="#FF9500",
#         alpha=0.45,
#         linewidths=0.6,
#         label="Exploration-Density",
#         zorder=1,
#     )

#     ax.scatter(
#         embB_rigid[:, 0],
#         embB_rigid[:, 1],
#         s=12,
#         marker=".",
#         c="#845EC2",
#         alpha=0.45,
#         linewidths=0.6,
#         label="Exploration-Rigid",
#         zorder=1,
#     )

#     # # ====== Path C：最关键点 ======
#     # ax.scatter(
#     #     embC[:, 0],
#     #     embC[:, 1],
#     #     s=15,
#     #     marker="v",
#     #     c="#FF9500",
#     #     edgecolors="k",
#     #     linewidths=0.6,
#     #     label="Selected",
#     #     zorder=4,
#     # )

#     ax.set_xlabel("Learned atomic representation 1")
#     ax.set_ylabel("Learned atomic representation 2")

#     ax.legend(
#         loc="best",
#         frameon=True,
#         framealpha=0.9,
#         handlelength=1.2,
#         fontsize=5.8,
#     )

#     plt.tight_layout()
#     plt.savefig("./plot_node_energy2v1.pdf")
#     # plt.show()

# print("Congratulations!!")
