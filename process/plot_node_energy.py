#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
安装：　　conda install scikit-learn plotly umap-learn scienceplots
方法：　使用PCA、TSNE、UMAP降维，使用KMeans聚类，找到最接近聚类中心的原子，保存为extxyz文件．
结果：
有机体系存在共价键，降维后像曲线．
金属／合金／电池体系降维后偏向团簇．
"""

import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import colormaps
import scienceplots
import pandas as pd
import umap
from ase.io import read, write
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import plotly.express as px


if __name__ == "__main__":
    ############### 1 ###############
    n_clusters = 30
    # path = r"/home/giga/BIG/data/AmmoniumNitrate/al/work_an_revpbe_4090_c8352Y_v1/data/ff600_label.xyz"
    path = r"/home/giga/BIG/data/AmmoniumNitrate/al/work_an_revpbe_4090_c8352Y_v1/data/v1d10_label.xyz"

    num_feats = 64
    atoms = read(path, format="extxyz", index=":")

    # data = []
    # max_nums = [len(atom) for atom in atoms]
    # max_num = max(max_nums)
    # print(f"max_atoms_num : {max_num}")
    # data = np.zeros((len(atoms), max_num, num_feats))
    # for ii, atom in enumerate(atoms):
    #     num = len(atom)
    #     data[ii, :num, :] = atom.arrays["MACE_node_feats"]
    # # data = data.reshape(-1, max_num * num_feats)
    # data = data.mean(1)

    data_list = []
    for ii, atom in enumerate(atoms):
        num = len(atom)
        data_list.append(atom.arrays["MACE_node_feats"])
        # data_list.append(atom.arrays["MACE_node_feats"].mean(0))

    data = np.vstack(data_list)
    # # # save data
    # df = pd.DataFrame(data)
    # save_path = f"{path.split('.')[0]}_data.csv"
    # df.to_csv(save_path, index=True, header=True)
    # atoms = None

    ############### 2 ###############
    # save_path_low = f"{path.split('.')[0]}_data_low.csv"
    ## load data
    # data = pd.read_csv(save_path, index_col=0)


    # # # TSNE
    # tsne = TSNE(n_components=2, random_state=42)
    # embedding = tsne.fit_transform(data)
    # df = pd.DataFrame(embedding)
    # df.to_csv(save_path_low, index=True, header=True)

    # # # PCA
    # pca = PCA(n_components=2)
    # pca.fit(data)
    # print(pca.explained_variance_ratio_)
    # embedding = pca.transform(data)
    # df = pd.DataFrame(embedding)
    # df.to_csv(save_path_low, index=True, header=True)

    # # # ## UMAP
    scaled_data = StandardScaler().fit_transform(data)
    reducer = umap.UMAP(
        n_components=2,
        random_state=42,
        n_neighbors=20,
        min_dist=1.0,
    )
    embedding = reducer.fit_transform(scaled_data)
    # # save data and embedding
    # df = pd.DataFrame(embedding)
    # df.to_csv(save_path_low, index=True, header=True)

    ############### 3 KMeans ###############
    # save_path_kmeans = f"{path.split('.')[0]}_data_kmeans.csv"

    # # # load data
    # embedding = pd.read_csv(save_path_low, index_col=0)
    # # KMeans
    # kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(embedding)
    # cluster_centers_ = kmeans.cluster_centers_
    # # 离聚类中心最近的原子序号
    # values = embedding.values
    # closest_pt_idx = []
    # for iclust in range(kmeans.n_clusters):
    #     # get all points assigned to each cluster:
    #     cluster_pts_indices = np.where(kmeans.labels_ == iclust)[0]
    #     cluster_pts = values[cluster_pts_indices]
    #     cluster_cen = kmeans.cluster_centers_[iclust]
    #     r_dist = np.linalg.norm(cluster_cen[np.newaxis, :] - cluster_pts, axis=1)
    #     min_idx = np.argmin(r_dist)
    #     closest_pt_idx.append(cluster_pts_indices[min_idx])
    # #
    # df = pd.DataFrame(closest_pt_idx)
    # df.to_csv(save_path_kmeans, index=True, header=True)
    # #
    # # ############### 4  save ###############
    # save_path_final = f"{path.split('.')[0]}_data_final.xyz"

    # # load data
    # closest_pt_idx = pd.read_csv(save_path_kmeans, index_col=0)
    # atoms = read(path, index=":")
    # atoms_save = [atoms[ii] for ii in closest_pt_idx.values.flatten()]
    # print(len(atoms_save))
    # print(save_path_final)
    # write(save_path_final, atoms_save, format="extxyz", append=False)

    # ############### 5  plot ###############
    ## plot
    # embedding = pd.read_csv(save_path_low, index_col=0)
    # embedding = embedding.values
    c = list(range(len(embedding)))
    x = embedding[:, 0]
    y = embedding[:, 1]
    # z = embedding[:, 2]
    # k = pd.read_csv(save_path_kmeans, index_col=0).values.flatten()
    # embeddingk = embedding[k]
    # x1 = embeddingk[:, 0]
    # y1 = embeddingk[:, 1]
    # z1 = embeddingk[:, 2]

    with plt.style.context(["science"]):
        # fig = px.scatter_3d(
        #     x=x,
        #     y=y,
        #     z=z,
        #     color=c,
        #     size_max=8,
        #     opacity=0.8,
        #     color_continuous_scale="Jet",
        # )  # 颜色越深，代表MD越往后

        # fig.update_layout(
        #     scene=dict(xaxis_title="X轴", yaxis_title="Y轴", zaxis_title="Z轴")
        # )
        # fig.add_trace(
        #     px.scatter_3d(
        #         x=x1,
        #         y=y1,
        #         z=z1,
        #         color=np.zeros_like(x1) - len(embedding),
        #         size_max=15,
        #         opacity=0.8,
        #         color_continuous_scale="brbg",
        #     ).data[0]
        # )  # 0 代表采样点
        # # "Viridis", "jet" "RdBu" "Bluered" "Picnic"
        # # "Portland" "Jet" "Hot" "Blackbody"
        # # "Earth" "Electric" "YIOrRd" "YIGnBu"
        # fig.show()

        # 2D
        plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=c,
            cmap=plt.cm.get_cmap("jet", len(embedding)),
        )
        # ax.colorbar()
        plt.show()

    print("Congratulations!!")
