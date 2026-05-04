#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""

"""
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from ase.io import read, write


def get_forces(path, prefix="", index=":"):
    label = f"{prefix}forces"
    elabel = f"{prefix}energy"
    atoms = read(path, format="extxyz", index=index)
    if not isinstance(atoms, list):
        atoms = [atoms]
    try:
        forces = [atom.arrays[label] for atom in atoms]
    except:
        forces = [atom._calc.results[label] for atom in atoms]
    try:
        energy = [atom.info[elabel] for atom in atoms]
    except:
        energy = [atom._calc.results[elabel] for atom in atoms]
    forces = np.array(forces).flatten()
    energy = np.array(energy).flatten()
    energy -= energy.min()
    pos = np.array([atom.positions for atom in atoms]).flatten()
    return energy, forces, pos

if __name__ == "__main__":
    # # 原始数据路径（DFT参考）
    # xyz_ref = "/home/giga/BIG/data/md22/bond_test/md22_DHA_3000_bond_oh_gfn2.xyz"
    # energy_true, force_true, pos_true = get_forces(xyz_ref, prefix="REF_", index=":")
    # info = "rebond"  # rebond  original
    # bond_index = [1,56]
    # # 所有预测路径和前缀，添加更多模型只需在此添加路径即可
    # model_paths = [
    #     (
    #         f"/home/giga/BIG/data/md22/bond_test/md22_DHA_3000_bond_oh_gfn2_1{info}.xyz",
    #         "MACE_",
    #     ),
    #     (
    #         f"/home/giga/BIG/data/md22/bond_test/md22_DHA_3000_bond_oh_gfn2_2{info}.xyz",
    #         "MACE_",
    #     ),
    #     (
    #         f"/home/giga/BIG/data/md22/bond_test/md22_DHA_3000_bond_oh_gfn2_3{info}.xyz",
    #         "MACE_",
    #     ),
    #     # 添加更多模型时在此添加路径和前缀
    #     # ("/path/to/other_model.xyz", "MACE_"),
    # ]
    #

    # # 原始数据路径（DFT参考）
    # xyz_ref = "/home/giga/BIG/data/md22/bond_test/AT_NH5_gfn2.xyz"
    # energy_true, force_true, pos_true = get_forces(xyz_ref, prefix="REF_", index=":")
    # info = "original"  # rebond  original
    # bond_index = [16, 25]
    # # 所有预测路径和前缀，添加更多模型只需在此添加路径即可
    # model_paths = [
    #     (
    #         f"/home/giga/BIG/data/md22/bond_test/AT_NH5_gfn2_1{info}.xyz",
    #         "MACE_",
    #     ),
    #     (
    #         f"/home/giga/BIG/data/md22/bond_test/AT_NH5_gfn2_2{info}.xyz",
    #         "MACE_",
    #     ),
    #     (
    #         f"/home/giga/BIG/data/md22/bond_test/AT_NH5_gfn2_3{info}.xyz",
    #         "MACE_",
    #     ),
    #     # 添加更多模型时在此添加路径和前缀
    #     # ("/path/to/other_model.xyz", "MACE_"),
    # ]

    # 原始数据路径（DFT参考）
    xyz_ref = "/home/giga/BIG/data/md22/bond_test/aspirin_300k_NVT_gaff_bond_gfn2.xyz"
    energy_true, force_true, pos_true = get_forces(xyz_ref, prefix="REF_", index=":")
    info = "rebond"  # rebond  original
    bond_index = [10, 14]
    # 所有预测路径和前缀，添加更多模型只需在此添加路径即可
    model_paths = [
        (
            f"/home/giga/BIG/data/md22/bond_test/aspirin_300k_NVT_gaff_bond_gfn2_1{info}.xyz",
            "MACE_",
        ),
        (
            f"/home/giga/BIG/data/md22/bond_test/aspirin_300k_NVT_gaff_bond_gfn2_2{info}.xyz",
            "MACE_",
        ),
        (
            f"/home/giga/BIG/data/md22/bond_test/aspirin_300k_NVT_gaff_bond_gfn2_3{info}.xyz",
            "MACE_",
        ),
        (
            f"/home/giga/BIG/data/md22/bond_test/aspirin_300k_NVT_gaff_bond_gfn2_4{info}.xyz",
            "MACE_",
        ),
        (
            f"/home/giga/BIG/data/md22/bond_test/aspirin_300k_NVT_gaff_bond_gfn2_5{info}.xyz",
            "MACE_",
        ),
        # 添加更多模型时在此添加路径和前缀
        # ("/path/to/other_model.xyz", "MACE_"),
    ]

    # # 原始数据路径（DFT参考）
    # xyz_ref = "/home/giga/BIG/data/md22/bond_test/md22_ATv3_300k_NVT_gaff_bond_gfn2.xyz"
    # energy_true, force_true, pos_true = get_forces(xyz_ref, prefix="REF_", index=":")
    # info = "original"  # rebond  original
    # bond_index = [1, 10]
    # # 所有预测路径和前缀，添加更多模型只需在此添加路径即可
    # model_paths = [
    #     (
    #         f"/home/giga/BIG/data/md22/bond_test/md22_ATv3_300k_NVT_gaff_bond_gfn2_1{info}.xyz",
    #         "MACE_",
    #     ),
    #     (
    #         f"/home/giga/BIG/data/md22/bond_test/md22_ATv3_300k_NVT_gaff_bond_gfn2_2{info}.xyz",
    #         "MACE_",
    #     ),
    #     (
    #         f"/home/giga/BIG/data/md22/bond_test/md22_ATv3_300k_NVT_gaff_bond_gfn2_3{info}.xyz",
    #         "MACE_",
    #     ),
    #     # 添加更多模型时在此添加路径和前缀
    #     # ("/path/to/other_model.xyz", "MACE_"),
    # ]

    atoms = read(xyz_ref, format="extxyz", index=":")
    bond_length = [
        atom.get_distance(bond_index[0] - 1, bond_index[1] - 1) for atom in atoms
    ]
    with plt.style.context(["science", "grid"]):
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        ax[0].plot(energy_true, energy_true, "k-", label="DFT")
        ax[1].plot(force_true, force_true, "k-", label="DFT")
        ax[2].plot(bond_length,energy_true, "k-", label="DFT")

        for i, (path, prefix) in enumerate(model_paths):
            label = f"Model{i+1}"
            try:
                energy_pred, force_pred, _ = get_forces(path, prefix=prefix, index=":")
                # energy_pred -= energy_pred[-1]  # 归一化能量

                # 计算指标
                r2_e = np.corrcoef(energy_true, energy_pred)[0, 1] ** 2
                mae_e = np.mean(np.abs(energy_true - energy_pred))
                rmse_e = np.sqrt(np.mean((energy_true - energy_pred) ** 2))

                r2_f = np.corrcoef(force_true, force_pred)[0, 1] ** 2
                mae_f = np.mean(np.abs(force_true - force_pred))
                rmse_f = np.sqrt(np.mean((force_true - force_pred) ** 2))

                print(
                    f"{label} -> Energy: r2={r2_e:.4f}, mae={mae_e:.4f}, rmse={rmse_e:.4f}"
                )
                print(
                    f"{label} -> Force : r2={r2_f:.4f}, mae={mae_f:.4f}, rmse={rmse_f:.4f}"
                )

                # 绘图
                ax[0].plot(energy_true, energy_pred, "+", label=label)
                ax[1].plot(force_true, force_pred, "+", label=label)
                ax[2].plot(bond_length, energy_pred, "+", label=label)

            except Exception as e:
                print(f"Error processing {label}: {e}")

        ax[0].set_xlabel("DFT energy (eV)")
        ax[0].set_ylabel("Predicted energy (eV)")
        ax[0].set_title("Energy Prediction")
        ax[0].legend(loc="best")

        ax[1].set_xlabel("DFT force (eV/Å)")
        ax[1].set_ylabel("Predicted force (eV/Å)")
        ax[1].set_title("Force Prediction")
        ax[1].legend(loc="best")

        ax[2].set_xlabel("Frame Index")
        ax[2].set_ylabel("Energy (eV)")
        ax[2].set_title("Bond length vs Frame")
        ax[2].legend(loc="best")
        ax[2].set_ylim(-0.1, 4)

        plt.tight_layout()
        plt.savefig(f"force_energy_{info}.png")
        plt.show()

    print("Congratulations!!")
