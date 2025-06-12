#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
绘制能量、力（散点图）和能量/应力随帧变化的曲线图（支持多个模型）
"""

import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from ase.io import read


def get_forces(path, prefix="", index=":"):
    label = f"{prefix}forces"
    elabel = f"{prefix}energy"
    stress_label = f"{prefix}stress"

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

    try:
        stress = [atom.info[stress_label] for atom in atoms]
    except:
        stress = [atom._calc.results[stress_label] for atom in atoms]

    forces = np.array(forces).flatten()
    energy = np.array(energy).flatten()
    stress = np.array(stress).reshape(len(atoms), -1)  # shape: (Nframes, 6)
    pos = np.array([atom.positions for atom in atoms]).flatten()
    energy -= energy[-1]
    return energy, forces, stress, pos


if __name__ == "__main__":
    # 原始数据路径（DFT参考）
    # 原始数据路径（DFT参考）
    xyz_ref = "/home/giga/BIG/data/3M_F/3/scan/gfn.xyz"
    energy_true, force_true, stress_true, pos_true = get_forces(
        xyz_ref, prefix="REF_", index=":"
    )
    # stress_true *= -1
    # 所有预测路径和前缀
    model_paths = [
        ("/home/giga/BIG/data/3M_F/3/scan/gfn_out1.xyz", "MACE_"),
        ("/home/giga/BIG/data/3M_F/3/scan/gfn_out2.xyz", "MACE_"),
        # 可添加更多模型路径
    ]

    stress_labels = ["xx", "yy", "zz", "yz", "xz", "xy"]
    nframes = len(energy_true)
    x = np.arange(nframes)

    with plt.style.context(["science", "grid"]):
        # 图 1: energy, force, energy-vs-frame (3个子图)
        fig_main, ax_main = plt.subplots(1, 3, figsize=(18, 5))
        ax_energy, ax_force, ax_energy_vs_frame = ax_main

        # 图 2: 应力随帧变化（曲线图）
        fig_stress, ax_stress = plt.subplots(2, 3, figsize=(18, 10))
        ax_stress = ax_stress.flatten()

        # 画 DFT 曲线（实线）
        ax_energy.plot(energy_true, energy_true, "k-", label="DFT")
        ax_force.plot(force_true, force_true, "k-", label="DFT")
        ax_energy_vs_frame.plot(x, energy_true, "k-", label="DFT")

        for j in range(6):
            ax_stress[j].plot(x, stress_true[:, j], "k-", label="DFT")

        # 遍历模型预测
        for i, (path, prefix) in enumerate(model_paths):
            label = f"Model{i+1}"
            try:
                energy_pred, force_pred, stress_pred, _ = get_forces(
                    path, prefix=prefix, index=":"
                )
                energy_pred -= energy_pred[-1]

                # 输出评估指标（可选）
                r2_e = np.corrcoef(energy_true, energy_pred)[0, 1] ** 2
                mae_e = np.mean(np.abs(energy_true - energy_pred))
                r2_f = np.corrcoef(force_true, force_pred)[0, 1] ** 2
                mae_f = np.mean(np.abs(force_true - force_pred))
                print(f"{label} -> Energy: r2={r2_e:.4f}, MAE={mae_e:.4f}")
                print(f"{label} -> Force : r2={r2_f:.4f}, MAE={mae_f:.4f}")

                ax_energy.plot(energy_true, energy_pred, "+", label=label)
                ax_force.plot(force_true, force_pred, "+", label=label)
                ax_energy_vs_frame.plot(x, energy_pred, "+", label=label)

                for j in range(6):
                    ax_stress[j].plot(x, stress_pred[:, j], "+", label=label)

            except Exception as e:
                print(f"Error processing {label}: {e}")

        # 图 1 标签设置
        ax_energy.set_xlabel("DFT Energy (eV)")
        ax_energy.set_ylabel("Predicted Energy (eV)")
        ax_energy.set_title("Energy Prediction")
        ax_energy.legend()

        ax_force.set_xlabel("DFT Force (eV/Å)")
        ax_force.set_ylabel("Predicted Force (eV/Å)")
        ax_force.set_title("Force Prediction")
        ax_force.legend()

        ax_energy_vs_frame.set_xlabel("Frame Index")
        ax_energy_vs_frame.set_ylabel("Energy (eV)")
        ax_energy_vs_frame.set_title("Energy vs Frame")
        ax_energy_vs_frame.legend()

        # 图 2 标签设置
        for j in range(6):
            ax_stress[j].set_title(f"Stress Component: {stress_labels[j]}")
            ax_stress[j].set_xlabel("Frame Index")
            ax_stress[j].set_ylabel("Stress (GPa)")
            ax_stress[j].legend()

        fig_main.tight_layout()
        fig_stress.tight_layout()
        fig_main.savefig("force_energy_main.png")
        fig_stress.savefig("stress_vs_frame.png")
        plt.show()

    print("All plots generated successfully.")
