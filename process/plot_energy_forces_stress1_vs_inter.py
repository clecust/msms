#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
绘制能量、力（散点图）和能量/应力随帧变化的曲线图（支持多个模型）
"""

import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from ase.io import read


def get_forces(path, prefix="forces", index=":", force_label="forces"):
    # label = f"{prefix}forces"
    label = force_label

    atoms = read(path, format="extxyz", index=index)
    if not isinstance(atoms, list):
        atoms = [atoms]

    try:
        forces = [atom.arrays[label] for atom in atoms]
    except:
        forces = [atom._calc.results[label] for atom in atoms]

    return np.vstack(forces).flatten()

if __name__ == "__main__":
    index = ":"
    # forces_type = "forces_rot"
    # forces_type = "forces_trans"

    forces_type = "forces_inter"
    # forces_type = "forces_intra"

    # # # # ############ ec/emc data
    xyz_ref = "/home/giga/BIG/data/ML_TrainTest_ECEMC/al-new/rigid_test/pbe_tzv2p_nomol_ot_d3bj_virial_cut800_LiquidConfigs_test40_split.xyz"
    model_paths = [
        (
            "/home/giga/BIG/data/ML_TrainTest_ECEMC/al-new/rigid_test/pbe_tzv2p_nomol_ot_d3bj_virial_cut800_LiquidConfigs_test40_label_split.xyz",
            "v1d2_s9",
        ),
        # 可添加更多模型路径
    ]

    force_true = get_forces(
        xyz_ref, force_label=forces_type, index=index
    )  # MACE_    REF_

    with plt.style.context(["science", "grid"]):
        fig_all, ax_all = plt.subplots(1, 1, figsize=(6, 6),dpi=250)

        # 画 DFT 参考线
        ax_all.plot(force_true, force_true, "k-", label="DFT")

        title_lines = []

        if len(model_paths) > 0:
            for i, (path, prefix) in enumerate(model_paths):
                label = prefix if prefix.strip() else f"Model{i+1}"
                force_pred = get_forces(path, force_label=forces_type, index=index)

                rmse_f = np.sqrt(np.mean((force_true - force_pred) ** 2)) *1e3
                r2_f = np.corrcoef(force_true, force_pred)[0, 1] ** 2  
                mae_f = np.mean(np.abs(force_true - force_pred)) * 1e3

                print(
                    f"{label} -> Force : RMSE={rmse_f:.4f}, R2={r2_f:.4f}, MAE={mae_f:.4f}"
                )

                ax_all.plot(force_true, force_pred, "+", label=label)

                title_lines.append(
                    f"{label}: RMSE={rmse_f:.4f}, R$^2$={r2_f:.4f}, MAE={mae_f:.4f}"
                )

        ax_all.set_xlabel("DFT Force (eV/Å)")
        ax_all.set_ylabel("Predicted Force (eV/Å)")

        if title_lines:
            ax_all.set_title("\n".join(title_lines))
        else:
            ax_all.set_title("Force Prediction")

        ax_all.legend()
        plt.tight_layout()
        plt.show()

    print("All plots generated successfully.")
