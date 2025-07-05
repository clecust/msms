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
    # max_pos = max([len(atom) for atom in atoms])
    # forces_list = []

    try:
        forces = [atom.arrays[label] for atom in atoms]
    except:
        forces = [atom._calc.results[label] for atom in atoms]

    try:
        energy = [atom.info[elabel] for atom in atoms]
    except:
        energy = [atom._calc.results[elabel] for atom in atoms]

    try:
        if stress_label in atoms[0].info:
            stress = [atom.info[stress_label] for atom in atoms]
        elif stress_label in atoms[0]._calc.results:
            stress = [atom._calc.results[stress_label] for atom in atoms]
        else:
            # 当stress_label不在上述两个位置时，直接初始化stress
            stress = np.zeros((len(atoms), 3, 3))
    except:
        print(f"Error:")
        # 出现异常时初始化stress
        stress = np.zeros((len(atoms), 3, 3))

    forces = np.vstack(forces).flatten()
    energy = np.array(energy).flatten()
    stress = np.array(stress)  # shape: (Nframes, 3, 3)
    stress6 = np.stack([
        stress[:, 0, 0],  # xx
        stress[:, 1, 1],  # yy
        stress[:, 2, 2],  # zz
        stress[:, 1, 2],  # yz
        stress[:, 0, 2],  # xz
        stress[:, 0, 1],  # xy
    ], axis=1)  # shape: (Nframes, 6)
    # stress = np.array(stress).reshape(len(atoms), -1)  # shape: (Nframes, 6)
    pos = np.vstack([atom.positions for atom in atoms]).flatten()
    energy -= energy[-1]
    return energy, forces, stress6, pos


if __name__ == "__main__":
    # xyz_ref = "/home/giga/BIG/data/ML_TrainTest_ECEMC/ec1emc2/al/work_emc3_b3lyp_4090_c8352Y_v0/iter_5/label/labeled.xyz"
    # model_paths = [
    #     (
    #         "/home/giga/code/Draft/al_inter_forces/all/b3lyp_admm_pob-TZVP_ot_cut800-d3bjvirial_r4_labeled/label/labeled.xyz",
    #         "REF_",
    #     ),
    #     # (
    #     #     "/home/giga/BIG/data/ML_TrainTest_ECEMC/TrainingData/al/b3lyp_scan_test/cp2k_pob800_need_label_10/label.bak/labeled.xyz",
    #     #     "REF_",
    #     # ),
    #     # (
    #     #     "/home/giga/BIG/data/ML_TrainTest_ECEMC/TrainingData/al/b3lyp_scan_test/cp2k_pob_admm600_need_label_10/label/labeled.xyz",
    #     #     "REF_",
    #     # ),
    # ]
    # xyz_ref = "/home/giga/BIG/data/ML_TrainTest_ECEMC/TrainingData/al/work_ecemc_b3lyp_rigid_3090_c8352Y_v32/iter_18/label/labeled_10.xyz"
    # xyz_ref = (
    #     # "/home/giga/BIG/data/ML_TrainTest_ECEMC/TrainingData/al/work_ecemc_b3lyp_rigid_3090_c8352Y_v33/data/v33d13.xyz"
    #     "/home/giga/BIG/data/ML_TrainTest_ECEMC/TrainingData/al/pbe_scan_test/pbe_tzvp800_VScan_EMC_config01w101_15.xyz"
    # )

    # model_paths = [
    #     # (
    #     #     "/home/giga/code/Draft/al_inter_forces/all/cp2k_admm600_need_label_10/label/labeled.xyz",
    #     #     "REF_",
    #     # ),
    #     (
    #         "/home/giga/code/Draft/al_inter_forces/all/cp2k_pbe_tzv2p800_v33d13/label/labeled.xyz",
    #         "REF_",
    #     ),
    #     # (
    #     #     "/home/giga/BIG/data/ML_TrainTest_ECEMC/TrainingData/al/pbe_scan_test/pbe_tzvp600_VScan_EMC_config01w101_15.xyz",
    #     #     "REF_",
    #     # ),
    #     # (
    #     #     "/home/giga/BIG/data/ML_TrainTest_ECEMC/TrainingData/al/pbe_scan_test/pbe_tzvp800_VScan_EMC_config01w101_15.xyz",
    #     #     "REF_",
    #     # ),
    #     # (
    #     #     "/home/giga/BIG/data/ML_TrainTest_ECEMC/TrainingData/al/pbe_scan_test/pbe_tzv2p_nomol_d3bj_virial_cut600.xyz",
    #     #     "REF_",
    #     # ),
    # ]

    # # #### ec
    # xyz_ref = (
    #     "/home/giga/BIG/data/ML_TrainTest_ECEMC/GAPtests/DFT_PBED2/VScan_EC_config01w101.xyz"
    # )
    # model_paths = [
    #     # (
    #     #     "/home/giga/BIG/data/ML_TrainTest_ECEMC/GAPtests/DFT_PBED2/VScan_EC_config01w101_out1.xyz",
    #     #     "MACE_",
    #     # ),
    #     (
    #         "/home/giga/BIG/data/ML_TrainTest_ECEMC/GAPtests/DFT_PBED2/VScan_EC_config01w101_out2.xyz",
    #         "MACE_",
    #     ),
    #     # 可添加更多模型路径
    # ]

    # # # # ##### emc
    # xyz_ref = "/home/giga/BIG/data/ML_TrainTest_ECEMC/GAPtests/DFT_PBED2/VScan_EMC_config01w101.xyz"
    # model_paths = [
    #     # (
    #     #     "/home/giga/BIG/data/ML_TrainTest_ECEMC/GAPtests/DFT_PBED2/VScan_EMC_config01w101_out1.xyz",
    #     #     "MACE_",
    #     # ),
    #     (
    #         "/home/giga/BIG/data/ML_TrainTest_ECEMC/GAPtests/DFT_PBED2/VScan_EMC_config01w101_out2.xyz",
    #         "MACE_",
    #     ),
    #     # 可添加更多模型路径
    # ]

    # # # # ############ ec/emc data
    xyz_ref = "/home/giga/BIG/data/ML_TrainTest_ECEMC/GAPtests/DFT_PBED2/VScan_66EC_33EMC_config01w101.xyz"
    model_paths = [
        (
            "/home/giga/BIG/data/ML_TrainTest_ECEMC/GAPtests/DFT_PBED2/VScan_66EC_33EMC_config01_out1.xyz",
            "MACE_",
        ),
        (
            "/home/giga/BIG/data/ML_TrainTest_ECEMC/GAPtests/DFT_PBED2/VScan_66EC_33EMC_config01_out2.xyz",
            "MACE_",
        ),
        # 可添加更多模型路径
    ]

    # # ############ scan data
    # xyz_ref = "/home/giga/BIG/data/ML_TrainTest_ECEMC/TrainingData/al/pbe_scan_test/pbe_tzvp800_VScan_EMC_config01w101_15.xyz"
    # model_paths = [
    #     (
    #         "/home/giga/BIG/data/ML_TrainTest_ECEMC/TrainingData/al/pbe_scan_test/pbe_tzvp800_VScan_EMC_config01w101_15_out1.xyz",
    #         "MACE_",
    #     ),
    #     (
    #         "/home/giga/BIG/data/ML_TrainTest_ECEMC/TrainingData/al/pbe_scan_test/pbe_tzvp800_VScan_EMC_config01w101_15_out2.xyz",
    #         "MACE_",
    #     ),
    #     # 可添加更多模型路径
    # ]

    # ############ train data
    # xyz_ref = "/home/giga/BIG/data/ML_TrainTest_ECEMC/TrainingData/al/work_ecemc_b3lyp_rigid_3090_c8352Y_v33/data/v33d13_pbe_tzv2p_ot_d3bj_virial_cut800.xyz"
    # model_paths = [
    #     (
    #         "/home/giga/BIG/data/ML_TrainTest_ECEMC/TrainingData/al/work_ecemc_b3lyp_rigid_3090_c8352Y_v33/data/v33d13_pbe_tzv2p_ot_d3bj_virial_cut800_out1.xyz",
    #         "MACE_",
    #     ),
    #     (
    #         "/home/giga/BIG/data/ML_TrainTest_ECEMC/TrainingData/al/work_ecemc_b3lyp_rigid_3090_c8352Y_v33/data/v33d13_pbe_tzv2p_ot_d3bj_virial_cut800_out2.xyz",
    #         "MACE_",
    #     ),
    #     # 可添加更多模型路径
    # ]
    energy_true, force_true, stress_true, pos_true = get_forces(
        xyz_ref, prefix="REF_", index=":"
    )
    stress_labels = ["xx", "yy", "zz", "yz", "xz", "xy"]
    nframes = len(energy_true)
    x = np.arange(nframes)

    with plt.style.context(["science", "grid"]):
        # 创建一张画布，共 3 行 3 列子图（前 3 个用于 fig_main，后 6 个用于 stress）
        fig_all, ax_all = plt.subplots(3, 3, figsize=(18, 15))
        ax_all = ax_all.flatten()

        # 分配子图
        ax_energy = ax_all[0]
        ax_force = ax_all[1]
        ax_energy_vs_frame = ax_all[2]
        ax_stress = ax_all[3:]  # 6 个 stress 分量

        # 画 DFT 曲线（实线）
        ax_energy.plot(energy_true, energy_true, "k-", label="DFT")
        ax_force.plot(force_true, force_true, "k-", label="DFT")
        ax_energy_vs_frame.plot(x, energy_true, "k-", label="DFT")

        for j in range(6):
            ax_stress[j].plot(x, stress_true[:, j], "k-", label="DFT")

        # 遍历模型预测
        if len(model_paths) > 0:
            for i, (path, prefix) in enumerate(model_paths):
                label = f"Model{i+1}"
                try:
                    energy_pred, force_pred, stress_pred, _ = get_forces(
                        path, prefix=prefix, index=":"
                    )

                    # 输出评估指标
                    rmse_e = np.sqrt(np.mean((energy_true - energy_pred) ** 2))
                    r2_e = np.corrcoef(energy_true, energy_pred)[0, 1] ** 2
                    mae_e = np.mean(np.abs(energy_true - energy_pred))

                    rmse_f = np.sqrt(np.mean((force_true - force_pred) ** 2))
                    r2_f = np.corrcoef(force_true, force_pred)[0, 1] ** 2
                    mae_f = np.mean(np.abs(force_true - force_pred))

                    rmse_s = np.sqrt(np.mean((stress_true - stress_pred) ** 2))
                    r2_s = (
                        np.corrcoef(stress_true.flatten(), stress_pred.flatten())[0, 1]
                        ** 2
                    )
                    mae_s = np.mean(np.abs(stress_true - stress_pred))

                    print(
                        f"{label} -> Energy: RMSE={rmse_e:.4f}, r2={r2_e:.4f}, MAE={mae_e:.4f}"
                    )
                    print(
                        f"{label} -> Force : RMSE={rmse_f:.4f}, r2={r2_f:.4f}, MAE={mae_f:.4f}"
                    )
                    print(
                        f"{label} -> Stress: RMSE={rmse_s:.4f}, r2={r2_s:.4f}, MAE={mae_s:.4f}"
                    )

                    ax_energy.plot(energy_true, energy_pred, "+", label=label)
                    ax_force.plot(force_true, force_pred, "+", label=label)
                    ax_energy_vs_frame.plot(x, energy_pred, "+", label=label)

                    for j in range(6):
                        ax_stress[j].plot(x, stress_pred[:, j], "+", label=label)

                except Exception as e:
                    print(f"Error processing {label}: {e}")

        # 设置标签和标题
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

        stress_labels = ["xx", "yy", "zz", "yz", "xz", "xy"]
        for j in range(6):
            ax_stress[j].set_title(f"Stress Component: {stress_labels[j]}")
            ax_stress[j].set_xlabel("Frame Index")
            ax_stress[j].set_ylabel("Stress (GPa)")
            ax_stress[j].legend()

        fig_all.tight_layout()
        fig_all.savefig("combined_force_energy_stress.png")
        # plt.show()

    print("All plots generated successfully.")
