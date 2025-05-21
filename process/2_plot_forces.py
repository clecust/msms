#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""

"""
import numpy as np
from ase.io import read
import scienceplots
import matplotlib.pyplot as plt

def get_forces(path,label="forces", index=":"):
    atoms = read(path, format="extxyz", index=index)
    if not isinstance(atoms, list):
        atoms = [atoms]
    forces = [atom.arrays[label] for atom in atoms]
    forces = np.array(forces).reshape(-1, 1)
    return forces


if __name__ == "__main__":

    # path = "/home/giga/code/msms/run/checkpoints/mace_r4_run-1.model-mliap_lammps.pt_lmp_out.xyz"
    # path2 = "/home/giga/code/msms/run/checkpoints/mace_r4_run-1.model-mliap_lammps.pt_lmp.xyz"

    ## msmace
    # path = "/home/giga/code/msms/run/checkpoints/msmace_r4_debug_run-1.model-mliap_lammps.pt_lmp_out.xyz"
    # path2 = "/home/giga/code/msms/run/checkpoints/msmace_r4_debug_run-1.model-mliap_lammps.pt_lmp.xyz"

    ## msmacecso
    path = "/home/giga/code/msms/run/checkpoints/msmacecso_r10_debug_run-1.model-mliap_lammps.pt_lmp_out.xyz"
    path2 = "/home/giga/code/msms/run/checkpoints/msmacecso_r10_debug_run-1.model-mliap_lammps.pt_lmp.xyz"
    forces = get_forces(path, label="MACE_forces", index=":")
    forces2 = get_forces(path2, label="REF_forces", index=":")

    with plt.style.context(["science"]):
        fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.0), dpi=250)
        plt.plot(forces ,forces ,'black')
        plt.plot(
            forces,
            forces2,
            "+",
            label=f"rr2={np.corrcoef(forces.flatten(), forces2.flatten())[0, 1]:.4f},mae={np.mean(np.abs(forces.flatten()-forces2.flatten())):.4f}",
        )
        plt.legend()

        plt.tight_layout(pad=0.0)
        # plt.grid()
        plt.show()

    print("Congratulations!!")
