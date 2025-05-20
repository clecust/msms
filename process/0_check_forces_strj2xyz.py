#!/usr/bin/env python

"""
读取lammps的保存轨迹文件，转化成extxyz文件，再测试力．
"""
import os
import numpy as np
import ase.io
from ase_old_2_new_ref import old2new_ref


if __name__ == "__main__":

    lammps_path = r"/home/giga/code/msms/run/checkpoints/mace_r4_run-1.model-mliap_lammps.pt.lammpstrj"
    specorder = [ "O", "H"]

    if "pdb" in lammps_path:
        atoms = ase.io.read(lammps_path, format="proteindatabank", index="0")
        save_name = lammps_path.replace(".pdb", "_lmp.xyz")

    else:
        atoms = ase.io.read(
            lammps_path, format="lammps-dump-text", index=":", specorder=specorder
        )
        save_name = lammps_path.replace(".lammpstrj", "_lmp.xyz")

    print(len(atoms))
    # save to extxyz with force

    atoms = old2new_ref(atoms)

    print(f"save to {save_name}")
    ase.io.write(
        save_name,
        atoms,
        format="extxyz",
        append=False,
        # properties=["forces"],
    )

    print("Congratulations!!")
