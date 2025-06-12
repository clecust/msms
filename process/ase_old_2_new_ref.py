import numpy as np
import glob,re
from ase.io import read, write


def old2new_ref(atoms_list):
    for atoms in atoms_list:
        try:
            atoms.arrays["REF_forces"] = atoms.get_forces()
        except Exception as e:  # pylint: disable=W0703
            pass
        try:
            atoms.info["REF_energy"] = atoms.get_potential_energy()
        except Exception as e:  # pylint: disable=W0703
            pass
        try:
            atoms.info["REF_stress"] = atoms.get_stress()
        except Exception as e:  # pylint: disable=W0703
            pass
        try:
            if 'virial' in atoms.info :
                atoms.info["REF_virials"] = atoms.info['virial']
                del atoms.info["virial"]
            elif "REF_virial" in atoms.info:
                atoms.info["REF_virials"] = atoms.info["REF_virial"]
                del atoms.info["REF_virial"]
        except Exception as e:  # pylint: disable=W0703
            pass
        try:
            #  ASE 默认定义为压缩为正，而有些软件（如 DFTB和LAMMPS,cp2k）是张力为正。
            atoms.info["REF_stress"] = -atoms.info["REF_virials"] / atoms.get_volume()
        except Exception as e:  # pylint: disable=W0703
            pass
        if 'config_weight' in atoms.info:
            if atoms.info['config_weight'] == 50.0:
                atoms.info['config_weight'] = 1.01
        atoms.calc = None
    return atoms_list


if __name__ == "__main__":
    paths = [r"/home/giga/BIG/al/param_rigid_3090_c6238_v14_pbe_vis/data/AL_23.xyz"]
    path_lists = []
    for path in paths:
        if "*" in path:
            temp_path = glob.glob(path)
            path_lists.extend(temp_path)
        else:
            path_lists.append(path)
    # 依据数字排序
    # if len(path_lists) > 1:
    #     path_lists.sort(key=lambda x: int(re.findall(r"\d+", x)[-1]))
    for pls in path_lists:
        print(pls)
        atoms_list = read(pls, format="extxyz", index=":")
        atoms_list = old2new_ref(atoms_list)

        save_pls = pls.replace('.xyz',"w101.xyz")
        print(f"save to {save_pls}")
        write(save_pls, atoms_list, append=False)
