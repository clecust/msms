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
            atoms.info["REF_virial"] = atoms.info['virial']
        except Exception as e:  # pylint: disable=W0703
            pass
        atoms.calc = None
    return atoms_list


if __name__ == "__main__":
    paths = [r"/home/giga/code/ms4mace/code/ms4mace/data/data_*.xyz"]
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

        save_pls = pls.replace('.xyz',"ref.xyz")
        write(save_pls, atoms_list, append=False)
