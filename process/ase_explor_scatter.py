import numpy as np
import glob,re
from ase.io import read, write


if __name__ == "__main__":
    paths = (
        r"/home/giga/BIG/data/ML_TrainTest_ECEMC/al-new/work_ecemc_pbe_4090_c8352Y_v1/iter_*/model_devi/wrap_dump.xyz"
    )
    only_num = 160 # 168 # 216

    save_paths = f"/home/giga/BIG/data/ML_TrainTest_ECEMC/al-new/work_ecemc_pbe_4090_c8352Y_v1/data/expolration_{str(only_num)}.xyz"
    path_list = glob.glob(paths)
    save_atoms_list = []
    for pl in path_list:
        print(pl)
        atoms = read(pl, format="extxyz", index=":")
        if only_num is not None:
            sel_atoms = []
            for ii, atom in enumerate(atoms):
                num = len(atom)
                if only_num == num :
                    sel_atoms.append(atom)
        else:
            sel_atoms = atoms

        save_atoms_list.extend(sel_atoms)
    print(len(save_atoms_list))
    write(save_paths,save_atoms_list)
    print()
