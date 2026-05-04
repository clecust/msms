import numpy as np
import glob,re
from ase.io import read, write


def old2new_ref(
    atoms_list,
    prefix="REF_",
    energy_key="energy",
    forces_key="forces",
    stress_key="stress",
    kcal_to_ev=False,
):
    """
    Convert and store reference energy/force/stress/virial into atoms.info and atoms.arrays.

    Parameters:
    - atoms_list: List[ase.Atoms] or ase.Atoms
    - prefix: Prefix for new keys in atoms.info and atoms.arrays
    - energy_key: key to access potential energy (default 'energy')
    - forces_key: key to access forces (default 'forces')
    - stress_key: key to access stress (default 'stress')
    - kcal_to_ev: if True, convert energy/force from kcal/mol to eV
    """
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]
    print(f"len=({len(atoms_list)})")

    factor = 0.0433641 if kcal_to_ev else 1.0

    for atoms in atoms_list:
        # --- forces ---
        forces = None
        try:
            forces = atoms.get_forces()
        except Exception:
            forces = atoms._calc.results.get(forces_key, None)
        if forces is None:
            forces = atoms.info.get(forces_key, None)
        atoms.arrays[f"{prefix}forces"] = forces * factor if forces is not None else None

        # --- energy ---
        energy = None
        try:
            energy = atoms.get_potential_energy()
        except Exception:
            energy = atoms._calc.results.get(energy_key, None)
        if energy is None:
            energy = atoms.info.get(energy_key, None)
        atoms.info[f"{prefix}energy"] = energy * factor if energy is not None else None

        # --- stress ---
        stress = None
        try:
            stress = atoms.get_stress()
        except Exception:
            stress = atoms._calc.results.get(stress_key, None)
        if stress is None:
            stress = atoms.info.get(stress_key, None)
        if stress is not None:
            atoms.info[f"{prefix}stress"] = stress

        # --- virials → stress ---
        try:
            if "virial" in atoms.info:
                atoms.info[f"{prefix}virials"] = atoms.info["virial"]
                del atoms.info["virial"]
            elif f"{prefix}virial" in atoms.info:
                atoms.info[f"{prefix}virials"] = atoms.info[f"{prefix}virial"]
                del atoms.info[f"{prefix}virial"]
        except Exception:
            pass

        try:
            if f"{prefix}virials" in atoms.info:
                atoms.info[f"{prefix}stress"] = -atoms.info[f"{prefix}virials"] / atoms.get_volume()
        except Exception:
            pass

        if atoms.info.get("config_weight") == 50.0:
            atoms.info["config_weight"] = 1.01

        atoms.calc = None

    return atoms_list


if __name__ == "__main__":
    paths = [
        r"/home/giga/BIG/data/AmmoniumNitrate/lmp-md/lammps_3w10/traj_10ns_lmp_gfn1_stress.xyz"
    ]
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
        atoms_list = old2new_ref(
            atoms_list, energy_key="Energy", prefix="REF_", kcal_to_ev=False
        )

        save_pls = pls.replace('.xyz',"_stress.xyz")
        print(f"save to {save_pls}")
        write(save_pls, atoms_list, append=False)
