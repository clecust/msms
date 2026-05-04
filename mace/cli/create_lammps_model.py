# pylint: disable=wrong-import-position
import argparse
import copy
import os

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import torch
from e3nn.util import jit

from mace.calculators import LAMMPS_MACE
from mace.calculators.lammps_mliap_mace import LAMMPS_MLIAP_MACE
from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
from mace.cli.convert_e3nn_cueq import mace2macecso 

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the model to be converted to LAMMPS",
    )
    parser.add_argument(
        "--head",
        type=str,
        nargs="?",
        help="Head of the model to be converted to LAMMPS",
        default=None,
    )
    parser.add_argument(
        "--dtype",
        type=str,
        nargs="?",
        help="Data type of the model to be converted to LAMMPS",
        default="float64",
    )
    parser.add_argument(
        "--coefficient",
        type=float,
        help=" coefficient ",
        default=None,
    )
    parser.add_argument(
        "--cso_r",
        type=float,
        help=" cso_r ",
        default=None,
    )
    parser.add_argument(
        "--c6",
        type=float,
        help=" c6 ",
        default=None,
    )
    parser.add_argument(
        "--a1",
        type=float,
        help=" a1 ",
        default=None,
    )
    parser.add_argument(
        "--format",
        type=str,
        help="Old libtorch format, or new mliap format",
        default="libtorch",
    )
    parser.add_argument(
        "--mace2macecso",
        help="mace2macecso",
        action="store_true",
        default=False,
    )
    return parser.parse_args()


def select_head(model):
    if hasattr(model, "heads"):
        heads = model.heads
    else:
        heads = [None]

    if len(heads) == 1:
        print(f"Only one head found in the model: {heads[0]}. Skipping selection.")
        return heads[0]

    print("Available heads in the model:")
    for i, head in enumerate(heads):
        print(f"{i + 1}: {head}")

    # Ask the user to select a head
    selected = input(
        f"Select a head by number (Defaulting to head: {len(heads)}, press Enter to accept): "
    )

    if selected.isdigit() and 1 <= int(selected) <= len(heads):
        return heads[int(selected) - 1]
    if selected == "":
        print("No head selected. Proceeding without specifying a head.")
        return None
    print(f"No valid selection made. Defaulting to the last head: {heads[-1]}")
    return heads[-1]


def has_dftd_submodule(model, submodule_name="dftd"):
    for name, _ in model.named_modules():
        if name.endswith("." + submodule_name) or name == submodule_name:
            return True
    return False


def main():
    args = parse_args()
    model_path = args.model_path  # takes model name as command-line input
    model = torch.load(
        model_path,
        map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    if args.cso_r is not None:
        ## 不能转化jit的，checkpoint可以转化
        # if hasattr(model,'r_mid'):
        #     model = mace2macecso(copy.deepcopy(model))
        #     print(f"to msmacecso")
        # else:
        model = mace2macecso(copy.deepcopy(model))
        print(f"to macecso")
    # print(model)

    if has_dftd_submodule(model, "dispersion_correction"):
        print(
            f"model.dispersion_correction.coefficient={model.dispersion_correction.coefficient},with a1 = {model.dispersion_correction.a1} "
        )

        if args.coefficient is not None:
            model.dispersion_correction.coefficient = (
                torch.ones_like(model.dispersion_correction.coefficient)
                * args.coefficient
            )
            print(f"Change coefficient to  {model.dispersion_correction.coefficient} ")

        # print(model)
        if args.cso_r is not None:
            model.r_max = torch.ones_like(model.r_max) * float(args.cso_r)

            model.dispersion_correction.cutoff = (
                float(args.cso_r) / model.dispersion_correction.d3_autoang
            )
        # print(model)

        if args.c6 is not None:
            model.dispersion_correction.c6_emb.weight[8] *= float(args.c6)
            print(
                f"Change to model.dispersion_correction.c6_emb.weight[8]={model.dispersion_correction.c6_emb.weight[8]}, by {args.c6}"
            )
        if args.a1 is not None:
            print(f"model.dispersion_correction.a1={model.dispersion_correction.a1}")
            model.dispersion_correction.a1 = (
                torch.ones_like(model.dispersion_correction.a1) * args.a1
            )
            print(
                f"Change to model.dispersion_correction.a1={model.dispersion_correction.a1}, by {args.a1}"
            )
    if args.dtype == "float64":
        model = model.double().to("cpu")
    elif args.dtype == "float32":
        print("Converting model to float32, this may cause loss of precision.")
        model = model.float().to("cpu")

    if args.format == "mliap":
        # Enabling cuequivariance by default. TODO: switch?
        model = run_e3nn_to_cueq(copy.deepcopy(model))
        model.lammps_mliap = True

    if args.head is None:
        head = select_head(model)
    else:
        head = args.head
        print(
            f"Selected head: {head} from command line in the list available heads: {model.heads}"
        )

    lammps_class = LAMMPS_MLIAP_MACE if args.format == "mliap" else LAMMPS_MACE
    lammps_model = (
        lammps_class(model, head=head) if head is not None else lammps_class(model)
    )
    if args.format == "mliap":
        label_info = ''
        if args.coefficient is not None:
            label_info = f"-coef={str(args.coefficient).replace('.','_')}"
        if args.cso_r is not None:
            label_info += f"-csor={str(args.cso_r).replace('.','_')}"
        if args.c6 is not None:
            label_info += f"-c6={str(args.c6).replace('.','_')}"
        if args.a1 is not None:
            label_info += f"-a1={str(args.a1).replace('.','_')}"
        model_path = f"{model_path.replace('.model',f'{label_info}')}" + ".model"
        # _negativecso
        torch.save(
            lammps_model,
            model_path + "-mliap_lammps.pt",
        )
        if label_info !=  '':
            torch.save(
                model,
                model_path,
            )
    else:
        lammps_model_compiled = jit.compile(lammps_model)
        lammps_model_compiled.save(model_path + "-lammps.pt")


if __name__ == "__main__":
    main()
