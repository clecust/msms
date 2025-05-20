#!/bin/bash 

conda activate py310pt250cu124

# python3 /home/giga/code/msms/mace/cli/create_lammps_model.py \
#   /home/giga/code/msms/run/checkpoints/mace_r8_run-1.model \
#   --dtype=float32 \
#     --format=mliap 

python3 /home/giga/code/msms/mace/cli/eval_configs.py \
--configs /home/giga/code/msms/run/checkpoints/mace_r4_run-1.model-mliap_lammps.pt_lmp.xyz \
--model /home/giga/code/msms/run/checkpoints/mace_r4_run-1.model \
--output /home/giga/code/msms/run/checkpoints/mace_r4_run-1.model-mliap_lammps.pt_lmp_out.xyz \
--device cuda --default_dtype float32  --info_prefix MACE_