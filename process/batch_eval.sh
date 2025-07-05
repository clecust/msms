#!/bin/bash 

conda activate py310pt250cu124


# # # # # # # 处理所有匹配的文件 pfpef
# for model_file in /home/giga/BIG/al/param_rigid_3090_c6238_v20_gfn2/data/msmace_v20d*gfn2virial_64l3r35_64l0r8_n2_b32_ea_swa_stress1001000_run-6240*_stagetwo.model; do
#     echo "处理文件: $model_file"
#     python3 /home/giga/code/msms/mace/cli/create_lammps_model.py \
#       "$model_file" \
#       --dtype=float32   --format=mliap
# done

# --format libtorch mliap

# # # 处理所有匹配的文件 ecemc
for model_file in /home/giga/BIG/data/ML_TrainTest_ECEMC/ecemc/al/work_ecemc_b3lyp_4090_c8352Y_v5/data/mace_ecemcv5d15virial_cut800_64l2r6n2_b32_e0_swa_stress1001000_run-7050*_stagetwo.model ; do
# for model_file in /home/giga/BIG/data/ML_TrainTest_ECEMC/TrainingData/al/work_ecemc_pbe_4090_c8352Y_v42/data/mace_pbev42d14virial_cut800_64l2r6n2_b32_e0_swa_stress1001000_run-7040*_stagetwo.model ; do
    echo "处理文件: $model_file"
    python3 /home/giga/code/msms/mace/cli/create_lammps_model.py \
      "$model_file" \
      --dtype=float32  --format=mliap
done
      
      