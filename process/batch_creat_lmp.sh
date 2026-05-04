#!/bin/bash 

conda activate py310pt250cu124

# python3 /home/giga/code/msms/mace/cli/create_lammps_model.py \
#   "/home/giga/code/mi/code/lammps_bak/2023-12-10-mace-128-L0_energy_epoch-249.model" \
#   --dtype=float32  #  --format=mliap

# # # # # # # 处理所有匹配的文件 pfpef
# for model_file in /home/giga/BIG/al/param_rigid_3090_c6238_v20_gfn2/data/msmace_v20d*gfn2virial_64l3r35_64l0r8_n2_b32_ea_swa_stress1001000_run-6240*_stagetwo.model; do
#     echo "处理文件: $model_file"
#     python3 /home/giga/code/msms/mace/cli/create_lammps_model.py \
#       "$model_file" \
#       --dtype=float32   --format=mliap
# done

# --format libtorch mliap

# 处理所有匹配的文件 ecemc
# for model_file in /home/giga/BIG/data/md22/zip/model/md22_AT-AT_rebond_modification_v3_200_gfn2_epoch5000_ea_l3_b5_run-*_stagetwo.model ; do

# # for model_file in /home/giga/BIG/data/md22/gfnff300K/model/md22_*v7*200*epoch2000*l3*b32*h64*.model ; do
# # for model_file in /home/giga/BIG/data/md22/gfnff300K/model/1000v10/md22_*v13*400*l3*.model ; do
# for model_file in /home/giga/BIG/data/md22/openmmgaff300K/model/*l_v55_50*l3*b10*h256*.model ; do
# # for model_file in /home/giga/BIG/data/chignolin/5-percent-small-dataset/smaller_Chig_AIMD/data/model/*v20*l3*b10*.model ; do
# # for model_file in /home/giga/BIG/data/chignolin/md17_aspirin/data/model/*v21*l3*b10*h256.model ; do

# # for model_file in /home/giga/BIG/data/ML_TrainTest_ECEMC/TrainingData/al/npj_data/model/mace_Train16_PBED3w101nos_64l2r5_n2_b32_e0_swa_stress1001000_epoch10000_run-2025*_stagetwo.model ; do
# # for model_file in /home/giga/BIG/data/chignolin/5-percent-small-dataset/smaller_Chig_AIMD/lmp/mace_chig_rebond960gfn2_64l*n2_b32_ea_run-7100*.model ; do



# # for model_file in /home/giga/BIG/data/ML_TrainTest_ECEMC/ecemc/al/work_ecemc_b3lyp_4090_c8352Y_v5/data/mace_npjTrain16_PBED3w101_v*shear*_64l2r6n2_b32_ea_swa_stress1001000_run-7050*_stagetwo.model ; do
# for model_file in /home/giga/BIG/data/ML_TrainTest_ECEMC/al-new/work_ecemc_pbe_4090_c8352Y_v1/data/*_32e3_16e0_r10_b64_stress_run-9_stagetwo.model ; do
#     echo "处理文件: $model_file"
#     python3 /home/giga/code/msms/mace/cli/create_lammps_model.py \
#       "$model_file" \
#       --dtype=float32      --format=mliap
# done


for model_file in /home/giga/BIG/data/ML_TrainTest_ECEMC/al-new/work_ecemc_pbe_4090_c8352Y_v5v0/data/model/*d13*b10*1024_stagetwo.model ; do
    echo "处理文件: $model_file"
    python3 /home/giga/code/msms/mace/cli/create_lammps_model.py \
      "$model_file" \
      --dtype=float32      --format=mliap
done    

# # ####  a1 
# model_file="/home/giga/BIG/data/ML_TrainTest_ECEMC/TrainingData/al/npj_data/model/MACE-OFF23_small.model"
# model_file="/home/giga/BIG/data/ML_TrainTest_ECEMC/TrainingData/al/npj_data/model/mace_Train16_PBED3w101nos_64l2r5_n2_b32_e0_swa_stress1001000_epoch10000_run-20253_stagetwo.model"
# model_file="/home/giga/BIG/data/AmmoniumNitrate/al/work_an_revpbe_4090_c8352Y_v1/data/v1d10ff600_32e2_16e0_r7_stress_stagetwo.model"
# model_file="/home/giga/BIG/data/lipfecpc/ffmd/ffmd_32e2_16e0_r7_stress_stagetwo.model"
# model_file="/home/giga/BIG/data/lipfecpc/work_lipfecpc_pbe_4090_c8352Y_v7/data/v7d13_32e2_16e0_r7_stress_run-*_stagetwo.model"

# # for x in  0.2 ; do
# for x in  1.2; do

# #  0.8 1.2 1.5
#     echo "处理文件: $model_file"
#     python3 /home/giga/code/msms/mace/cli/create_lammps_model.py \
#       "$model_file" \
#       --dtype=float32   --format=mliap  #  --coefficient 0.05   --a1 ${x} # --cso_r=10
#             # --dtype=float32  --format=mliap --cso_a1 ${a1} --cso_r=10 --c6 ${c6}

# done
      

