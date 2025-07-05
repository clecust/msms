#!/bin/bash 

conda activate py310pt250cu124

# python3 /home/giga/code/msms/mace/cli/create_lammps_model.py \
#   /home/giga/code/msms/run/checkpoints/mace_r4_run-1.model \
#   --dtype=float32 --format=mliap 

# python3 /home/giga/code/msms/mace/cli/eval_configs.py \
# --configs /home/giga/code/msms/run/checkpoints/mace_r4_run-1.model-mliap_lammps.pt_lmp.xyz \
# --model /home/giga/code/msms/run/checkpoints/mace_r4_run-1.model \
# --output /home/giga/code/msms/run/checkpoints/mace_r4_run-1.model-mliap_lammps.pt_lmp_out.xyz \
# --device cuda --default_dtype float32  --info_prefix MACE_ --compute_stress



##### msmace:

# python3 /home/giga/code/msms/mace/cli/eval_configs.py \
# --configs /home/giga/code/msms/run/checkpoints/msmace_r4_debug_run-1.model-mliap_lammps.pt_lmp.xyz \
# --model /home/giga/code/msms/run/checkpoints/msmace_r4_debug_run-1.model \
# --output /home/giga/code/msms/run/checkpoints/msmace_r4_debug_run-1.model-mliap_lammps.pt_lmp_out.xyz \
# --device cuda --default_dtype float32  --info_prefix MACE_  --batch_size  10 --compute_stress

##### msmacecso:

# python3 /home/giga/code/msms/mace/cli/eval_configs.py \
# --configs /home/giga/code/msms/run/checkpoints/msmacecso_r10_debug_run-1.model-mliap_lammps.pt_lmp.xyz \
# --model /home/giga/code/msms/run/checkpoints/msmacecso_r10_debug_run-1.model \
# --output /home/giga/code/msms/run/checkpoints/msmacecso_r10_debug_run-1.model-mliap_lammps.pt_lmp_out.xyz \
# --device cuda --default_dtype float32  --info_prefix MACE_  --batch_size  10 --compute_stress

# python3 /home/giga/code/msms/mace/cli/eval_configs.py \
# --configs /home/giga/code/msms/run/checkpoints/msmacecso_r10_debug_run-1.model-lammps.pt_lmp.xyz \
# --model /home/giga/code/msms/run/checkpoints/msmacecso_r10_debug_run-1.model \
# --output /home/giga/code/msms/run/checkpoints/msmacecso_r10_debug_run-1.model-lammps.pt_lmp_out.xyz \
# --device cuda --default_dtype float32  --info_prefix MACE_  --batch_size  10 --compute_stress


# python3 /home/giga/code/msms/mace/cli/eval_configs.py \
# --configs /home/giga/code/msms/run/checkpoints/msmacecsor_water_run-1.model-mliap_lammps.pt_lmp.xyz \
# --model /home/giga/code/msms/run/checkpoints/msmacecsor_water_run-1.model \
# --output /home/giga/code/msms/run/checkpoints/msmacecsor_water_run-1.model-mliap_lammps.pt_lmp_out.xyz  \
# --device cuda --default_dtype float32  --info_prefix MACE_  --batch_size  10 --compute_stress

########### mace libtorch
# python3 /home/giga/code/msms/mace/cli/eval_configs.py \
# --configs /home/giga/code/msms/run/checkpoints/msmacecsor_water_run-1.model-lammps.pt_lmp.xyz \
# --model /home/giga/code/msms/run/checkpoints/msmacecsor_water_run-1.model \
# --output /home/giga/code/msms/run/checkpoints/msmacecsor_water_run-1.model-lammps.pt_lmp_out.xyz  \
# --device cuda --default_dtype float32  --info_prefix MACE_  --batch_size  10 --compute_stress


# ############ scan 
# ## b3lyp
# # ## emc
# python3 /home/giga/code/msms/mace/cli/eval_configs.py \
# --configs /home/giga/BIG/data/ML_TrainTest_ECEMC/TrainingData/VScan_EMC_config05_b3lyp_admm_ot_d3bj.xyz \
# --model /home/giga/BIG/data/ML_TrainTest_ECEMC/TrainingData/al/work_ecemc_b3lyp_rigid_3090_c8352Y_v22/data/macecso_v22d4virial_b3lypadmmotd3bj_64l3r5_csor10_n2_b32_e0_swa_stress1001000_run-62002_stagetwo.model \
# --output /home/giga/BIG/data/ML_TrainTest_ECEMC/TrainingData/VScan_EMC_config05_b3lyp_admm_ot_d3bj_out2.xyz  \
# --device cuda --default_dtype float32  --info_prefix MACE_  --batch_size  50 --compute_stress

# # ec/emc
# python3 /home/giga/code/msms/mace/cli/eval_configs.py \
# --configs /home/giga/BIG/data/ML_TrainTest_ECEMC/TrainingData/VScan_66EC_33EMC_config05_b3lyp_admm_ot_d3bj.xyz \
# --model /home/giga/BIG/data/ML_TrainTest_ECEMC/TrainingData/al/work_ecemc_b3lyp_rigid_3090_c8352Y_v22/data/macecso_v22d4virial_b3lypadmmotd3bj_64l3r5_csor10_n2_b32_e0_swa_stress1001000_run-62002_stagetwo.model \
# --output /home/giga/BIG/data/ML_TrainTest_ECEMC/TrainingData/VScan_66EC_33EMC_config05_b3lyp_admm_ot_d3bj_2.xyz  \
# --device cuda --default_dtype float32  --info_prefix MACE_  --batch_size  50 --compute_stress

# # ec/emc test
# python3 /home/giga/code/msms/mace/cli/eval_configs.py \
# --configs /home/giga/BIG/data/ML_TrainTest_ECEMC/TrainingData/al/work_ecemc_b3lyp_rigid_3090_c8352Y_v31/iter_7/label/labeled.xyz \
# --model /home/giga/BIG/data/ML_TrainTest_ECEMC/TrainingData/al/work_ecemc_b3lyp_rigid_3090_c8352Y_v33/data/mace_v33d13_pbe_tzv2p_ot_d3bj_virial_cut800_64l2r6n2_b32_e0_swa_stress1001000_run-62701_stagetwo.model \
# --output /home/giga/BIG/data/ML_TrainTest_ECEMC/TrainingData/al/work_ecemc_b3lyp_rigid_3090_c8352Y_v31/iter_7/label/labeled_out1.xyz  \
# --device cuda --default_dtype float32  --info_prefix MACE_  --batch_size  50 --compute_stress

# python3 /home/giga/code/msms/mace/cli/eval_configs.py \
# --configs /home/giga/BIG/data/ML_TrainTest_ECEMC/TrainingData/al/work_ecemc_b3lyp_rigid_3090_c8352Y_v31/iter_7/label/labeled.xyz \
# --model /home/giga/BIG/data/ML_TrainTest_ECEMC/TrainingData/al/work_ecemc_b3lyp_rigid_3090_c8352Y_v33/data/mace_v33d13_pbe_tzv2p_ot_d3bj_virial_cut800_64l2r6n2_b32_e0_swa_stress1001000_run-62701_stagetwo.model  \
# --output /home/giga/BIG/data/ML_TrainTest_ECEMC/TrainingData/al/work_ecemc_b3lyp_rigid_3090_c8352Y_v31/iter_7/label/labeled_out1.xyz  \
# --device cuda --default_dtype float32  --info_prefix MACE_  --batch_size  50 --compute_stress

# #### pbe
# # # # ## emc
# id=1  ## ${id}
# model="/home/giga/BIG/data/ML_TrainTest_ECEMC/TrainingData/al/work_ecemc_b3lyp_rigid_3090_c8352Y_v35/data/mace_v35d4virial_cut800_64l2r6n2_b32_e0_swa_stress1001000_run-7010${id}_stagetwo.model"

template_model="/home/giga/BIG/data/ML_TrainTest_ECEMC/TrainingData/al/work_ecemc_pbe_4090_c8352Y_v42/data/mace_pbev42d14virial_cut800_64l2r6n2_b32_e0_swa_stress1001000_run-70401_stagetwo.model"

for id in 1 2  ; do
    # 替换路径中的 run-70101 为 run-7010${id}
    model="${template_model/1_stagetwo/${id}_stagetwo}"

    echo "Evaluating id=${id}"
    echo "Using model: $model"
    # # emc
    python3 /home/giga/code/msms/mace/cli/eval_configs.py \
    --configs /home/giga/BIG/data/ML_TrainTest_ECEMC/GAPtests/DFT_PBED2/VScan_EMC_config01w101.xyz \
    --model ${model}  \
    --output /home/giga/BIG/data/ML_TrainTest_ECEMC/GAPtests/DFT_PBED2/VScan_EMC_config01w101_out${id}.xyz  \
    --device cuda --default_dtype float32  --info_prefix MACE_  --batch_size  50 --compute_stress

    # # ec
    python3 /home/giga/code/msms/mace/cli/eval_configs.py \
    --configs /home/giga/BIG/data/ML_TrainTest_ECEMC/GAPtests/DFT_PBED2/VScan_EC_config01w101.xyz \
    --model  ${model}   \
    --output /home/giga/BIG/data/ML_TrainTest_ECEMC/GAPtests/DFT_PBED2/VScan_EC_config01w101_out${id}.xyz \
    --device cuda --default_dtype float32  --info_prefix MACE_  --batch_size  50 --compute_stress

    # # ec/emc
    python3 /home/giga/code/msms/mace/cli/eval_configs.py \
    --configs /home/giga/BIG/data/ML_TrainTest_ECEMC/GAPtests/DFT_PBED2/VScan_66EC_33EMC_config01w101.xyz \
    --model  ${model}   \
    --output /home/giga/BIG/data/ML_TrainTest_ECEMC/GAPtests/DFT_PBED2/VScan_66EC_33EMC_config01_out${id}.xyz \
    --device cuda --default_dtype float32  --info_prefix MACE_  --batch_size  50 --compute_stress

done
# # # SCAN
# python3 /home/giga/code/msms/mace/cli/eval_configs.py \
# --configs /home/giga/BIG/data/ML_TrainTest_ECEMC/TrainingData/al/pbe_scan_test/pbe_tzvp800_VScan_EMC_config01w101_15.xyz \
# --model  ${model}   \
# --output /home/giga/BIG/data/ML_TrainTest_ECEMC/TrainingData/al/pbe_scan_test/pbe_tzvp800_VScan_EMC_config01w101_15_out${id}.xyz \
# --device cuda --default_dtype float32  --info_prefix MACE_  --batch_size  50 --compute_stress


# # TRAIN
# python3 /home/giga/code/msms/mace/cli/eval_configs.py \
# --configs /home/giga/BIG/data/ML_TrainTest_ECEMC/TrainingData/al/work_ecemc_b3lyp_rigid_3090_c8352Y_v33/data/v33d13_pbe_tzv2p_ot_d3bj_virial_cut800.xyz \
# --model  ${model}   \
# --output /home/giga/BIG/data/ML_TrainTest_ECEMC/TrainingData/al/work_ecemc_b3lyp_rigid_3090_c8352Y_v33/data/v33d13_pbe_tzv2p_ot_d3bj_virial_cut800_out${id}.xyz \
# --device cuda --default_dtype float32  --info_prefix MACE_  --batch_size  50 --compute_stress


# ## test
# python3 /home/giga/code/msms/mace/cli/eval_configs.py \
# --configs /home/giga/BIG/data/ML_TrainTest_ECEMC/TrainingData/al/work_ecemc_pbe_rigid_3090_c8352Y_v13/iter_8.bak/model_devi/test/dump_lmp.xyz \
# --model /home/giga/BIG/data/ML_TrainTest_ECEMC/TrainingData/al/work_ecemc_pbe_rigid_3090_c8352Y_v13/data/msmacecso_v13d4refb3lypw101_64l2n2r3_64l0n2r6_csor10_b64_ea_run-60102.model \
# --output /home/giga/BIG/data/ML_TrainTest_ECEMC/TrainingData/al/work_ecemc_pbe_rigid_3090_c8352Y_v13/iter_8.bak/model_devi/test/dump_lmp_out.xyz \
# --device cuda --default_dtype float32  --info_prefix MACE_  --batch_size  50 --compute_stress

# #################scan
# ### pfpef
# python3 /home/giga/code/msms/mace/cli/eval_configs.py \
# --configs /home/giga/BIG/data/3M_F/3/scan/gfn.xyz \
# --model "/home/giga/BIG/al/param_rigid_3090_c6238_v14_pbe_vis/data/4090mace_pbev14d23_gfn1_64l2n2r6_b5_ea_run-52702_stagetwo.model" \
# --output /home/giga/BIG/data/3M_F/3/scan/gfn_out2.xyz  \
# --device cuda --default_dtype float32  --info_prefix MACE_  --batch_size  50 --compute_stress

#################### test

# python3 /home/giga/code/msms/mace/cli/eval_configs.py \
# --configs /home/giga/code/competition/data/competition_round2.xyz \
# --model /home/giga/code/competition/run/train_mace/competition_h8n2l1c3r128b64_compiled.model \
# --output /home/giga/code/competition/data/competition_round2_h8.xyz  \
# --device cuda --default_dtype float32  --info_prefix REF_  --batch_size  128 --return_node_feats