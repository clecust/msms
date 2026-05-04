#!/bin/bash 

# conda activate py310pt250cu124

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
# model="/home/giga/BIG/data/ML_TrainTest_ECEMC/TrainingData/al/work_ecemc_b3lyp_rigid_3090_c8352Y_v35/data/macecso_Train16_PBED3w101nos_64l2r5_csor10_n2_b32_ea_swa_stress1001000_run-70701_stagetwo.model"

# template_model="/home/giga/BIG/data/ML_TrainTest_ECEMC/TrainingData/al/npj_data/model/mace_Train16_PBED3w101nos_64l2r5_n2_b32_e0_swa_stress1001000_epoch10000_run-20255_stagetwo.model"
# # template_model="/home/giga/BIG/data/ML_TrainTest_ECEMC/ecemc/al/work_ecemc_b3lyp_4090_c8352Y_v5/data/mace_npjTrain16_PBED3w101_64l2r6n2_b32_ea_swa_stress1001000_run-70501_stagetwo.model"
# #     -coef=0_05-csor=10_0-a1=0_6 
# config="config05w101"

# for id in 5  ; do
#     # 替换路径中的 run-70101 为 run-7010${id}
#     model="${template_model/1_stagetwo/${id}_stagetwo}"
#     info=''

#     echo "Evaluating id=${id}"
#     echo "Using model: $model"
#     # # emc
#     python3 /home/giga/code/msms/mace/cli/eval_configs.py \
#     --configs /home/giga/BIG/data/ML_TrainTest_ECEMC/GAPtests/DFT_PBED2/VScan_EMC_${config}.xyz \
#     --model ${model}  \
#     --output /home/giga/BIG/data/ML_TrainTest_ECEMC/GAPtests/DFT_PBED2/VScan_EMC_${config}_out${id}${info}.xyz  \
#     --device cuda --default_dtype float32  --info_prefix MACE_  --batch_size  50 --compute_stress

#     # # ec
#     python3 /home/giga/code/msms/mace/cli/eval_configs.py \
#     --configs /home/giga/BIG/data/ML_TrainTest_ECEMC/GAPtests/DFT_PBED2/VScan_EC_${config}.xyz \
#     --model  ${model}   \
#     --output /home/giga/BIG/data/ML_TrainTest_ECEMC/GAPtests/DFT_PBED2/VScan_EC_${config}_out${id}${info}.xyz \
#     --device cuda --default_dtype float32  --info_prefix MACE_  --batch_size  50 --compute_stress

#     # # ec/emc
#     python3 /home/giga/code/msms/mace/cli/eval_configs.py \
#     --configs /home/giga/BIG/data/ML_TrainTest_ECEMC/GAPtests/DFT_PBED2/VScan_66EC_33EMC_${config}.xyz \
#     --model  ${model}   \
#     --output /home/giga/BIG/data/ML_TrainTest_ECEMC/GAPtests/DFT_PBED2/VScan_66EC_33EMC_${config}_out${id}${info}.xyz \
#     --device cuda --default_dtype float32  --info_prefix MACE_  --batch_size  50 --compute_stress

#     # # ec/emc
#     python3 /home/giga/code/msms/mace/cli/eval_configs.py \
#     --configs /home/giga/BIG/data/ML_TrainTest_ECEMC/GAPtests/DFT_PBED2/VScan_33EC_66EMC_${config}.xyz \
#     --model  ${model}   \
#     --output /home/giga/BIG/data/ML_TrainTest_ECEMC/GAPtests/DFT_PBED2/VScan_33EC_66EMC_${config}_out${id}${info}.xyz \
#     --device cuda --default_dtype float32  --info_prefix MACE_  --batch_size  50 --compute_stress

# done
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


# # template_model="/home/giga/BIG/data/md22/gfnff300K/model/1000v10/md22_DHA_300K_3000_rebond_v12_600_gfn2_epoch1000_ea_l3_b32_h64_run-20251.model"
# # template_model="/home/giga/BIG/data/md22/openmmgaff300K/model/aspirin_300k_NVT_gaff_rebond_v37_80_gfn2_epoch1000_ea_l3_b10_h256_run-20251.model"
# template_model="/home/giga/BIG/data/md22/openmmgaff300K/model/aspirin_300k_NVT_gaff_rebond_v54_50_gfn2_epoch1000_ea_l3_b10_h256_run-20251.model"
# # template_model="/home/giga/BIG/data/md22/gfnff300Kn3/model/md22_DHA_n3_1000_rebond_original_v4_300_gfn2_epoch1000_ea_l3_b5_run-20251.model"
# for id in 1 2 3 4 5  ; do
#     # 替换路径中的 run-70101 为 run-7010${id}
#     model="${template_model/1.model/${id}.model}"
#     info='rebond'  # rebond  original

#     # python3 /home/giga/code/msms/mace/cli/eval_configs.py \
#     # --configs /home/giga/BIG/data/md22/bond_test/md22_DHA_3000_bond_oh_gfn2.xyz \
#     # --model  ${model}   \
#     # --output /home/giga/BIG/data/md22/bond_test/md22_DHA_3000_bond_oh_gfn2_${id}${info}.xyz \
#     # --device cuda --default_dtype float32  --info_prefix MACE_  --batch_size  50 --compute_stress

#     # python3 /home/giga/code/msms/mace/cli/eval_configs.py \
#     # --configs /home/giga/BIG/data/md22/bond_test/AT_NH5_gfn2.xyz \
#     # --model  ${model}   \
#     # --output /home/giga/BIG/data/md22/bond_test/AT_NH5_gfn2_${id}${info}.xyz \
#     # --device cuda --default_dtype float32  --info_prefix MACE_  --batch_size  50 --compute_stress

#     python3 /home/giga/code/msms/mace/cli/eval_configs.py \
#     --configs /home/giga/BIG/data/md22/bond_test/aspirin_300k_NVT_gaff_bond_gfn2.xyz \
#     --model  ${model}   \
#     --output /home/giga/BIG/data/md22/bond_test/aspirin_300k_NVT_gaff_bond_gfn2_${id}${info}.xyz \
#     --device cuda --default_dtype float32  --info_prefix MACE_  --batch_size  50 --compute_stress

#     # python3 /home/giga/code/msms/mace/cli/eval_configs.py \
#     # --configs /home/giga/BIG/data/md22/bond_test/md22_ATv3_300k_NVT_gaff_bond_gfn2.xyz \
#     # --model  ${model}   \
#     # --output /home/giga/BIG/data/md22/bond_test/md22_ATv3_300k_NVT_gaff_bond_gfn2_${id}${info}.xyz \
#     # --device cuda --default_dtype float32  --info_prefix MACE_  --batch_size  50 --compute_stress
# done

# ec/emc37
# python3 /home/giga/code/msms/mace/cli/eval_configs.py \
# --configs /home/giga/BIG/data/ML_TrainTest_ECEMC/save_model/VScan_33EC_66EMC_config05w101.xyz \
# --model /home/giga/BIG/data/ML_TrainTest_ECEMC/save_model/mace_Train16_PBED3w101nos_64l2r5_n2_b32_e0_swa_stress1001000_epoch10000_run-20255_stagetwo-coef=0_05-csor=10_0-a1=0_6.model \
# --output /home/giga/BIG/data/ML_TrainTest_ECEMC/save_model/VScan_33EC_66EMC_config05w101_after.xyz  \
# --device cuda --default_dtype float32  --info_prefix MACE_  --batch_size  50 --compute_stress

# python3 /home/giga/code/msms/mace/cli/eval_configs.py \
# --configs /home/giga/BIG/data/ML_TrainTest_ECEMC/save_model/VScan_EMC_config05w101.xyz \
# --model /home/giga/BIG/data/ML_TrainTest_ECEMC/save_model/MACE-OFF23_small-coef=-0_05-csor=10_0-a1=1_07.model \
# --output /home/giga/BIG/data/ML_TrainTest_ECEMC/save_model/VScan_EMC_config05w101_after.xyz  \
# --device cuda --default_dtype float32  --info_prefix MACE_  --batch_size  50 --compute_stress --return_node_feats



# python3 /home/giga/code/msms/mace/cli/eval_configs.py \
# --configs /home/giga/BIG/data/AmmoniumNitrate/al/work_an_revpbe_4090_c8352Y_v1/data/v1d10.xyz \
# --model /home/giga/BIG/data/AmmoniumNitrate/al/work_an_revpbe_4090_c8352Y_v1/data/v1d10ff600_32e2_16e0_r7_stress_stagetwo.model \
# --output /home/giga/BIG/data/AmmoniumNitrate/al/work_an_revpbe_4090_c8352Y_v1/data/v1d10_label.xyz  \
# --device cuda --default_dtype float32  --info_prefix MACE_  --batch_size  50 --compute_stress --return_node_feats


# python3 /home/giga/code/msms/mace/cli/eval_configs.py \
# --configs /home/giga/BIG/data/AmmoniumNitrate/al/work_an_revpbe_4090_c8352Y_v1/data/ff600.xyz \
# --model /home/giga/BIG/data/AmmoniumNitrate/al/work_an_revpbe_4090_c8352Y_v1/data/v1d10ff600_32e2_16e0_r7_stress_stagetwo.model \
# --output /home/giga/BIG/data/AmmoniumNitrate/al/work_an_revpbe_4090_c8352Y_v1/data/ff600_label.xyz  \
# --device cuda --default_dtype float32  --info_prefix MACE_  --batch_size  50 --compute_stress --return_node_feats


# data="/home/giga/BIG/data/lipfecpc/work_lipfecpc_pbe_4090_c8352Y_v7/data/AL_17_sel_100_al"
# data="/home/giga/BIG/data/lipfecpc/work_lipfecpc_pbe_4090_c8352Y_v7/data/AL_17_sel_130_al"
# data="/home/giga/BIG/data/lipfecpc/work_lipfecpc_pbe_4090_c8352Y_v7/data/AL_17_sel_216_al"
# data="/home/giga/BIG/data/lipfecpc/work_lipfecpc_pbe_4090_c8352Y_v7/data/AL_17_sel_168_al"

# data="/home/giga/BIG/data/ML_TrainTest_ECEMC/al-new/work_ecemc_pbe_4090_c8352Y_v1/data/test/AL_17_120_al"
# data="/home/giga/BIG/data/ML_TrainTest_ECEMC/al-new/work_ecemc_pbe_4090_c8352Y_v1/data/test/AL_17_140_al"
# data="/home/giga/BIG/data/ML_TrainTest_ECEMC/al-new/work_ecemc_pbe_4090_c8352Y_v1/data/test/AL_17_150_al"
# data="/home/giga/BIG/data/ML_TrainTest_ECEMC/al-new/work_ecemc_pbe_4090_c8352Y_v1/data/test/AL_17_160_al"
# data="/home/giga/BIG/data/ML_TrainTest_ECEMC/al-new/work_ecemc_pbe_4090_c8352Y_v1/data/test/AL_17_180_al"
# data="/home/giga/BIG/data/ML_TrainTest_ECEMC/al-new/work_ecemc_pbe_4090_c8352Y_v1/data/AL_17_sel"
# # data="/home/giga/BIG/data/ML_TrainTest_ECEMC/al-new/work_ecemc_pbe_4090_c8352Y_v1/data/v1d13_160_al"

# python3 /home/giga/code/msms/mace/cli/eval_configs.py \
# --configs ${data}.xyz \
# --model /home/giga/BIG/data/ML_TrainTest_ECEMC/al-new/work_ecemc_pbe_4090_c8352Y_v1/data/v1d13_32e3_16e0_r10_b64_stress_run-2026_stagetwo.model \
# --output ${data}_label.xyz \
# --device cuda --default_dtype float32  --info_prefix MACE_  --batch_size  200 --compute_stress  # --return_node_feats



data="/home/giga/BIG/data/ML_TrainTest_ECEMC/al-new/rigid_test/pbe_tzv2p_nomol_ot_d3bj_virial_cut800_LiquidConfigs_test40"
# model="/home/giga/BIG/data/ML_TrainTest_ECEMC/al-new/work_ecemc_pbe_4090_c8352Y_v1/data/v1d13_32e3_16e0_r10_b64_stress_run-9_stagetwo.model"
# model="/home/giga/BIG/data/ML_TrainTest_ECEMC/al-new/work_ecemc_pbe_4090_c8352Y_v1/data/add-model/v1d9_32e3_16e0_r10_b64_stress_run-9_stagetwo.model"
# model="/home/giga/BIG/data/ML_TrainTest_ECEMC/al-new/work_ecemc_pbe_4090_c8352Y_v1/data/add-model/v1d5_32e3_16e0_r10_b64_stress_run-9_stagetwo.model"
model="/home/giga/BIG/data/ML_TrainTest_ECEMC/al-new/work_ecemc_pbe_4090_c8352Y_v1/data/add-model/v1d2_32e3_16e0_r10_b64_stress_run-9_stagetwo.model"



python3 /home/giga/code/msms/mace/cli/eval_configs.py \
--configs ${data}.xyz \
--model ${model} \
--output ${data}_label.xyz \
--device cuda --default_dtype float32  --info_prefix MACE_  --batch_size  200 --compute_stress # --return_node_feats


