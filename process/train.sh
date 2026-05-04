#!/bin/bash 

source /home/giga/soft/mambaforge/condabin/mamba/activate /home/giga/soft/mambaforge/envs/py310pt250cu124

epoch=1000
seed=2026
data_path="/home/giga/BIG/data/AmmoniumNitrate/lmp-ff/lammps_3w10/traj_10ns_lmp_gfn1_stress"
run_name="test"

python3 /home/giga/code/msms/mace/cli/run_train.py \
--name ${run_name} --seed ${seed} \
--train_file ${data_path}_train.xyz  --valid_file ${data_path}_val.xyz  \
--test_file ${data_path}_test.xyz \
--forces_weight 100.0 --energy_weight 1.0 --stress_weight 100.0 --swa_stress_weight 1000.0 --config_type_weights='{"Default":1.0}' \
--batch_size 10 --valid_batch_size 32 \
--energy_key REF_energy --forces_key REF_forces --virials_key REF_virial \
--interaction RealAgnosticResidualInteractionBlock \
--interaction_first RealAgnosticResidualInteractionBlock \
--max_num_epochs ${epoch}  --model ScaleShiftMSMACECSO --loss weighted --gate silu \
--r_max 10.0 --r_mid 7.0 --r_min 3.5 \
--max_ell 3 --long_max_ell 0 \
--hidden_irreps 64x0e  --long_node_feats_irreps 16x0e \
--radial_MLP "[64,64,64]" --long_radial_MLP "[16,16,64]" \
--correlation 3  --error_table PerAtomRMSE \
--num_interactions 2 --lr 0.01 --optimizer adam --device cuda --num_radial_basis 8 --num_cutoff_basis=6 \
--patience 200 --E0s 'average' --default_dtype='float32'  \
--eval_interval=2 --scheduler_patience=5 \
--ema --ema_decay=0.99 --amsgrad --clip_grad=10.0 --restart_latest --enable_cueq True