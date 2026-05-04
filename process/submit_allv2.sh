#!/bin/bash

# 固定 epoch 和种子
epoch=1000
#seed_list=(20251 20252 20253  20254 20255)
#seed_list=(20251 20252 20253 )
#seed_list=( 20255 )
seed_list=( 20251 20252 20253  20254 20255)
# 多个数据集路径
data_sets=(
"paracetamol_300k_NVT_gaff_rebond_v55_50_gfn2"
"7000_300k_NVT_gaff_rebond_v55_50_gfn2"
"md22_Ac-Ala3-NHMe_300k_NVT_gaff_rebond_v55_50_gfn2"
#"aspirin_300k_NVT_gaff_rebond_v54_50_gfn2"
#"md22_DHA_300k_NVT_gaff_rebond_v53_50_gfn2"
)

# 循环组合提交
for data_path in "${data_sets[@]}"; do
    for seed in "${seed_list[@]}"; do
        dataset_name=$(basename "${data_path}")
        run_name="${dataset_name}_epoch${epoch}_ea_l3_b10_h256"
        job_name="${dataset_name}_epoch${epoch}_ea_l3_b10_h256_s${seed}"
        slurm_script="slurm_${job_name}.sh"

        cat <<EOF > $slurm_script
#!/bin/bash 
#SBATCH -J ${job_name}
#SBATCH --output=${job_name}.out 
#SBATCH --error=${job_name}.out  
#SBATCH -p gpu4090
##SBATCH -p a800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

module purge
module load compiler/cuda/12.8

source /public/software/apps/anaconda3/2024.10/bin/activate /public/home/junbao/.conda/envs/py310pt250cuda124
which python3

python3 /public/home/junbao/code/msms/mace/cli/run_train.py \\
--name ${run_name} --seed ${seed} \\
--train_file ${data_path}_train.xyz  --valid_file ${data_path}_val.xyz  \\
--test_file ${data_path}.xyz \\
--forces_weight 100.0 --energy_weight 1.0 --stress_weight 100.0 --swa_stress_weight 1000.0 --config_type_weights='{"Default":1.0}' \\
--batch_size 10 --valid_batch_size 32 \\
--energy_key REF_energy --forces_key REF_forces --virials_key REF_virial \\
--interaction RealAgnosticResidualInteractionBlock \\
--interaction_first RealAgnosticResidualInteractionBlock \\
--max_num_epochs ${epoch}  --model ScaleShiftMACE --loss weighted --gate silu \\
--r_max 6.0 --hidden_irreps 256x0e --default_dtype='float32' \\
--correlation 3 --max_ell 3 --error_table PerAtomRMSE --radial_MLP "[64,64,64]" \\
--num_interactions 2 --lr 0.01 --optimizer adam --device cuda --num_radial_basis 8 --num_cutoff_basis=6 \\
--patience 200 --E0s 'average' \\
--eval_interval=2 --scheduler_patience=5 \\
--ema --ema_decay=0.99 --amsgrad --clip_grad=10.0 --restart_latest --enable_cueq True
EOF

        # 提交作业
        sbatch $slurm_script
    done
done

