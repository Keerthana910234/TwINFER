#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --nodes=1
#SBATCH --ntasks=33
#SBATCH --mem=10GB
#SBATCH --time=48:00:00
#SBATCH --job-name=A_to_B
#SBATCH --output=/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/parameter_scan_simulations/slurm_log/slurmLog-%A-%x.out
#SBATCH --error=/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/parameter_scan_simulations/slurm_log/slurmLog-%A-%x.err
#SBATCH --array=0-9

eval "$(conda shell.bash hook)"
conda activate twinfer
SLURM_ARRAY_TASK_ID=1
start_index=$((100 * SLURM_ARRAY_TASK_ID))
end_index=$((start_index + 200))
# start_rows=( 446 1066 1655 2286 2990 3975 4588 5250 5904 6310 6543 7672 8210 8722 8955 9584 10775 11996 12436 13136 13747 14849 15477 16124 17148 17392 18363 18992 19581 20184 20688 20932 21510 22576 23194 23765 24011 24252 24506 24748 )
# end_rows=( 1065 1654 2285 2989 3973 4587 5249 5903 6309 6542 7671 8209 8721 8954 9583 10774 11995 12435 13135 13746 14848 15476 16123 17147 17391 18362 18991 19580 20182 20687 20931 21509 22575 23193 23764 24010 24251 24505 24747 24999 )

task_id=$SLURM_ARRAY_TASK_ID
# start_index=${start_rows[$task_id]}

path_to_parameter="/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/parameter_scan_simulations/simulation_details/parameters_2genes_positive_k_add_10_k_on.csv"
path_to_connectivity_matrix="/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/parameter_scan_simulations/simulation_details/connectivity_matrix_A_to_B.txt"
path_to_output_folder="/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/parameter_scan_simulations/A_to_B_k_add_10_k_on/"
path_to_log_file="/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/parameter_scan_simulations/logs/A_to_B_k_add_10_k_on.jsonl"
type_of_interaction="A_to_B"

# Run Python script with matching CLI arguments
python /home/gzu5140/Keerthana_b1042/grnInference/code/TwINFER/TwINFER_function_scripts/gillespie_script.py \
    --path_to_connectivity_matrix "$path_to_connectivity_matrix" \
    --param_csv "$path_to_parameter" \
    --row_to_start "$start_index" \
    --row_to_end "$end_index" \
    --output_folder "$path_to_output_folder" \
    --log_file "$path_to_log_file" \
    --type "$type_of_interaction" \
    --number_parallel_processes 3 \
    --number_of_cores_per_parameter 10\
    --n_genes 2 \
    --n_cells 6000 \
    --simulation_time_before_division 1500 \
    --twin_simulation_time_after_division 48 \
    --twin_measurement_resolution 1