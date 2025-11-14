#!/bin/bash -l
#SBATCH --account=b1042
#SBATCH --partition=genomics
#SBATCH --nodes=1
#SBATCH --ntasks=42
#SBATCH --mem=4GB
#SBATCH --time=12:00:00
#SBATCH --job-name=Parameter-Analysis
#SBATCH --output=/home/gzu5140/Keerthana_b1042/grnInference/logs/slurmLog-%A_%a-%x.out
#SBATCH --error=/home/gzu5140/Keerthana_b1042/grnInference/logs/slurmLog-%A_%a-%x.err

module purge
module load python-miniconda3/4.10.3
source /home/gzu5140/minicondacurl/etc/profile.d/conda.sh
conda activate twinfer
which python

# path="/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/parameter_scan_simulations/A_B"
# out="/home/gzu5140/Keerthana_b1042/grnInference/analysisData/parameter_scan_random_null_1_20/A_B_2_states"

# # Each array task processes a different chunk
# # Make job-specific subfolder
# job_out="${out}"
# mkdir -p "$job_out"

# ~/.conda/envs/twinfer/bin/python /home/gzu5140/Keerthana_b1042/grnInference/code/TwINFER/parameter_scan/calculate_corr.py \
#   --path_to_simulations "$path" \
#   --output "$job_out" \
#   --genes gene_1_mRNA gene_2_mRNA \
#   --timepoints 1 20 \
#   --jobs 8 \
#   --shuffles_gene_gene 10000 \
#   --shuffles_random_diff 10000 \
#   --shuffles_directed 10000 \
#   --batch_size 500 \
#   --save_interval 500 \
#   --seed 2025 \
#   --mode "pair" \
#   --csv "/home/gzu5140/Keerthana_b1042/grnInference/analysisData/parameter_scan_1_20/A_B_2_states/two_state_pairs_final.csv"

# path="/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/parameter_scan_simulations/A_to_B"
# out="/home/gzu5140/Keerthana_b1042/grnInference/analysisData/parameter_scan_random_null_1_20/A_to_B_2_states"

# # Each array task processes a different chunk
# # Make job-specific subfolder
# job_out="${out}"
# mkdir -p "$job_out"

# ~/.conda/envs/twinfer/bin/python /home/gzu5140/Keerthana_b1042/grnInference/code/TwINFER/parameter_scan/calculate_corr.py \
#   --path_to_simulations "$path" \
#   --output "$job_out" \
#   --genes gene_1_mRNA gene_2_mRNA \
#   --timepoints 1 20 \
#   --jobs 8 \
#   --shuffles_gene_gene 10000 \
#   --shuffles_random_diff 10000 \
#   --shuffles_directed 10000 \
#   --batch_size 500 \
#   --save_interval 500 \
#   --seed 2025 \
#   --mode "pair" \
#   --csv "/home/gzu5140/Keerthana_b1042/grnInference/analysisData/parameter_scan_1_20/A_to_B_2_states/two_state_pairs_final.csv"

# path="/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/parameter_scan_simulations/A_and_B"
# out="/home/gzu5140/Keerthana_b1042/grnInference/analysisData/parameter_scan_random_null_1_20/A_and_B_2_states"

# # Each array task processes a different chunk
# # Make job-specific subfolder
# job_out="${out}"
# mkdir -p "$job_out"

# ~/.conda/envs/twinfer/bin/python /home/gzu5140/Keerthana_b1042/grnInference/code/TwINFER/parameter_scan/calculate_corr.py \
#   --path_to_simulations "$path" \
#   --output "$job_out" \
#   --genes gene_1_mRNA gene_2_mRNA \
#   --timepoints 1 20 \
#   --jobs 8 \
#   --shuffles_gene_gene 10000 \
#   --shuffles_random_diff 10000 \
#   --shuffles_directed 10000 \
#   --batch_size 500 \
#   --save_interval 500 \
#   --seed 2025 \
#   --mode "pair" \
#   --csv "/home/gzu5140/Keerthana_b1042/grnInference/analysisData/parameter_scan_1_20/A_and_B_2_states/two_state_pairs_final.csv"

# path="/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/parameter_scan_simulations/A_rep_B/"
# out="/home/gzu5140/Keerthana_b1042/grnInference/analysisData/parameter_scan_random_null_1_20/A_rep_B_2_states"


# # Make job-specific subfolder
# job_out="${out}"
# mkdir -p "$job_out"

# ~/.conda/envs/twinfer/bin/python /home/gzu5140/Keerthana_b1042/grnInference/code/TwINFER/parameter_scan/calculate_corr.py \
#   --path_to_simulations "$path" \
#   --output "$job_out" \
#   --genes gene_1_mRNA gene_2_mRNA \
#   --timepoints 1 20 \
#   --jobs 8 \
#   --shuffles_gene_gene 10000 \
#   --shuffles_random_diff 10000 \
#   --shuffles_directed 10000 \
#   --batch_size 500 \
#   --save_interval 500 \
#   --seed 2025 \
#   --mode "pair" \
#   --csv "/home/gzu5140/Keerthana_b1042/grnInference/analysisData/parameter_scan_1_20/A_rep_B_2_states/two_state_pairs_final.csv"

# path="/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/parameter_scan_simulations/A_rep_B"
# out="/home/gzu5140/Keerthana_b1042/grnInference/analysisData/parameter_scan_random_null_1_20/A_rep_B"

# # Each array task processes a different chunk
# # Make job-specific subfolder
# job_out="${out}"
# mkdir -p "$job_out"

# ~/.conda/envs/twinfer/bin/python /home/gzu5140/Keerthana_b1042/grnInference/code/TwINFER/parameter_scan/calculate_corr.py \
#   --path_to_simulations "$path" \
#   --output "$job_out" \
#   --genes gene_1_mRNA gene_2_mRNA \
#   --timepoints 1 20 \
#   --jobs 8 \
#   --shuffles_gene_gene 10000 \
#   --shuffles_random_diff 10000 \
#   --shuffles_directed 10000 \
#   --batch_size 500 \
#   --save_interval 500 \
#   --seed 2025 \
#   --mode "single"



# # Fixed input/output paths
# path="/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/parameter_scan_simulations/A_and_B"
# out="/home/gzu5140/Keerthana_b1042/grnInference/analysisData/parameter_scan_random_null_1_20/A_and_B"

# # Each array task processes a different chunk
# # Make job-specific subfolder
# job_out="${out}"
# mkdir -p "$job_out"

# ~/.conda/envs/twinfer/bin/python /home/gzu5140/Keerthana_b1042/grnInference/code/TwINFER/parameter_scan/calculate_corr.py \
#   --path_to_simulations "$path" \
#   --output "$job_out" \
#   --genes gene_1_mRNA gene_2_mRNA \
#   --timepoints 1 20 \
#   --jobs 8 \
#   --shuffles_gene_gene 10000 \
#   --shuffles_random_diff 10000 \
#   --shuffles_directed 10000 \
#   --batch_size 500 \
#   --save_interval 500 \
#   --seed 2025 \


# Fixed input/output paths
path="/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/parameter_scan_simulations/A_B"
out="/home/gzu5140/Keerthana_b1042/grnInference/analysisData/parameter_scan_random_null_1_20/A_B"

Each array task processes a different chunk
Make job-specific subfolder
job_out="${out}"
mkdir -p "$job_out"

~/.conda/envs/twinfer/bin/python /home/gzu5140/Keerthana_b1042/grnInference/code/TwINFER/parameter_scan/calculate_corr.py \
  --path_to_simulations "$path" \
  --output "$job_out" \
  --genes gene_1_mRNA gene_2_mRNA \
  --timepoints 1 20 \
  --jobs 8 \
  --shuffles_gene_gene 10000 \
  --shuffles_random_diff 10000 \
  --shuffles_directed 10000 \
  --batch_size 500 \
  --save_interval 500 \
  --seed 2025 \

  # Fixed input/output paths
path="/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/parameter_scan_simulations/A_to_B"
out="/home/gzu5140/Keerthana_b1042/grnInference/analysisData/parameter_scan_random_null_1_20/A_to_B"

# Each array task processes a different chunk
# Make job-specific subfolder
job_out="${out}"
mkdir -p "$job_out"

~/.conda/envs/twinfer/bin/python /home/gzu5140/Keerthana_b1042/grnInference/code/TwINFER/parameter_scan/calculate_corr.py \
  --path_to_simulations "$path" \
  --output "$job_out" \
  --genes gene_1_mRNA gene_2_mRNA \
  --timepoints 1 20 \
  --jobs 8 \
  --shuffles_gene_gene 10000 \
  --shuffles_random_diff 10000 \
  --shuffles_directed 10000 \
  --batch_size 500 \
  --save_interval 500 \
  --seed 2025 \