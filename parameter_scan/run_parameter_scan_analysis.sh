#!/bin/bash
#SBATCH --account=p32655
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=22
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
# Fixed input/output paths
path="/home/gzu5140/Keerthana_b1042/grnInference/simulation_data/parameter_scan_simulations/A_to_B"
out="/home/gzu5140/Keerthana_b1042/grnInference/analysisData/parameter_scan/A_to_B_redo"

# Each array task processes a different chunk
chunk_size=4500
start_index=$((chunk_size * SLURM_ARRAY_TASK_ID))

# Make job-specific subfolder
job_out="${out}/seed_2025"
mkdir -p "$job_out"

python /home/gzu5140/Keerthana_b1042/grnInference/code/TwINFER/parameter_scan/calculate_corr.py \
  --path_to_simulations "$path" \
  --output "$job_out" \
  --genes gene_1_mRNA gene_2_mRNA \
  --timepoints 1 10 \
  --jobs 4 \
  --shuffles_gene_gene 10000 \
  --shuffles_random_diff 10000 \
  --shuffles_directed 10000 \
  --batch_size 500 \
  --save_interval 500 \
  --seed 2025 \
  --start_index "$start_index"

job_out="${out}/seed_101010"
mkdir -p "$job_out"

python /home/gzu5140/Keerthana_b1042/grnInference/code/TwINFER/parameter_scan/calculate_corr.py \
  --path_to_simulations "$path" \
  --output "$job_out" \
  --genes gene_1_mRNA gene_2_mRNA \
  --timepoints 1 10 \
  --jobs 4 \
  --shuffles_gene_gene 10000 \
  --shuffles_random_diff 10000 \
  --shuffles_directed 10000 \
  --batch_size 500 \
  --save_interval 500 \
  --seed 101010 \
  --start_index "$start_index"