#!/bin/bash
#SBATCH --job-name=1n1k_vit_s_jepa         # A descriptive job name
#SBATCH --output=1n1k_vit_s_jepa_%j.out     # Output file; %j will be replaced with the job ID
#SBATCH --nodes=1                           # Use a single node
#SBATCH --gres=gpu:2                        # Request 2 GPUs on that node
#SBATCH --cluster=gpu                       # Specify the GPU cluster
#SBATCH --partition=ls40s                    # Use the ls40s partition (adjust if needed)
#SBATCH --time=24:00:00                     # Walltime (days-HH:MM:SS); adjust as necessary
#SBATCH --mail-user=DTK28@pitt.edu     # Your Pitt email for notifications
#SBATCH --mail-type=END,FAIL                # Send email when job ends or fails

# Clean environment and load required modules
module purge
module load python/ondemand-jupyter-python3.11  # Replace with your specific Python/PyTorch module if needed
source .venv/bin/activate

# Change to the directory where you submitted the job
cd $SLURM_SUBMIT_DIR

# Run the training command with torchrun on 2 GPUs
torchrun --nproc_per_node=2 main.py \
  --fname configs/in1k_vits16-224_ep300.yaml \
  --devices cuda:0 cuda:1
