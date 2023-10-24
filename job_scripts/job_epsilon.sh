#!/bin/bash
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=5:0:0    
#SBATCH --mail-user=grondin.suzie@courrier.uqam.ca
#SBATCH --mail-type=ALL

cd ~/$project
module purge
module load python/3.9.0 scipy-stack
source ~/venvtest/bin/activate

python RLandPricing/cluster/efficency_beta.py
<:wq

