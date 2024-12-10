#!/bin/bash

#SBATCH -p general
#SBATCH --nodes=12
#SBATCH --ntasks-per-node=8
#SBATCH -o base_out_%j.out
#SBATCH -e base_error_%j.err
#SBATCH --mail-type=ALL
#SBATCH --time=2-30:00:00
#SBATCH --job-name=NER_processor


#Load any modules that your program needs
module load python
#pip install transformers 
#pip install seqeval
pip install transformers[torch]
pip install unicodedata
echo "Modules loaded and required Python packages installed successfully."

python NER.py 
