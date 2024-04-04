#!/bin/bash -l
#SBATCH -N 1
#SBATCH -A p200130
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
##SBATCH --mem-per-cpu=128GB
#SBATCH -p gpu
#SBATCH -q short
#SBATCH --time 04:00:00
#SBATCH --mail-user fred@zortify.com
#SBATCH --mail-type END,FAIL

cd fred/luxembourgish_ZSC

#Load Python module
module load Python

#Check Python version
python -c  'import sys; print(sys.version)'

#Create the virtual environment
#python -m venv my_python-env

#Source to activate the virtual environment
source my_python-env/bin/activate

#Install the dependencies (listed in requirement.txt) with pip in the virtual environment
python -m pip install -r requirements.txt

#Execute the program
for i in 0 1 2 3; do
  srun --ntasks=1 --exclusive --output=slurm-$i.out python src/run_training.py \
    --model_type luxembert \
    --training_data ours_hr \
    --seed $i &
done
wait