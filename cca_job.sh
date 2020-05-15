#!/bin/bash                      
#SBATCH -t 11:59:00                  # walltime = 1 hours and 30 minutes
#SBATCH -N 1                         # one node
#SBATCH -n 2                         #  two CPU (hyperthreaded) cores
#SBATCH --mem=8G
#SBATCH --array=[0-27:4]%10
module load openmind/singularity/3.2.0        # load singularity module
singularity exec -B /om2,/om3  /om2/user/drmiguel/vagrant/newVag/geometricneuro_pytorch.simg python openmind_swapsimilarity.py $SLURM_ARRAY_TASK_ID dep   # Run the job steps
