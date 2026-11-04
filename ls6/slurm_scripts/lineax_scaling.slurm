#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script
#   for TACC Lonestar6 AMD Milan nodes
#
#   *** MPI Job in Normal Queue ***
# 
# Last revised: October 22, 2021
#
# Notes:
#
#   -- Launch this script by executing
#      "sbatch milan.mpi.slurm" on a Lonestar6 login node.
#
#   -- Use ibrun to launch MPI codes on TACC systems.
#      Do NOT use mpirun or mpiexec.
#
#   -- Max recommended MPI ranks per Milan node: 128
#      (start small, increase gradually).
#
#   -- If you're running out of memory, try running
#      fewer tasks per node to give each task more memory.
#
#----------------------------------------------------

#SBATCH -J lx_scaling           # Job name
#SBATCH -o lx_scaling.o%j       # Name of stdout output file
#SBATCH -e lx_scaling.e%j       # Name of stderr error file
#SBATCH -p gpu-a100           # Queue (partition) name
#SBATCH -N 1               # Total # of nodes 
#SBATCH -n 32              # Total # of mpi tasks
#SBATCH -t 00:05:00        # Run time (hh:mm:ss)
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH -A MCB24085       # Project/Allocation name (req'd if you have more than 1)
#SBATCH --mail-user=kmeyer2@utexas.edu

# Any other commands must follow all #SBATCH directives...
module load cuda/12.8 # load cuda
cd $SCRATCH           
source $SCRATCH/virtual_environments/jax_env/bin/activate

# navigate to lineax scaling code (assumes installed already)
cd $SCRATCH/soft/jax_testing/demos/
python --version
python sparse_lineax_sharding.py

# try to launch GPU run with 2 GPUs on one node - need to configure this.
# srun can be used as only 1 node is used here.


# Launch MPI code... 
# ibrun ./myprogram         # Use ibrun instead of mpirun or mpiexec
