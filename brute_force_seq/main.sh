#!/usr/bin/env bash
#SBATCH -t 0-03:00:00 -p slurm_shortgpu
#SBATCH --gres=gpu:1
#SBATCH --output=output-%j.txt

module load gcc cuda cmake
cd $SLURM_SUBMIT_DIR
cmake .
make

mesh=../sample_mesh.obj
spheref=../sample_spheres.csv
radius=3.0
outfile=outfileTATime

: '
mesh=../mycube.obj
spheref=../toyspheres.csv
radius=0.5
outfile=outfileToyTime
'
cuda-memcheck ./collide $mesh $spheref $radius $outfile

: '
cuda-memcheck ./collide $mesh ../Samples/sample_spheres_10.csv $radius outfileTA_10
cuda-memcheck ./collide $mesh ../Samples/sample_spheres_11.csv $radius outfileTA_11
cuda-memcheck ./collide $mesh ../Samples/sample_spheres_12.csv $radius outfileTA_12
cuda-memcheck ./collide $mesh ../Samples/sample_spheres_13.csv $radius outfileTA_13
cuda-memcheck ./collide $mesh ../Samples/sample_spheres_14.csv $radius outfileTA_14
cuda-memcheck ./collide $mesh ../Samples/sample_spheres_15.csv $radius outfileTA_15
cuda-memcheck ./collide $mesh ../Samples/sample_spheres_16.csv $radius outfileTA_16
cuda-memcheck ./collide $mesh ../Samples/sample_spheres_17.csv $radius outfileTA_17
cuda-memcheck ./collide $mesh ../Samples/sample_spheres_18.csv $radius outfileTA_18
cuda-memcheck ./collide $mesh ../Samples/sample_spheres_19.csv $radius outfileTA_19
cuda-memcheck ./collide $mesh ../Samples/sample_spheres_20.csv $radius outfileTA_20
'
