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
outfile=outfileTATimeParaSha

: '
mesh=../mycube.obj
spheref=../toyspheres.csv
radius=0.5
outfile=outfileToyTimeParaSha
'
cuda-memcheck ./collide $mesh $spheref $radius $outfile

: '
cuda-memcheck ./collide $mesh ../Samples/sample_spheres_10.csv $radius outfileTAParaSha_10
cuda-memcheck ./collide $mesh ../Samples/sample_spheres_11.csv $radius outfileTAParaSha_11
cuda-memcheck ./collide $mesh ../Samples/sample_spheres_12.csv $radius outfileTAParaSha_12
cuda-memcheck ./collide $mesh ../Samples/sample_spheres_13.csv $radius outfileTAParaSha_13
cuda-memcheck ./collide $mesh ../Samples/sample_spheres_14.csv $radius outfileTAParaSha_14
cuda-memcheck ./collide $mesh ../Samples/sample_spheres_15.csv $radius outfileTAParaSha_15
cuda-memcheck ./collide $mesh ../Samples/sample_spheres_16.csv $radius outfileTAParaSha_16
cuda-memcheck ./collide $mesh ../Samples/sample_spheres_17.csv $radius outfileTAParaSha_17
cuda-memcheck ./collide $mesh ../Samples/sample_spheres_18.csv $radius outfileTAParaSha_18
cuda-memcheck ./collide $mesh ../Samples/sample_spheres_19.csv $radius outfileTAParaSha_19
cuda-memcheck ./collide $mesh ../Samples/sample_spheres_20.csv $radius outfileTAParaSha_20
'
