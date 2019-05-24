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
outfile=outfileTATimePara

: '
mesh=../mycube.obj
spheref=../toyspheres.csv
radius=0.5
outfile=outfileToyTimePara
'
cuda-memcheck ./collide $mesh $spheref $radius $outfile

: '
cuda-memcheck ./collide $mesh ../Samples/sample_spheres_10.csv $radius outfileTAPara_10
cuda-memcheck ./collide $mesh ../Samples/sample_spheres_11.csv $radius outfileTAPara_11
cuda-memcheck ./collide $mesh ../Samples/sample_spheres_12.csv $radius outfileTAPara_12
cuda-memcheck ./collide $mesh ../Samples/sample_spheres_13.csv $radius outfileTAPara_13
cuda-memcheck ./collide $mesh ../Samples/sample_spheres_14.csv $radius outfileTAPara_14
cuda-memcheck ./collide $mesh ../Samples/sample_spheres_15.csv $radius outfileTAPara_15
cuda-memcheck ./collide $mesh ../Samples/sample_spheres_16.csv $radius outfileTAPara_16
cuda-memcheck ./collide $mesh ../Samples/sample_spheres_17.csv $radius outfileTAPara_17
cuda-memcheck ./collide $mesh ../Samples/sample_spheres_18.csv $radius outfileTAPara_18
cuda-memcheck ./collide $mesh ../Samples/sample_spheres_19.csv $radius outfileTAPara_19
cuda-memcheck ./collide $mesh ../Samples/sample_spheres_20.csv $radius outfileTAPara_20
'
