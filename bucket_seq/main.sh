#!/usr/bin/env bash
#SBATCH -t 0-00:59:00 -p slurm_shortgpu
#SBATCH --output=output_bin-%j.txt

module load cuda cmake
cd $SLURM_SUBMIT_DIR
cmake .
make

mesh=../sample_mesh.obj #sample_mesh.obj #mycube.obj
spheref=../sample_spheres.csv #sample_spheres.csv #toyspheres.csv
radius=3.0 #50.0 #10.0 #3.0 #0.5
outfile=outfile

./collide $mesh $spheref $radius $outfile
base=2
for pow in {10..20}; do
    spheref=../sample_spheres_$pow.csv
    ./collide $mesh $spheref $radius "${outfile}_${pow}"
done

#cat $outfile | head -n 1
: '
pow=24
n=2
N=$((n**$pow))
./collide $N 1
'
: '
n=2
seed=99

for pow in {1..24}; do
    N=$((n**$pow))
    ./collide $N $seed
done
'
