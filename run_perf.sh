#!/usr/bin/env bash

computearch=cpu #gpu 
computer_name=$(hostname -s)
num_cores=$(grep -c ^processor /proc/cpuinfo)

slurm_file=run_tigergpu.sh

nsamples_computenode=10

branch_name=$(git symbolic-ref --short HEAD)
commit_sha1=$(git rev-parse HEAD)
commit_sha1_short=$(git rev-parse --short HEAD)

version_suffix=''
version_description=''
summary_version="${computer_name}_${branch_name}_${commit_sha1_short}_${version_suffix}"

computenode_summary_filename="${summary_version}.txt"

make clean

make

 # Create split CSV files in addition to master summary file
csv_result_dir="${computer_name}"
mkdir -p ${csv_result_dir}

export OMP_NESTED=TRUE
export OMP_PLACES=cores
export OMP_PROC_BIND=spread,close
export KMP_HOT_TEAMS_MODE=1
export KMP_HOT_TEAMS_MAX_LEVEL=2

for events_concurrency in 1 2 4 8 14 28 # 1 2 4 8 16 32 
do 
let innerloop_concurrency=${num_cores}/${events_concurrency}
export OMP_NUM_THREADS=${events_concurrency},${innerloop_concurrency}
# Erase old results files                                                                                                                                                                        
csv_result_file_computenode="${csv_result_dir}/${commit_sha1}_${computearch}_eventsconcurrency_${events_concurrency}.csv"
rm -f ${csv_result_file_computenode}

echo " ${computearch} performance test on ${computer_name} with events concurrency=${events_concurrency} " >> ${computenode_summary_filename}
    for sample in $(seq 1 $nsamples_computenode)
    do 
	echo "timing sample=${sample}/${nsamples_computenode}"
	jobid=$(sbatch --parsable --wait ${slurm_file})
        tail -n 1 slurm-${jobid}.out | xargs -n 1 | tail -n 1 | tee -a ${csv_result_file_computenode} >> ${computenode_summary_filename}
    done 
    echo " " >> ${computenode_summary_filename}
done 

mkdir -p "old_slurm/"
mv ./slurm-* old_slurm/
