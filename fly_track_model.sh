#!/bin/bash
#SBATCH -N 1               # request one node
#SBATCH -t 0-20:30:000        # request two hours
#SBATCH -p gpu    # in single partition (queue)
#SBATCH -A loni_quantum01
#SBATCH -o slurm.out # optional, name of the stdout, using the job number (%j) and the hostname of the node (%N)
#SBATCH -e slurm.err # optional, name of the stderr, using job and hostname values
# below are job commands


python3 $Yolo/detect_220101.py --weight $Yolo/runs/train/$Model/weights/best.pt --source $VIDEO --conf 0.4 --bh-count --tar-track --head-bind --img-size 1280 --num-fly $NUM
