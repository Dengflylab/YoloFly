#!/bin/bash
#SBATCH -N 1               # request one node
#SBATCH -t 2-09:00:00	        # request two hours
#SBATCH -p gpu    # in single partition (queue)
#SBATCH -A loni_quantum01
#SBATCH -o slurm.out # optional, name of the stdout, using the job number (%j) and the hostname of the node (%N)
#SBATCH -e slurm.err # optional, name of the stderr, using job and hostname values
# below are job commands

#bash tmp
Model=2022_03_01_p529_1280_5l_e700_b128
Yolo=/work/ken/yolov5

N_ROW=$(wc -l Video_list|awk '{print $1}')
for (( i=1; i<=$N_ROW; i++ ));do TMP=$(awk 'NR=='$i'{print}' Video_list); NUM=$(echo $TMP|awk '{print $1}');
    VIDEO=$(echo $TMP|awk '{print $2}'); echo $VIDEO $NUM;
    python3 $Yolo/detect_220101.py --weight $Yolo/runs/train/$Model/weights/best.pt --source $VIDEO --conf 0.4 --bh-count --tar-track --head-bind --img-size 1280 --num-fly $NUM
done
