#!/bin/bash
#SBATCH -N 1               # request one node
#SBATCH -t 0-08:00:00	        # request two hours
#SBATCH -p single   # in single partition (queue)
#SBATCH -A loni_denglab
#SBATCH -o slurm_Pre_analysis.out # optional, name of the stdout, using the job number (%j) and the hostname of the node (%N)
#SBATCH -e slurm_Pre_analysis.err # optional, name of the stderr, using job and hostname values


#python Post_data/QC.py

YOLO_DIR=/work/ken/yolov5
python $YOLO_DIR/Post_data/QC.py
#python Post_data/Plot_1.py -p 60
