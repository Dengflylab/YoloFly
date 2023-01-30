
for i in $(ls ../FliesDetect/Behavior_movie/png/fly_20210804.mp4_frame_*.txt| head -n 60); do pic=$(echo $i| sed "s/txt/png/") ;  cp $pic fly/train/images; cp $i fly/train/labels/; done

rm fly/test/images/* fly/test/labels/* fly/valid/images/* fly/valid/labels/*

for i in $(ls ../FliesDetect/Behavior_movie/png/fly_20210804.mp4_frame_*.txt| head -n 70| tail -n 10); do pic=$(echo $i| sed "s/txt/png/") ;  cp $pic fly/test/images; cp $i fly/test/labels/; done
for i in $(ls ../FliesDetect/Behavior_movie/png/fly_20210804.mp4_frame_*.txt| head -n 80| tail -n 10); do pic=$(echo $i| sed "s/txt/png/") ;  cp $pic fly/valid/images; cp $i fly/valid/labels/; done
