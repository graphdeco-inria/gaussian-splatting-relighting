: ${1:?"Please provide a trained model path e.g. output/real/garage_wall/00"}


MODEL=$(dirname $1)
LABEL=$(basename $1)
NAME=$(basename $MODEL)

CRF=${CRF:-17}

set -e

for k in 0 1; do 
    python _render_relighting_video.py -m $MODEL/$LABEL --camera_mode ken_burns --view_number $k
    ffmpeg -y -framerate 60 -i $MODEL/$LABEL/train/renders/%05d.png -pix_fmt yuv420p -c:v h264 -crf $CRF -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" out_${NAME}_${LABEL}_ken_burns_$k.mp4 
done

python _render_relighting_video.py -m $MODEL/$LABEL --camera_mode orbit --view_number 0
ffmpeg -y -framerate 60 -i $MODEL/$LABEL/train/renders/%05d.png -pix_fmt yuv420p -c:v h264 -crf $CRF -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" out_${NAME}_${LABEL}_orbit_0.mp4

python _render_relighting_video.py -m $MODEL/$LABEL --camera_mode static --view_number 0
ffmpeg -y -framerate 60 -i $MODEL/$LABEL/train/renders/%05d.png -pix_fmt yuv420p -c:v h264 -crf $CRF -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" out_${NAME}_${LABEL}_static_0.mp4
