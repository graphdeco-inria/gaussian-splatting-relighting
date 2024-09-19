: ${1:?"Please provide a trained model path e.g. output/synthetic/hard_kitchen/00"}

CRF=${CRF:-17}
ITERATIONS=${ITERATIONS:-30000}

echo 👉 Rendering videos for synthetic scene $1 &&

target_dirs="23 14 18 10" && 

# this renders the videos for the principal directions
python render.py --video -m $1 &&
for dir_id in $target_dirs; do 
    ffmpeg -y -framerate 30 -i "$1/test/video_frames/%05d_dir_${dir_id}.png" -c:v libx264 -pix_fmt yuv420p -crf $CRF $1/video_dir_${dir_id}.mp4
done && 

# this renders the light sweep
python render.py --sweep -m $1 &&
ffmpeg -y -framerate 60 -pattern_type glob -i "$1/test/sweep_frames/*.png" -c:v libx264 -pix_fmt yuv420p $1/sweep.mp4 &&

true
