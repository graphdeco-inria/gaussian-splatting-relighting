set -e
mkdir -p colmap/real
for scene in {chest_of_drawers,garage_wall,hot_plates,kettle,paint_gun}; do
    wget https://repo-sam.inria.fr/fungraph/generative-radiance-field-relighting/controlnet_samples/$scene.zip
    python -m zipfile -e $scene.zip colmap/real/$scene/tmp
    mv colmap/real/$scene/tmp/color colmap/real/$scene/train/relit_images
    rm -r colmap/real/$scene/tmp
    rm $scene.zip
done