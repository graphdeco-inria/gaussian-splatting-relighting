set -e
mkdir -p colmap/real
for scene in {chest_of_drawers,garage_wall,hot_plates,kettle,paint_gun}; do
    wget https://repo-sam.inria.fr/fungraph/generative-radiance-field-relighting/datasets/real/$scene.zip
    python -m zipfile -e $scene.zip colmap/real/
    rm $scene.zip
done