set -e
mkdir -p output
for scene in {chest_of_drawers,garage_wall,hot_plates,kettle,paint_gun}; do
    wget https://repo-sam.inria.fr/fungraph/generative-radiance-field-relighting/pretrained-scenes/$scene.zip
    python -m zipfile -e $scene.zip output/
    rm $scene.zip
done