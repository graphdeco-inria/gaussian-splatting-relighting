set -e
mkdir -p output
for scene in {easy,hard}_{bedroom,kitchen,livingroom,office}; do
    wget https://repo-sam.inria.fr/fungraph/generative-radiance-field-relighting/pretrained-scenes/$scene.zip
    python -m zipfile -e $scene.zip output/
    rm $scene.zip
done