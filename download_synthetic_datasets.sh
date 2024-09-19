set -e
mkdir -p colmap/synthetic/
for scene in {easy,hard}_{bedroom,kitchen,livingroom,office}; do
    wget https://repo-sam.inria.fr/fungraph/generative-radiance-field-relighting/datasets/synthetic/$scene.zip
    python -m zipfile -e $scene.zip colmap/synthetic/
    rm $scene.zip
done