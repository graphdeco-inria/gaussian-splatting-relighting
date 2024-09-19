set -e
mkdir -p colmap/synthetic/
for scene in {easy,hard}_{bedroom,kitchen,livingroom,office}; do
    wget https://repo-sam.inria.fr/fungraph/generative-radiance-field-relighting/controlnet_samples/$scene.zip
    python -m zipfile -e $scene.zip colmap/synthetic/$scene/tmp
    mv colmap/synthetic/$scene/tmp/color colmap/synthetic/$scene/train/relit_images
    rm -r colmap/synthetic/$scene/tmp
    rm $scene.zip
done