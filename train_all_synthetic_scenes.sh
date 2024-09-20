for SCENE in colmap/synthetic/*; do 
    python train.py --halfres -s $SCENE
    python render.py -m ${SCENE/colmap/output}
done