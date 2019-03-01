PREFIX=$1

RESIZE_DIR=$PREFIX'256'
MAT_DIR=$PREFIX'mat'
EDGE_DIR=$PREFIX'edge'
DATA_DIR=$PREFIX'data'

python resize.py --input-dir $PREFIX --output-dir $RESIZE_DIR

python batch_hed.py --images_dir $RESIZE_DIR --hed_mat_dir $MAT_DIR

python mat2jpg.py --input-dir $MAT_DIR --output-dir $EDGE_DIR

python combine_A_and_B.py --fold_A $RESIZE_DIR --fold_B $EDGE_DIR --fold_AB $DATA_DIR