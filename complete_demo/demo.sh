# Assume you have the folling env:
# OPENPOSE_DIR, CUDA_VISIBLE_DEVICES(optional)
CURRENT_DIR=$(pwd)
rm -r cache
rm -r output
mkdir cache
cd $OPENPOSE_DIR
./build/examples/openpose/openpose.bin --image_dir $CURRENT_DIR/input/ --no_display --write_keypoint=$CURRENT_DIR/cache
cd $CURRENT_DIR
matlab -nodesktop -nojvm -r "script1; exit"
SEG_MODEL=./latest_2.t7 th test_seg.lua
matlab -nodesktop -nojvm -r "script2; exit"
python test_lang_initial.py
mkdir output
th test_gan.lua
echo "Finished! If no errors are observed above then everything is smooth, and the results are expected to be saved in the output folder."

