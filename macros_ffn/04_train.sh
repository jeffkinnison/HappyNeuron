export SAMPLE=cell_goodpeeps
export T_FOLDER=training_points_8_11_1_1/

tensorboard --logdir $T_FOLDER &

train.py \
    --train_coords tf_record_file \
    --data_volumes $SAMPLE:training_data.h5:image \
    --label_volumes $SAMPLE:training_data.h5:labels \
    --model_name convstack_3d.ConvStack3DFFNModel \
    --model_args "{\"depth\": 8, \"fov_size\": [11, 11, 11], \"deltas\": [1, 1, 1]}" \
    --image_mean 126 \
    --image_stddev 41 \
    --train_dir $T_FOLDER \
    --max_steps 10000000

