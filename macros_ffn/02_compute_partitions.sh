compute_partitions.py \
    --input_volume training_data.h5:labels \
    --output_volume training_data_partitions.h5:af \
    --thresholds 0.025,0.05,0.075,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
    --lom_radius 5,5,5 \
    --min_size 50
