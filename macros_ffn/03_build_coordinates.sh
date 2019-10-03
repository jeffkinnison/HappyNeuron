export SAMPLE=cell_goodpeeps

build_coordinates.py \
     --partition_volumes $SAMPLE:training_data_partitions.h5:af \
     --coordinate_output tf_record_file \
     --margin 5,5,5
