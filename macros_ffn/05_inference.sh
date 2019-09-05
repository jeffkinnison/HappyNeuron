run_inference.py \
    --inference_request="$(cat 05_config.pbtxt)" \
    --bounding_box 'start { x:0 y:0 z:0 } size { x:128 y:128 z:128 }'
