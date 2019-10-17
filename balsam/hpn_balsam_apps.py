import sys
sys.path.insert(0,'/soft/datascience/Balsam/0.3.5.1/env/lib/python3.6/site-packages/')
sys.path.insert(0,'/soft/datascience/Balsam/0.3.5.1/')


sys.path.insert(0,'/gpfs/mira-home/keceli/ffn/keceli_ffn/')
sys.path.insert(0,'/lus/theta-fs0/projects/connectomics_aesp/keceli/pip_ffn/')
sys.path.insert(0,'/soft/datascience/tensorflow/tf1.13/')


import balsam
from balsam_helper import *

# APP DATABASE
## No need to edit those. We will keep then up to date.

env_preamble = '/lus/theta-fs0/projects/connectomics_aesp/software/HappyNeuron/macros_theta/theta_build_preamble.sh'




##TRAKEM2 APPS

##phasing out to be a single function inside the montage job
##can be run on a single stupid node
# add_app(name='trakem2_pre_tiles',
#         executable='python /lus/theta-fs0/projects/connectomics_aesp/software/klab_utils/trakem2/preprocess_tiles.py',
#         description='TRAKEM2 Create Montage script',
#         envscript='/lus/theta-fs0/projects/connectomics_aesp/software/macros_theta/theta_balsam_preamble.sh')


##need 1 job / node
add_app(name='trakem_montage',
        executable='python /lus/theta-fs0/projects/connectomics_aesp/software/klab_utils/trakem2/mpi_montage.py',
        description='TRAKEM2 MPI montage script',
        envscript=env_preamble)

##can be run on a single stupid node
add_app(name='trakem2_proc_folder',
        executable='python /lus/theta-fs0/projects/connectomics_aesp/software/klab_utils/trakem2/preprocess_stack.py',
        description='TRAKEM2 create pre aligment script',
        envscript=env_preamble)

##need 1 job / node
add_app(name='trakem2_align',
        executable='python /lus/theta-fs0/projects/connectomics_aesp/software/klab_utils/trakem2/align.py',
        description='TRAKEM2 aligment script',
        envscript=env_preamble)

##need 1 job / node
add_app(name='trakem2_export',
        executable='python /lus/theta-fs0/projects/connectomics_aesp/software/klab_utils/trakem2/mpi_export.py',
        description='TRAKEM2 MPI export script',
        envscript=env_preamble)

##ALIGNTK APPS

add_app(name='aligntk_apply_map',
        executable='python -m klab_utils.aligntk_mpi_apply_map',
        description='Distributed FFN training script',
        envscript=env_preamble)


##U-NET APPS

#add_app(name='unet_train')

#add_app(name='unet_infer')

##FLORIN APPS

#add_app(name='florin_')

##CLOUDVOLUME APPS

#add_app(name='cv_create_layer',
## --data_path --layer_type --mags --resolution --offset 

#add_app(name='cv_extract_block')
## --info --mag --offset --volume --file --key

#add_app(name='get_layer_properties')
##--histogram

#add_app(name='classify_objects')


#add_app(name='cv_create_mesh',

#add_app(name='cv_create_skeleton')

##AUTOMO APPS

#add_app(name='automo_preview',



##FFN PART!!


#add_#add_app(name='ffn_build_coordinates',

#add_app(name='ffn_potato2',


add_app(name='ffn_trainer',
        executable='python /lus/theta-fs0/projects/connectomics_aesp/software/ffn/train_hvd.py',
        description='Distributed FFN training script',
        envscript=env_preamble)

add_app(name='ffn_inference',
        executable='python /lus/theta-fs0/projects/connectomics_aesp/software/ffn/run_inference.py',
        description='FFN inference script',
        envscript=env_preamble)


from ffn.utils import bounding_box
from ffn.utils import geom_utils

def create_inference_config(pars, file_name):
    request_par = ('''image {
                  hdf5: "%s:%s"
                }
                image_mean: %s
                image_stddev: %s
                checkpoint_interval: 1800
                seed_policy: "PolicyPeaks"
                model_checkpoint_path: "%s"
                model_name: "convstack_3d.ConvStack3DFFNModel"
                model_args: "{\\"depth\\": 2, \\"fov_size\\": [5,5,5], \\"deltas\\": [1, 1, 1]}"
                segmentation_output_dir: "%s"
                inference_options {
                  init_activation: 0.9
                  pad_value: 0.05
                  move_threshold: 0.1
                  min_boundary_dist { x: 1 y: 1 z: 1}
                  segment_threshold: 0.08
                  min_segment_size: 10
                }''') % pars
    print (request_par)

def divide_bounding_box(bbox, subvolume_size, overlap):
    """
    Returns a list of bounding boxes that divides the given bounding box into subvolumes.
    Parameters
    ----------
    bbox: BoundingBox object,
    subvolume_size: list or tuple
    overlap: list or tuple
    """
    start = geom_utils.ToNumpy3Vector(bbox.start)
    size = geom_utils.ToNumpy3Vector(bbox.size)
    bbox = bounding_box.BoundingBox(start, size)
    calc = bounding_box.OrderlyOverlappingCalculator(outer_box=bbox, 
                                                    sub_box_size=subvolume_size, 
                                                    overlap=overlap, 
                                                    include_small_sub_boxes=True,
                                                    back_shift_small_sub_boxes=False)
    return [bb for bb in calc.generate_sub_boxes()]

def check_balsam_jobs(bbox, config_file):
    for i,box in enumerate(boxes):
        start = box.start
        size  = box.size
        print(f" --bounding_box 'start {{ x:{start[0]} y:{start[1]} z:{start[2]} }} size {{ x:{size[0]} y:{size[1]} z:{size[2]} }}' ")
    print(f" --inference_request=\"$(cat "+config_file+")\" ")


def generate_balsam_inference_jobs(bbox_list, config_file, workflow_name='ffn_sub_inference'):
    for i,box in enumerate(bbox_list):
        start = box.start
        size  = box.size
        inference_args  = f" --inference_request=\"$(cat "+config_file+")\" "
        inference_args += f" --bounding_box 'start {{ x:{start[0]} y:{start[1]} z:{start[2]} }} size {{ x:{size[0]} y:{size[1]} z:{size[2]} }}' "
        add_job(name=f'sub_inference_{i}',
                workflow=workflow_name,
                application='inference',
                args=inference_args,
                ranks_per_node=1,
                environ_vars='OMP_NUM_THREADS=32')


#app(name='automo_center',

#add_app(name='automo_search_center',

#add_app(name='automo_recon',

#add_app(name='automo_preview_recon')



#add_app(name='ffn_build_coordinates',

#add_app(name='ffn_potato2',


add_app(name='ffn_trainer',
        executable='python /lus/theta-fs0/projects/connectomics_aesp/software/ffn/train_hvd.py',
        description='Distributed FFN training script',
        envscript=env_preamble)

add_app(name='ffn_inference',
        executable='python /lus/theta-fs0/projects/connectomics_aesp/software/ffn/run_inference.py',
        description='FFN inference script',
        envscript=env_preamble)