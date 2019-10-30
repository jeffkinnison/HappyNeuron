import sys


##BALSAM IMPORTS
sys.path.insert(0,'/soft/datascience/Balsam/0.3.5.1/env/lib/python3.6/site-packages/')
sys.path.insert(0,'/soft/datascience/Balsam/0.3.5.1/')
import balsam
from balsam_helper import *


##FFN IMPORTS
sys.path.insert(0,'/gpfs/mira-home/keceli/ffn/keceli_ffn/')
sys.path.insert(0,'/lus/theta-fs0/projects/connectomics_aesp/keceli/pip_ffn/')
sys.path.insert(0,'/soft/datascience/tensorflow/tf1.13/')


##HPN IMPORTS
sys.path.insert(0,'/lus/theta-fs0/projects/connectomics_aesp/software/HappyNeuron/')

import happyneuron as hpn




#path = hpn.__PATH__


# APP DATABASE
## No need to edit those. We will keep then up to date.
## We should get rid of this hardcode locations on the balsam app database

env_preamble = '/lus/theta-fs0/projects/connectomics_aesp/software/HappyNeuron/macros_theta/theta_build_preamble.sh'




##TRAKEM2 APPS


##############
##I don't like that but I need to add cv2 here and this env just don't have it.
sys.path.insert(0,'/lus/theta-fs0/projects/connectomics_aesp/software/neuro_env_36/lib/python3.6/site-packages/')
import happyneuron as hpn
from happyneuron.trakem2.preprocess_tiles import *
##############

##need 4 job / node
add_app(name='trakem_montage',
        executable='python /lus/theta-fs0/projects/connectomics_aesp/software/HappyNeuron/happyneuron/trakem2/mpi_montage.py',
        description='TRAKEM2 MPI montage script',
        envscript=env_preamble)

def trakem_montage_job(workflow_name, raw_folder, process_folder, target='', min=1024, max=2048, fiji="/lus/theta-fs0/projects/connectomics_aesp/software/Fiji.app/ImageJ-linux64", num_nodes=1):
    emp = EMTilePreprocessor(raw_folder, os.path.join(process_folder,'align_raw.txt'))
    emp.run()
    
    if target=='':
        target=process_folder
    
    #todo.. implement workflow pass for nule workflow names + uid
#     if workflow=='':
#         pass
    
    montage_args = ''
    montage_args += f' {process_folder}/align_raw.txt '
    montage_args += f' {target} '
    montage_args += f' --min {min} '
    montage_args += f' --max {max} '
    montage_args += f' --fiji {fiji} '
    print(montage_args)
    job = add_job(name=f'montage',
        workflow=workflow_name,
        application='trakem_montage',
        num_nodes=num_nodes,
        args=montage_args,
        ranks_per_node=4,
        environ_vars='OMP_NUM_THREADS=32')
    print('Trakem2 Montage Job added')
    return job



##can be run on a single stupid node
##will be phased out soon
add_app(name='trakem2_proc_folder',
        executable='python /lus/theta-fs0/projects/connectomics_aesp/software/HappyNeuron/happyneuron/trakem2/preprocess_stack.py',
        description='TRAKEM2 create pre aligment script',
        envscript=env_preamble)

##need 1 job / node
add_app(name='trakem2_align',
        executable='python /lus/theta-fs0/projects/connectomics_aesp/software/HappyNeuron/happyneuron/trakem2/align.py',
        description='TRAKEM2 aligment script',
        envscript=env_preamble)

##need 1 job / node
add_app(name='trakem2_export',
        executable='python /lus/theta-fs0/projects/connectomics_aesp/software/HappyNeuron/happyneuron/trakem2/mpi_export.py',
        description='TRAKEM2 MPI export script',
        envscript=env_preamble)

##ALIGNTK APPS


add_app(name='aligntk_gen_mask',
        executable='python /lus/theta-fs0/projects/connectomics_aesp/software/HappyNeuron/happyneuron/aligntk/gen_mask.py',
        description='AlignTK mask generator',
        envscript=env_preamble)

add_app(name='aligntk_apply_map',
        executable='python /lus/theta-fs0/projects/connectomics_aesp/software/HappyNeuron/happyneuron/aligntk/mpi_apply_map.py',
        description='AlignTK Apply Map',
        envscript=env_preamble)



##FLORIN APPS

#add_app(name='florin_')

##FFN PART!!

add_app(name='ffn_build_coordinates',
        executable='python /lus/theta-fs0/projects/connectomics_aesp/software/ffn/build_coordinates_mpi.py',
        description='Distributed FFN build coordinates script',
        envscript=env_preamble)

add_app(name='ffn_compute_partitions',
        executable='python /lus/theta-fs0/projects/connectomics_aesp/software/ffn/compute_partitions_mpi.py',
        description='Distributed FFN compute partitions scripts script',
        envscript=env_preamble)

add_app(name='ffn_trainer',
        executable='python /lus/theta-fs0/projects/connectomics_aesp/software/ffn/train_hvd.py',
        description='Distributed FFN training script',
        envscript=env_preamble)

add_app(name='ffn_inference',
        executable='python /lus/theta-fs0/projects/connectomics_aesp/software/ffn/run_inference.py',
        description='FFN inference script',
        envscript=env_preamble)



def ffn_compute_partitions_job(INPUT_VOLUME, INPUT_VOLUME_DSET, OUTPUT_VOLUME, THRESHOLDS, LOM_RADIUS, MIN_SIZE, workflow='',num_nodes=1):
    comp_args = ''
    comp_args += f' --input_volume {INPUT_VOLUME}:{INPUT_VOLUME_DSET} '
    comp_args += f' --output_volume {OUTPUT_VOLUME}:af '
    comp_args += f' --thresholds {THRESHOLDS} '
    comp_args += f' --lom_radius {LOM_RADIUS}  '
    comp_args += f' --min_size {MIN_SIZE} '

    job = add_job(name=f'comp_parts',
        workflow=workflow,
        num_nodes=num_nodes,
        application='ffn_compute_partitions',
        args=comp_args,
        ranks_per_node=1)
    return job

def ffn_build_coordinates_job(SAMPLE, PARTITION_DATA, TFRECORDFILE, MARGIN, workflow, num_nodes=1):
    build_args = ''
    build_args += f' --partition_volumes {SAMPLE}:{PARTITION_DATA}:af '
    build_args += f'  --coordinate_output {TFRECORDFILE} '
    build_args += f' --margin {MARGIN} '

    job = add_job(name=f'build_coords',
        workflow=workflow,
        num_nodes=num_nodes,
        application='ffn_build_coordinates',
        args=build_args,
        ranks_per_node=1)
    return job

def ffn_train_network_job(TFRECORDFILE, GROUNDTRUTH, GRAYSCALE, BATCHSIZE, OPTIMIZER, TIMESTAMP, TRAINDIR, DEPTH, FOV, DELTA, workflow):
    train_args = ''
    train_args += f' --train_coords {TFRECORDFILE} '
    train_args += f' --data_volumes valdation1:{GRAYSCALE}:raw '
    train_args += f' --label_volumes valdation1:{GROUNDTRUTH}:stack '
    train_args += f' --model_name convstack_3d.ConvStack3DFFNModel '
    #myargs += f' --model_args \"\{\\"depth\\": {DEPTH}, \\"fov_size\\": [{FOV}], \\"deltas\\": [{DELTA}]\}\"'
    train_args += ''' --model_args "{\\"depth\\": 12, \\"fov_size\\": [33, 33, 33], \\"deltas\\": [8, 8, 8]}"'''
    train_args += ' --image_mean 128 --image_stddev 33 '
    train_args += ' --max_steps 400 --summary_rate_secs 60 ' 
    train_args += f' --batch_size {BATCHSIZE} '
    train_args += f' --optimizer {OPTIMIZER} '
    train_args += ' --num_intra_threads 64 --num_inter_threads 1 '
    train_args += f' --train_dir {TRAINDIR} '

    job = add_job(name='test_train',
            workflow=workflow,
            application='ffn_trainer',
            args=train_args)
    return job


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
    jobs = []
    for i,box in enumerate(bbox_list):
        start = box.start
        size  = box.size
        inference_args  = f" --inference_request=\"$(cat "+config_file+")\" "
        inference_args += f" --bounding_box 'start {{ x:{start[0]} y:{start[1]} z:{start[2]} }} size {{ x:{size[0]} y:{size[1]} z:{size[2]} }}' "
        job = add_job(name=f'sub_inference_{i}',
                workflow=workflow_name,
                application='ffn_inference',
                args=inference_args,
                ranks_per_node=1,
                environ_vars='OMP_NUM_THREADS=32')
        jobs.append(job)
    return jobs

##CLOUDVOLUME APPS

add_app(
    'HappyNeuron_img2cv',
    'python /lus/theta-fs0/projects/software/HappyNeuron/happyneuron/io/img_to_cloudvolume.py',  # 'img_to_cloudvolume',
    description='Convert images to a CloudVolume layer.',
    envscript=env_preamble
)

#importlib.get_path_of('happyneuron.mesh.mesh_generator')

add_app(
    'HappyNeuron_meshing',
    'python /home/kinnison/HappyNeuron/happyneuron/mesh/mesh_generator.py',  # 'mesh_generator',
    description='Create a 3D segmentation mesh.',
    envscript=env_preamble
)

add_app(
    'HappyNeuron_h52cv',
    'python /lus/theta-fs0/projects/software/HappyNeuron/happyneuron/io/hdf5_to_cloudvolume.py',  # 'hdf5_to_cloudvolume',
    description='Convert images to a CloudVolume layer.',
    envscript=env_preamble
)


#add_app(name='cv_create_layer',
## --data_path --layer_type --mags --resolution --offset 

#add_app(name='cv_extract_block')
## --info --mag --offset --volume --file --key

#add_app(name='get_layer_properties')
##--histogram

#add_app(name='classify_objects')


#add_app(name='cv_create_mesh',

#add_app(name='cv_create_skeleton')
