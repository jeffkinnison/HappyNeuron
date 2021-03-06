from pathlib import Path
from balsam.launcher import dag
import happyneuron as hpn
from happyneuron.trakem2 import preprocess_tiles
import happyneuron.balsam
import os
from ffn.utils import bounding_box
from ffn.utils import geom_utils
import numpy as np


def trakem_montage_job(
    workflow_name, 
    raw_folder, 
    process_folder, 
    target='', 
    min=1024, max=2048, 
    num_nodes=1
):
    """
    Creates a Trakem2 Montage Job
    """
    emp = preprocess_tiles.EMTilePreprocessor(
        raw_folder, 
        os.path.join(process_folder,'align_raw.txt')
    )
    emp.run()
    
    if target=='':
        target=process_folder
    
    montage_args = ''
    montage_args += f' {process_folder}/align_raw.txt '
    montage_args += f' {target} '
    montage_args += f' --min {min} '
    montage_args += f' --max {max} '
    montage_args += f' --fiji {happyneuron.balsam.context.fiji} '
    job = dag.add_job(
        name=f'montage',
        workflow=workflow_name,
        application='trakem2_montage',
        num_nodes=num_nodes,
        args=montage_args,
        ranks_per_node=4,
        environ_vars='OMP_NUM_THREADS=32'
    )
    return job

def trakem_check_folders (folder_list):
    for kfolder in folder_list:
        print(os.path.isdir(kfolder),kfolder)

def trakem_create_targets (target_path, folder_list,keyword='target'):
    list_size = len(folder_list)
    if (list_size==0):
        print ('No folders in folder_list')
        return 
    else:
        targets = []
        for k in np.arange(0,list_size):
            ktarget = os.path.join(target_path,keyword+'_'+str(k))
            targets.append(ktarget)
            os.makedirs(ktarget,exist_ok=True)
        return targets
    
        
def trakem_create_montage_list(folder_list, target_list, MIN=1000, MAX=2000,create_jobs=False, num_nodes=128, workflow=''):
    list_size =  (len(folder_list))
    target_size = (len(target_list))
    
    ##TODO Check sizes and existance..
#     if (list_size != target_size & target_size != 0):
#         print ('lists and targets must have the same size')
    
    for k in np.arange(0,list_size):
        print(k)
        RAW_INPUT=folder_list[k]
        PROCESS_FOLDER=os.path.join(RAW_INPUT,'montage_dir')
        TARGET_FOLDER =target_list[k]
        
        jobs = []
        if(create_jobs):
            job = hpn_balsam_apps.trakem_montage_job(workflow, RAW_INPUT, PROCESS_FOLDER,target=TARGET_FOLDER, min=MIN, max=MAX, num_nodes=num_nodes)
            jobs.append(job)
        else:
            print('Debug mode.......')
            print(RAW_INPUT, PROCESS_FOLDER, TARGET_FOLDER, MIN, MAX, workflow, num_nodes)
    return jobs


def aligntk_create_dirs(OUTPUT_DIR, MASK_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MASK_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR+'/cmaps', exist_ok=True)
    os.makedirs(OUTPUT_DIR+'/logs', exist_ok=True)
    os.makedirs(OUTPUT_DIR+'/amaps', exist_ok=True)
    os.makedirs(OUTPUT_DIR+'/grids', exist_ok=True)
    os.makedirs(OUTPUT_DIR+'/maps', exist_ok=True)
    os.makedirs(OUTPUT_DIR+'/aligned', exist_ok=True)
    

def aligntk_make_schedule(OUTPUT_DIR):
    
    schedule = ['10   1.0  0.1',
                '9   1.0  0.1',
                '8   1.0  0.3',
                '7   1.0  1.0',
                '7   1.0  2.0',
                '7   1.0  5.0',
                '6   1.0  5.0']
    
    file = open(OUTPUT_DIR+"/schedule.lst","w+") 
    for L in schedule:
        file.writelines(L)
        file.writelines('\n')
    file.close() 
 


def aligntk_rst_job(
    workflow_name,
    aligntk_path,
    images,
    mask,
    output,
    rotation,
    max_res,
    scale,
    tx,
    ty,
    n_proc):
    for n in range(n_proc):
        pairs = os.path.join(output, 'pairs%d.lst' % n)
        cmaps_output = os.path.join(output, 'cmaps/')
        summary = os.path.join(cmaps_output, 'summary%d.out' % n)
        
        find_rst_args = (f'--aligntk_path={aligntk_path}'
            f' --pairs={pairs}'
            f' --images={images}'
            f' --mask={mask}'
            f' --output={cmaps_output}'
            f' --rotation="{rotation}"'
            f' --max_res={max_res}'
            f' --scale="{scale}"'
            f' --tx="{tx}"'
            f' --ty="{ty}"'
            f' --summary={summary}')

        print(find_rst_args)

        dag.add_job(name=f'find_rst',                                                                                                                                                                                                                                                                     
            workflow=workflow_name,                                                                                                                                                                                                                                                                  
            application='aligntk_findrst',                                                                                                                                                                                                                                                            
            num_nodes=1,                                                                                                                                                                                                                                                                     
            args=find_rst_args,                                                                                                                                                                                                                                                                       
            ranks_per_node=n_proc,                                                                                                                                                                                                                                                                        
            environ_vars='OMP_NUM_THREADS=32') 


def ffn_compute_partitions_job(
    input_volume, 
    input_volume_dset, 
    output_volume, 
    thresholds, 
    lom_radius, 
    min_size, 
    workflow='',
    num_nodes=1
):
    comp_args = ''
    comp_args += f' --input_volume {input_volume}:{input_volume_dset} '
    comp_args += f' --output_volume {output_volume}:af '
    comp_args += f' --thresholds {thresholds} '
    comp_args += f' --lom_radius {lom_radius}  '
    comp_args += f' --min_size {min_size} '

    job = dag.add_job(name=f'comp_parts',
        workflow=workflow,
        num_nodes=num_nodes,
        application='ffn_compute_partitions',
        args=comp_args,
        ranks_per_node=1
    )
    return job

def ffn_build_coordinates_job(SAMPLE, PARTITION_DATA, TFRECORDFILE, MARGIN, workflow, num_nodes=1):
    build_args = ''
    build_args += f' --partition_volumes {SAMPLE}:{PARTITION_DATA}:af '
    build_args += f'  --coordinate_output {TFRECORDFILE} '
    build_args += f' --margin {MARGIN} '

    job = dag.add_job(name=f'build_coords',
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

    job = dag.add_job(name='test_train',
            workflow=workflow,
            application='ffn_trainer',
            args=train_args)
    return job



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
        job = dag.add_job(name=f'sub_inference_{i}',
                workflow=workflow_name,
                application='ffn_inference',
                args=inference_args,
                ranks_per_node=1,
                environ_vars='OMP_NUM_THREADS=32')
        jobs.append(job)
    return jobs
