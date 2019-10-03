import sys
sys.path.insert(0,'/soft/datascience/Balsam/0.3.5.1/env/lib/python3.6/site-packages/')
sys.path.insert(0,'/soft/datascience/Balsam/0.3.5.1/')

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