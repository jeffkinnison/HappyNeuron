from pathlib import Path
import os
import sys
import balsam

HPN_TOP = Path(__file__).absolute().parent.parent
context = None

def validate_path(path, top=HPN_TOP):
    path = (top / path).absolute()
    if not path.exists():
        raise ValueError(f'{path} does not exist')
    return path

def pyexe_path(python, path, top=HPN_TOP):
    abspath = validate_path(path, top)
    return f'{python} {path}'

class BalsamContext:
    def __init__(
        self,
        database,
        fiji="/projects/connectomics_aesp/software/Fiji.app/ImageJ-linux64",
        ffn="/lus/theta-fs0/projects/connectomics_aesp/software/ffn",
        preamble="macros_theta/theta_build_preamble.sh",
        python= sys.executable,
    ):
        """
        Set up DB & Apps
        """
        global context
        if not Path(python).exists():
            raise ValueError(f'No such Python {python}')

        os.environ["BALSAM_DB_PATH"] = database_path
        from balsam.launcher import dag
        env_preamble = validate_path(preamble)
        self.fiji = fiji

        # *******
        # TRAKEM2
        # *******
        dag.add_app(
            name = "trakem_montage",
            executable = pyexe_path('happyneuron/trakem2/mpi_montage.py'),
            description = 'TRAKEM2 MPI montage',
            envscript = env_preamble
        )
        dag.add_app(
            name = 'trakem2_proc_folder',
            executable = pyexe_path('happyneuron/trakem2/preprocess_stack.py'),
            description = 'TRAKEM2 create pre aligment script',
            envscript = env_preamble
        )
        dag.add_app(
            name = 'trakem2_align',
            executable = pyexe_path('happyneuron/trakem2/align.py'),
            description = 'TRAKEM2 aligment script',
            envscript = env_preamble
        )
        dag.add_app(
            name = 'trakem2_export',
            executable = pyexe_path('happyneuron/trakem2/mpi_export.py'),
            description = 'TRAKEM2 MPI export script',
            envscript = env_preamble
        )
        
        # *******
        # ALIGNTK
        # *******
        add_app(
            name = 'aligntk_gen_mask',
            executable = pyexe_path('happyneuron/aligntk/gen_mask.py'),
            description = 'AlignTK mask generator',
            envscript = env_preamble
        )

        add_app(
            name = 'aligntk_apply_map',
            executable = pyexe_path('happyneuron/aligntk/mpi_apply_map.py'),
            description = 'AlignTK Apply Map',
            envscript = env_preamble
        )

        # ***
        # FFN
        # ***
        dag.add_app(
            name = 'ffn_build_coordinates',
            executable = pyexe_path('build_coordinates_mpi.py', top=ffn),
            description = 'Distributed FFN build coordinates script',
            envscript = env_preamble
        )

        dag.add_app(
            name = 'ffn_compute_partitions',
            executable = pyexe_path('compute_partitions_mpi.py', top=ffn),
            description = 'Distributed FFN compute partitions scripts script',
            envscript = env_preamble
        )

        dag.add_app(
            name = 'ffn_trainer',
            executable = pyexe_path('train_hvd.py', top=ffn),
            description = 'Distributed FFN training script',
            envscript = env_preamble
        )

        dag.add_app(
            name = 'ffn_inference',
            executable = pyexe_path('run_inference.py', top=ffn),
            description = 'FFN inference script',
            envscript = env_preamble
        )

        # ***********
        # CLOUDVOLUME
        # ***********
        dag.add_app(
            'HappyNeuron_img2cv',
            pyexe_path('happyneuron/io/img_to_cloudvolume.py'),
            description = 'Convert images to a CloudVolume layer.',
            envscript = env_preamble
        )
        dag.add_app(
            'HappyNeuron_meshing',
            pyexe_path('happyneuron/mesh/mesh_generator.py'),
            description = 'Create a 3D segmentation mesh.',
            envscript = env_preamble
        )
        dag.add_app(
            'HappyNeuron_h52cv',
            pyexe_path('happyneuron/io/hdf5_to_cloudvolume.py'),
            description = 'Convert images to a CloudVolume layer.',
            envscript = env_preamble
        )
        context = self
