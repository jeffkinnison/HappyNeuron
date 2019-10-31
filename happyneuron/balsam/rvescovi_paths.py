"""
This module is a temporary workaround due to our inability to use a custom
environment with Jupyter
"""
import sys

# Balsam
sys.path.insert(0,'/soft/datascience/Balsam/0.3.5.1/env/lib/python3.6/site-packages/')
sys.path.insert(0,'/soft/datascience/Balsam/0.3.5.1/')


#FFN
sys.path.insert(0,'/gpfs/mira-home/keceli/ffn/keceli_ffn/')
sys.path.insert(0,'/lus/theta-fs0/projects/connectomics_aesp/keceli/pip_ffn/')
sys.path.insert(0,'/soft/datascience/tensorflow/tf1.13/')


#HPN 
sys.path.insert(0,'/lus/theta-fs0/projects/connectomics_aesp/software/HappyNeuron/')

sys.path.insert(0,'/lus/theta-fs0/projects/connectomics_aesp/software/neuro_env_36/lib/python3.6/site-packages/')
