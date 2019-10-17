import sys
sys.path.insert(0,'/soft/datascience/Balsam/0.3.5.1/env/lib/python3.6/site-packages/')
sys.path.insert(0,'/soft/datascience/Balsam/0.3.5.1/')

import balsam

def get_database_paths(verbose=True):
    """
    Prints the paths for existing balsam databases
    """
    try:
        from balsam.django_config.db_index import refresh_db_index
        databasepaths = refresh_db_index()
    except:
        databasepaths = None
    if verbose:
        if len(databasepaths) > 0:
            print(f'Found {len(databasepaths)} balsam database location')
            for db in databasepaths:
                print(db)
        else:
            print('No balsam database found')
    return databasepaths

def get_active_database(verbose=True):
    """
    Gets the activate database set in environment variable BALSAM_DB_PATH
    Parameters:
    verbose: Boolean, (True): Prints verbose info (False): No print
    Returns
    -------
    str, path for the active database
    """
    try:
        db = os.environ["BALSAM_DB_PATH"]
        if verbose: print(f'Active balsam database path: {db}')
    except:
        if verbose: print('BALSAM_DB_PATH is not set')
        db = None
    return db
    
def add_app(name, executable, description='', envscript='', preprocess='', postprocess='', checkexe=False):
    """
    Adds a new app to the balsam database.
    """
    from balsam.core.models import ApplicationDefinition as App
    import shutil
    
    if checkexe:
        if shutil.which(executable):        
            print('{} is found'.format(executable))
        else:
            print('{} is not found'.format(executable))
            return newapp
    newapp, created = App.objects.get_or_create(name=name)
    newapp.name        = name
    newapp.executable  = executable
    newapp.description = description
    newapp.envscript   = envscript
    newapp.preprocess  = preprocess
    newapp.postprocess = postprocess
    newapp.save()
    if created: print("Created new app")
    else: print("Updated existing app")
    return newapp

def get_apps(verbose=True):
    """
    Returns all apps as a list
    """
    try:
        from balsam.core.models import ApplicationDefinition as App
        apps = App.objects.all()
    except:
        apps = None
    return apps
        
def get_job():
    from balsam.launcher.dag import BalsamJob
    return BalsamJob()

def add_job(
    name, workflow, application, 
    description='', args='', 
    num_nodes=1, ranks_per_node=1,
    cpu_affinity='depth', threads_per_rank=1, threads_per_core=1,
    data={},
    environ_vars= {},
    tf_environ_vars={
        'KMP_BLOCKTIME':'0',
        'KMP_AFFINITY': 'granularity=fine,verbose,compact,1,0',
    }):
    from balsam.launcher.dag import BalsamJob
    #environ_vars.update(tf_environ_vars)
    #environ_vars = ':'.join(f"{k}={v}" for k,v in environ_vars.items())
    job                  = BalsamJob()
    job.name             = name
    job.workflow         = workflow
    job.application      = application
    job.description      = description
    job.args             = args
    job.num_nodes        = num_nodes
    job.ranks_per_node   = ranks_per_node
    job.threads_per_rank = threads_per_rank
    job.threads_per_core = threads_per_core
    job.cpu_affinity   = cpu_affinity
    job.environ_vars   = environ_vars
    job.data           = {}
    job.save()
    
def submit(project='datascience',queue='debug-flat-quad',nodes=1,wall_minutes=30,job_mode='mpi',wf_filter=''):
    """
    Submits a job to the queue with the given parameters.
    Parameters
    ----------
    project: str, name of the project to be charged
    queue: str, queue name, can be: 'default', 'debug-cache-quad', or 'debug-flat-quad'
    nodes: int, Number of nodes, can be any integer from 1 to 4096.
    wall_minutes: int, max wall time in minutes
    job_mode: str, Balsam job mode, can be 'mpi', 'serial'
    wf_filter: str, Selects Balsam jobs that matches the given workflow filter.
    """
    from balsam import setup
    setup()
    from balsam.service import service
    from balsam.core import models
    QueuedLaunch = models.QueuedLaunch
    mylaunch = QueuedLaunch()
    mylaunch.project = project
    mylaunch.queue = queue
    mylaunch.nodes = nodes
    mylaunch.wall_minutes = wall_minutes
    mylaunch.job_mode = job_mode
    mylaunch.wf_filter = wf_filter
    mylaunch.prescheduled_only=False
    mylaunch.save()
    service.submit_qlaunch(mylaunch, verbose=True)