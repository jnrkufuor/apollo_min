#make sure pip is in your path, for Windows Users
#install torch-sparse, scatter and geometric on their own
import sys
import subprocess
import pkg_resources

#list of required packages to be installed into system
required = {'numpy','pandas','GoogleNews','goose3','IPython','sklearn','networkx','tqdm','matplotlib','seaborn','py3plex','louvain','python-igraph','leidenalg','flair','nltk','squarify','pandas-datareader'} 
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

print(missing)

if "torch" in missing:
     subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch'])

if missing:
    # implement pip as a subprocess:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install',"--user",*missing])
    
from flair.models import SequenceTagger
tagger = SequenceTagger.load('ner')

#add file to where python package export PYTHONPATH=$PYTHONPATH:/home/jay/apollo 
