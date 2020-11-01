
'''
Adds relevant directories to path
'''
import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        #sys.path.insert(0, path)
        sys.path.append(path)

cur_dir = osp.dirname(__file__)


add_path(osp.join(cur_dir, '../gromov')) #Can be obtained from https://github.com/tvayer/FGW
add_path(osp.join(cur_dir, './got')) #Can be obtained from https://github.com/Hermina/GOT. Already added to this repo.

