import importlib
from os import path as osp

from basicsr.utils import scandir

# automatically scan and import data modules for registry
# scan all the files that end with '_dataset.py' under the data folder
data_folder = osp.dirname(osp.abspath(__file__))
data_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(data_folder) if v.endswith('_dataset.py')]
# import all the data modules
_data_modules = [importlib.import_module(f'sst.data.{file_name}') for file_name in data_filenames]