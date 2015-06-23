#!/usr/bin/python
import sys

sys.path.append('../.')
from scaffoldutils import utils

cfg = utils.load_component_cfgs(".././")
collection_dir = cfg['collection']['paths']['public_data']

print str(collection_dir),

