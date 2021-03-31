from .cars import Cars
from .cub import CUBirds
from .SOP import SOP
from .import utils
from .base import BaseDataset


_type = {
    'cars': Cars,
    'cub': CUBirds,
    'SOP': SOP
}

def load(name, root, mode, scale = 0.8, transform = None):
    if name == 'cub':
        return _type[name](root = root, mode = mode, scale = scale, transform = transform)
    else:
        return _type[name](root = root, mode = mode, transform = transform)
