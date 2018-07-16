import torch
import torch.nn as nn
import bootstrap.lib.utils as utils
from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger
from bootstrap.models.networks.data_parallel import DataParallel

from .net import Net

def factory(engine=None):

    Logger()('Creating mnist network...')

    if Options()['model']['network']['name'] == 'net':
        network = Net()

        if Options()['misc']['cuda'] and len(utils.available_gpu_ids()) >= 2:
            network = DataParallel(network)

    else:
        raise ValueError()

    return network