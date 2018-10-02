from bootstrap.lib.options import Options
from .mnist import MNIST


def factory(engine=None):
    dataset = {}

    if Options()['dataset']['name'] == 'mnist':
        if Options()['dataset']['train_split']:
            dataset['train'] = factory_mnist(Options()['dataset']['train_split'])

        if Options()['dataset']['eval_split']:
            dataset['eval'] = factory_mnist(Options()['dataset']['eval_split'])
    else:
        raise ValueError()

    return dataset


def factory_mnist(split):
    dataset = MNIST(
        dir_data=Options()['dataset']['dir'],
        split=split,
        batch_size=Options()['dataset']['batch_size'],
        nb_threads=Options()['dataset']['nb_threads'],
        pin_memory=Options()['misc']['cuda'])
    return dataset
