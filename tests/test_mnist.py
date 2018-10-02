import os
import sys
from bootstrap.run import run
from bootstrap.run import main
from bootstrap.lib.options import Options


def reset_options_instance():
    Options._Options__instance = None
    sys.argv = [sys.argv[0]] # reset command line args

def test_run():
    reset_options_instance()
    sys.argv += [
        '-o', 'mnist/options/sgd.yaml',
        '--engine.nb_epochs', '1',
        '--engine.debug', 'True',
        '--exp.dir', 'logs/mnist/test_run'
    ]
    main(run=run)