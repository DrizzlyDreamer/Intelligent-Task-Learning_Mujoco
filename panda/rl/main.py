""" Launch RL training and evaluation. """
import re
import sys
import signal
import os
import json
import numpy as np
import torch
from mpi4py import MPI
sys.path.append("/home/zhangshidi/GitHub/1_Reproduce/Intelligent-Task-Learning_Mujoco/panda/")
from config import argparser
from trainer import Trainer
from utils.logger import logger



np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)


def run(config):
    """
    Runs Trainer.
    """
    # for parallel workers training (distributed training)
    rank = MPI.COMM_WORLD.Get_rank() # Each process is assigned a rank that is unique within Communicator(group of processes)
    config.rank = rank
    config.is_chef = rank == 0 # Binary
    config.seed = config.seed + rank
    config.num_workers = MPI.COMM_WORLD.Get_size() # total no. of processes is a size of communicator

    if config.is_chef:
        logger.warn('Run a base worker.')
        make_log_files(config)
    else:
        logger.warn('Run worker %d and disable logger.', config.rank)
        import logging
        logger.setLevel(logging.CRITICAL)

    def shutdown(signal, frame):
        logger.warn('Received signal %s: exiting', signal)
        sys.exit(128+signal)

    signal.signal(signal.SIGHUP, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # set global seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    # set the display no. configured with gpu
    os.environ["DISPLAY"] = ":0"
    # use gpu or cpu
    if config.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(config.gpu)
        assert torch.cuda.is_available()
        config.device = torch.device("cuda")
    else:
        config.device = torch.device("cpu")

    # build a trainer
    trainer = Trainer(config)
    if config.is_train:
        trainer.train()
        logger.info("Finish training")
    else:
        trainer.evaluate()
        logger.info("Finish evaluating")


def make_log_files(config):
    """
    Sets up log directories and saves git diff and command line.
    """
    config.run_name = '{}.{}.{}'.format(config.prefix, config.seed, config.suffix)

    config.log_dir = os.path.join(config.log_root_dir, config.run_name)
    logger.info('Create log directory: %s', config.log_dir)
    os.makedirs(config.log_dir, exist_ok=True)

    if config.is_train:
        # log config
        param_path = os.path.join(config.log_dir, 'params.json')
        logger.info('Store parameters in %s', param_path)
        with open(param_path, 'w') as fp:
            json.dump(config.__dict__, fp, indent=4, sort_keys=True)


if __name__ == '__main__':
    args, unparsed = argparser()
    if len(unparsed):
        logger.error('Unparsed argument is detected:\n%s', unparsed)
    else:
        run(args)
