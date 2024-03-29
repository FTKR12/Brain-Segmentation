import logging
import sys
import os
from os import path

def setup_logger(name, save_dir, is_Train):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    if is_Train:
        log_name = 'log_train.txt'
    else:
        log_name = 'log_test.txt'
    fh = logging.FileHandler(os.path.join(save_dir, log_name), mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger