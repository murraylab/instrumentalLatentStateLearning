"""
Copyright 2021, Warren Woodrich Pettine

This contains code pertaining to neural networks used in "Pettine, W. W., Raman, D. V., Redish, A. D., Murray, J. D.
“Human latent-state generalization through prototype learning with discriminative attention.” December 2021. PsyArXiv

https://psyarxiv.com/ku4fr

The code in this module facilitates the functioning of various other modules in the package
"""

import numpy as np
import pickle as pickle
import os
import multiprocessing as mp

PYCHARM_DEBUG=True # Helps if you're running pycharm


def saveData(file_name,save_dict):
    ''''Save method used in the classes'''
    testString(file_name)
    # Add extension, if necessary
    if not file_name.endswith('.pickle'):
        file_name = (file_name + '.pickle')
    # Break data up and save it
    max_bytes = 2 ** 31 - 1
    bytes_out = pickle.dumps(save_dict)
    with open(file_name, 'wb') as f_out:
        for idx in range(0, len(bytes_out), max_bytes):
            f_out.write(bytes_out[idx:idx + max_bytes])


def loadData(file_name):
    '''Load method used in the classes'''
    testString(file_name)
    # Add extension, if necessary
    if not file_name.endswith('.pickle'):
        file_name = (file_name + '.pickle')
    # Check to see if file exists
    if not os.path.isfile(file_name):
        raise ValueError('file ' + file_name + ' does not exist in workspace')
    #Rebuild it
    max_bytes = 2 ** 31 - 1
    bytes_in = bytearray(0)


def listDim(a):
    if not type(a) == list:
        raise ValueError('Must past a list')
    return np.array(a).ndim


def testBool(variable,none_valid=False):
    if none_valid:
        if variable is None:
            return 0
    try:
        assert isinstance(variable, bool)
    except:
        raise ValueError(f'{variable} must be either True or False')
    return 0


def testString(variable,none_valid=False):
    if none_valid:
        if variable is None:
            return 0
    try:
        assert isinstance(variable, str)
    except:
        raise ValueError(f'{variable} must be either string type')
    return 0


def testInt(variable,none_valid=False):
    if none_valid:
        if variable is None:
            return 0
    try:
        assert isinstance(variable, (int,np.integer))
    except:
        raise ValueError(f'{variable} must be an int')
    return 0


def testFloat(variable,none_valid=False):
    if none_valid:
        if variable is None:
            return 0
    try:
        assert isinstance(variable, (np.floating,float,int,np.integer))
    except:
        raise ValueError(f'{variable} must be either float or int')
    return 0


def testArray(variable,none_valid=False):
    if none_valid:
        if variable is None:
            return 0
    try:
        assert (type(variable) == np.ndarray) or ((type(variable) == list) and (listDim(variable)<2))
    except:
        raise ValueError(f'{variable} must be a numpy array')
    return 0


def testDict(variable,none_valid=False):
    if none_valid:
        if variable is None:
            return 0
    try:
        assert type(variable) == dict
    except:
        raise ValueError(f'{variable} must be a dictionary')
    return 0

def softmax(values, beta=2):
    """
    Applies softmax to input vector
    :param action_values: vector of current values
    :return: soft_max_vals: the vector of probabilities
    """
    testArray(values)
    testFloat(beta)
    exp_vals = np.exp(beta * values)
    exp_vals[exp_vals == float("inf")] = 1000000  # Correct for explosions
    soft_max_vals = exp_vals / np.nansum(exp_vals)
    return soft_max_vals

def getPool():
    """ Produces the parallel pool, both on cluster and laptop """
    on_cluster = os.getenv("SLURM_JOB_ID")
    if on_cluster:
        print('Obtaining number of cores using os.getenv')
        num_workers = int(os.getenv("SLURM_CPUS_PER_TASK"))
    else:
        print('Obtaining number of cores using mp.cpu_count')
        num_workers = mp.cpu_count()
    pool = mp.Pool(num_workers)
    return pool
