#!/usr/bin/python
####
# simple SUSTAIN code (psych review version)
# programmed: July 12, 2004
# updated: Oct. 1, 2007
# author: Todd Gureckis (gureckis@nyu.edu)
#
# adapted: July 7th, 2022
# author: Warren Woodrich Pettine
####

####
#  this code is provided to researchers interested in working
#  with SUSTAIN.  i make no claim that this code is the most efficient
#  or clear way to do things.  the primary simulation code used for
#  papers on SUSTAIN is done in C++.  This python version is just to
#  provide interested parties with a simple version of the code for
#  their own investigations.
#
#  the code doesn't provide any file input routines or any other bells and whistles.
#  the model is just simply applied to the shepard, hovland, jenkins
#  (1961) replication by nosofsky, et al. (1993) as reported in
#
#   Love, B.C., Medin, D.L, and Gureckis, T.M (2004) SUSTAIN: A Network Model
#   of Category Learning.  Psychological Review, 11, 309-332.
#
#  note:
#  i use a lot of functional programming so for the uninitiated:
#  map(f, x) where x is a list, will return [f(x[0]), f(x[1]), ...]
#  so map(lambda y: y+2, x) will return [x[0]+2, x[1]+2, x[2]+2, ...]
#  it's a simple error-free way to iterate (as opposed to for loops) and is
#  optimized in many cases by the interpreter to be fast.  the trade off
#  is that it can be conceptually bulky at times because a single line can
#  accomplish so much.
#
####

###########################################################
# import modules
###########################################################

import os, sys
import math
import tempfile
from time import sleep
import string
# from Numeric import *
# from MLab import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from random import random, randint, shuffle
import copy
from simulations.experiments import getContextGenPrototypes, getExampleGenPrototypes
# from experiments import getContextGenPrototypes, getExampleGenPrototypes


###########################################################
# defines
###########################################################
# this is the abstract structure of the six shepard problems
# for simplicity they are just defined to be arrays instead of
# read in from a file.  note that i invert the columns of these
# matricies so that the first value in each column is always 0.
# this arbitrary mapping makes the code simpler.


def initializeValueMap(data):
    s = set([])
    maxNDimValues = np.max(list(map(lambda x: len(s.union(set(x))), np.transpose(data))))
    valueMap = np.identity(maxNDimValues).astype(np.float64)
    return valueMap


###########################################################
# SUSTAIN Class
###########################################################
class SUSTAIN:

    ###########################################################
    # __init__: initializes and reset the network structure
    ###########################################################
    def __init__(self, r, beta, d, threshold, learn, initalphas):

        self.R = r
        self.BETA = beta
        self.D = d
        self.THRESHOLD = threshold
        self.LEARN = learn
        self.LAMBDAS = initalphas

        self.clusters = []
        self.activations = []
        self.connections = []
        self.catunitacts = []
        self.coutputs = []

        self.maxValue = 0.0
        self.minValue = 0.0

    ###########################################################
    # stimulate: present item and env for forward stimulation
    ###########################################################
    def stimulate(self, item, env):

        itemflat = np.resize(item, (1, len(item) * len(item[0])))[0]
        self.maxValue = np.max(itemflat)
        self.minValue = np.min(itemflat)

        # this binary mask will block out queried or missing dims from the calcs
        maskhash = {'k': 1, '?': 0, 'm': 0}
        mask = np.array(list(map(lambda x: maskhash[x], env)), np.float64)

        # compute distances between item and each cluster (Equation #4 in Psych Review)
        self.distances = []
        for cluster in self.clusters:
            self.distances.append(np.array(list(map(lambda x, y: sum(abs(x - y)) / 2.0, item, cluster)), np.float64))

        # compute activation of each cluser  (Equation #5 in Psych. Review)
        lambda2r = np.array(mask * pow(self.LAMBDAS, self.R), np.float64)
        sumlambda2r = np.sum(lambda2r)
        self.activations = []
        for clustdist in self.distances:
            self.activations.append(np.sum(lambda2r * np.exp(-1.0 * self.LAMBDAS * clustdist)) / sumlambda2r)

        # calculate output of most activated cluster after competition (Equation #6 in Psych Review)
        if len(self.activations) > 0:
            a = np.array(list(map(lambda x: pow(x, self.BETA), self.activations)), np.float64)
            b = sum(a)
            self.coutputs = list(map(lambda x, y: (float(x) * float(y)) / float(b), a, self.activations))
            winnerindex = self.coutputs.index(np.max(self.coutputs))
            # passing winner's output over connection weights (Equation #7 in Psych Review)
            self.catunitacts = np.array(float(self.coutputs[winnerindex]) * self.connections[winnerindex], np.float64)
            self.catunitacts = np.resize(self.catunitacts, (len(item), len(item[0])))
        else:
            # set all category unit outputs to zero
            self.catunitacts = np.resize(np.array([0., 0.]), (len(item), len(item[0])))

        # compute output probabilities via luce choice rule (Equation #8 in Psych Review)
        a = list(map(lambda x: np.exp(self.D * x), self.catunitacts))
        b = list(map(lambda x: np.sum(x), a))
        outputprobs = np.array(list(map(lambda x, y: x / y, a, b)))

        # compute probability of making correct response
        outputprobs = np.array(list(map(lambda x, y: x * y, outputprobs, 1.0 - mask)))
        outputprobsflat = np.resize(outputprobs, (1, len(outputprobs) * len(outputprobs[0])))[0]
        probofcorrect = np.max(itemflat * outputprobsflat)

        # generate a response
        if random() > probofcorrect:
            response = False
        else:
            response = True

        return [response, probofcorrect, outputprobs, self.catunitacts, self.activations, self.distances]

    ###########################################################
    # learn: recruits cluster and updates weights
    ###########################################################
    def learn(self, item, env):

        if len(self.clusters) == 0:
            # create new cluster
            self.clusters.append(item)
            self.connections.append(np.array([0.0] * len(item) * len(item[0])))
            self.stimulate(item, env)
            winnerindex = self.activations.index(np.max(self.activations))
            self.adjustcluster(winnerindex, item, env)
        else:
            # is most activated cluster in the correct category? (Equation #10 in Psych Review)
            winnerindex = self.activations.index(max(self.activations))

            # binary "masks" again force learning only on queried dimensions
            maskhash = {'k': 0, '?': 1, 'm': 0}
            mask = np.array(list(map(lambda x: maskhash[x], env)), np.float64)
            maskitem = list(map(lambda x, y: x * y, item, mask))
            maskclus = list(map(lambda x, y: x * y, self.clusters[winnerindex], mask))
            tmpdist = list(map(lambda x, y: sum(abs(x - y)) / 2.0, maskitem, maskclus))

            if (max(self.activations) < self.THRESHOLD) or (sum(tmpdist) != 0.0):  # (Equation #11 in Psych Review)
                # create new cluster
                self.clusters.append(item)
                self.connections.append(np.array([0.0] * len(item) * len(item[0])))
                self.stimulate(item, env)
                winnerindex = self.activations.index(np.max(self.activations))
                self.adjustcluster(winnerindex, item, env)

            else:
                self.adjustcluster(winnerindex, item, env)

        return [self.LAMBDAS, self.connections, self.clusters]

    ###########################################################
    # humbleteach: adjusts winning cluster (Equation #9 in Psych Review)
    ###########################################################
    def humbleteach(self, a, m):
        if (((m > self.maxValue) and (a == self.maxValue)) or
                ((m < self.minValue) and (a == self.minValue))):
            return 0
        else:
            return a - m

    ###########################################################
    # adjustcluster: adjusts winning cluster
    ###########################################################
    def adjustcluster(self, winner, item, env):

        catactsflat = np.resize(self.catunitacts, (1, len(self.catunitacts) * len(self.catunitacts[0])))[0]
        itemflat = np.resize(item, (1, len(item) * len(item[0])))[0]

        # find connection weight errors
        deltas = list(map(lambda x, y: self.humbleteach(x, y), itemflat, catactsflat))

        # mask to only update queried dimensions (Equation #14 in Psych Review)
        maskhash = {'k': 0, '?': 1, 'm': 0}
        mask = np.array(list(map(lambda x: maskhash[x], env)), np.float64)
        deltas = list(map(lambda x, y: x * y, np.resize(deltas, (len(item), len(item[0]))), mask))
        deltas = np.resize(deltas, (1, len(item) * len(item[0])))[0]
        self.connections[winner] += self.LEARN * deltas * self.coutputs[winner]

        # update cluster position (Equation #12 in Psych Review)
        deltas = list(map(lambda x, y: x - y, item, self.clusters[winner]))
        self.clusters[winner] = list(map(lambda x, y: x + (self.LEARN * y), self.clusters[winner], deltas))

        # update lambdas (Equation #13 in Psych Review)
        a = list(map(lambda x, y: x * y, self.distances[winner], self.LAMBDAS))
        b = list(map(lambda x: np.exp(-1.0 * x), a))
        # print map(lambda x,y: self.LEARN*x*(1.0-y), b, a)
        self.LAMBDAS += list(map(lambda x, y: self.LEARN * x * (1.0 - y), b, a))


###########################################################
# END SUSTAIN Class
###########################################################

###########################################################
# mode : really BAD function to compute the mode of a list
# DO NOT REUSE THIS ANYWHERE SINCE IT DOESNT HANDLE TIES
# OR ANYTHING ELSE.  i suggest SciPy
###########################################################
def mode(a):
    counts = {}
    for item in a:
        counts[item] = 0
    for item in a:
        counts[item] += 1
    items = [(v, k) for k, v in counts.items()]
    items.sort()
    return items[-1][1]


###########################################################
# trainModel : function to simulate sustain in this exp.
###########################################################

def trainModel(model, data, env, valueMap=None, nblocks=32, pOmission=0):
    if valueMap is None:
        valueMap = initializeValueMap(data)

    trainingitems = list(map(lambda x: list(map(lambda y: valueMap[int(y)], x)), data))
    clusters_report = []

    lc_full = np.zeros(nblocks)
    nblockscorrect = 0
    quickfinish = False
    for i in range(nblocks):
        randepoch = np.arange(len(trainingitems))  # shuffle items presentations
        shuffle(randepoch)
        lc_across_cats = np.zeros(len(randepoch))
        nitemscorrect = 0

        if nblockscorrect >= 4:  # if reached criterion (4 blocks in a row with 100% acc)
            quickfinish = True;

        for j in randepoch:
            trainingitem = copy.deepcopy(trainingitems[j])
            # if (random() < pOmission): # Randomly omit the feedback (probabalistic reward)
            #     trainingitem[-1] *= 0
            if not quickfinish:
                [res, prob, outunits, outacts, act, dist] = model.stimulate(trainingitem, env)
                if (res == True):
                    nitemscorrect += 1
                #                 lc[problem][i] += (1.0 - prob)
                lc_across_cats[j] = (1.0 - prob)
                if (random() < pOmission):  # Randomly omit the feedback (probabalistic reward)
                    trainingitem[-1] *= 0
                [lambdas, clus, conn] = model.learn(trainingitem, env)
            else:  # if reached criterion, assume 100% for the rest of experiment
                nitemscorrect += 1;

        lc_full[i] = np.mean(lc_across_cats)

        if (nitemscorrect == len(randepoch)) or quickfinish:
            nblockscorrect += 1
        else:
            nblockscorrect = 0
    clusters_report.append(len(model.clusters))
    return model, lc_full


def runBasicInstrumental(n_models=1,version=1,nBlocks=36,learn=0.092327,r=9.01245,pOmission=0.20):
    if version == 1:
        # Single stimulus per state task
        data = np.array([
            [0, 0],
            [1, 1]
        ])
    elif version == 2:
        # # Two stimuli per state task
        data = np.array([
            [0, 1, 0],
            [0, 0, 0],
            [1, 0, 1],
            [1, 1, 1]
        ])
    else:
        raise ValueError(f'Parameter "version" must be 1 or 2')

    env = ['k'] * (data.shape[1] - 1) + ['?']
    valueMap = initializeValueMap(data)

    P = {}
    P['n_models'] = n_models
    P['version'] = version
    P['nBlocks'] = nBlocks
    P['learn'] = learn
    P['r'] = r
    P['pOmission'] = pOmission
    P['data'] = data
    # Run the models
    lc_full_all = []
    models = []
    for n in range(n_models):
        print(f'Training model {n+1} of {n_models}')
        model = SUSTAIN(r=r, beta=1.252233, d=16.924073,
                                        threshold=0.0, learn=learn,
                                        initalphas=np.array([1.0] * len(data[0, :]), np.float64))
        model, lc_full = trainModel(model, data, env, valueMap=valueMap, nblocks=nblocks, pOmission=pOmission)
        lc_full_all.append(lc_full)
        models.append(model)
    lc_full_all = np.array(lc_full_all)

    return models, lc_full_all, P


###########################################################
# main
###########################################################
if __name__ == '__main__':
    # main()

    # stim_prototypes_1, ans_prototypes_1 = getContextGenPrototypes(version=1, block='generalization')
    # data = [np.zeros((stim_prototypes_1.shape[0],stim_prototypes_1.shape[1]+1))]
    # data[0][:,:-1], data[0][:,-1] = stim_prototypes_1, ans_prototypes_1
    # env = ['k']*stim_prototypes_1.shape[1] + ['?']
    # shepard_six(data, env)
    # print('here')

    #############################
    #   BASIC INSTRUMENTAL
    #############################
    n_models = 2
    version = 2
    nblocks = int(np.round(500/(version*2)))

    learn = 0.092327 # 0.7, 0.092327
    r = 9.01245 # 20, 9.01245
    pOmission = 0.20

    models, lc_full_all, P = runBasicInstrumental(n_models=n_models, version=version, nBlocks=nblocks, learn=learn, r=r, pOmission=pOmission)

    df = pd.DataFrame({
        'Mean': np.mean(1-lc_full_all, axis=0),
        'STD': np.std(1-lc_full_all, axis=0),
        'Model': ['SUSTAIN'] * lc_full_all.shape[1],
        'Task Version': [version] * lc_full_all.shape[1],
        'P(Reward)': [1 - pOmission] * lc_full_all.shape[1]
    })

    save_dir = '/Users/wpettine/Dropbox/_Murray/Project/state_creation_RL/Paper/code/data'
    f_name = f'sustain_BasicInstrumental_{version*2}stim_pReward-{1-pOmission}.csv'
    df.to_csv(os.path.join(save_dir,f_name))


    fig_dir = '/Users/wpettine/Dropbox/_Murray/Project/state_creation_RL/Paper/code/repos/MultiLatentStateInstrumentalLearning/plots'
    fig_name = f'sustain_basicInstrumental_nStim-{P["data"].shape[0]/2}_eta-{learn}_f-{r}_pReward-{1 - pOmission}.png'

    fig, ax = plt.subplots()
    ax.plot(np.arange(len(lc_full_all[0]))*P['data'].shape[0],1 - np.mean(lc_full_all, axis=0))
    ax.set_xlabel('Trial')
    # ax.set_xlabel('Cycle')
    ax.set_ylabel('P(Largest Chosen)')
    ax.set_title(f'Basic Insturmental, nStim={P["data"].shape[0]/2}, pReward={1 - pOmission}')
    # ax.set_title('Action Generalization Performance')

    # fig.savefig(os.path.join(fig_dir, fig_name), dpi=300)

    fig.show()

    print('here')


    ######################################
    #  ACTION GENERALIZATION EXPERIMENT
    ######################################

    # stim_prototypes_1, ans_prototypes_1 = getExampleGenPrototypes(block='categorize')
    # data_categorize = np.zeros((stim_prototypes_1.shape[0], stim_prototypes_1.shape[1] + 1))
    # data_categorize[:, :-1], data_categorize[:, -1] = stim_prototypes_1, ans_prototypes_1
    #
    # stim_prototypes_1, ans_prototypes_1 = getExampleGenPrototypes(block='generalize')
    # data_generalization = np.zeros((stim_prototypes_1.shape[0], stim_prototypes_1.shape[1] + 1))
    # data_generalization[:, :-1], data_generalization[:, -1] = stim_prototypes_1, ans_prototypes_1
    #
    # nblocks = 36
    # env = ['k'] * stim_prototypes_1.shape[1] + ['?']
    #
    # learn = 0.092327 # 0.7, 0.092327
    # r = 9.01245 # 20, 9.01245
    #
    # pOmission = 0.2
    #
    # model = SUSTAIN(r=r, beta=1.252233, d=16.924073,
    #                 threshold=0.0, learn=learn,
    #                 initalphas=np.array([1.0] * len(data_categorize[0, :]), np.float64))
    #
    # valueMap = initializeValueMap(data_generalization)
    #
    # model, lc_full_categorize = trainModel(model, data_categorize, env, valueMap=valueMap, nblocks=nblocks, pOmission=pOmission)
    # model, lc_full_generalization = trainModel(model, data_generalization, env, valueMap=valueMap, nblocks=nblocks, pOmission=pOmission)
    #
    #
    # fig_dir = '/Users/wpettine/Dropbox/_Murray/Project/state_creation_RL/Paper/code/repos/MultiLatentStateInstrumentalLearning/plots'
    # fig_name = f'sustain_actGen_eta-{learn}_f-{r}_pReward-{1 - pOmission}.png'
    #
    # fig, ax = plt.subplots()
    # ax.plot(lc_full_categorize, label='Initial learning')
    # ax.plot(lc_full_generalization, label='Generalization')
    # ax.set_xlabel('Cycle')
    # ax.set_ylabel('Error')
    # ax.legend()
    # ax.set_title(f'Action Gen, $\\eta={learn}$, r={r}, pReward={1 - pOmission}')
    # # ax.set_title('Action Generalization Performance')
    #
    # fig.savefig(os.path.join(fig_dir, fig_name), dpi=300)
    #
    # fig.show()
    #
    # print('here')

    ######################################
    ## SET GENERALIZATION EXPERIMENT
    ######################################

    # stim_prototypes_1, ans_prototypes_1 = getContextGenPrototypes(version=1, block='context_1')
    # data_set_1 = np.zeros((stim_prototypes_1.shape[0], stim_prototypes_1.shape[1] + 1))
    # data_set_1[:, :-1], data_set_1[:, -1] = stim_prototypes_1, ans_prototypes_1
    #
    # stim_prototypes_1, ans_prototypes_1 = getContextGenPrototypes(version=1, block='context_2')
    # data_set_2 = np.zeros((stim_prototypes_1.shape[0], stim_prototypes_1.shape[1] + 1))
    # data_set_2[:, :-1], data_set_2[:, -1] = stim_prototypes_1, ans_prototypes_1
    #
    # stim_prototypes_1, ans_prototypes_1 = getContextGenPrototypes(version=1, block='generalization')
    # data_generalization = np.zeros((stim_prototypes_1.shape[0], stim_prototypes_1.shape[1] + 1))
    # data_generalization[:, :-1], data_generalization[:, -1] = stim_prototypes_1, ans_prototypes_1
    #
    # env = ['k'] * stim_prototypes_1.shape[1] + ['?']
    #
    # learn = 0.7 # 0.7, 0.092327
    # r = 20 # 20, 9.01245
    #
    # pOmission = 0.2
    #
    # model = SUSTAIN(r=r, beta=1.252233, d=16.924073,
    #                 threshold=0.0, learn=learn,
    #                 initalphas=np.array([1.0] * len(data_set_1[0, :]), np.float64))
    #
    # valueMap = initializeValueMap(data_generalization)
    #
    # model, lc_full_set_1 = trainModel(model, data_set_1, env, valueMap=valueMap, nblocks=32, pOmission=pOmission)
    # model, lc_full_set_2 = trainModel(model, data_set_2, env, valueMap=valueMap, nblocks=32, pOmission=pOmission)
    # model, lc_full_generalization = trainModel(model, data_generalization, env, valueMap=valueMap, nblocks=32, pOmission=pOmission)
    #
    # fig_dir = '/Users/wpettine/Dropbox/_Murray/Project/state_creation_RL/Paper/code/repos/MultiLatentStateInstrumentalLearning/plots'
    # fig_name = f'sustain_setGen_eta-{learn}_f-{r}_pReward-{1 - pOmission}.png'
    #
    # fig, ax = plt.subplots()
    # ax.plot(lc_full_set_1, label='Set 1')
    # ax.plot(lc_full_set_2, label='Set 2')
    # ax.set_xlabel('Cycle')
    # ax.set_ylabel('Error')
    # ax.set_title(f'Set Gen, $\\eta={learn}$, r={r}, pReward={1 - pOmission}')
    # ax.plot(lc_full_generalization, label='Generalization')
    # ax.legend()
    #
    # fig.savefig(os.path.join(fig_dir, fig_name), dpi=300)
    # fig.show()