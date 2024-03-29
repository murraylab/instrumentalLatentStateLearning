"""
Copyright 2021, Warren Woodrich Pettine

This contains code pertaining to neural networks used in "Pettine, W. W., Raman, D. V., Redish, A. D., Murray, J. D.
“Human latent-state generalization through prototype learning with discriminative attention.” December 2021. PsyArXiv

https://psyarxiv.com/ku4fr

The code in this module is central to the functioning of the algorithmic models featured in the paper.
"""

import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from scipy.spatial import distance

#Import specific functions
from utils.utils import saveData, loadData
from simulations.parameters import genParams, worldStateParams, agentParams, agentStateParams
from analysis.general import smooth
from utils.utils import testBool, testString, testInt, testArray, testFloat, testDict, softmax

PYCHARM_DEBUG=True


######################################
# Functions that aid in smooth running
######################################


def calcStimDeviation(mu,trial_vec,blur_states_param_linear=0):
    """
    Calculates difference between observed cue/stimulus and expected cue. It has functionality to blur that difference
    to increase confusion between similar states.
    :param mu: vector of values for expected cue
    :param trial_vec: vector of values for observed cue
    :param blur_states_param_linear: linearly blurs the difference between mu and trial_vec (default=0)
    :return: stim_deviation
    """
    #Check inputs
    testArray(mu)
    testArray(trial_vec)
    testFloat(blur_states_param_linear)
    #Perform operations
    stim_deviation = (1 - blur_states_param_linear) * (trial_vec - mu)
    #Return results
    return stim_deviation


def calcExemplarOtherStateActivation(states,P_agent,indx_target,indx_others,w_k=None,distance_method='activation'):
    if w_k is None:
        w_k = states[0].stimulus_record[0,:]*0+0.5
    activations = []
    for indx in indx_others:
        trials = states[indx].getUpdateTrials(update_state_n_trials=states[indx].P['update_state_n_trials'])
        stimuli_clean = np.abs(np.round(trials))
        stimulus_exemplars, exemplar_counts = np.unique(stimuli_clean,axis=0,return_counts=True)
        exemplar_weights = exemplar_counts / np.sum(exemplar_counts)
        exemplar_activations = exemplar_weights*0
        trials_state = states[indx].stimulus_record
        stimuli_clean_state = np.abs(np.round(trials_state))
        if distance_method == 'euclidean':
            weights = np.diag(states[indx].precision_mat)
            weights = weights / np.max(weights)
        for e in range(stimulus_exemplars.shape[0]):
            indx_exemp = np.where((stimuli_clean_state == stimulus_exemplars[e,:]).all(axis=1))[0][-P_agent['update_exemplar_n_trials']:]
            mu = np.mean(trials_state[indx_exemp,:],axis=0)
            if distance_method == 'activation':
                activation = states[indx_target].calcActivation(P_agent,mu,w_k)
            elif distance_method == 'euclidean':
                #
                stimuli_clean_target_state = np.abs(np.round(states[indx_target].stimulus_record))
                stimulus_exemplars_target, exemplar_counts_target = np.unique(stimuli_clean_target_state, axis=0, return_counts=True)
                exemplar_weights_target = exemplar_counts_target / np.sum(exemplar_counts_target)
                activations_target = np.zeros(stimulus_exemplars_target.shape[0])
                # Loop through each of the target exemplars
                for s in range(stimulus_exemplars_target.shape[0]):
                    indx_exemp_target = (stimulus_exemplars_target[s, :] == stimuli_clean_target_state).all(1)
                    if sum(indx_exemp_target)>1:
                        mu_target = np.mean(states[indx_target].stimulus_record[indx_exemp_target,:],axis=0)
                    else:
                        mu_target = states[indx_target].stimulus_record[indx_exemp_target, :]
                    distance_target = distance.euclidean(mu, mu_target)
                    # distance_target = distance.euclidean(mu * weights, mu_target * weights)
                    if distance_target < 0.00000000000001:
                        activations_target[s] = 100000000
                    else:
                        activations_target[s] = 1 / distance_target
                # activation = np.max(activations_target)
                activation = np.sum(activations_target * exemplar_weights_target)
            if (type(activation) is float) or (type(activation) is int) or (type(activation) is np.nan):
                exemplar_activations[e] = activation
            else:
                exemplar_activations[e] = activation.flatten()[0]
        activations.append(np.sum(exemplar_activations*exemplar_weights))
        # activations.append(np.max(exemplar_activations))
    activations = np.array(activations)
    return activations


def calcPrototypOtherStateActivation(states,P_agent,indx_target,indx_others,w_k=None,distance_method='activation'):
    if w_k is None:
        w_k = states[0].stimulus_record[0,:]*0+0.5
    activations = []
    for indx in indx_others:
        # trial_vec = states[indx].mu
        # activation = states[indx_target].calcActivation(P_agent,trial_vec,w_k)
        trial_vec = states[indx_target].mu
        if distance_method == 'activation':
            activation = states[indx].calcActivation(P_agent, trial_vec, w_k)
        elif distance_method == 'euclidean':
            weights = np.diag(states[indx].precision_mat)
            weights = weights / np.max(weights)
            dist = distance.euclidean(states[indx].mu*weights, trial_vec*weights)
            if dist == 0: # Make sure the values don't explode
                activation = 10000
            else:
                activation = 1/distance.euclidean(states[indx].mu*weights, trial_vec*weights)
        if (type(activation) is float) or (type(activation) is int):
            activations.append(activation)
        else:
            activations.append(activation.flatten()[0])
    activations = np.array(activations)
    return activations


def calcOtherStateActivation(states,P_agent,indx_target,indx_others,w_k=None,distance_method='activation'):
    if P_agent['state_kernel'] == 'prototype':
        activations = calcPrototypOtherStateActivation(states,P_agent,indx_target,indx_others,w_k=w_k,distance_method=distance_method)
    elif P_agent['state_kernel'] == 'exemplar':
        activations = calcExemplarOtherStateActivation(states,P_agent,indx_target,indx_others,w_k=w_k,distance_method=distance_method)
    else:
        raise ValueError('state_kernel must be "prototype" or "exemplar"')
    return activations


def induceStateConfusion(state_indices,states,P_agent,w_k=None,beta=15,distance_method='euclidean',
                         p_state_confusion=0):
        if w_k is None:
            w_k = states[0].stimulus_record[0,:] * 0 + 0.5

        # #FOR DEVELOPMENT, MANUALLY SET A FEW VALUES
        # beta = 15 # 2 exemplar, 15 prototype
        distance_method = 'euclidean'

        # Roll the dice and see if the state indices should be swapped
        if np.random.rand() < p_state_confusion:
            # Confuse state with the most similar
            confusion_indices = []
            for state_index in state_indices:
                indx_others = np.arange(len(state_indices)).astype(int)[
                    np.arange(len(state_indices)).astype(int) != state_index]
                activations_others = calcOtherStateActivation(states, P_agent, state_index, indx_others,
                                                              w_k=w_k,distance_method=distance_method)
                if max(activations_others) == 0:  # If no states are similar, just give back the state index
                    confusion_indices.append(state_index)
                else:
                    posterior_probs_others = np.array(activations_others) / sum(activations_others)
                    # if distance_method == 'activation':
                    posterior_probs_others = softmax(posterior_probs_others, beta=beta)
                    confusion_indices.append(np.random.choice(indx_others, p=posterior_probs_others))
            state_indices = np.array(confusion_indices)
        return state_indices

def getTrialvecLen(P):
    """
    Identifies how long the trial vector is to be. Conserved function
    :param P:
    :return:
    """
    if P['session_type'] == 'cue_tone':
        trial_vec_len = 2 * round(P['n_stim'] / 2) + P['n_distractors'] + P['prior_reward_as_cue']
    elif P['session_type'] == 'igt':
        trial_vec_len = 5
    elif P['session_type'] == 'image':
        if (P['set_type'] == 'shannon') or (P['set_type'] == 'distractFisher'):
            trial_vec_len = P['n_stim']*P['n_actions'] + P['n_distractors'] + P['prior_reward_as_cue']
        elif (P['set_type'] == 'conditionViolation'):
            trial_vec_len = P['n_stim'] * (P['n_actions']-1) + P['n_distractors'] + P['prior_reward_as_cue']
        elif P['set_type'] == 'setGeneralization':
            trial_vec_len = 9 # 6, Really need to automate this one. Lazy to manually set it like this
        else:
            trial_vec_len = P['n_stim'] + P['n_distractors'] + P['prior_reward_as_cue']
    else:
        raise ValueError('Invalid session_type')
    return trial_vec_len


###############################
# Classes used in the algorithm
###############################

class agentState():
    """
    Creates a state as indentified by the agent
    """
    def __init__(self,P={},mu=None,trial_vec_len=None):
        """
        Initialization method
        :param P: parameter dicationary as generated by agentStateParams()
        :param mu: starting prototype vector (default=None)
        :param trial_vec_len: length of trial vector. if none, state initializes it with first call (default=None)
        """
        testDict(P)
        testArray(mu, none_valid=True)
        self.mu = mu
        if len(P) == 0: P = agentStateParams()
        self.P = P
        self.P['trial_vec_len'] = trial_vec_len
        if trial_vec_len is not None:
            testInt(trial_vec_len)
            self.initializeTrialVecLen(trial_vec_len)
        self.n_trials = 0
        self.action_values = np.zeros((self.P['n_actions']))
        self.activation_record = np.array([])
        self.choice_record = np.zeros((1, 2))*np.nan # Chosen option | correct option

    def initializeTrialVecLen(self,trial_vec_len):
        """
        Once the state has been called, this initializes several variables according to the length of the trial vector
        :param trial_vec_len: length of trial vector. Must be integer
        :return:
        """
        testInt(trial_vec_len)
        if self.mu is None:
            self.mu = np.ones(trial_vec_len) * 0.5 * self.P['stim_scale']
        self.cov_mat = np.eye(trial_vec_len) * (self.P['sigma_0'])
        self.stimulus_record = np.zeros((1, trial_vec_len)) * np.nan
        self.precision_mat = np.linalg.inv(self.cov_mat)

    def updateMu(self):
        """Update the prototypical vector for the state based on observations"""
        if np.isnan(self.P['update_state_n_trials']): #Use all trials
            self.mu = np.mean(self.stimulus_record, axis=0)
        else:
            trials= self.getUpdateTrials(update_state_n_trials=self.P['update_state_n_trials'])
        if len(trials)>0: #Need some trials to take their mean
            self.mu = np.mean(trials, axis=0)

    def getUpdateTrials(self,update_state_n_trials=None):
        """
        Obtains the set of trials used to update state varaibles (mu, covariance, etc.)
        :param update_state_n_trials: number of trials to go back. If none, it uses all trials in the state (default=None)
        :return: trials
        """
        # Check inputs
        testInt(update_state_n_trials,none_valid=True)
        correct_indx = np.where(self.choice_record[:, 0] == self.choice_record[:, 1])[0].astype(int)
        if (update_state_n_trials is None) or (np.isnan(update_state_n_trials)):
            update_state_n_trials = len(correct_indx)
        incorrect_indx = np.where(self.choice_record[:, 0] != self.choice_record[:, 1])[0].astype(int)
        n_incorrect = int(np.round(update_state_n_trials * self.P['prop_wrong_update_include']))
        n_correct = int(update_state_n_trials - n_incorrect)
        if n_incorrect > 0:
            trials = self.stimulus_record[np.concatenate((correct_indx[-n_correct:], incorrect_indx[-n_incorrect:])), :]
        else:
            trials = self.stimulus_record[correct_indx[-n_correct:], :]
        return trials

    def updateCov(self,update_state_n_trials=None,polish_cov=True,polish_noise_sigma=0.00000001):
        """
        Update the covariance matrix of the state based on observations.
        :param update_state_n_trials: number of trials to go back. If none, it uses all trials in the state (default=None)
        :param polish_cov: whether to remove noise from the trials, then inject a small amount (default=True)
        :param polish_noise_sigma: If polishing, the amount of noise to inject (default=0.00000001)
        :returns: update of internal covariance and precision matrices
        """
        # Unit testing
        testInt(update_state_n_trials,none_valid=True)
        testBool(polish_cov)
        testFloat(polish_noise_sigma)
        #Make sure the minimum isn't more than what is asked
        if update_state_n_trials > self.P['update_state_n_trials']:
            raise ValueError('covariance will not compute as the minimum number of trials is greater than those obtained from the state')
        trials = self.getUpdateTrials(update_state_n_trials=self.P['update_state_n_trials'])
        if (update_state_n_trials is None) or (len(trials) >= update_state_n_trials):
            if polish_cov:
                trials = np.abs(np.round(trials))
                trials = trials + np.random.randn(trials.shape[0],trials.shape[1]) * polish_noise_sigma
            self.cov_mat = np.cov(trials.transpose()) + \
                          (1 / trials.shape[0]) * np.eye(trials.shape[1]) * (10 ** -4)
            self.precision_mat = np.linalg.inv(self.cov_mat)

    def updateValue(self,action,delta):
        """
        Updates the value of an action based on the delta
        :param action: Index of action to be updated
        :param delta: deviation between received and expected reward
        :return:
        """
        # Check inputs
        testInt(action)
        testFloat(delta)
        self.action_values[action] += self.P['eta']*delta
        pass

    def addRecord(self,vec):
        """
        Adds observation to stimulus record
        :param vec: cue vector for recent observation
        :return:
        """
        # Check input
        testArray(vec)
        if self.P['trial_vec_len'] is None:
            self.initializeTrialVecLen(trial_vec_len=len(vec))
        if len(vec) != self.stimulus_record.shape[1]:
            raise ValueError('Vector size not compatible with record')
        if len(vec.shape)==1:
            vec = vec.reshape(1,len(vec))
        if np.isnan(self.stimulus_record[0,0]):
            self.stimulus_record[0, :] = vec
        else:
            self.stimulus_record = np.append(self.stimulus_record,vec,axis=0)

    def updateActivationRecord(self,A):
        """
        Includes latest activation in activation record
        :param A: Activation value
        :return:
        """
        if (type(A) == float) or (type(A) == int) or (type(A) == np.float64):
            self.activation_record = np.append(self.activation_record, A)
        elif type(A) == np.ndarray:
            self.activation_record = np.concatenate((self.activation_record, A[0]))
        else:
            raise ValueError('Incorrect type passed for activation')

    def updateNtrails(self):
        """
        Updates the internal trial count
        :return:
        """
        self.n_trials += 1

    def updateChoiceRecord(self,vec):
        """
        Updates the choice record with the choice made and the correct choice
        :param vec: vector containing the choice and the correct choice
        :return:
        """
        # Check inputs
        testArray(vec)
        # Do the operations
        if np.isnan(self.choice_record[0,0]):
            self.choice_record[0,:] = vec
        else:
            self.choice_record = np.append(self.choice_record, [vec], axis=0)

    def weightMod(self,weights,param_val,method='norm'):
        """
        Shifts the discribution of weights to alter use of attention
        :param weights: Vector or matrix of weights
        :param param_val: Value used to shift the weights
        :param method: Method used. Must be 'tanh', or 'norm' (default='tanh')
        :return: mod_weights
        """
        # Check inputs
        testArray(weights)
        testFloat(param_val)
        testString(method)
        if method == 'tanh':
            mod_weights = 0.5 + 0.5 * np.tanh((weights - 0.5) / param_val)
        elif method == 'norm':
            if weights.ndim == 1:
                mod_weights = (1-param_val)*weights + param_val
            elif weights.ndim == 2:
                left = (1 - param_val) * weights
                right = param_val * np.eye(len(weights)) * np.sum(weights.flatten()) / len(weights)
                mod_weights = left + right
            else:
                raise ValueError('Weight matrix has more than two dimensions')
        else:
            raise ValueError("Method must be 'tanh', or 'norm'")
        return mod_weights

    def calcActivation(self,P_agent,trial_vec,w_k,w_A=None,delta_bar=0):
        """For a given state, calculate the activation from the trial vector"""
        if self.P['trial_vec_len'] is None:
            self.initializeTrialVecLen(trial_vec_len=len(trial_vec))
        if w_A is None:
            w_A = trial_vec*0+.5
        precision_mat = self.precision_mat
        # If using a prototype kernel, just go with all the examples
        if self.P['state_kernel'] == 'prototype':
            stim_deviation = calcStimDeviation(self.mu,trial_vec,P_agent['blur_states_param_linear'])
            A = self.calcA(P_agent,stim_deviation,precision_mat,w_k,w_A,delta_bar)
        #If using an exemplar kernel, segregate based on stimuli
        elif self.P['state_kernel'] == 'exemplar':
            #Not enough examples for exemplars
            if np.isnan(self.stimulus_record[0,0]) or (self.stimulus_record.shape[0] < self.P['n_trials_burn_in']):
                stim_deviation = calcStimDeviation(self.mu,trial_vec,P_agent['blur_states_param_linear'])
                A = self.calcA(P_agent, stim_deviation, precision_mat, w_k, w_A, delta_bar)
            else:
                trials = self.getUpdateTrials(update_state_n_trials=self.P['update_state_n_trials'])
                stimuli_clean = np.abs(np.round(trials))
                stimulus_exemplars = np.unique(stimuli_clean,axis=0)
                activations = np.zeros(len(stimulus_exemplars))
                n_examples = activations.copy()
                for s in range(stimulus_exemplars.shape[0]):
                    indx = (stimuli_clean == stimulus_exemplars[s,:]).all(axis=1)
                    n_examples[s] = sum(indx)
                    trials_stim = trials[indx,:]
                    if (n_examples[s] > 1):
                        exemplar_mu = np.mean(trials_stim,axis=0)
                    else:
                        exemplar_mu = trials_stim.flatten()
                    if (n_examples[s] > self.P['update_exemplar_n_trials']):
                        exemplar_cov_mat = np.cov(trials_stim.transpose()) + \
                                       (1 / trials_stim.shape[0]) * np.eye(trials_stim.shape[1]) * (10 ** -4)
                    else:
                        exemplar_cov_mat = np.eye(len(trial_vec)) * (self.P['sigma_0'])
                    exemplar_precision_mat = np.linalg.inv(exemplar_cov_mat)
                    stim_deviation = calcStimDeviation(exemplar_mu,trial_vec,P_agent['blur_states_param_linear'])
                    activations[s] = self.calcA(P_agent, stim_deviation, exemplar_precision_mat, w_k, w_A, delta_bar)
                prop_examples = n_examples / sum(n_examples)
                A = sum(prop_examples * activations)
        elif self.P['state_kernel'] == 'gcm_exemplar':
            # Not enough examples for exemplars
            if np.isnan(self.stimulus_record[0, 0]):
                stim_deviation = (1 - P_agent['blur_states_param']) * (trial_vec - self.mu) + P_agent[
                    'blur_states_param'] * \
                                 np.zeros(len(trial_vec))
                A = self.calcA(P_agent, stim_deviation, precision_mat, w_k, w_A, delta_bar)
            else:
                stimuli_clean = np.abs(np.round(self.stimulus_record))
                stimulus_exemplars = np.unique(stimuli_clean, axis=0)
                A = []
                for s in range(stimulus_exemplars.shape[0]):
                    indx = (stimuli_clean == stimulus_exemplars[s,:]).all(axis=1)
                    trials = self.stimulus_record[indx, :]
                    if len(trials) > 1:
                        exemplar_mu = np.mean(trials,axis=0)
                    else:
                        exemplar_mu = trials.flatten()
                    A_ = self.calcGCMDistance(exemplar_mu, trial_vec, discrim_param=P_agent['blur_states_param'])
                    A.append(A_)
                A = sum(A)
        #Update the activation record
        self.updateActivationRecord(A)
        return A

    def calcA(self,P_agent,stim_deviation,precision_mat,w_k,w_A,delta_bar):
        """
        Calculates the activation value
        :param P_agent: Parameter dictionary produced by agentParams
        :param stim_deviation: Deviation of stimulus cues from expected. Calculated using calcStimDeviation
        :param precision_mat: Precision matrix
        :param w_k: Mutual information values for discriminative attention
        :param w_A: Mutual information values for discriminative attention modified by reward history
        :param delta_bar: Integrated reward history
        :return: A (activation)
        """
        #Check inputs
        testDict(P_agent)
        testArray(stim_deviation)
        testArray(precision_mat)
        testArray(w_k)
        testArray(w_A)
        testFloat(delta_bar)
        #Get determinant of covariance
        precision_mat_det = np.linalg.det(np.linalg.inv(precision_mat))
        if P_agent['db_use_method'] == 'w_A':
            Z = w_A * stim_deviation
            D2 = np.dot(Z.reshape(1, len(Z)), np.dot(precision_mat, Z.reshape(len(Z), 1)))
            R = D2
        elif P_agent['db_use_method'] == 'exponential':
            Z = w_k * stim_deviation
            D2 = np.dot(Z.reshape(1, len(Z)), np.dot(precision_mat, Z.reshape(len(Z), 1)))
            R = np.exp(-1 * delta_bar / P_agent['xi_DB']) * D2
        elif P_agent['db_use_method'] == 'linear':
            Z = w_k * stim_deviation
            D2 = np.dot(Z.reshape(1, len(Z)), np.dot(precision_mat, Z.reshape(len(Z), 1)))
            R = ((P_agent['xi_DB'] - delta_bar) / P_agent['xi_DB']) * D2
        elif P_agent['db_use_method'] == 'log':
            Z = w_k * stim_deviation
            D2 = np.dot(Z.reshape(1, len(Z)), np.dot(precision_mat, Z.reshape(len(Z), 1)))
            R = np.log(P_agent['xi_DB'] - delta_bar[-1]) * D2
        else:
            raise ValueError(f'db_use_method {P_agent["db_use_method"]} invalid. must be "w_A", "exponential" or "linear" or "log"')
        #Calculate activation
        A = 1 / (np.sqrt(2 * np.pi * precision_mat_det)) * np.exp((-1 / 2) * R)
        return A

class agent():
    """
    Class of a single agent.
    """
    def __init__(self,P={},P_states={}):
        """
        Initialization method.
        :param P: Parameters for the agent as generated by agentParams()
        :param P_states:  Parameters to be used when creating new states, as generated by agentStateParams()
        """
        #Check inputs
        testDict(P)
        testDict(P_states)
        #Build the agent varaibles
        if len(P) == 0: P = agentParams()
        self.P = P
        if len(P_states) == 0: P_states = agentStateParams(P_gen=P,eta=self.P['eta'],
                                                           precision_distortion=self.P['precision_distortion'])
        self.P['trial_vec_len'] = None
        self.n_trials = 0
        self.choice_record = np.zeros((1, 2))  # Chosen option | correct option
        self.state_P_default = P_states
        self.soft_max_activations = np.zeros((1,self.P['n_actions'])) #Record of softmax activations
        self.state_soft_max_activations = np.array([]) # Record of winning state activation
        self.states = []
        self.n_states = np.array([])
        self.delta_bar = np.array([self.P['delta_bar_0']])
        self.delta_bar_slow = self.delta_bar.copy()
        self.session_number = 0  # Whether the agent been through any trials
        self.session_record = np.array([])
        self.trial_state = np.array([])
        self.activation_record = np.array([])
        self.rewards = np.array([])
        self.reward_probs = np.array([])
        self.state_candidates = [] # List of candidate states for each trial

    def initializeTrialVecVars(self,trial_vec_len=None):
        """
        Initializes the trial vector
        :param trial_vec_len: length of trial vector, must be int (default=None)
        """
        testInt(trial_vec_len,none_valid=True)
        self.P['trial_vec_len'] = trial_vec_len
        self.stimulus_record = np.zeros((1, self.P['trial_vec_len']))
        self.w_A = np.ones((1, self.P['trial_vec_len'])) * .5
        self.MI = np.zeros([1, self.P['trial_vec_len']]) * np.nan

    def session(self,stimuli,ans_key,reward_mag=1,reward_prob=None,learn_states=True):
        """
        Main call for running the simulation
        :param stimuli: A matrix of stimuli where rows are trials and columns are cues
        :param ans_key: Most rewarded action(s) on each trial. Either a vector with indices, or a matrix with boolean
        :param reward_mag: Magnitude of reward (default=1)
        :param reward_prob: Probability of reward. Can be a single value, or matrix for each action each trial (default=None)
        :param learn_states: Boolean of whether to update the state values on each drial (default=True)
        :return:
        """
        # Check inputs
        testArray(stimuli)
        testArray(ans_key)
        testFloat(reward_mag)
        testBool(learn_states)
        # Check if reward probability was manually passed (can be either an array or a float).
        try: testArray(reward_prob,none_valid=True)
        except: testFloat(reward_prob,none_valid=True)
        if reward_prob is None:
            reward_prob = self.P['reward_probability'] # Use a float value.
        # Check if the trial vector length variable has been initialized
        if self.P['trial_vec_len'] is None:
            self.initializeTrialVecVars(trial_vec_len=stimuli.shape[1])
        # Check if there are any potential states in the bank
        if len(self.states) == 0:
            self.states = [agentState(P=self.state_P_default,trial_vec_len=stimuli.shape[1])]  # Initialize with a single state
        # Initialize the probability of state confusion
        p_state_confusion = self.P['p_state_confusion']
        # Loop through each stimulus
        for s in range(stimuli.shape[0]):
            # Record the current session
            self.session_record = np.concatenate((self.session_record, [self.session_number]))
            # Extract the feature vector, and add previous reward as a cue
            trial_vec = stimuli[s,:].copy()
            # Identify the state, calculate the softmax, the action and determine the outcome
            if learn_states: # This updates the state
                state_index, state_probs = self.identifyState(trial_vec=trial_vec,p_state_confusion=p_state_confusion)
            else: # We're not updating the states
                state_index, state_probs = self.identifyStateNoLearn(trial_vec=trial_vec,p_state_confusion=p_state_confusion)
            self.state_soft_max_activations = np.append(self.state_soft_max_activations, state_probs[state_index])
            if isinstance(reward_prob, (np.floating,float,int,np.integer)):
                # Get the action, and the associated soft max values
                action, soft_max_vals = self.getActionSoftmax(state_index)
                #Multiple equivalent answers
                if ans_key.ndim > 1:
                    ans_poss = np.arange(0, ans_key.shape[1])[ans_key[s, :] > 0]
                    ans_match = (action in ans_poss)
                    if len(ans_poss) == 0:
                        ans = np.nan
                    elif ans_match:
                        ans = action
                    else:
                        ans = ans_poss[0]
                else:
                    ans_match = (action == ans_key[s])
                    ans = ans_key[s]
                # Update record of reward
                self.rewards = np.append(self.rewards,ans_match * (np.random.rand() < \
                                  reward_prob)) #Provide reward on a probablistic schedule
                self.reward_probs = np.append(self.reward_probs, reward_prob)
            else: #The reward probabilities are specific for each value and each trial
                #Get actions, append reward values
                action, soft_max_vals = self.getActionSoftmax(state_index, valid_actions=(np.isnan(reward_prob[s,:])<1))
                self.rewards = np.append(self.rewards, int(np.random.rand() < reward_prob[s,action]))
                if ans_key.ndim > 1:
                    ans_poss = np.arange(0, ans_key.shape[1])[ans_key[s, :] > 0]
                    ans_match = (action in ans_poss)
                    if len(ans_poss) == 0:
                        ans = np.nan
                    elif ans_match:
                        ans = action
                    else:
                        ans = ans_poss[0]
                else:
                    ans = ans_key[s]
                if (s == 0) and (self.session_number == 0):
                    self.reward_probs = reward_prob[s,:]
                else:
                    self.reward_probs = np.vstack((self.reward_probs, reward_prob[s,:]))
            #Calculate deviation from expected reward
            delta = self.calcDelta(int(self.rewards[-1]),reward_mag,state_index,action)
            #Update values and delta bar
            self.updateDeltaBar(delta)
            self.states[state_index].updateValue(action,delta)
            previous_reward = self.rewards[-1]*reward_mag
            #Update the record
            self.updateNtrials()
            if (self.session_number == 0) and (s==0):
                self.soft_max_activations[0,:] = soft_max_vals
                self.stimulus_record[0,:] = trial_vec
                self.choice_record[0,:] = [action,ans]
            else:
                self.soft_max_activations = np.append(self.soft_max_activations,[soft_max_vals],axis=0)
                self.stimulus_record = np.append(self.stimulus_record,[trial_vec],axis=0)
                self.choice_record = np.append(self.choice_record,[[action,ans]],axis=0)
            if learn_states:
                self.states[state_index].addRecord(trial_vec)
                self.states[state_index].updateChoiceRecord([action,ans])
                if self.checkUpdateBool():
                    self.states[state_index].updateCov(update_state_n_trials=self.P['n_trials_burn_in'], polish_cov=True,
                                                       polish_noise_sigma=self.P['cov_noise_sigma'])
                    variable_cue_bool = np.var(self.stimulus_record,axis=0) > 0.1
                    self.states[state_index].updateMu()
            self.trial_state = np.concatenate((self.trial_state, [state_index]))
            self.n_states = np.append(self.n_states, len(self.states))
            p_state_confusion = self.updateProbStateConfusion(p_state_confusion=p_state_confusion)
        #Record the session
        self.session_number += 1

    def checkUpdateBool(self,param='n_trials_burn_in'):
        """
        Checks if an update is appropriate.
        :param param: what param to use to check for update
        :return: True/False
        """
        testString(param)
        if (self.P[param] is None) or np.isnan(self.P[param]) or \
            (self.stimulus_record.shape[0] > self.P[param]):
            return True
        else:
            return False


    def updateNtrials(self):
        """
        Update the number of trials that have been experienced.
        :return:
        """
        self.n_trials += 1

    def updateActivationRecord(self,A):
        """
        Add latest activation to the activation record
        :param A: Activation value from that trial
        :return:
        """
        # Check input
        testFloat(A)
        # Update record
        self.activation_record = np.append(self.activation_record, A)

    def calcWK(self,trial_vec):
        """
        Modify the mutual information according to how much the agent utilizes it
        :param trial_vec: Vector of trial stimulus cue values
        :return: w_k
        """
        # Check input
        testArray(trial_vec)
        # Only modify if enough trials have passed.
        if self.stimulus_record.shape[0] >= self.P['n_trials_update_wA']:
            w_k = self.weightMod(self.MI[-1, :], self.P['xi_CW'], method='norm')
        #Otherwise, default to 0.5
        else:
            w_k = trial_vec * 0 + .5
        return w_k

    def stateCandidates(self,trial_vec):
        """
        Identifies indexes of candidate states for a trial
        :param trial_vec: Vector of trial stimulus cue values
        :return: indx_states
        """
        # Check inputs
        testArray(trial_vec)
        # Only do so if the burn in trial count has been reached.
        if self.stimulus_record.shape[0] > self.P['update_state_n_trials']:
            w_k = self.calcWK(trial_vec)
            activations = np.zeros((len(self.states)))
            # Calculate the activations for each state
            for state_index in range(len(self.states)):
                activations[state_index] = self.states[state_index].calcActivation(self.P,trial_vec,w_k,w_A=None,
                                                                                   delta_bar=self.delta_bar[-1])
            bayesian_suprise = -1 * np.log(activations)
            # Determine the state indexes
            if self.P['upsilon_candidates'] is not None:
                indx_states = np.where(bayesian_suprise <= (-np.log(self.P['upsilon_candidates'])))[0]
            else:
                indx_states = np.arange(len(bayesian_suprise)).astype(int)
        else:
            indx_states = np.arange(0,len(self.states))
        return indx_states

    def createNewState(self,trial_vec):
        """
        Go through the steps of birthing a new state
        :param trial_vec: Vector of trial stimulus cue values
        :return: state_index, state_probs
        """
        # Check inputs
        testArray(trial_vec)
        # Get current max index for states
        state_index = len(self.states)-1
        # Only create a new state if the minimum number of update trials has passed.
        if self.stimulus_record.shape[0] > self.P['update_state_n_trials']:
            # Either place the new state at the trial vector, or the center of the space
            if self.P['new_state_mu_prior'] == 'trial_vec':
                new_states = self.createNewStates(mu=trial_vec)
            elif self.P['new_state_mu_prior'] == 'center':
                new_states = self.createNewStates()
            else:
                raise ValueError(f'{self.P["new_state_mu_prior"]} invalid. new_state_mu_prior must be "center" or "trial_vec"')
            # Update variables
            self.states += new_states # Add to the list
            self.delta_bar = np.concatenate((self.delta_bar, [0]))  # Reset the record
            self.delta_bar_slow = np.concatenate((self.delta_bar_slow, [0]))  # Reset the record
            state_index += 1
        state_probs = np.zeros(state_index + 1) * np.nan
        return state_index, state_probs

    def identifyState(self,trial_vec,p_state_confusion=0):
        """
        Indentify the state of the current trial. If none pass activation, it creates a new state
        :param trial_vec: Vector of cue information for a single trial
        :param p_state_confusion: float indicating probability that state will be confused (default=0)
        :return: state_index, state_probs
        """
        # Check inputs
        testArray(trial_vec)
        testFloat(p_state_confusion)
        if (p_state_confusion<0) or (p_state_confusion>1):
            raise ValueError(f'{p_state_confusion} invalid. p_state_confusion must be between 0 and 1')
        # Get indices for the state candidates
        candidate_states_indx = self.stateCandidates(trial_vec)
        # Update variables
        self.state_candidates.append(candidate_states_indx)
        self.updateMI(candidate_states_indx=candidate_states_indx)
        self.updateWA()
        # If no state makes the cut, create a new one. Otherwise, choose between candidates
        if len(candidate_states_indx) < 1:
            state_index, state_probs = self.createNewState(trial_vec=trial_vec)
            self.updateActivationRecord(np.nan)
        else:
            #Get activation values (likelihoods)
            w_k = self.calcWK(trial_vec)
            activations = np.zeros(len(self.states))
            state_indices = np.arange(len(self.states)).astype(int)
            for state_index in candidate_states_indx:
                activations[state_index] = self.states[state_index].calcActivation(self.P, trial_vec, w_k,
                                                                                   self.w_A[-1, :], self.delta_bar[-1])
            #Implemenent state confusion
            if (self.P['p_state_confusion'] is not None) and (p_state_confusion > 0) and (len(state_indices)>1):
                state_indices = induceStateConfusion(state_indices, self.states, self.P, w_k=w_k,
                                        beta=self.P['beta_state_confusion'],p_state_confusion=p_state_confusion)
            #Record maximum activation value
            self.updateActivationRecord(np.max(activations))
            # Covert to Bayesian surprise and figure out the winner!
            bayesian_suprise = -1 * np.log(activations)
            if min(bayesian_suprise) > -np.log(self.P['upsilon']):
                state_index, state_probs = self.createNewState(trial_vec)
            else:
                posterior_probs = activations / sum(activations)
                state_probs = self.softmax(posterior_probs, self.getBetaState())
                state_index = np.random.choice(state_indices, p=state_probs)
        return state_index, state_probs

    def getBetaState(self):
        """
        Updates beta state, allowing it to adapt over the course of a session
        """
        if (self.P['beta_state_min'] is None) or (self.P['beta_state_max'] is None):
            beta_state = self.P['beta_state']
        elif (self.P['n_trials_burn_in'] is not None) and (np.isnan(self.P['n_trials_burn_in'])<1) and \
                (np.sum(np.max(self.session_record) == self.session_record) < self.P['n_trials_burn_in']):
            beta_state = self.P['beta_state_min']
        else:
            delta_bar = self.delta_bar[-1]
            db_factor = min(1, min(0, delta_bar - self.P['xi_DB_low']) / (self.P['xi_DB_high'] - self.P['xi_DB_low']))
            beta_state = (1 - db_factor) * self.P['beta_state_max'] + db_factor * self.P['beta_state_min']
        return beta_state

    def updateProbStateConfusion(self,p_state_confusion=0):
        """
        Allow probability of state confusion adapt over the course of a session
        :param p_state_confusion: probability of state confusion (default=0)
        :return p_state_confusion
        """
        testFloat(p_state_confusion)
        if (p_state_confusion < 0) or (p_state_confusion > 1):
            raise ValueError(f'{p_state_confusion} invalid. p_state_confusion must be between 0 and 1')
        if (self.P['p_state_confusion'] is None) or (self.P['p_state_confusion_end'] is None) or (self.P['p_state_confusion'] == 0):
            p_state_confusion = p_state_confusion
        elif np.abs(p_state_confusion - self.P['p_state_confusion_end']) < 0.001:
            p_state_confusion = self.P['p_state_confusion_end']
        else:
            p_state_confusion = p_state_confusion + self.P['p_state_confusion']*self.P['p_state_confusion_rate_change']
        return p_state_confusion

    def identifyStateNoLearn(self,trial_vec,p_state_confusion=0):
        """
        Indentify the state of the current trial. If none pass activation, it creates a new state. All states are known,
        so the state creation methods have been removed
        :param trial_vec: Vector of cue information for a single trial
        :param p_state_confusion: probability of state confusion (default=0)
        :return: state_index, state_probs
        """
        # Check inputs
        testArray(trial_vec)
        testFloat(p_state_confusion)
        if (p_state_confusion < 0) or (p_state_confusion > 1):
            raise ValueError(f'{p_state_confusion} invalid. p_state_confusion must be between 0 and 1')
        #Get the candidate list and update the values
        candidate_states_indx = self.stateCandidates(trial_vec)
        self.state_candidates.append(candidate_states_indx)
        self.updateMI(candidate_states_indx=candidate_states_indx)
        self.updateWA()
        w_k = self.calcWK(trial_vec)
        # Get activations. Convert to Bayesian and choose a state
        activations = np.zeros(len(self.states))
        state_indices = np.arange(0, len(self.states))
        # Get activations for each state
        for state_index in candidate_states_indx:
            activations[state_index] = self.states[state_index].calcActivation(self.P,trial_vec,w_k,self.w_A[-1,:],self.delta_bar[-1])
        # Probability fi state confusion
        if (self.P['p_state_confusion'] is not None) and (p_state_confusion > 0) and (len(state_indices) > 1):
            state_indices = induceStateConfusion(state_indices, self.states, self.P, w_k=w_k,
                                                 p_state_confusion=p_state_confusion,
                                                 beta=self.P['beta_state_confusion'])

        posterior_probs = activations / sum(activations)
        state_probs = self.softmax(posterior_probs, self.getBetaState())
        state_index = np.random.choice(state_indices, p=state_probs)
        return state_index, state_probs

    def createNewStates(self,mu=np.array([])):
        """
        Initialize new state
        :param mu: Vector for centering the new state
        :return: new_states
        """
        # Check input
        testArray(mu)
        # Create the new state object
        new_states = [agentState(P=self.state_P_default,mu=mu,trial_vec_len=len(mu))]
        return new_states

    def softmax(self,values,beta=None,valid_actions=None):
        """
        Applies softmax to input vector
        :param action_values: vector of current values
        :return: soft_max_vals (the vector of probabilities)
        """
        #Check inputs
        testArray(values)
        testFloat(beta,none_valid=True)
        # Get default values
        if beta is None:
            beta = self.P['beta_value']
        if valid_actions is not None:
            testArray(valid_actions)
            values[(valid_actions < 1)] = np.nan
        #Calculate and return softmax values
        exp_vals = np.exp(beta * values)
        exp_vals[exp_vals == float("inf")] = 1000000 # Correct for explosions
        soft_max_vals = exp_vals / np.nansum(exp_vals)
        return soft_max_vals

    def getActionSoftmax(self,state_index,valid_actions=None):
        """
        Exctracts the values from the given state and calculates the softmax activation. Only chooses from valid actions
        :param state_index: Index of the state being assessed
        :return: action (chosen action of the agent), soft_max_vals (those calculated by the agent)
        """
        # Check inputs
        testInt(state_index)
        testArray(valid_actions,none_valid=True)
        # Calculate the softmax values
        soft_max_vals = self.softmax(self.states[state_index].action_values.copy(), valid_actions=valid_actions)
        # Check if only a certain subset of actions are valid on that trial. Then determine action
        if valid_actions is None:
            action = np.random.choice(np.arange(0, len(soft_max_vals)), p=soft_max_vals)
        else:
            action = np.random.choice(np.arange(0, len(soft_max_vals))[(valid_actions > 0)],
                                      p=soft_max_vals[(valid_actions > 0)])
        return int(action), soft_max_vals

    def calcMI(self,n_trials_back=None,candidate_states_indx=[]):
        """
        Calculates the mutual information for the cues to deterimine which are useful and which are distractors
        :return: MI, mutual information measure of each cue in the vector
        """
        # Check inputs
        testInt(n_trials_back, none_valid=True)
        testArray(candidate_states_indx)
        # determine list of states from which to compute MI
        if len(candidate_states_indx) < 2: # There's no special list of states, so give them all equal value
            MI = np.ones((self.P['trial_vec_len'])) * 0.5
            return MI
        MI = np.ones((self.P['trial_vec_len']))
        # Create list that includes only candidate states.
        states = [self.states[indx] for indx in candidate_states_indx]
        # Loop through each cue
        for c in range(self.P['trial_vec_len']):
            # Look back far enough to cover all the states, but not over the total number of trials
            if n_trials_back is None:
                n_trials_back_0 = self.stimulus_record.shape[0]
            else:
                n_trials_back_0 = min(n_trials_back * len(states), self.stimulus_record.shape[0])
            H_s = np.zeros((len(states))) # Store entropy for each individual state
            all_trials = []
            # Calculate weighted cue entropy for each state
            for s in range(len(states)):
                # Only calculate if the number of required trials have been achieved
                if not self.checkUpdateBool(param='n_trials_burn_in'):
                    H_s[s] = 0
                else:
                    state_trials = np.abs(np.round(states[s].getUpdateTrials(update_state_n_trials=None)))
                    if len(state_trials) < 1:
                        H_s[s] = 0
                        continue
                    # Look back on the specified number of trials
                    if n_trials_back is None:
                        n_trials_back_s = state_trials.shape[0]
                    else:
                        n_trials_back_s = min(n_trials_back, state_trials.shape[0])
                    # Get proporation of trials for which this state was observed
                    p = n_trials_back_s / n_trials_back_0
                    # Calculate the entropy and weight by the observed proportion
                    H_s[s] = p * self.calcEntropy(state_trials[-n_trials_back_s:, c].copy())
                    # Get cue across all trials for calculating overall entropy
                    if len(all_trials) == 0:
                        all_trials = state_trials[:,c].copy()
                    else:
                        all_trials = np.concatenate((all_trials, state_trials[:,c]))
            if (len(all_trials) == 0) or (len(all_trials) == state_trials.shape[0]):
                return MI * 0.5
            # Calculate entropy for all trials
            H_0 = self.calcEntropy(all_trials)
            # Subtract weighted state entropy from overall entropy to determine cue MI
            MI[c] = H_0 - sum(H_s)
        return MI

    def calcEntropy(self,vec):
        """
        Calculates the entropy of an input vector. The entropy is measured using log2
        :param vec: Input vector
        :return: entropy
        """
        # Check inputs
        testArray(vec)
        # Trim values to fit distribution
        vec[vec<0], vec[vec>self.P['stim_scale']] = 10**-15, 0.9999999*self.P['stim_scale']
        if sum(np.isnan(vec))>0:
            raise ValueError(f'Found a nan in the vec {vec}')
        # Convert vector to a distribution and calculate entropy across it.
        vec_h, _ = (np.histogram(vec, np.linspace(0, self.P['stim_scale'], 200)))
        vec_h = vec_h / len(vec) + 10 ** -15
        entropy = -1 * sum( vec_h * np.log2(vec_h) )
        return entropy

    def calcDelta(self,result,reward_mag,state_index,action):
        """
        Calculate the delta (error) on the trial
        :param result: Result of the previous trial (0/1)
        :param reward_mag: Magnitude for the reward for that world
        :param state_index: Index of the state used in the decision
        :param action: Action chosen by the agent
        :return: delta (the RPE)
        """
        # Check inputs
        testInt(result)
        testFloat(reward_mag)
        testInt(state_index)
        testInt(action)
        # Subtract observed reward from expected reward.
        delta = result * reward_mag - \
                    self.states[state_index].action_values[int(action)]
        return delta

    def updateDeltaBar(self,delta):
        """
        Updates value of delta bar variables. Of note, "delta_bar" refers to the fast integration
        :param delta: Deviation from expexted reward
        :return:
        """
        """Calculate update of delta bar"""
        # test input
        testFloat(delta)
        # Get the new integrated value
        delta_bar_new = self.P['xi_0']*self.delta_bar[-1] + self.P['xi_1']* min(delta,0)
        self.delta_bar = np.concatenate((self.delta_bar,[delta_bar_new]))
        # Incorporate slow timescale
        delta_bar_new = self.P['xi_0_slow'] * self.delta_bar_slow[-1] + self.P['xi_1_slow'] * min(delta, 0)
        self.delta_bar_slow = np.concatenate((self.delta_bar_slow, [delta_bar_new]))

    def weightMod(self,weights,param_val=0,method='tanh'):
        """
        Modifies attention weights according to a pram value and a method
        :param weights: Matrix or vector of weights
        :param param_val: value of parameter (default=0)
        :param method: Method used to modify weights, "tahh" or "norm" (default='tanh')
        :return: mod_weights
        """
        """Shifts the distribution of weights"""
        #Check inputs
        testArray(weights)
        testFloat(param_val)
        testString(method)
        # Determine method and modify the weights
        if method == 'tanh':
            mod_weights = 0.5 + 0.5 * np.tanh((weights - 0.5) / param_val)
        elif method == 'norm':
            if weights.ndim == 1:
                mod_weights = (1-param_val)*weights + param_val
            elif weights.ndim == 2:
                left = (1 - param_val) * weights
                right = param_val * np.eye(len(weights)) * np.sum(weights.flatten()) / len(weights)
                mod_weights = left + right
            else:
                raise ValueError('Weight matrix has more than two dimensions')
        return mod_weights

    def calcWA(self,xi_CW=None):
        """
        Modulates the mutual information according to the integrated reward history (delta_bar)
        :param xi_CW: Parameter dictating the strength of discriminative attention, 0-1
        :return: w_A
        """
        testFloat(xi_CW,none_valid=True)
        if xi_CW is None:
            xi_CW = self.P['xi_CW']
        # Modulate MI by the weight to the variable
        w_k = self.weightMod(self.MI[-1, :], xi_CW, method='norm')
        delta_bar = min(0, self.delta_bar[-1] - self.delta_bar_slow[-1])
        if self.P['w_A_update_method'] == 'tanh':
            w_A = ((-np.tanh(delta_bar / self.P['xi_DB'])) + (1 + np.tanh(delta_bar / self.P['xi_DB'])) * w_k).reshape(1,
                                                                                                self.w_A.shape[1])
        elif self.P['w_A_update_method'] == 'linear':
            db_factor = min(1, min(0, delta_bar - self.P['xi_DB_low']) / (self.P['xi_DB_high'] - self.P['xi_DB_low']))
            w_A = ((1 - db_factor) * w_k + db_factor).reshape(1, self.w_A.shape[1])
        return w_A

    def updateWA(self,xi_CW=None):
        """
        Calculate and update new value for w_A. Much of the functionality is in the calcWA function.
        :param xi_CW: Degree to which discriminative attention is incorporated.
        :return:
        """
        """"""
        testFloat(xi_CW,none_valid=True)
        if xi_CW is None:
            xi_CW = self.P['xi_CW']
        if self.stimulus_record.shape[0]>=self.P['n_trials_update_wA']: #Only change it if threshold has passed
            w_A  = self.calcWA(xi_CW=xi_CW)
            self.w_A = np.concatenate((self.w_A, w_A),axis=0)
        else:
            self.w_A = np.concatenate((self.w_A,self.w_A[-1,:].reshape(1,len(self.w_A[-1,:]))),axis=0)

    def updateMI(self,candidate_states_indx=[]):
        """
        Calculates and updates the MI value
        :param candidate_states_indx: List of candidate states
        :return:
        """
        # Test input
        testArray(candidate_states_indx)
        # Calculate MI
        MI = self.calcMI(n_trials_back=None,candidate_states_indx=candidate_states_indx)
        if np.isnan(self.MI[0,0]):
            self.MI[0,:] = MI
        else:
            self.MI = np.concatenate((self.MI, MI.reshape(1, len(MI))), axis=0)

    def getSmoothedPerformance(self,window_len=11,I=None):
        """
        Smooths the performance data create a vector. The "smooth" function is custom.
        :param window_len: smoothing window size
        :return: performance (smoothed vector)
        """
        testArray(I,none_valid=True)
        if I is None:
            I = np.ones(self.choice_record.shape[0])>0
        else:
            try:
                assert len(I) == self.choice_record.shape[0]
            except:
                raise ValueError('mismatch in vector lengths')
        performance = smooth(self.choice_record[I, 0] == self.choice_record[I, 1],window_len=window_len)
        return performance

    def getMeanPerformance(self):
        """
        Obtain mean performance over the trial
        :return: performance (mean performance level)
        """
        performance = np.mean(self.choice_record[:,0] == self.choice_record[:,1])
        return performance

    def saveAgent(self,file_name="agent_data.pickle"):
        # Check input
        testString(file_name)
        # Create dictionary to be saved
        save_dict = {'P':                           self.P,
                     'choice_record':               self.choice_record,
                     'state_P_default':             self.state_P_default,
                     'soft_max_activations':        self.soft_max_activations,
                     'state_soft_max_activations':  self.state_soft_max_activations,
                     'states':                      self.states,
                     'n_states':                    self.n_states,
                     'state_candidates':            self.state_candidates,
                     'stimulus_record':             self.stimulus_record,
                     'delta_bar':                   self.delta_bar,
                     'delta_bar_slow':              self.delta_bar_slow,
                     'w_A':                         self.w_A,
                     'session_number':              self.session_number,
                     'session_record':              self.session_record,
                     'trial_state':                 self.trial_state,
                     'activation_record':           self.activation_record,
                     'MI':                          self.MI,
                     'reward_probs':                self.reward_probs,
                     'rewards':                     self.rewards}
        saveData(file_name=file_name, save_dict=save_dict)

    def loadAgent(self,file_name="agent_data.pickle"):
        # Check input
        testString(file_name)
        # Load it
        save_dict = loadData(file_name=file_name)
        self.P =                                    save_dict['P']
        self.choice_record =                        save_dict['choice_record']
        self.state_P_default =                      save_dict['state_P_default']
        self.soft_max_activations =                 save_dict['soft_max_activations']
        self.state_soft_max_activations =           save_dict['state_soft_max_activations']
        self.states =                               save_dict['states']
        self.n_states =                             save_dict['n_states']
        self.state_candidates =                     save_dict['state_candidates']
        self.stimulus_record =                      save_dict['stimulus_record']
        self.delta_bar =                            save_dict['delta_bar']
        self.delta_bar_slow =                       save_dict['delta_bar_slow']
        self.w_A =                                  save_dict['w_A']
        self.session_number =                       save_dict['session_number']
        self.session_record =                       save_dict['session_record']
        self.reward_probs =                         save_dict['reward_probs']
        self.rewards =                              save_dict['rewards']
        self.trial_state =                          save_dict['trial_state']
        self.activation_record =                    save_dict['activation_record']
        self.MI =                                   save_dict['MI']
        del save_dict


if __name__ == '__main__':
    ## Set parameters
    upsilon = 1.0  # 0.1
    upsilon_candidates = 10 ** -10  # 10**-10
    reward_probability = 1
    detailed_reward_prob = None
    beta_value = 15  # 15
    beta_state = 50
    n_trials = 800
    eta = 0.05
    stim_scale = 1
    n_stim = 4
    n_actions = 2
    stimulus_noise_sigma = 0.05  # 0.02, 0.05
    update_state_n_trials = 75 # 100 if exemplar', 75 if prototype
    xi_0 = 0.99
    xi_shift_1 = -0.009
    xi_shift_2 = 0.004
    xi_DB_low = -4  # -2, -4
    xi_DB_high = -12
    w_A_update_method = 'linear'
    xi_CW = 0
    state_kernel = 'prototype' # 'exemplar', 'prototype'

    ## Create parameters dictionaries
    P_gen = genParams(stim_scale=stim_scale, n_stim=n_stim, n_actions=n_actions, update_state_n_trials=update_state_n_trials,
                      reward_probability=reward_probability, detailed_reward_prob=detailed_reward_prob,
                      state_kernel=state_kernel)
    P_agent = agentParams(P_gen=P_gen, upsilon=upsilon, beta_value=beta_value, eta=eta, xi_0=xi_0,xi_CW=xi_CW,
                          xi_shift_1=xi_shift_1, xi_shift_2=xi_shift_2, xi_DB_low=xi_DB_low, xi_DB_high=xi_DB_high,
                          w_A_update_method=w_A_update_method, xi_DB=5, upsilon_candidates=upsilon_candidates,
                          new_state_mu_prior="trial_vec")

    ## Create the basic instrumental stimuli (adapted from experiments.genBasicInstrumentalStimuli)
    # Build the examples
    dims = []
    for a in range(n_stim):
        dims.append(np.eye(n_actions))
    indx = np.array(list(product(set(range(n_actions)), repeat=n_stim)))
    stim_prototypes = np.zeros((len(indx), (n_actions) * n_stim))
    ans_key_prototypes = indx[:, 0]
    for s in range(len(indx)):
        for a in range(n_stim):
            stim_prototypes[s, (a * n_actions):(a * n_actions + n_actions)] = dims[a][indx[s, a], :]
    # Apply to each trial
    trial_order = np.random.randint(0, len(ans_key_prototypes), n_trials)
    ans_key = np.random.randint(0, n_actions, n_trials)
    stimuli = np.zeros((n_trials, stim_prototypes.shape[1]))
    reward_probs = np.zeros((n_trials, n_actions))
    for a in range(len(ans_key_prototypes)):
        stimuli[trial_order == a, :] = stim_prototypes[a, :]
        ans_key[trial_order == a] = ans_key_prototypes[a]
        reward_probs[trial_order == a, ans_key_prototypes[a]] = 1
    # Add the noise
    stimuli += np.random.randn(stimuli.shape[0], stimuli.shape[1]) * stimulus_noise_sigma

    ## Create and agent and run the session
    A = agent(P_agent)
    A.session(stimuli=stimuli, ans_key=ans_key, reward_mag=1, reward_prob=1)

    ## Plot the smoothed performance
    plt.plot(A.getSmoothedPerformance())
    plt.show()

    print('here')