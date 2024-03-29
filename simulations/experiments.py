"""
Copyright 2021, Warren Woodrich Pettine

This contains code pertaining to neural networks used in "Pettine, W. W., Raman, D. V., Redish, A. D., Murray, J. D.
“Human latent-state generalization through prototype learning with discriminative attention.” December 2021. PsyArXiv

https://psyarxiv.com/ku4fr

This moduele features code that creates the experiments described in the paper
"""

import numpy as np
from itertools import product
from scipy.spatial import distance
#Import specific functions
from simulations.parameters import worldStateParams
from simulations.algorithmicmodels import agent
from analysis.general import figurePrototypesConversion
# from parameters import worldStateParams
# from algorithmicmodels import agent


from utils.utils import testBool, testString, testInt, testArray, testFloat, testDict


class worldState():
    """
    Creates a state of the world
    """
    def __init__(self,P=None):
        testDict(P)
        if P is None:
            P = worldStateParams()
        self.P = P

    def generateExampleGenStimuli_old(self,n_trials=None,structured=False,block='categorize'):
        testInt(n_trials,none_valid=True)
        testBool(structured)
        testString(block)
        if n_trials is None:
            n_trials = self.P['n_trials']  # If not trial count specified, use default
        stim_prototypes, ans_indx = getExampleGenPrototypes(block)
        #Stimulus reward probabilities
        reward_prob_prototypes = self.P['detailed_reward_prob'][ans_indx, :]
        mx = np.max(reward_prob_prototypes, axis=1)
        ans_prototypes = (reward_prob_prototypes == np.tile(mx, (reward_prob_prototypes.shape[1], 1)).T).astype(int)
        # Structured or not
        if structured:  # Make it so all are seen before one is seen again. Randomized within rounds
            n_reps = np.floor(n_trials / len(stim_prototypes)).astype(int)
            n_trials = (len(stim_prototypes) * n_reps).astype(int)
            trial_stim = []
            stim_indx = np.arange(0, stim_prototypes.shape[0])
            for t in range(n_reps):
                np.random.shuffle(stim_indx)
                trial_stim += stim_indx.tolist()
            trial_stim = np.array(trial_stim)
        else:
            trial_stim = np.random.randint(0, len(stim_prototypes), n_trials)
        # Build the returned stimuli and answer keys
        stimuli = np.zeros((n_trials, stim_prototypes.shape[1]))
        ans_key = np.zeros((n_trials, self.P['n_actions']))
        reward_prob = ans_key.copy()
        for a in range(len(stim_prototypes)):
            stimuli[trial_stim == a, :] = stim_prototypes[a, :]
            ans_key[trial_stim == a, :] = ans_prototypes[a, :]
            reward_prob[trial_stim == a, :] = reward_prob_prototypes[a, :]
        # Add noise
        stimuli += np.random.randn(stimuli.shape[0], stimuli.shape[1]) * self.P['stimulus_noise_sigma'] \
                   * self.P['stim_scale']
        return stimuli, ans_key, reward_prob

    def generateExampleGenStimuli(self,n_trials=None,structured=False,block='categorize'):
        testInt(n_trials,none_valid=True)
        testBool(structured)
        testString(block)
        if n_trials is None: n_trials = self.P['n_trials']  # If not trial count specified, use default
        #Spell out the stimuli to make it clear
        if block == 'categorize':
            stim_prototypes = np.array([[1, 0, 0, 1, 0, 0, 1, 0],
                                        [0, 1, 0, 1, 0, 0, 1, 0],
                                        [0, 0, 1, 1, 0, 0, 1, 0],
                                        [0, 0, 1, 0, 1, 0, 1, 0],
                                        [0, 0, 1, 0, 0, 1, 1, 0],
                                        [1, 0, 0, 0, 1, 0, 1, 0],
                                        [1, 0, 0, 0, 0, 1, 1, 0],
                                        [0, 1, 0, 0, 1, 0, 1, 0],
                                        [0, 1, 0, 0, 0, 1, 1, 0]])
            ans_indx = np.array([0, 0, 2, 1, 1, 3, 3, 3, 3])
        elif block == 'generalize':
            stim_prototypes = np.array([[1, 0, 0, 1, 0, 0, 1, 0],
                                        [0, 1, 0, 1, 0, 0, 1, 0],
                                        [0, 0, 1, 1, 0, 0, 1, 0],
                                        [0, 0, 1, 0, 1, 0, 1, 0],
                                        [0, 0, 1, 0, 0, 1, 1, 0],
                                        [1, 0, 0, 0, 1, 0, 1, 0],
                                        [1, 0, 0, 0, 0, 1, 1, 0],
                                        [0, 1, 0, 0, 1, 0, 1, 0],
                                        [0, 1, 0, 0, 0, 1, 1, 0],
                                        [1, 0, 0, 1, 0, 0, 0, 1],
                                        [0, 1, 0, 1, 0, 0, 0, 1],
                                        [0, 0, 1, 1, 0, 0, 0, 1],
                                        [0, 0, 1, 0, 1, 0, 0, 1],
                                        [0, 0, 1, 0, 0, 1, 0, 1],
                                        [1, 0, 0, 0, 1, 0, 0, 1],
                                        [1, 0, 0, 0, 0, 1, 0, 1],
                                        [0, 1, 0, 0, 1, 0, 0, 1],
                                        [0, 1, 0, 0, 0, 1, 0, 1]])
            ans_indx = np.array([0, 0, 2, 1, 1, 3, 3, 3, 3, 0, 0, 2, 1, 1, 3, 3, 3, 3])
        else:
            raise ValueError(f'block value of {block} is invalid. Must be "categorize" or "generalize".')
        #Stimulus reward probabilities
        reward_prob_prototypes = self.P['detailed_reward_prob'][ans_indx, :]
        mx = np.max(reward_prob_prototypes, axis=1)
        ans_prototypes = (reward_prob_prototypes == np.tile(mx, (reward_prob_prototypes.shape[1], 1)).T).astype(int)
        # Structured or not
        if structured:  # Make it so all are seen before one is seen again. Randomized within rounds
            n_reps = np.floor(n_trials / len(stim_prototypes)).astype(int)
            n_trials = (len(stim_prototypes) * n_reps).astype(int)
            trial_stim = []
            stim_indx = np.arange(0, stim_prototypes.shape[0])
            for t in range(n_reps):
                np.random.shuffle(stim_indx)
                trial_stim += stim_indx.tolist()
            trial_stim = np.array(trial_stim)
        else:
            trial_stim = np.random.randint(0, len(stim_prototypes), n_trials)
        # Build the returned stimuli and answer keys
        stimuli = np.zeros((n_trials, stim_prototypes.shape[1]))
        ans_key = np.zeros((n_trials, self.P['n_actions']))
        reward_prob = ans_key.copy()
        for a in range(len(stim_prototypes)):
            stimuli[trial_stim == a, :] = stim_prototypes[a, :]
            ans_key[trial_stim == a, :] = ans_prototypes[a, :]
            reward_prob[trial_stim == a, :] = reward_prob_prototypes[a, :]
        # Add noise
        if self.P['stimulus_noise_sigma'] >0:
            stimuli += np.random.randn(stimuli.shape[0], stimuli.shape[1]) * self.P['stimulus_noise_sigma'] \
                   * self.P['stim_scale']
        # Prior reward as cue

        return stimuli, ans_key, reward_prob

    def generateContextGenStimuli(self, n_trials=None, block='context_1', version=1):
        testInt(n_trials,none_valid=True)
        testString(block)
        testInt(version)
        if n_trials is None:
            n_trials = self.P['n_trials']  # If not trial count specified, use default
        stim_prototypes, ans_prototypes = getContextGenPrototypes(version, block)
        reward_prob_prototypes = np.zeros((stim_prototypes.shape[0], 4))
        reward_prob_prototypes[np.arange(len(ans_prototypes)).astype(int), ans_prototypes.astype(int)] = \
            self.P['reward_probability']
        # Randomly select them for each trial
        trial_stim = np.random.choice(len(ans_prototypes), n_trials).astype(int)
        stimuli, ans_key, reward_prob = stim_prototypes[trial_stim, :].astype(float), \
                                        ans_prototypes[trial_stim].astype(int), reward_prob_prototypes[trial_stim, :]
        stimuli += np.random.randn(stimuli.shape[0], stimuli.shape[1]) * self.P['stimulus_noise_sigma'] \
                   * self.P['stim_scale']
        # Return the lot of it
        return stimuli, ans_key, reward_prob

    def genBasicInstrumentalStimuli(self,n_trials=None):
        """
        Creates stimuli for the basic instrumental learning task. In this case, n_stim refers to the number of stimulus
        attributes.
        :param n_trials: Total number of trials for the task
        :return:
        """
        testInt(n_trials,none_valid=True)
        n_act = self.P['n_actions']
        n_stim = self.P['n_stim']
        #Build the prototypes
        dims = []
        for a in range(n_stim):
            dims.append(np.eye(n_act))
        indx = np.array(list(product(set(range(n_act)), repeat=n_stim)))
        stim_prototypes = np.zeros((len(indx), (n_act) * n_stim))
        ans_key_prototypes = indx[:, 0]
        for s in range(len(indx)):
            for a in range(n_stim):
                stim_prototypes[s, (a * n_act):(a * n_act + n_act)] = dims[a][indx[s, a], :]
        #Apply to each trial
        trial_order = np.random.randint(0,len(ans_key_prototypes),n_trials)
        ans_key = np.random.randint(0, n_act, n_trials)
        stimuli = np.zeros((n_trials,stim_prototypes.shape[1]))
        reward_probs = np.zeros((n_trials, n_act))
        for a in range(len(ans_key_prototypes)):
            stimuli[trial_order==a,:] = stim_prototypes[a,:]
            ans_key[trial_order==a] = ans_key_prototypes[a]
            reward_probs[trial_order == a, ans_key_prototypes[a]] = 1
        #Add the noise
        stimuli += np.random.randn(stimuli.shape[0], stimuli.shape[1]) * self.P['stimulus_noise_sigma']

        return stimuli, ans_key, reward_probs


def genBasicInstrumentalStimuli(n_trials=100,n_actions=2,n_stim=1,noise_sigma=0,reward_prob=1):
    """
    Creates stimuli for the basic instrumental learning task. In this case, n_stim refers to the number of stimulus
    attributes.
    :param n_trials: Total number of trials for the task
    :return:
    """
    #Build the prototypes
    dims = []
    for a in range(n_stim):
        dims.append(np.eye(n_actions))
    indx = np.array(list(product(set(range(n_actions)), repeat=n_stim)))
    stim_prototypes = np.zeros((len(indx), (n_actions) * n_stim))
    ans_key_prototypes = indx[:, 0]
    for s in range(len(indx)):
        for a in range(n_stim):
            stim_prototypes[s, (a * n_actions):(a * n_actions + n_actions)] = dims[a][indx[s, a], :]
    #Apply to each trial
    trial_order = np.random.randint(0,len(ans_key_prototypes),n_trials)
    ans_key = np.random.randint(0, n_actions, n_trials)
    stimuli = np.zeros((n_trials,stim_prototypes.shape[1]))
    reward_probs = np.zeros((n_trials, n_actions))
    for a in range(len(ans_key_prototypes)):
        stimuli[trial_order==a,:] = stim_prototypes[a,:]
        ans_key[trial_order==a] = ans_key_prototypes[a]
        reward_probs[trial_order == a, ans_key_prototypes[a]] = reward_prob
    #Add the noise
    stimuli += np.random.randn(stimuli.shape[0], stimuli.shape[1]) * noise_sigma

    return stimuli, ans_key, reward_probs


def getExampleGenPrototypes(block):
    # Spell out the stimuli to make it clear
    if block == 'categorize':
        stim_prototypes = np.array([[1, 0, 0, 1, 0, 0, 1, 0],
                                    [0, 1, 0, 1, 0, 0, 1, 0],
                                    [0, 0, 1, 1, 0, 0, 1, 0],
                                    [0, 0, 1, 0, 1, 0, 1, 0],
                                    [0, 0, 1, 0, 0, 1, 1, 0],
                                    [1, 0, 0, 0, 1, 0, 1, 0],
                                    [1, 0, 0, 0, 0, 1, 1, 0],
                                    [0, 1, 0, 0, 1, 0, 1, 0],
                                    [0, 1, 0, 0, 0, 1, 1, 0]])
        ans_indx = np.array([0, 0, 2, 1, 1, 3, 3, 3, 3])
    elif block == 'generalize':
        stim_prototypes = np.array([[1, 0, 0, 1, 0, 0, 1, 0],
                                    [0, 1, 0, 1, 0, 0, 1, 0],
                                    [0, 0, 1, 1, 0, 0, 1, 0],
                                    [0, 0, 1, 0, 1, 0, 1, 0],
                                    [0, 0, 1, 0, 0, 1, 1, 0],
                                    [1, 0, 0, 0, 1, 0, 1, 0],
                                    [1, 0, 0, 0, 0, 1, 1, 0],
                                    [0, 1, 0, 0, 1, 0, 1, 0],
                                    [0, 1, 0, 0, 0, 1, 1, 0],
                                    [1, 0, 0, 1, 0, 0, 0, 1],
                                    [0, 1, 0, 1, 0, 0, 0, 1],
                                    [0, 0, 1, 1, 0, 0, 0, 1],
                                    [0, 0, 1, 0, 1, 0, 0, 1],
                                    [0, 0, 1, 0, 0, 1, 0, 1],
                                    [1, 0, 0, 0, 1, 0, 0, 1],
                                    [1, 0, 0, 0, 0, 1, 0, 1],
                                    [0, 1, 0, 0, 1, 0, 0, 1],
                                    [0, 1, 0, 0, 0, 1, 0, 1]])
        ans_indx = np.array([0, 0, 2, 1, 1, 3, 3, 3, 3, 0, 0, 2, 1, 1, 3, 3, 3, 3])
    else:
        raise ValueError(f'block value of {block} is invalid. Must be "categorize" or "generalize".')
    return stim_prototypes, ans_indx


def getContextGenPrototypes(version,block):
    testInt(version)
    testString(block)
    if version == 1:
        if block == 'context_1':
            stim_prototypes = np.array([[1, 0, 1, 0, 0, 1, 0, 1, 0],
                                        [1, 0, 1, 0, 0, 0, 1, 1, 0],
                                        [0, 1, 0, 1, 0, 1, 0, 1, 0],
                                        [0, 1, 0, 1, 0, 0, 1, 1, 0]])
            ans_prototypes = np.array([0, 0, 1, 1])
        elif block == 'context_2':
            stim_prototypes = np.array([[1, 0, 1, 0, 0, 1, 0, 0, 1],
                                        [1, 0, 0, 0, 1, 1, 0, 0, 1],
                                        [0, 1, 1, 0, 0, 1, 0, 0, 1],
                                        [0, 1, 0, 0, 1, 1, 0, 0, 1]])
            ans_prototypes = np.array([2, 2, 3, 3])
        elif block == 'generalization':
            stim_prototypes = np.array([[1, 0, 1, 0, 0, 1, 0, 1, 0],
                                        [1, 0, 1, 0, 0, 0, 1, 1, 0],
                                        [0, 1, 0, 1, 0, 1, 0, 1, 0],
                                        [0, 1, 0, 1, 0, 0, 1, 1, 0],
                                        [1, 0, 1, 0, 0, 1, 0, 0, 1],
                                        [1, 0, 0, 0, 1, 1, 0, 0, 1],
                                        [0, 1, 1, 0, 0, 1, 0, 0, 1],
                                        [0, 1, 0, 0, 1, 1, 0, 0, 1]])
            ans_prototypes = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        else:
            raise ValueError('block must be "set_1", "set_2" or "generalization"')
    elif version == 2:
        if block == 'context_1':
            stim_prototypes = np.array([[1, 0, 0, 1, 0, 1, 0, 1, 0],
                                        [1, 0, 1, 0, 0, 0, 1, 1, 0],
                                        [0, 1, 1, 0, 0, 1, 0, 1, 0],
                                        [0, 1, 0, 1, 0, 1, 0, 1, 0]])
            ans_prototypes = np.array([0, 0, 1, 1])
        elif block == 'context_2':
            stim_prototypes = np.array([[1, 0, 1, 0, 0, 1, 0, 0, 1],
                                        [1, 0, 0, 0, 1, 1, 0, 0, 1],
                                        [0, 1, 1, 0, 0, 0, 1, 0, 1],
                                        [0, 1, 0, 0, 1, 1, 0, 0, 1]])
            ans_prototypes = np.array([2, 2, 3, 3])
        elif block == 'generalization':
            stim_prototypes = np.array([[0, 1, 0, 0, 1, 1, 0, 0, 1],
                                        [0, 1, 0, 1, 0, 1, 0, 1, 0],
                                        [0, 1, 1, 0, 0, 0, 1, 0, 1],
                                        [0, 1, 1, 0, 0, 1, 0, 1, 0],
                                        [1, 0, 0, 0, 1, 1, 0, 0, 1],
                                        [1, 0, 0, 1, 0, 1, 0, 1, 0],
                                        [1, 0, 1, 0, 0, 0, 1, 1, 0],
                                        [1, 0, 1, 0, 0, 1, 0, 0, 1]])
            ans_prototypes = np.array([3, 1, 3, 1, 2, 0, 0, 2])
        else:
            raise ValueError('block must be "set_1", "set_2" or "generalization"')
    else:
        raise ValueError('version must be 1 or 2')
    return stim_prototypes, ans_prototypes


def createGroupedStimStates(A, stimulus_prototypes, ans_key, reward_prototypes, n_trials=76, stimulus_noise_sigma=0.05):
    # Test inputs
    try:
        assert isinstance(A, agent)
    except:
        raise ValueError("The first passed variable 'A' must be of agent() class")
    testArray(stimulus_prototypes)
    testArray(ans_key)
    testArray(reward_prototypes)
    testInt(n_trials)
    testFloat(stimulus_noise_sigma)
    # Get into the function
    n_states_0 = len(A.states)
    ans_unq = np.unique(ans_key)
    for s in range(len(ans_unq)):
        indx = ans_key == ans_unq[s]
        stim_record = np.tile(stimulus_prototypes[indx,:].T,n_trials).T + \
                      np.random.randn(n_trials * sum(indx), stimulus_prototypes.shape[1]) * stimulus_noise_sigma
        choice_record = np.zeros((stim_record.shape[0], 2)) + ans_unq[s]
        activations = np.zeros(stim_record.shape[0])
        reward_probs = np.zeros((stim_record.shape[0], reward_prototypes.shape[1])) + reward_prototypes[s, :].reshape(1, -1)
        # Overal record
        A.state_candidates.append([s] * stim_record.shape[0])
        if A.stimulus_record.shape[0] < 2:
            A.stimulus_record = stim_record
            A.choice_record = choice_record
            A.activation_record = activations
            A.trial_state = np.zeros(stim_record.shape[0]) + s + n_states_0
            A.reward_probs = reward_probs
        else:
            A.stimulus_record = np.vstack((A.stimulus_record, stim_record))
            A.choice_record = np.vstack((A.choice_record, choice_record))
            A.activation_record = np.concatenate((A.activation_record,activations))
            A.trial_state = np.concatenate((A.trial_state, np.zeros(stim_record.shape[0]) + s + n_states_0))
            A.reward_probs = np.vstack((A.reward_probs, reward_probs))
        # Create State
        A.states += A.createNewStates()
        A.states[s+n_states_0].initializeTrialVecLen(trial_vec_len=stim_record.shape[1])
        # Add stim record to state
        A.states[s+n_states_0].stimulus_record = stim_record
        # Choice Array
        A.states[s+n_states_0].choice_record = choice_record
        #Activations
        A.states[s+n_states_0].activation_record = activations
        # Action values
        A.states[s+n_states_0].action_values[ans_key[s]] = 1
        # Compute mu prototype
        A.states[s+n_states_0].updateMu()
        # Compute covariance matrix
        A.states[s+n_states_0].updateCov(update_n_trials=A.P['update_n_trials'])
    A.session_number += 1

    return A


def createIndividualStimStates(A, stimulus_prototypes, ans_key, reward_prototypes, n_trials=76, stimulus_noise_sigma=0.05,
                               pre_train_action_vals=True):
    # Test inputs
    try:
        assert isinstance(A, agent)
    except:
        raise ValueError("The first passed variable 'A' must be of agent() class")
    testArray(stimulus_prototypes)
    testArray(ans_key)
    testArray(reward_prototypes)
    testInt(n_trials)
    testFloat(stimulus_noise_sigma)
    testBool(pre_train_action_vals)
    # Get into function
    n_states_0 = len(A.states)
    for s in range(stimulus_prototypes.shape[0]):
        stim_record = np.tile(stimulus_prototypes[s, :].reshape(-1, 1), n_trials).T + \
                      np.random.randn(n_trials, stimulus_prototypes.shape[1]) * stimulus_noise_sigma
        choice_record = np.zeros((stim_record.shape[0], 2)) + ans_key[s]
        activations = np.zeros(n_trials)
        reward_probs = np.zeros((n_trials, reward_prototypes.shape[1])) + reward_prototypes[s, :].reshape(1, -1)
        # Overal record
        A.state_candidates.append([s] * n_trials)
        if A.stimulus_record.shape[0] < 2:
            A.stimulus_record = stim_record
            A.choice_record = choice_record
            A.activation_record = activations
            A.trial_state = np.zeros(n_trials) + s + n_states_0
            A.reward_probs = reward_probs
        else:
            A.stimulus_record = np.vstack((A.stimulus_record, stim_record))
            A.choice_record = np.vstack((A.choice_record, choice_record))
            A.activation_record = np.concatenate((A.activation_record,activations))
            A.trial_state = np.concatenate((A.trial_state, np.zeros(n_trials) + s + n_states_0))
            A.reward_probs = np.vstack((A.reward_probs, reward_probs))
        # Create State
        A.states += A.createNewStates()
        A.states[s+n_states_0].initializeTrialVecLen(trial_vec_len=stim_record.shape[1])
        # Add stim record to state
        A.states[s+n_states_0].stimulus_record = stim_record
        # Choice Array
        A.states[s+n_states_0].choice_record = choice_record
        #Activations
        A.states[s+n_states_0].activation_record = activations
        # Action values
        if pre_train_action_vals:
            A.states[s+n_states_0].action_values[ans_key[s]] = 1
        # Compute mu prototype
        A.states[s+n_states_0].updateMu()
        # Compute covariance matrix
        A.states[s+n_states_0].updateCov(update_n_trials=A.P['update_n_trials'])
    A.session_number += 1

    return A


def createContextGenPreMadeAgent(P_agent, P_world, n_trials=76, agent_type='individual',context_gen_version=1,
                             pre_train_action_vals=True):
    testDict(P_agent)
    testDict(P_world)
    testInt(n_trials)
    testString(agent_type)
    testInt(context_gen_version)
    testBool(pre_train_action_vals)
    # Set 1
    stim_prototypes_1, ans_prototypes_1 = getContextGenPrototypes(version=context_gen_version, block='context_1')

    reward_prototypes_1 = np.zeros((4, 4))
    for a in range(len(ans_prototypes_1)):
        reward_prototypes_1[a, ans_prototypes_1[a]] = 1
    # Set 2
    stim_prototypes_2, ans_prototypes_2 = getContextGenPrototypes(version=context_gen_version, block='context_2')

    reward_prototypes_2 = np.zeros((4, 4))
    for a in range(len(ans_prototypes_2)):
        reward_prototypes_2[a, ans_prototypes_2[a]] = 1
    # Create agent
    A = agent(P_agent)
    A.initializeTrialVecVars(trial_vec_len=stim_prototypes_1.shape[1])
    # Bake it
    if agent_type == 'individual':
        A = createIndividualStimStates(A, stim_prototypes_1, ans_prototypes_1, reward_prototypes_1, n_trials=n_trials, \
                                       stimulus_noise_sigma=P_world['stimulus_noise_sigma'],
                                       pre_train_action_vals=pre_train_action_vals)

        A = createIndividualStimStates(A, stim_prototypes_2, ans_prototypes_2, reward_prototypes_2, n_trials=n_trials, \
                                       stimulus_noise_sigma=P_world['stimulus_noise_sigma'],
                                       pre_train_action_vals=pre_train_action_vals)
    elif agent_type == 'grouped':
        A = createGroupedStimStates(A, stim_prototypes_1, ans_prototypes_1, reward_prototypes_1, n_trials=n_trials, \
                                    stimulus_noise_sigma=P_world['stimulus_noise_sigma'],
                                    pre_train_action_vals=pre_train_action_vals)

        A = createGroupedStimStates(A, stim_prototypes_2, ans_prototypes_2, reward_prototypes_2, n_trials=n_trials, \
                                    stimulus_noise_sigma=P_world['stimulus_noise_sigma'],
                                    pre_train_action_vals=pre_train_action_vals)
    else:
        raise ValueError('agent_type must be "individual" or "grouped"')

    A.session_record = np.concatenate((np.zeros(n_trials * 4), np.ones(n_trials * 4)))

    return A


def stimDistance(stims, att_indices, att_order, stim_indx=None):
    if stim_indx is None:
        stim_indx = np.arange(stims.shape[0])
    att_differs = np.zeros((stims.shape[0], len(att_order), stims.shape[0]))
    for i in stim_indx:
        stim = stims[i, :]
        for a in range(len(att_order)):
            for s in range(stims.shape[0]):
                att_differs[s, a, i] = (stims[s, att_indices[att_order[a]]] != stim[att_indices[att_order[a]]]).any()

    return att_differs


def representationToRDM(representation):
    rdm = np.zeros((representation.shape[0], representation.shape[0]))

    for i in range(representation.shape[0]):
        for l in range(representation.shape[0]):
            rdm[i, l] = distance.euclidean(representation[i, :], representation[l, :])
    return rdm


def getPredictedRDMs(version=2,att_indices=None):
    if att_indices is None:
        att_indices = {
            'shape': [0, 1],
            'color': [2, 3, 4],
            'size': [5, 6],
            'texture': [7, 8]
        }

    #Get prototypes
    stimulus_prototypes, ans_key = getContextGenPrototypes(version, 'generalization')
    # Put them in order for fitting and the RDM plot
    conversions = figurePrototypesConversion(experimental_set='models', context_gen_version=version)
    indx = [np.where(conversions['figure'] == (i + 1))[0][0] for i in range(8)]
    stimulus_prototypes, ans_key = np.unique(stimulus_prototypes, axis=0)[indx, :], ans_key[indx]

    # features in common for each state
    state_A_common = (stimulus_prototypes[0, :] * stimulus_prototypes[1, :]) > 0
    state_B_common = (stimulus_prototypes[2, :] * stimulus_prototypes[3, :]) > 0
    state_C_common = (stimulus_prototypes[4, :] * stimulus_prototypes[5, :]) > 0
    state_D_common = (stimulus_prototypes[6, :] * stimulus_prototypes[7, :]) > 0

    # Exemplar representations
    representations_E = stimulus_prototypes.copy()
    rdm_exemp = representationToRDM(representations_E)

    # Prototype representations
    representations_P = stimulus_prototypes * 0
    representations_P[[0, 1], :] = stimulus_prototypes[[0, 1], :] * state_A_common
    representations_P[[2, 3], :] = stimulus_prototypes[[2, 3], :] * state_B_common
    representations_P[[4, 5], :] = stimulus_prototypes[[4, 5], :] * state_C_common
    representations_P[[6, 7], :] = stimulus_prototypes[[6, 7], :] * state_D_common
    rdm_proto = representationToRDM(representations_P)

    # Discriminative Boundaries
    set_1_boundaries, set_2_boundaries = state_A_common * 0, state_A_common * 0
    for indices in att_indices.values():
        if np.sum(np.abs(state_A_common[indices]) + np.abs(state_B_common[indices])) > 1 and \
                np.sum(state_A_common[indices] * state_B_common[indices]) < 1:
            set_1_boundaries[indices] = True
        if np.sum(np.abs(state_C_common[indices]) + np.abs(state_D_common[indices])) > 1 and \
                np.sum(state_C_common[indices] * state_D_common[indices]) < 1:
            set_2_boundaries[indices] = True

    # Discriminative 1
    representations_D1 = stimulus_prototypes * (set_1_boundaries * set_2_boundaries)
    rdm_discrim = representationToRDM(representations_D1)

    # Normalize
    rdm_exemp, rdm_proto, rdm_discrim = rdm_exemp / np.max(rdm_exemp), rdm_proto / np.max(
        rdm_proto), rdm_discrim / np.max(rdm_discrim)
    # Make list for returning
    return_list = [rdm_exemp, rdm_proto, rdm_discrim]

    # Discriminative 2
    if version == 1:
        boundaries = (set_1_boundaries + set_2_boundaries) > 0
        representations_D2 = stimulus_prototypes * boundaries
        rdm_discrim2 = representationToRDM(representations_D2)
        rdm_discrim2 = rdm_discrim2/np.max(rdm_discrim2)
        return_list.append(rdm_discrim2)

    return return_list