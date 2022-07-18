"""
Copyright 2021, Warren Woodrich Pettine

This contains code pertaining to neural networks used in "Pettine, W. W., Raman, D. V., Redish, A. D., Murray, J. D.
“Human latent-state generalization through prototype learning with discriminative attention.” December 2021. PsyArXiv

https://psyarxiv.com/ku4fr

This module contains example usage of the code in the context of the main experiments
"""

import numpy as np
import matplotlib.pyplot as plt
import os
#Import specific functions
from simulations.parameters import genParams, worldStateParams, agentParams, nnParams
from simulations.experiments import worldState
from simulations.algorithmicmodels import agent
from utils.utils import testBool, testString, testInt, testFloat, getPool, saveData
from analysis.plotfunctions import removeSpines, plotRDM
from analysis.general import forwardSmooth
from simulations.neuralnetwork import makeRLModel, trainRLNetwork, saveTrainedNetworks

def setExampleGenAx(ax,n_trials):
    """
    Helper function for cleaning up the axes when plotting action generalization data
    :param ax: matplotlib subplot axis object
    :param n_trials: number of trials in the first block
    :return:
    """
    testInt(n_trials)
    ax.axvline(n_trials, linestyle=':', color='k')
    ax.set_xlabel('Trial')
    ax = removeSpines(ax)
    return ax

def setContextGenAx(ax,n_trials_block1,n_trials_block2):
    """
    Helper function for cleaning up the axes when plotting set generalization data
    :param ax: matplotlib subplot axis object
    :param n_trials_block1: number of trials in the first block
    :param n_trials_block2: number of second in the first block
    :return:
    """
    testInt(n_trials_block1)
    testInt(n_trials_block2)
    ax.axvline(n_trials_block1, linestyle=':', color='k')
    ax.axvline(n_trials_block1 + n_trials_block2, linestyle=':', color='k')
    ax.set_xlabel('Trial')
    ax = removeSpines(ax)
    return ax


def exampleGeneralization(state_kernel='prototype',attention_distortion=0,save_fig=True,fig_dir='plots/',
                         fig_name_base='',fig_extension='png'):
    """
    Use this area to play with the action generalization task. It is functionalized for organization purposes.
    :param state_kernel: Kernel for internal states "prototype" or 'exemplar' (default='prototype')
    :param discriminative_component: Boolean of whether or not to include discriminative component (default=True)
    :param save_fig: Boolean for whether to save figure (default=True)
    :param fig_dir: String specifying directory for saving figures (default='plots/')
    :param fig_name_base: Specify a base name for the figure (fefault='')
    :param fig_extension: Specify and extension (default='png')
    :return: A (trained agent class)
    """
    ## CHECK INPUTS
    testFloat(attention_distortion)
    testBool(save_fig)
    testString(fig_dir)
    testString(fig_name_base)
    testString(fig_extension)
    testString(state_kernel)
    if state_kernel == 'exemplar':
        update_state_n_trials = 100 
        blur_states_param_linear = 0.95
        upsilon_candidates = 10 ** -150  

    elif state_kernel == 'prototype':
        update_state_n_trials = 75 
        blur_states_param_linear = 0.98
        upsilon_candidates = 10 ** -20  
    else:
        raise ValueError('state_kernel must be "exemplar" or "prototype"')
    xi_CW = attention_distortion
    ## SET PARAMETERS (play around here)
    upsilon = 0.5 # 0.05 0.1
    beta_value = 50 #15
    beta_state = 50
    n_trials = 800
    eta = 0.05
    n_actions = detailed_reward_prob.shape[1]
    update_state_n_trials = update_state_n_trials
    n_trials_burn_in = 15
    update_exemplar_n_trials = 15
    detailed_reward_prob = np.array([[01.0, 0.00, 0.0, 0.0],  # Cue only for first action
                                     [0.00, 01.0, 0.0, 0.0],  # Cue only for second action
                                     [01.0, 01.0, 0.0, 0.0],  # Cue for both actions
                                     [0.00, 0.00, 1.0, 0.0]])  # No cue for actions
    state_kernel = state_kernel

    ## RUN THE SIMULATION
    P_gen = genParams(n_actions=n_actions,update_state_n_trials=update_state_n_trials, reward_probability=None,
                      detailed_reward_prob=detailed_reward_prob,state_kernel=state_kernel,
                      update_exemplar_n_trials=update_exemplar_n_trials,n_trials_burn_in=n_trials_burn_in)
    A = agent(
        agentParams(P_gen=P_gen, upsilon=upsilon, beta_value=beta_value, eta=eta,xi_CW=xi_CW,beta_state=beta_state,
                    upsilon_candidates=upsilon_candidates,blur_states_param_linear=blur_states_param_linear))
    # Session with stable texture
    P_gen['stable_distractors'] = True
    P = worldStateParams(P_gen=P_gen)
    W_1 = worldState(P=P)
    # Session with random texture
    P_gen['stable_distractors'] = False
    P = worldStateParams(P_gen=P_gen)
    W_2 = worldState(P=P)
    # generate stimuli
    stim_1, ans_1, reward_prob_block1 = W_1.generateExampleGenStimuli(n_trials=n_trials,structured=True,
                                                                      block='categorize')
    stim_2, ans_2, reward_prob_block2 = W_2.generateExampleGenStimuli(n_trials=n_trials, structured=True,
                                                                      block='generalize')
    # Run the session
    A.session(stimuli=stim_1, ans_key=ans_1, reward_mag=W_1.P['reward_mag'], reward_prob=reward_prob_block1)
    A.session(stimuli=stim_2, ans_key=ans_2, reward_mag=W_2.P['reward_mag'], reward_prob=reward_prob_block2)

    ## PLOT THE RESULTS
    n_candidates = np.zeros(len(A.state_candidates))
    for n in range(len(A.state_candidates)):
        n_candidates[n] = len(A.state_candidates[n])

    fig, ax = plt.subplots(3, 2, figsize=(10, 10))
    ax[0, 0].plot(A.getSmoothedPerformance(window_len=40),color='k')
    ax[0, 0].set_title('Learning Curve')
    ax[0, 0].set_ylabel('P(Largest Chosen)')
    ax[0, 0] = setExampleGenAx(ax[0, 0],n_trials)
    ax[0, 1].plot(A.delta_bar, label='Fast')
    ax[0, 1].plot(A.delta_bar_slow, label='Slow')
    ax[0, 1].plot(A.delta_bar - A.delta_bar_slow, label='Effective')
    ax[0, 1].set_title('$\\bar{\delta}$ Components')
    ax[0, 1].set_ylabel('Integrated RPE')
    ax[0, 1] = setExampleGenAx(ax[0, 1], n_trials)
    ax[0, 1].legend()
    ax[1, 0].scatter(np.arange(len(A.activation_record)),-1*np.log(A.activation_record),color='k')
    ax[1, 0].set_title('State Estimation Surprise')
    ax[1, 0].set_ylabel('Surprise')
    ax[1, 0] = setExampleGenAx(ax[1, 0], n_trials)
    ax[1, 1].plot(A.MI)
    ax[1, 1].set_title('Cue Feature Mutual Information')
    ax[1, 1].set_ylabel('Mutual Information')
    ax[1, 1] = setExampleGenAx(ax[1, 1],n_trials)
    ax[2, 0].scatter(np.arange(len(A.trial_state)),n_candidates,color='k')
    ax[2, 0] = setExampleGenAx(ax[2, 0],n_trials)
    ax[2, 0].set_title('State Candidate Set')
    ax[2, 0].set_ylabel('Set Length')
    ax[2, 1].scatter(np.arange(len(A.trial_state)), A.trial_state,color='k')
    ax[2, 1].set_title('Final State Estimation')
    ax[2, 1].set_ylabel('State Label')
    ax[2, 1] = setExampleGenAx(ax[2, 0], n_trials)
    discrim_ttl = f'Attention Distortion = {attention_distortion}'
    discrim_save = f'discrim-{attention_distortion}'
    fig.suptitle(f'Action Generalization, {state_kernel.capitalize()} States\n{discrim_ttl}')
    fig.tight_layout()
    if save_fig:
        f_name = f'{fig_dir}{fig_name_base}exampleGeneralization_{state_kernel}States_{discrim_save}.{fig_extension}'
        fig.savefig(f_name,dpi=300)
    fig.show()

    return A


def contextGeneralization(state_kernel='prototype',context_gen_version=1,p_state_confusion=0,save_fig=True,
                      fig_dir='plots/',fig_name_base='',fig_extension='png',beta_state_confusion=5,upsilon_candidates=None):
    """
    Use this area to play with the set generalization task. It is functionalized for organization purposes.
    :param state_kernel: Kernel for internal states "prototype" or 'exemplar' (default='prototype')
    :param context_gen_version: Version of the experiment. 1 or 2 (default=1)
    :param increased_state_confusion: Boolean for whether to implement increased state confusion (default=False)
    :param save_fig: Boolean for whether to save figure (default=True)
    :param fig_dir: String specifying directory for saving figures (default='plots/')
    :param fig_name_base: Specify a base name for the figure (fefault='')
    :param fig_extension: Specify and extension (default='png')
    :return: A (trained agent class)
    """
    ## CHECK INPUTS
    testInt(context_gen_version)
    testBool(save_fig)
    testFloat(p_state_confusion)
    if p_state_confusion > 1 or p_state_confusion < 0:
        raise ValueError('p_state_confusion must be between 0 and 1')
    testFloat(beta_state_confusion,none_valid=True)
    testFloat(upsilon_candidates,none_valid=True)
    testString(fig_dir)
    testString(fig_name_base)
    testString(fig_extension)
    testString(state_kernel)


    if state_kernel == 'exemplar':
        update_state_n_trials = 100  # 75 (prototype) 100 (exemplar)
        blur_states_param_linear = 0.95
        if upsilon_candidates is None:
            upsilon_candidates = 10 ** -150 
        if beta_state_confusion is None:
            beta_state_confusion = 6
    elif state_kernel == 'prototype':
        update_state_n_trials = 75  # 75 (prototype) 100 (exemplar)
        blur_states_param_linear = 0.98
        if upsilon_candidates is None:
            upsilon_candidates = 10 ** -20 
        if beta_state_confusion is None:
            beta_state_confusion = 16
    else:
        raise ValueError('state_kernel must be "exemplar" or "prototype"')

    ## SET PARAMETERS
    update_state_n_trials = update_state_n_trials # 75 (prototype)
    update_exemplar_n_trials = 15
    detailed_reward_prob = None
    n_trials_burn_in=15
    n_trials_block1 = 800
    n_trials_block2 = 500
    # #Agent parameters
    upsilon = 0.5 # 0.05 0.1
    beta_value = 50  # 15
    beta_state = 50 # 3, 50
    eta = 0.05
    xi_CW = 0
    n_actions = 4
    context_gen_version = 1
    reward_probability=1

    ## CREATE WORLD
    #Create parameters
    P_gen = genParams(n_actions=n_actions,update_state_n_trials=update_state_n_trials, reward_probability=reward_probability,
                      detailed_reward_prob=detailed_reward_prob,state_kernel=state_kernel,update_exemplar_n_trials=update_exemplar_n_trials,
                      n_trials_burn_in=n_trials_burn_in,beta_state_confusion=beta_state_confusion)
    #Create agent
    A = agent(
        agentParams(P_gen=P_gen, upsilon=upsilon, beta_value=beta_value, eta=eta,xi_CW=xi_CW,beta_state=beta_state,
                    upsilon_candidates=upsilon_candidates,blur_states_param_linear=blur_states_param_linear))


    P = worldStateParams(P_gen=P_gen)
    W = worldState(P=P)

    stimuli_set_1, ans_key_set_1, reward_prob_set_1 = W.generateContextGenStimuli(n_trials=n_trials_block1,
                                                                                block='context_1',version=context_gen_version)
    stimuli_set_2, ans_key_set_2, reward_prob_set_2 = W.generateContextGenStimuli(n_trials=n_trials_block1,
                                                                                block='context_2',version=context_gen_version)
    stimuli_gen, ans_key_gen, reward_prob_gen = W.generateContextGenStimuli(n_trials=n_trials_block2,
                                                                        block='generalization',version=context_gen_version)


    # RUN SIMULATION
    A.session(stimuli=stimuli_set_1, ans_key=ans_key_set_1, reward_mag=W.P['reward_mag'], reward_prob=reward_prob_set_1)
    A.session(stimuli=stimuli_set_2, ans_key=ans_key_set_2, reward_mag=W.P['reward_mag'], reward_prob=reward_prob_set_2)
    A.P['p_state_confusion'] = p_state_confusion # Turn on the confusion
    A.session(stimuli=stimuli_gen, ans_key=ans_key_gen, reward_mag=W.P['reward_mag'],
              reward_prob=reward_prob_gen,learn_states=False)

    # PLOT THE RESULTS
    ## PLOT THE RESULTS
    n_candidates = np.zeros(len(A.state_candidates))
    for n in range(len(A.state_candidates)):
        n_candidates[n] = len(A.state_candidates[n])

    fig, ax = plt.subplots(3, 2, figsize=(10, 10))
    ax[0, 0].plot(A.getSmoothedPerformance(window_len=40),color='k')
    ax[0, 0] = setContextGenAx(ax[0, 0],n_trials_block1,n_trials_block2)
    ax[0, 0].set_title('Learning Curve')
    ax[0, 0].set_ylabel('P(Largest Chosen)')
    ax[0, 1].plot(A.delta_bar, label='Fast')
    ax[0, 1].plot(A.delta_bar_slow, label='Slow')
    ax[0, 1].plot(A.delta_bar - A.delta_bar_slow, label='Effective')
    ax[0, 1] = setContextGenAx(ax[0, 1],n_trials_block1,n_trials_block2)
    ax[0, 1].set_title('$\\bar{\delta}$ Components')
    ax[0, 1].set_ylabel('Integrated RPE')
    ax[0, 1].legend()
    ax[1, 0].scatter(np.arange(len(A.activation_record)),-1*np.log(A.activation_record),color='k')
    ax[1, 0].set_title('State Estimation Surprise')
    ax[1, 0].set_ylabel('Surprise')
    ax[1, 0] = setContextGenAx(ax[1, 0],n_trials_block1,n_trials_block2)
    ax[1, 1].plot(A.MI)
    ax[1, 1].set_title('Cue Feature Mutual Information')
    ax[1, 1].set_ylabel('Mutual Information')
    ax[1, 1] = setContextGenAx(ax[1, 1],n_trials_block1,n_trials_block2)
    ax[2, 0].scatter(np.arange(len(A.trial_state)),n_candidates,color='k')
    ax[2, 0] = setContextGenAx(ax[2, 0],n_trials_block1,n_trials_block2)
    ax[2, 0].set_title('State Candidate Set')
    ax[2, 0].set_ylabel('Set Length')
    ax[2, 1].scatter(np.arange(len(A.trial_state)), A.trial_state,color='k')
    ax[2, 1].set_title('Final State Estimation')
    ax[2, 1].set_ylabel('State Label')
    ax[2, 1] = setContextGenAx(ax[2, 1],n_trials_block1,n_trials_block2)
    fig.suptitle(f'Set Generalization Version {context_gen_version}, {state_kernel.capitalize()} States')
    fig.tight_layout()
    if save_fig:
        f_name = f'{fig_dir}{fig_name_base}contextGeneralization{context_gen_version}_{state_kernel}States.{fig_extension}'
        fig.savefig(f_name,dpi=300)
    fig.show()


def contextGeneralizationANN(n_networks=1,context_gen_version=1,save_fig=True,parallel=False,weight_stdev=None,
                      fig_dir='plots/',fig_name_base='',fig_extension='png'):
    # Unit testing
    testInt(context_gen_version)
    testInt(n_networks)
    testBool(save_fig)
    testBool(parallel)
    testString(fig_dir)
    testString(fig_name_base)
    testString(fig_extension)
    if (context_gen_version != 1) and (context_gen_version != 2):
        raise ValueError(f'context_gen_version must be 1 or 2. {context_gen_version} invalid')
    # create the parameters
    P = nnParams(task=f'context_generalizationV{context_gen_version}',calc_rdm=True,weight_stdev=weight_stdev,
                 n_trials_per_bloc=np.array([1000, 1000, 1000]).astype(int),rdm_layers=['common','action_intermediate'])
    # Train the networks
    train = trainRLNetworks(P=P)
    train.run(parallel=parallel)
    # Plot the smoothed choice
    fig, ax = plt.subplots(figsize=(5, 3))
    for i in range(len(train.choice_records)):
        largest_chosen = train.choice_records[i][:, 0] == train.choice_records[i][:, 1]
        y = forwardSmooth(largest_chosen, window_len=10)[:-10]
        ax.plot(np.arange(len(y)), y)
    ax = removeSpines(ax)
    ax.set_xlabel('Trial')
    ax.set_ylabel('P(Largest Chosen)')
    ax.axvline(np.where(train.set_record == 1)[0][0], color='k', linestyle=':')
    ax.axvline(np.where(train.set_record == 2)[0][0], color='k', linestyle=':')
    ax.set_title('Raw Performance')
    if save_fig:
        f_name = f'{fig_dir}{fig_name_base}contextGeneralization{context_gen_version}_ANN.{fig_extension}'
        fig.savefig(f_name,dpi=300)
    fig.show()
    # Plot RDMs
    fig_name = f'{fig_name_base}ann_rdm_layer-hidden_version-{context_gen_version}.{fig_extension}'
    plotRDM(train.rdm[0][0], ttl=f"Hidden Layer", cbar_shrink=.65, fig_dir=fig_dir,
            version=context_gen_version, fig_name=fig_name, save_bool=save_fig)
    plt.show()

    fig_name = f'{fig_name_base}ann_rdm_layer-action_version-{context_gen_version}.{fig_extension}'
    plotRDM(train.rdm[0][1], ttl=f"Action Layer", cbar_shrink=.65, fig_dir=fig_dir,
            version=context_gen_version, fig_name=fig_name, save_bool=save_fig)
    plt.show()


class trainRLNetworks():
    def __init__(self,P,n_networks=None):
        self.P = P
        if n_networks is None:
            n_networks = P['n_networks']
        self.n_networks = n_networks
        self.weight_dicts = []
        self.choice_records = [] # Choice, label
        self.reward_histories = []
        self.inputs = []
        self.set_record = [] #np.zeros(sum(P['n_trials_per_bloc']))
        self.rdm = []
        self.weights = []

    def run(self,parallel=True):
        if parallel:
            print('Running networks in parallel')
            pool = getPool()
            results = pool.map(self.runCall, np.arange(0, self.n_networks))
            pool.close()
            for result in results:
                weights, inputs, labels, action_history, reward_history, set_record, rdm = result
                self.weights.append(weights)
                self.choice_records.append(np.vstack((action_history,labels)).T)
                self.inputs.append(inputs)
                self.reward_histories.append(reward_history)
                self.set_record = set_record
                self.rdm.append(rdm)
        else:
            for n in range(self.n_networks):
                weights, inputs, labels, action_history, reward_history, set_record, rdm = self.runCall()
                self.weights.append(weights)
                self.choice_records.append(np.vstack((action_history,labels)).T)
                self.inputs.append(inputs)
                self.reward_histories.append(reward_history)
                self.set_record = set_record
                self.rdm.append(rdm)
        print('Done training networks')

    def runCall(self,run_num=None):
        """
        Call to allow for parallelization
        :param P:
        :return:
        """
        if (self.P['task'] == 'context_generalizationV1') or (self.P['task'] == 'context_generalizationV2'):
            version = int(self.P['task'][-1])
            P_world = worldStateParams(stimulus_noise_sigma=self.P['noise_sigma'])
            W = worldState(P_world)
            X_0, Y_0, _ = W.generateContextGenStimuli(n_trials=self.P['n_trials_per_bloc'][0], block='context_1',
                                                      version=version)
            X_1, Y_1, _ = W.generateContextGenStimuli(n_trials=self.P['n_trials_per_bloc'][1], block='context_2',
                                                      version=version)
            X_2, Y_2, _ = W.generateContextGenStimuli(n_trials=self.P['n_trials_per_bloc'][2], block='generalization',
                                                      version=version)
            X, Y = [X_0, X_1, X_2], [Y_0, Y_1, Y_2]
            inputs = np.vstack((X_0, X_1, X_2))
            labels = np.concatenate((Y_0, Y_1, Y_2))
            num_actions, num_inputs = len(np.unique(np.concatenate((Y_0, Y_1, Y_2)))), X_0.shape[1]
        elif self.P['task'] == 'basicInstrumental':
            P_world = worldStateParams(stimulus_noise_sigma=self.P['stimulus_noise_sigma'],n_actions=\
                self.P['n_actions'], n_stim=self.P['n_basic_instrumental_stim'],reward_probability=\
                self.P['reward_probability'])
            W = worldState(P_world)
            X, Y, _ = W.genBasicInstrumentalStimuli(n_trials=self.P['n_trials_per_bloc'][0])
            num_actions, num_inputs = self.P['n_actions'], X.shape[1]
            inputs, labels = X, Y
        else:
            raise ValueError('Only context_generalization and basicInstrumental currently implemented')
        model = makeRLModel(num_inputs=num_inputs, num_actions=num_actions,
                            rl_type=self.P['rl_type'], weight_stdev=self.P['weight_stdev'])

        model, action_history, rewards_history, set_record, rdm = \
            trainRLNetwork(model, X, Y, task=self.P['task'], optimizer_type=self.P['optimizer_type'],
                           learning_rate=self.P['learning_rate'],rl_type=self.P['rl_type'],
                           rdm_layers=self.P['rdm_layers'],mask_actions=self.P['mask_actions'],
                           calc_rdm=self.P['calc_rdm'],reward_probability=self.P['reward_probability'])
        weights = model.get_weights()
        return weights, inputs, labels, action_history, rewards_history, set_record, rdm

    def saveResults(self,file_name='savedNetworks.pickle', save_dir=''):
        try:
            saveTrainedNetworks(self.weights, self.inputs, self.choice_records, self.reward_histories, self.set_record,
                                P=self.P, file_name=file_name, save_dir=save_dir, legacy=False)
            if (len(self.rdm) > 0) and (self.rdm[0] is not None):
                if file_name.split('.')[-1] == 'pickle':
                    file_name = file_name[:-7] + '_rdm'
                else:
                    file_name = file_name + '_rdm'
                saveData(os.path.join(save_dir,file_name), self.rdm)
            return 0
        except:
            return 1



if __name__ == '__main__':

    #######################
    ## ALGORITHMIC MODELS
    #######################

    ## EXAMPLE GENERALIZATION
    # state_kernel = 'exemplar' # 'prototype', 'exemplar'
    # attention_distortion = 0 # True, False
    # A = exampleGeneralization(state_kernel=state_kernel, attention_distortion=attention_distortion)


    ## CONTEXT GENERALIZATION
    state_kernel = 'exemplar'  # 'prototype' (ProDAtt), 'exemplar' (ExDAtt)
    context_gen_version = 1 # Version of the task. Must be 1 or 2
    p_state_confusion = 0 # For examining errors during block 3. Must be between 0 and 1
    beta_state_confusion = None # Controls whether errors are reandom, or biased by state. If None, it uses default for ProDAtt and ExDAtt
    upsilon_candidates = None # The threshold for context inclusion. If None, it uses default for ProDAtt and ExDAtt
    A = contextGeneralization(state_kernel=state_kernel, context_gen_version=context_gen_version,save_fig=True,
                      p_state_confusion=p_state_confusion,beta_state_confusion=beta_state_confusion)

    ########################
    ## NEURAL NETWORK MODEL
    ########################
    # n_networks = 1
    # context_gen_version = 1
    # parallel=False
    # weight_stdev = None
    # trained_networks = contextGeneralizationANN(n_networks=n_networks, context_gen_version=context_gen_version,
    #                                          parallel=parallel, weight_stdev=weight_stdev,)

    print('here')
