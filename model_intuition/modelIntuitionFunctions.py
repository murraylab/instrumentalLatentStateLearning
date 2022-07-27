
import numpy as np
from numpy.linalg import multi_dot
from scipy.spatial import distance
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
# from utils.utils import testBool, testString, testInt, testArray, testFloat, testDict


def createPrototypeState(state_examples,exemplar_props=np.array([0.5,0.5]),n_examples=100,noise_sigma=10**-8):
    if type(state_examples) != np.ndarray:
        raise ValueError('state_examples must be a numpy array')
    if not (state_examples % state_examples.astype(int) == 0).all():
        raise ValueError('values for cues and state exemplars must be of type int or int64')
    if type(exemplar_props) != np.ndarray:
        raise ValueError('exemplar_props must be a numpy array')
    if sum(exemplar_props) != 1:
        warnings.warn('The values in the vector do not sum to one. The values will be normalized.')
        exemplar_props = exemplar_props/sum(exemplar_props)
    n_examples_per_state = np.round(exemplar_props*n_examples).astype(int)
    examples = None
    for i in range(len(exemplar_props)):
        if n_examples_per_state[i] == 0:
            continue
        if examples is None:
            examples = np.tile(state_examples[i,:],(int(n_examples_per_state[i]),1))
        else:
            examples = np.vstack((examples,np.tile(state_examples[i,:],(int(n_examples_per_state[i]),1))))
    state_prototype_mu = np.mean(examples,axis=0)
    precision = calcPrecision(examples,noise_sigma=noise_sigma)
    state = {
        'mu': state_prototype_mu.astype(int),
        'precision': precision,
        'examples': examples.astype(int)
    }
    return state


def calcPrecision(examples,noise_sigma=10**-8):
    examples = examples + np.random.randn(examples.shape[0],examples.shape[1])*noise_sigma
    precision = np.linalg.inv(np.cov(examples,rowvar=False) +  \
                (1 / examples.shape[0]) * np.eye(examples.shape[1]) * (10 ** -4))
    return precision


def createExemplarState(state_exemplars,exemplar_props=np.array([0.5,0.5]),n_examples=100,noise_sigma=10**-8):
    '''
    Create a state from exemplars.
    '''
    if type(state_exemplars) != np.ndarray:
        raise ValueError('state_examples must be a numpy array')
    if not (state_exemplars % state_exemplars.astype(int) == 0).all():
        raise ValueError('values for cues and state exemplars must be of type int or int64')
    if type(exemplar_props) != np.ndarray:
        raise ValueError('exemplar_props must be a numpy array')
    if sum(exemplar_props) != 1:
        warnings.warn('The values in the vector do not sum to one. The values will be normalized.')
        exemplar_props = exemplar_props/sum(exemplar_props)
    n_examples_per_state = np.round(exemplar_props*n_examples).astype(int)
    state = []
    for s in range(2):
        state.append(
            {'mu': state_exemplars[s,:].astype(int),
            'examples': np.tile(state_exemplars[s,:],(n_examples_per_state[s],1)).astype(int)
            }
        )
        state[s]['precision'] = calcPrecision(state[s]['examples'],noise_sigma=noise_sigma)

    return state


def calcStateSurprise(cue,mu, precision=None, w_A=None,blur_states_param_linear=0.0):
    if (type(cue) != np.ndarray) or (type(mu) != np.ndarray):
        raise ValueError('cue and mu must be a numpy arrays')
    if (not (cue % cue.astype(int) == 0).all()) or (not (mu % mu.astype(int) == 0).all()):
        raise ValueError('values for cue and mu  exemplars must be of type int or int64')
    if w_A is None:
        w_A = np.ones(len(cue))*.5
    if precision is None:
        precision = np.eye(len(cue))*25
    Z = calcZ(w_A,cue,mu,blur_states_param_linear=blur_states_param_linear)
    D2 = calcD2(Z,precision)
    A = calcActivation(precision,D2)
    S = calcSurprise(A)[0][0]
    return S


def inferContext(cue, states, context_surprise_threshold=100,blur_states_param_linear=0.95):
    """
    Infer the context of a cue.
    :param cue: the cue to infer the context for
    :param states: the states of the model
    :param context_surprise_threshold: the threshold for the surprise of the context
    return state_context
    """
    state_context = []
    surprise = []
    for state in states:
        state_dict = createPrototypeState(state.reshape(1,-1),exemplar_props=np.array([1]),n_examples=100,noise_sigma=10**-8)
        surprise.append(calcStateSurprise(cue,state_dict['mu'], state_dict['precision'], state*0+.5,
            blur_states_param_linear=blur_states_param_linear))

    state_context = states[np.array(surprise)<context_surprise_threshold,:]

    if len(state_context) == 0:
        state_context = 'No existing states met threshold. New state created!'
    return state_context, surprise



def calcStimDeviation(mu,trial_vec,blur_states_param_linear=0.0):
    if (type(blur_states_param_linear) != float) and (type(blur_states_param_linear) != int):
        raise ValueError('blur_states_param_linear must be a float or int')
    if (blur_states_param_linear < 0) or (blur_states_param_linear > 1):
        raise ValueError('blur_states_param_linear must be between 0 and 1')
    stim_deviation = (1 - blur_states_param_linear) * (trial_vec - mu) + blur_states_param_linear * \
                    np.zeros(len(trial_vec))
    return stim_deviation



def calcMI(context_states):
    """
    Calculates the mutual information for the cues to deterimine which are useful and which are distractors. 
    It takes a matrix where each row is a state. 
    :context_state: List where each item is a matrix of examples from the states
    :return: MI, mutual information measure of each cue in the vector
    """
    examples_all = np.concatenate(context_states)
    MI = np.ones((examples_all.shape[1]))
    for c in range(examples_all.shape[1]):
        H_s = np.zeros((len(context_states)))
        for s in range(len(context_states)):
            H_s[s] = calcEntropy(context_states[s][:,c]) * context_states[s].shape[0]/examples_all.shape[0]
        H_0 = calcEntropy(examples_all[:,c])
        MI[c] = H_0 - sum(H_s)
    return MI


def calcZ(w_A,c_t,mu,blur_states_param_linear=0):
    """
    Calculates the deviation vector in the suprise equation
    :w_A: Vector of weights for each cue value
    :c_t: The stimulus on trial t
    :mu: The expected cue for the specific state
    :return: "Z", the deviation vector
    """
    stim_deviation = calcStimDeviation(mu,c_t,blur_states_param_linear=blur_states_param_linear)
    Z = w_A * stim_deviation
    return Z


def calcD2(Z,I_p):
    '''
    Calculates a single radial distance, using the Within-State weight matrix and the deviation vector
    :Z: The deviation vector, calculated using the calcZ function
    :I_p: The precision matrix
    :return: "D2" The radial distance
    '''
    if Z.ndim == 1:
        Z = Z.reshape(1,len(Z))
    D2 = multi_dot([Z,I_p,Z.T])
    return D2

def calcActivation(I_p,D2):
    '''
    Calculates the activation, ussing a multi-variate Gaussian
    :I_p: The precision matrix
    :D2: The radial distance
    :return: "A", the activation value
    '''
    det = np.linalg.det(np.linalg.inv(I_p))
    A = 1 / (np.sqrt(2 * np.pi * det)) * np.exp((-1 / 2) * D2)
    return A

def calcSurprise(A):
    '''
    Calculates the surprise index
    :A: the activation value of the multi-variate Gaussian
    :return: "S", the surprise index
    '''
    S = -1 * np.log(A)
    return S

def makeCovStims(n_trials_tot=100,prop_0=1,stims=np.array([[1,0],[0,1]]),seed=0):
    '''
    Creates the stimuli used to explore calculation of the precision matrix
    :n_trials_tot: Total number of trials to be created
    :prop_0: Reward proportion for action 0
    :stims: The basic stimuli
    :seed: Random seed value
    :return: "stimuli", a matrix of values where rows are trials and columns are cues
    '''
    #Check input values
    if (prop_0 > 1) or (prop_0 < 0):
        raise ValueError('prop_0 must be 0 <= prop_0 =< 1')
    if (n_trials_tot<0) or type(n_trials_tot) != int:
        raise ValueError('n_trials_tot must be an int > 0')
    if stims.shape != (2,2):
        raise valueError('stims must be a 2x2 numpy array')
    np.random.seed(seed) #Keep the results consistent
    #Get the number of trials
    n_prop_0 = round(n_trials_tot*prop_0)
    n_prop_1 = n_trials_tot-n_prop_0
    stimuli = np.vstack((np.tile(stims[0,:],(n_prop_0,1)), np.tile(stims[1,:],(n_prop_1,1))))
    stimuli = np.hstack((stimuli,np.random.randint(0,2,n_trials_tot).reshape(n_trials_tot,1)))
    stimuli = stimuli + np.random.rand(stimuli.shape[0], stimuli.shape[1])*.1
    return stimuli

def calcEntropy(vec):
    """
    Calculates the entropy of an input vector
    :param vec: Input vector
    :return: The entropy measure using log2
    """
    vec[vec<0]= 10**-15 #Trim to distribution
    if sum(np.isnan(vec))>0:
        print(vec)
        raise ValueError('Found a nan in the vec')
    vec_h, _ = (np.histogram(vec, np.linspace(0, 1, 200)))
    vec_h = vec_h / len(vec) + 10 ** -15
    entropy = -1 * sum( vec_h * np.log2(vec_h) )
    return entropy
    
def calcWA(w_k,delta_bar,xi_DB_floor=-5,xi_DB_ceil=-10):
    '''
    Calculates the w_A vector using the Cross-State weights and delta bar
    :w_k: Vector of values containing the Cross-State weights
    :delta_bar: Integrated negative reward
    :xi_DB_floor: Lowest magnitude delta_bar to consider
    :xi_DB_ceil: Highest magnitude delta_bar to consider
    :return: "w_A" the modified Cross-State weights
    '''
    if (type(delta_bar) != float) and (type(delta_bar) != int):
        raise ValueError('delta_bar must be a float')
    if delta_bar > 0:
        raise ValueError('delta_bar must be <= 0')
#     db_factor = min(1, (delta_bar-xi_DB_floor)/(xi_DB_ceil-xi_DB_floor))
    db_factor = min(1, min(0,delta_bar-xi_DB_floor)/(xi_DB_ceil-xi_DB_floor))
    w_A = (1 - db_factor) * w_k + db_factor
    return w_A

#Delta bar functions
def calcDB(delta_bar,delta,xi_0,xi_1):
    '''
    Updates the delta_bar value using the most recent delta
    :delta_bar: Running negative history of reward
    :delta: The trial's reward prediction error
    :xi_0: Leak parameter for delta_bar
    :xi_1: Integration parameter for delta
    :return: "delta_bar_new", the updated delta_bar value
    '''
    delta_bar_new = xi_0*delta_bar + xi_1*min(delta,0)
    return delta_bar_new

def calcAV(action_value,delta,eta=0.05):
    '''
    Updates the action value based on the RPE
    :action_value: The prior action value
    :delta: The trial's reward prediction error
    :eta: The learning rate
    :return: "action_value_new", the updated action value
    '''
    action_value_new = action_value + eta*delta
    return action_value_new

def calcDelta(action_value,reward):
    '''
    Calculates the reward prediction error
    :action_value: The stored action value
    :reward: Whether the agent received a reward on the trial
    :return: "delta", the RPE
    '''
    delta = reward - action_value
    return delta

def newXi(xi_0,xi_1,shft=0):
    '''
    Creates new parameters that change the timescale without changing the asymptote
    :xi_0: Leak parameter for delta_bar
    :xi_1: Integration parameter for delta
    :shft: Shifts xi_0
    :return: xi_0_new", xi_1_new", the updated parameter values    
    '''
    xi_0_new = xi_0 + shft
    xi_1_new = xi_1 * (1 - xi_0_new) / (1 - xi_0)
    return xi_0_new, xi_1_new

def calcDBs(xi_0=0.99,xi_1=1.5,shft_0=-0.009,shft_1=0.004,n_trials=1000,reward_prob=0.95,eta=0.05,seed=0):
    '''
    Tracks various delta bar values over the course of a session
    :xi_0: Leak parameter for delta_bar
    :xi_1: Integration parameter for delta
    :shft_0: The shift parameter for one delta_bar
    :shft_1: The shift parameter for another delta_bar
    :n_trials: Total number of trials
    :reward_prob: Reward probability on any given trial
    :eta: Learning rate
    :seed: Random seed value
    :RETURNS:
    :db_sim_baseline: The simulation with the initial parameters
    :db_sim_0: Simulation values for the first shift
    :db_sim_1: Simulation values for the second shift
    :db_ideal_baseline: Simulation without noise, baseline
    :db_ideal_shft_0: No noise simulation with first shift
    :db_ideal_shft_1: No noise simulation with second shift
    '''
    np.random.seed(seed) #Keep the results consistent
    #Calc new timescales
    xi_0_new_0, xi_1_new_0 = newXi(xi_0,xi_1,shft=-0.009)
    xi_0_new_1, xi_1_new_1 = newXi(xi_0,xi_1,shft=0.004)
    #Create arrays for storing values
    delta = np.zeros(n_trials)
    db_ideal_baseline = delta.copy()
    db_ideal_shft_0 = delta.copy()
    db_ideal_shft_1 = delta.copy()
    action_value = delta.copy()
    db_sim_baseline = delta.copy()
    db_sim_0 = delta.copy()
    db_sim_1 = delta.copy()
    #Loop through and calculate for each trial
    for t in range(n_trials):
        reward = np.random.rand() < reward_prob
        delta[t] = calcDelta(action_value[t-1],reward)
        action_value[t] = calcAV(action_value[t-1],delta[t],eta)
        #Idealized version
        db_ideal_baseline[t] = calcDB(db_ideal_baseline[t-1],reward_prob-1,xi_0,xi_1)
        db_ideal_shft_0[t] = calcDB(db_ideal_shft_0[t-1],reward_prob-1,xi_0_new_0,xi_1_new_0)
        db_ideal_shft_1[t] = calcDB(db_ideal_shft_1[t-1],reward_prob-1,xi_0_new_1,xi_1_new_1)
        #Noisy
        db_sim_baseline[t] = calcDB(db_sim_baseline[t-1],delta[t],xi_0,xi_1)
        db_sim_0[t] = calcDB(db_sim_0[t-1],delta[t],xi_0_new_0,xi_1_new_0)
        db_sim_1[t] = calcDB(db_sim_1[t-1],delta[t],xi_0_new_1,xi_1_new_1)
    return db_sim_baseline, db_sim_0, db_sim_1, db_ideal_baseline, db_ideal_shft_0, db_ideal_shft_1

def modPrecision(weights,xI_p=0):
    '''
    Modify the Within-State weight matrix
    :weights: Precision weight matrix
    :xI_p: Within-State distortion parameter
    :return: "mod", The modified weight matrix
    '''
    left = (1 - xI_p) * weights
    right = xI_p * np.eye(len(weights)) * np.sum(weights.flatten())/len(weights)
    mod = left + right
    return mod

def modMI(weights,xi_attention_distortion=0):
    '''
    Modify the Cross-State weight matrix
    :weights: array of values derived from MI
    :xi_attention_distortion: Feature attention distortion parameter
    :return: "mod", The modified weight array
    '''
    if (type(xi_attention_distortion) != float) and (type(xi_attention_distortion) != int):
        raise ValueError('xi_attention_distortion must be a float or int')
    if (xi_attention_distortion < 0) or (xi_attention_distortion > 1):
        raise ValueError('xi_attention_distortion must be between 0 and 1')
    mod = (1-xi_attention_distortion)*weights + xi_attention_distortion
    return mod

def modulateFeatureAttention(MI,delta_bar=0,attention_distortion=0):
    '''
    Modulate the feature attention matrix
    :MI: array of values derived from MI
    :delta_bar: The delta_bar value
    :xi_attention_distortion: Feature attention distortion parameter
    :return: "mod", The modified weight array
    '''
    w_k = modMI(MI,xi_attention_distortion=attention_distortion)
    w_A = calcWA(w_k,delta_bar,xi_DB_floor=-5,xi_DB_ceil=-10)
    return w_A


def softmax(values, beta=2):
        """
        Applies softmax to input vector
        :param action_values: vector of current values
        :return: soft_max_vals: the vector of probabilities
        """
        exp_vals = np.exp(beta * values)
        exp_vals[exp_vals == float("inf")] = 1000000  # Correct for explosions
        soft_max_vals = exp_vals / np.nansum(exp_vals)
        return soft_max_vals


def representationToRDM(representation):
    rdm = np.zeros((representation.shape[0], representation.shape[0]))

    for i in range(representation.shape[0]):
        for l in range(representation.shape[0]):
            rdm[i, l] = distance.euclidean(representation[i, :], representation[l, :])
    return rdm

def plotRDM(rdm,ax=None,fontsize=14,cbar_shrink=0.65,ttl='State Representation RDM'):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax = sns.heatmap(rdm, square=True, cmap='OrRd', ax=ax, cbar_kws={'shrink': cbar_shrink},
                        yticklabels=np.arange(1, rdm.shape[0] + 1),
                        xticklabels=np.arange(1, rdm.shape[0] + 1))
    ax.collections[0].colorbar.set_label("Euclidean Distance", fontsize=fontsize)
    ax.invert_yaxis()
    ax.set_xlabel("State Mu", fontsize=fontsize)
    ax.set_ylabel("State Mu", fontsize=fontsize)
    ax.set_title(ttl, fontsize=fontsize + 2)
    return ax