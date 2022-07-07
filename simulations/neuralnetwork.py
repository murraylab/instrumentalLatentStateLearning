"""
Copyright 2021, Warren Woodrich Pettine

This contains code pertaining to neural networks used in "Pettine, W. W., Raman, D. V., Redish, A. D., Murray, J. D.
“Human latent-state generalization through prototype learning with discriminative attention.” December 2021. PsyArXiv

https://psyarxiv.com/ku4fr

The code in this module has not been fully implemented, as it is not central to the conclusions of the paper, and all
implementations are standard. However, all the basic components are available, and it is being made available to
reviewers for completeness.

"""

import tensorflow as tf
from tensorflow.keras import initializers
import numpy as np
import os
from scipy.spatial import distance
from utils.utils import saveData, loadData
from simulations.experiments import getContextGenPrototypes, figurePrototypesConversion

os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'

def makeRLModel(num_inputs=6, num_actions=4, num_hidden=124, weight_stdev=None, rl_type='actor_critic',
                    mask=None):
    if mask is None:
        mask = np.ones(num_actions).astype(bool)

    # Make early layers of all models
    inputs = tf.keras.layers.Input(shape=(num_inputs,), name='input')

    if weight_stdev is None:
        common = tf.keras.layers.Dense(num_hidden, activation="relu", name='common')(inputs)
    else:
        stddev = np.sqrt(2 / (num_inputs + num_hidden))*weight_stdev
        common = tf.keras.layers.Dense(num_hidden, activation="relu", name='common',
                                       kernel_initializer=initializers.RandomNormal(stddev=stddev))(inputs)
        # common = tf.keras.layers.Dense(num_hidden, activation="relu", name='common',
        #                                kernel_initializer=tf.keras.initializers.GlorotUniform()*weight_stdev)(inputs)

    # Structure the different type of networks
    if rl_type == 'actor_critic':
        action_intermediate = tf.keras.layers.Dense(num_actions, activation="linear", name='action_intermediate')(
            common)
        softmax = tf.keras.layers.Softmax()(action_intermediate, mask)
        critic = tf.keras.layers.Dense(1, name='critic')(common)
        model = tf.keras.Model(inputs=inputs, outputs=[softmax, critic])
    elif (rl_type == 'deep_q_epsilon_greedy'):
        action = tf.keras.layers.Dense(num_actions, activation="linear", name='action')(common)
        model = tf.keras.Model(inputs=inputs, outputs=action)
    elif (rl_type == 'actor'):
        action_intermediate = tf.keras.layers.Dense(num_actions, activation="linear", name='action_intermediate')(
            common)
        softmax = tf.keras.layers.Softmax()(action_intermediate, mask)
        model = tf.keras.Model(inputs=inputs, outputs=softmax)
    else:
        raise ValueError(f'rl_type must be "actor_critic" or "deep_q_epsilon_greedy" or "actor". {rl_type} is invalid.')

    return model


def copyModelWeights(model1,model2):
    try:
        for i in range(len(model1.layers)):
            model2.layers[i].set_weights(model1.layers[i].get_weights())
    except:
        raise ValueError('The models passed are incompatible.')
    return model2


def validActionsBool(valid_actions,n_actions=None):
    """
    Make sure the valid actions are in a boolean array, and convert if integer indices were passed.
    :param valid_actions:
    :return:
    """
    if n_actions is None:
        n_actions = max(np.max(valid_actions),len(valid_actions))
    if valid_actions is None:
        valid_actions = np.ones(n_actions).astype(bool)
    elif len(valid_actions) > n_actions:
        raise ValueError('More valid_actions given than n_actions')
    elif (len(valid_actions) < n_actions) or (max(valid_actions) > 1):
        valid_actions_tmp = np.zeros(n_actions).astype(bool)
        valid_actions_tmp[valid_actions] = True
        valid_actions = valid_actions_tmp
    return valid_actions


def trainRLNetwork(model, inputs, labels, task='generic', weight_averaging=False,optimizer_type='adam',
                   learning_rate=0.01,rl_type='actor_critic',reset_epsilon_decay_step=True,mask_actions=False,
                   weight_stdev=None,calc_rdm=False,rdm_layers=['common'],reward_probability=1):
    if len(inputs) != len(labels):
        raise ValueError('Set count of inputs and labels are unequal')
    # Record of actions
    action_history = []
    rewards_history = []
    set_record = []
    decay_step = 0
    rdm = None
    if task == 'basicInstrumental':
        model, action_history, rewards_history, _, _, _, _ = trainRLNetworkModel(model, inputs, labels, rl_type=rl_type,
                    action_history=action_history,rewards_history=rewards_history,decay_step=decay_step,n_actions=2,
                    optimizer_type=optimizer_type,valid_actions=None,reward_probability=reward_probability)
    elif (task == 'generic') or (weight_averaging == False):
        # Loop through sets
        i = 0
        for X, Y in zip(inputs,labels):
            # HACK ADDED FOR GENERALIZATION
            if i == 2:
                reset_epsilon_decay_step = False
            if i>0 and (not reset_epsilon_decay_step):
                decay_step += len(labels[i-1])
            else:
                decay_step = 0
            if mask_actions:
                num_actions = model.layers[2].get_output_at(0).get_shape()[1]
                valid_actions = validActionsBool([int(val) for val in np.unique(Y)], n_actions=num_actions)
                # print(f'Masking actions for this block. Mask is {valid_actions}')
                if (rl_type == 'actor_critic') or (rl_type == 'actor'):
                    num_inputs = model.layers[0].get_output_at(0).get_shape()[1]
                    num_hidden = model.layers[1].get_output_at(0).get_shape()[1]
                    model2 = makeRLModel(num_inputs=num_inputs,num_hidden=num_hidden,num_actions=num_actions,
                                        weight_stdev=weight_stdev,rl_type=rl_type,mask=valid_actions)
                    model = copyModelWeights(model,model2)
            else:
                valid_actions = None
            # Set 1
            model, action_history, rewards_history, _, _, _, _ = trainRLNetworkModel(model, X, Y,rl_type=rl_type,
                                                    action_history=action_history,rewards_history=rewards_history,
                                                    optimizer_type=optimizer_type, decay_step=decay_step,
                                                    valid_actions=valid_actions,reward_probability=reward_probability)
            # Calculate the RDM after the first two blocks
            if calc_rdm and (i == 1):
                rdm = calcNNrdm(model,int(task[-1]),rdm_layers=rdm_layers)
            set_record += [i] * len(Y)
            i += 1

    elif ((task == 'context_generalizationV1') or (task == 'context_generalizationV2') or (task == 'context_generalizationV3') \
            ) and (weight_averaging == True):
        for i in range(len(inputs)):
            if i>0 and (not reset_epsilon_decay_step):
                decay_step += len(labels[i-1])
            if mask_actions:
                valid_actions = [int(val) for val in np.unique(labels[i])]
            else:
                valid_actions = None
            # on the last block, average weights
            if i == 2:
                # Get the weights from the second set
                weights_dict_2 = extractNetworkWeights(model)
                # Average them
                set_weights = []
                for l in range((int(len(weights_dict_1) / 2))):
                    w = (weights_dict_1[f'w{l}'][0] + weights_dict_2[f'w{l}'][0]) / 2
                    b = (weights_dict_1[f'b{l}'][0] + weights_dict_2[f'b{l}'][0]) / 2
                    set_weights += [w, b]
                # Set the model
                model.set_weights(set_weights)

            X, Y = inputs[i], labels[i]
            model, action_history, rewards_history, _, _, _, _ = trainRLNetworkModel(model, X, Y,decay_step=decay_step,
                                            action_history=action_history,rewards_history=rewards_history,
                                            optimizer_type=optimizer_type,learning_rate=learning_rate,
                                            valid_actions=valid_actions,reward_probability=reward_probability)
            set_record += [i] * len(Y)
            # Calculate the RDM after the first two blocks
            if calc_rdm and (i == 1):
                rdm = calcNNrdm(model,int(task[-1]),rdm_layers=rdm_layers)

            # Get weights from the first block of the task
            if i == 0:
                weights_dict_1 = extractNetworkWeights(model)

    action_history, rewards_history, set_record = np.array(action_history), np.array(rewards_history), \
                                                  np.array(set_record)

    return model, action_history, rewards_history, set_record, rdm


def validActionsMask(action_probs,valid_actions=None,norm=False,replace_value=np.nan,epsilon=10**-10):
    if valid_actions is None:
        return action_probs
    if tf.is_tensor(action_probs):
        action_probs = action_probs[0].numpy()
    if sum(np.isnan(action_probs)) == len(action_probs):
        action_probs = np.ones(len(action_probs))/len(action_probs)

    action_probs[valid_actions<1] = replace_value
    action_probs += epsilon
    if norm is True:
        action_probs = action_probs/np.nansum(action_probs)
    return action_probs


def trainRLNetworkModel(model, stimuli, ans_key, n_actions=4, action_history=[], rewards_history=[], learning_rate=0.01,
                        action_probs_history=[], critic_value_history=[], critic_losses=[], actor_losses=[],
                        optimizer_type='adam',rl_type='actor_critic', epsilon=1, epsilon_min=0.01, decay_step=0,
                        epsilon_decay=0.01,valid_actions=None,reward_probability=1):
    if (type(rl_type) != str) or ((rl_type != 'actor_critic') and (rl_type != 'deep_q_epsilon_greedy') and
                                  (rl_type != 'reinforce')):
        raise ValueError('rl_type must be "actor_critic", "deep_q_epsilon_greedy" or "reinforce"')
    if valid_actions is None:
        valid_actions = np.ones(n_actions)
    elif len(valid_actions) > n_actions:
        raise ValueError('More valid_actions given than n_actions')
    elif (len(valid_actions) < n_actions) or (max(valid_actions) > 1):
        valid_actions_tmp = np.zeros(n_actions)
        valid_actions_tmp[valid_actions] = 1
        valid_actions = valid_actions_tmp
    n_trials = stimuli.shape[0]
    if optimizer_type == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_type == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    huber_loss = tf.keras.losses.Huber()

    for t in range(n_trials):
        state = tf.convert_to_tensor(stimuli[t, :])
        state = tf.expand_dims(state, 0)
        if rl_type == 'actor_critic':
            with tf.GradientTape() as tape:
                # Predict action probabilities and estimated future rewards
                # from environment state
                action_probs, critic_value = model(state)
                critic_value_history.append(critic_value[0, 0])

                # Sample action from action probability distribution. Mask invalid actions
                action_probs_masked = validActionsMask(action_probs, valid_actions=valid_actions, replace_value=0,
                                                    norm=True)
                try:
                    action = np.random.choice(n_actions, p=np.squeeze(action_probs_masked))
                except:
                    print('here')

                # action = np.random.choice(n_actions, p=np.squeeze(action_probs))
                # action_probs_history.append(np.log(action_probs[action]))
                action_probs_history.append(tf.math.log(action_probs[0, action]))
                action_history.append(action)

                # Determine reward
                reward = action == ans_key[t]
                rewards_history.append(reward)

                # Calculate loss
                diff = rewards_history[-1] - critic_value_history[-1]
                actor_losses.append(-action_probs_history[-1] * diff)
                critic_losses.append(
                    huber_loss(tf.expand_dims(critic_value_history[-1], 0),
                               tf.expand_dims(rewards_history[-1], 0))
                )

                # Backpropagation
                loss_value = actor_losses[-1] + critic_losses[-1]
                grads = tape.gradient(loss_value, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

        elif (rl_type == 'deep_q_epsilon_greedy') or (rl_type == 'reinforce'):
            if rl_type == 'deep_q_epsilon_greedy':
                # Use epsilon-greedy for exploration
                explore_probability = epsilon_min + (epsilon - epsilon_min) * np.exp(
                    -1 * epsilon_decay * decay_step)
                decay_step += 1
                if explore_probability > np.random.rand(1)[0]:
                    # Take random action
                    action_probs = np.ones(n_actions) / n_actions
                    action_probs = validActionsMask(action_probs, valid_actions=valid_actions,norm=True,replace_value=0)
                    action = np.random.choice(n_actions, p=action_probs)
                else:
                    # Predict action Q-values and take action
                    action_probs = model(state, training=False)
                    action_probs = validActionsMask(action_probs, valid_actions=valid_actions,norm=False,replace_value=np.nan)
                    action = np.nanargmax(action_probs)
            elif rl_type == 'reinforce':
                action_probs = model(state, training=False)
                action_probs = validActionsMask(action_probs, valid_actions=valid_actions, norm=False,
                                                replace_value=np.nan)
            #Store actions
            action_probs_history.append(action_probs)
            action_history.append(action)

            # Determine reward and store values (include use of reward probability)
            reward = (action == ans_key[t]) * (np.random.rand() <= reward_probability)
            rewards_history.append(reward)
            if tf.is_tensor(action_probs):
                actor_losses.append(reward - action_probs[0, action].numpy())
            else:
                actor_losses.append(reward-action_probs[action])

            # # Decay probability of taking random action
            # epsilon -= epsilon_interval / epsilon_greedy_frames
            # epsilon = max(epsilon, epsilon_min)

            # Create a mask so we only calculate loss on the updated Q-values
            masks = tf.one_hot(action, n_actions)

            with tf.GradientTape() as tape:
                # Train the model on the states and updated Q-values
                q_values = model(state)

                # Apply the masks to the Q-values to get the Q-value for action taken
                try:
                    q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                except:
                    print('Here')

                # Calculate loss between new Q-value and old Q-value
                loss = huber_loss(reward, q_action)


            # Backpropagation
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return model, action_history, rewards_history, action_probs_history, critic_value_history, \
           critic_losses, actor_losses


def extractNetworkWeights(models):
    # Extract the weights
    weights_dict = {}
    if type(models) == list:
        for m in range(len(models)):
            weights = []
            for layer in models[m].layers:
                weights.append(layer.get_weights())
            weights = weights[1:]
            if m == 0:
                for l in range(len(models[m].layers) - 1):
                    weights_dict[f'w{l}'] = [weights[l][0]]
                    weights_dict[f'b{l}'] = [weights[l][1]]
            else:
                for l in range(len(models[m].layers) - 1):
                    weights_dict[f'w{l}'].append(weights[l][0])
                    weights_dict[f'b{l}'].append(weights[l][1])
    else:
        weights = []
        for layer in models.layers:
            weights.append(layer.get_weights())
        weights = weights[1:]
        for l in range(len(models.layers) - 1):
            weights_dict[f'w{l}'] = [weights[l][0]]
            weights_dict[f'b{l}'] = [weights[l][1]]
    return weights_dict


def saveTrainedNetworks(models, inputs, choice_records, reward_histories, set_record, P=None,
                        file_name='savedNetworks.pickle',save_dir='',legacy=True):
    if legacy:
        if type(models) == dict:
            weights = models
        else:
            if legacy:
                #Extract the weights
                weights = extractNetworkWeights(models)
    else:
        if type(models) == list:
            weights = models
        else:
            weights = []
            for model in models:
                weights.append(model.get_weights())

    save_dict = [weights, inputs, choice_records, set_record, reward_histories, P]
    saveData(os.path.join(save_dir,file_name), save_dict)


def loadTrainedNetworks(file_name,save_dir='',legacy=False):
    save_dict = loadData(os.path.join(save_dir, file_name))
    if legacy: # This is the old way of extracting weights. Keep it around for loading legacy trained networks.
        weights_dict, inputs, choice_records, set_record, reward_histories = save_dict
        num_inputs = weights_dict['w0'][0].shape[0]
        num_actions = weights_dict['w1'][0].shape[1]
        num_hidden = weights_dict['w0'][0].shape[1]
        models = []
        for m in range(len(weights_dict['w0'])):
            model = makeRLModel(num_inputs=num_inputs,num_actions=num_actions,num_hidden=num_hidden)
            set_weights = []
            for l in range((int(len(weights_dict) / 2))):
                set_weights.append(weights_dict[f'w{l}'][m])
                set_weights.append(weights_dict[f'b{l}'][m])
            model.set_weights(set_weights)
            models.append(model)
        return models, inputs, choice_records, set_record, reward_histories
    else:
        weights, inputs, choice_records, set_record, reward_histories, parameters = save_dict
        num_inputs = weights[0][0].shape[0]
        num_actions = weights[0][2].shape[1]
        num_hidden = weights[0][0].shape[1]
        models = []
        for w in range(len(weights)):
            model = makeRLModel(num_inputs=num_inputs, num_actions=num_actions, num_hidden=num_hidden,
                                weight_stdev=parameters['weight_stdev'],rl_type=parameters['rl_type'])
            model.set_weights(weights[w])
            models.append(model)
        return models, inputs, choice_records, set_record, reward_histories, parameters



def saveTrainedNetworks(models, inputs, choice_records, reward_histories, set_record,
                        file_name='savedNetworks.pickle',save_dir=''):
    if type(models) == dict:
        weights_dict = models
    else:
        #Extract the weights
        weights_dict = extractNetworkWeights(models)

    save_dict = [weights_dict, inputs, choice_records, set_record, reward_histories]
    saveData(save_dir + file_name, save_dict)


def loadTrainedNetworks(file_name,save_dir=''):
    save_dict = loadData(save_dir+file_name)
    weights_dict, inputs, choice_records, set_record, reward_histories = save_dict
    num_inputs = weights_dict['w0'][0].shape[0]
    num_actions = weights_dict['w1'][0].shape[1]
    num_hidden = weights_dict['w0'][0].shape[1]
    models = []
    for m in range(len(weights_dict['w0'])):
        model = makeRLModel(num_inputs=num_inputs,num_actions=num_actions,num_hidden=num_hidden)
        set_weights = []
        for l in range((int(len(weights_dict) / 2))):
            set_weights.append(weights_dict[f'w{l}'][m])
            set_weights.append(weights_dict[f'b{l}'][m])
        model.set_weights(set_weights)
        models.append(model)
    return models, inputs, choice_records, set_record, reward_histories


def rdmHelper(output):
    rdm = np.zeros((output.shape[0], output.shape[0]))
    for i in range(output.shape[0]):
        for l in range(output.shape[0]):
            rdm[i, l] = distance.euclidean(output[i, :], output[l, :])
    return rdm


def calcNNrdm(model, version=3,rdm_layers=['common']):
    stims, categories = getContextGenPrototypes(version=version,block='generalization')
    conversions = figurePrototypesConversion(experimental_set='models', context_gen_version=version)
    indx = [np.where(conversions['figure'] == (i + 1))[0][0] for i in range(8)]
    stims = np.unique(stims,axis=0)[indx, :]
    rdm = []
    for layer_name in rdm_layers:
        layer_model = tf.keras.Model(inputs=model.input,
                                              outputs=model.get_layer(layer_name).output)
        output = layer_model.predict(stims)
        rdm.append(rdmHelper(output))

    if len(rdm) == 1:
        rdm = rdm[0]

    return rdm


