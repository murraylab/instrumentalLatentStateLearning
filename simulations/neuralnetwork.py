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
import numpy as np
import os

from utils.utils import saveData, loadData, getPool

os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'

def makeRLModel(num_inputs=6, num_actions=4, num_hidden=124):
    # Make the model
    inputs = tf.keras.layers.Input(shape=(num_inputs,))
    common = tf.keras.layers.Dense(num_hidden, activation="relu")(inputs)
    action = tf.keras.layers.Dense(num_actions, activation="softmax")(common)
    critic = tf.keras.layers.Dense(1)(common)
    model = tf.keras.Model(inputs=inputs, outputs=[action, critic])

    return model

def trainRLmodel(model, stimuli, ans_key, n_trials, action_history=[], rewards_history=[],
                 action_probs_history=[], critic_value_history=[], critic_losses=[], actor_losses=[]):
    # Extract number of actions
    num_actions = model.output[0].shape[1]

    # Create optimizers and loss functions
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    huber_loss = tf.keras.losses.Huber()

    # Loop through timesteps
    for t in range(n_trials):
        with tf.GradientTape() as tape:
            # env.render(); Adding this line would show the attempts
            # of the agent in a pop up window.

            state = tf.convert_to_tensor(stimuli[t, :])
            state = tf.expand_dims(state, 0)

            # Predict action probabilities and estimated future rewards
            # from environment state
            action_probs, critic_value = model(state)
            critic_value_history.append(critic_value[0, 0])

            # Sample action from action probability distribution
            action = np.random.choice(num_actions, p=np.squeeze(action_probs))
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

    return model, action_history, rewards_history, action_probs_history, critic_value_history, \
           critic_losses, actor_losses


class trainRLNetworks():
    def __init__(self,P,n_networks=1):
        self.P = P
        self.n_networks = n_networks
        self.weight_dicts = []
        self.choice_records = [] # Choice, label
        self.reward_histories = []
        self.inputs = []
        self.set_record = [] #np.zeros(sum(P['n_trials_per_bloc']))

    def run(self,parallel=True):
        if parallel:
            print('Running networks in parallel')
            pool = getPool()
            results = pool.map(self.runCall, np.arange(0, self.n_networks))
            pool.close()
            weight_dicts = []
            for result in results:
                weight_dict, inputs, labels, action_history, reward_history, set_record = result
                weight_dicts.append(weight_dict)
                self.choice_records.append(np.vstack((action_history,labels)).T)
                self.inputs.append(inputs)
                self.reward_histories.append(reward_history)
                self.set_record = set_record
            self.weight_dicts = weight_dicts[0]
            for d in range(len(weight_dicts) - 1):
                for key in self.weight_dicts.keys():
                    self.weight_dicts[key] += weight_dicts[d + 1][key]

        else:
            for n in range(self.n_networks):
                weight_dict, inputs, labels, action_history, reward_history, set_record = self.runCall()
                self.weight_dicts.append(weight_dict)
                self.choice_records.append(np.vstack((action_history,labels)).T)
                self.inputs.append(inputs)
                self.reward_histories.append(reward_history)
                self.set_record = set_record
        print('Done training networks')

    def runCall(self,run_num=None):
        """
        Call to allow for parallelization
        :param P:
        :return:
        """
        if (self.P['task'] == 'set_generalizationV1') or (self.P['task'] == 'set_generalizationV2') or \
                (self.P['task'] == 'set_generalizationV3'):
            version = int(self.P['task'][-1])
            X_0, X_1, X_2, Y_0, Y_1, Y_2 = makeSetGeneralizationEnv(n_trials_block_0=self.P['n_trials_per_bloc'][0],
                                                                    n_trials_block_1=self.P['n_trials_per_bloc'][1],
                                                                    n_trials_block_2=self.P['n_trials_per_bloc'][2],
                                                                    version=version)
        else:
            raise ValueError('Only set_generalization currently implemented')
        model = makeRLModel(num_inputs=X_0.shape[1], num_actions=len(np.unique(np.concatenate((Y_0, Y_1, Y_2)))))

        model, action_history, rewards_history, set_record = \
            trainRLNetwork(model, [X_0, X_1, X_2], [Y_0, Y_1, Y_2],task=self.P['task'],
                           weight_averaging=self.P['weight_averaging'])

        weight_dict = extractNetworkWeights([model])
        inputs = np.vstack((X_0, X_1, X_2))
        labels = np.concatenate((Y_0, Y_1, Y_2))

        return weight_dict, inputs, labels, action_history, rewards_history, set_record

    def saveResults(self,file_name='savedNetworks.pickle', save_dir=''):
        try:
            saveTrainedNetworks(self.weight_dicts, self.inputs, self.choice_records, self.reward_histories,
                                self.set_record, file_name=file_name, save_dir=save_dir)
            return 0
        except:
            return 1


def trainRLNetwork(model, inputs, labels, task='generic', weight_averaging=False):
    if len(inputs) != len(labels):
        raise ValueError('Set count of inputs and labels are unequal')
    # Record of actions
    action_history = []
    rewards_history = []
    set_record = []
    if (task == 'generic') or (weight_averaging == False):
        # Loop through sets
        i = 0
        for X, Y in zip(inputs,labels):
            # Set 1
            model, action_history, rewards_history, _, _, _, _ = trainRLNetworkModel(model, X, Y,
                                                                              action_history=action_history,
                                                                              rewards_history=rewards_history)
            set_record += [i] * len(Y)
            i += 1

    elif ((task == 'set_generalizationV1') or (task == 'set_generalizationV2') or (task == 'set_generalizationV3') \
            ) and (weight_averaging == True):
        for i in range(len(inputs)):
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
            model, action_history, rewards_history, _, _, _, _ = trainRLNetworkModel(model, X, Y,
                                                                                     action_history=action_history,
                                                                                     rewards_history=rewards_history)
            set_record += [i] * len(Y)
            # Get weights from the first block of the task
            if i == 0:
                weights_dict_1 = extractNetworkWeights(model)

    action_history, rewards_history, set_record = np.array(action_history), np.array(rewards_history), \
                                                  np.array(set_record)

    return model, action_history, rewards_history, set_record


def trainRLNetworkModel(model, stimuli, ans_key, n_actions=4, action_history=[], rewards_history=[],
                        action_probs_history=[], critic_value_history=[], critic_losses=[], actor_losses=[]):
    n_trials = stimuli.shape[0]
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    huber_loss = tf.keras.losses.Huber()
    for t in range(n_trials):
        with tf.GradientTape() as tape:
            # env.render(); Adding this line would show the attempts
            # of the agent in a pop up window.

            state = tf.convert_to_tensor(stimuli[t, :])
            state = tf.expand_dims(state, 0)

            # Predict action probabilities and estimated future rewards
            # from environment state
            action_probs, critic_value = model(state)
            critic_value_history.append(critic_value[0, 0])

            # Sample action from action probability distribution
            action = np.random.choice(n_actions, p=np.squeeze(action_probs))
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

    return model, action_history, rewards_history, action_probs_history, critic_value_history, \
           critic_losses, actor_losses



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
