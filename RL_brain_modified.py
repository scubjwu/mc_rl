# -*- coding: utf-8 -*

import numpy as np
import tensorflow as tf
import os
from sklearn.preprocessing import LabelBinarizer
import sys

np.random.seed(1)
tf.set_random_seed(1)

# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions=1,
            #state = sensor_num * (3 + hotspot_num) + hotspot_num * 3
            #action = 42(hotspot_encode) + 1(wt)
            n_features=17 * 45 + 42 * 3 + 43,
            learning_rate=0.001,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=200,
            memory_size=30000,
            batch_size=256,
            #e_greedy_increment=0.001,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, (17 * 45 + 42 * 3) * 2 + 4))

        self.set_hotspot_valid_hour()

        self.hotspot_num_encode = LabelBinarizer()
        self.hotspot_num_encode.fit_transform([str(i) for i in range(1,43)])

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def random_action(self, hour):
        action = 2 * [None]

        actions = self.hotspot_valid[hour].items()
        index = np.random.randint(len(actions))
        res = actions[index]

        action[0] = int(res[0])
        action[1] = np.random.randint(1, res[1]+1)

        #print 'random action: ', action

        return action

    def _set_valid(self, i):
        self.hotspot_valid[i] = {}

        file_name = 'hotspot_valid/' + str(i+1) + '.txt'
        with open(file_name) as f:
            for line in f:
                data = line.strip().split(',')
                wtime = int(data[1])
                wtime = 4 if wtime > 4 else wtime
                self.hotspot_valid[i][data[0]] = wtime
                

    def set_hotspot_valid_hour(self):
        n = len(os.listdir('hotspot_valid/'))
        self.hotspot_valid = n * [None]

        for i in range(n):
            self._set_valid(i)


    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, n_l2, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 800, 800, \
                tf.glorot_uniform_initializer(1), tf.constant_initializer(0.05)  # config of layers
                #tf.random_normal_initializer(0., 0.1), tf.constant_initializer(0.1)  # config of layers

            # hidden layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, n_l2], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, n_l2], initializer=b_initializer, collections=c_names)
                l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

            # output layer. collections is used later when assign to target net
            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w3', [n_l2, self.n_actions], initializer=w_initializer, collections=c_names)
                b3 = tf.get_variable('b3', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l2, w3) + b3

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))

        with tf.variable_scope('train'):
            #self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
            self._train_op = tf.train.AdamOptimizer(epsilon=1e-4).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # hidden layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, n_l2], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, n_l2], initializer=b_initializer, collections=c_names)
                l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

            # output layer. collections is used later when assign to target net
            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w3', [n_l2, self.n_actions], initializer=w_initializer, collections=c_names)
                b3 = tf.get_variable('b3', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l2, w3) + b3

    def store_transition(self, s, a, r, t_, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, a, r, int(t_/1200), s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    ###########################################################
    def choose_action(self, observation, time):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]
        hour = int(time/1200)
        action = 2 * [None]

        # the init state is all 0
        if all(i == 0 for i in observation[0]):
            return self.random_action(hour)

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            choose_value = -sys.maxint - 1
            for h, wt in self.hotspot_valid[hour].items():
                h_encode = self.hotspot_num_encode.transform([h])[0]
                for wt_i in range(1, wt+1):
                    action_i = np.r_[h_encode, wt_i][np.newaxis, :]
                    action_observation = np.c_[action_i, observation]

                    action_value = self.sess.run(self.q_eval, feed_dict={self.s: action_observation})[0][0]

                    if action_value > choose_value:
                        choose_value = action_value
                        action[0] = int(h)
                        action[1] = wt_i

        else:
            return self.random_action(hour)
            #action = np.random.randint(0, self.n_actions)
            ###########################################################

        return action


    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        #construct batch data for training
        rows, cols = batch_memory.shape
        state_len = 17 * 45 + 42 * 3
        action_len = 2
        for row in range(rows):
            batch_memory_row = batch_memory[row]

            state_i = batch_memory_row[:state_len][np.newaxis, :]
            action_i = batch_memory_row[state_len: state_len + action_len]
            h_i = self.hotspot_num_encode.transform([str(action_i[0])])[0]
            wt_i = action_i[1]
            action_state_i = np.c_[np.r_[h_i, wt_i][np.newaxis, :], state_i]

            if row == 0:
                q_eval_all_action_state = action_state_i
            else:
                q_eval_all_action_state = np.vstack((q_eval_all_action_state, action_state_i))


            reward_i = batch_memory_row[state_len + action_len]
            hour_i = batch_memory_row[state_len + action_len + 1]

            state_i_1 = batch_memory_row[-state_len:][np.newaxis, :]
            reward_next = -sys.maxint - 1
            for h, wt in self.hotspot_valid[int(hour_i)].items():
                h_encode = self.hotspot_num_encode.transform([str(h)])[0]
                for wt_i in range(1, wt+1):
                    action_i_1 = np.r_[h_encode, wt_i][np.newaxis, :]
                    action_state_i_1 = np.c_[action_i_1, state_i_1]

                    action_value = self.sess.run(self.q_next, feed_dict={self.s_: action_state_i_1})[0][0]

                    reward_next = action_value if action_value > reward_next else reward_next

            y_i = reward_i + self.gamma * reward_next 
            if row == 0:
                q_target = np.array([y_i])[np.newaxis, :]
            else:
                q_target = np.vstack((q_target, np.array([y_i])[np.newaxis, :]))


        _, self.cost = self.sess.run([self._train_op, self.loss],
                feed_dict={self.s: q_eval_all_action_state,
                           self.q_target: q_target})

        self.cost_his.append(self.cost)
        print('loss, greedy-e', self.cost, self.epsilon)
        
        # increasing epsilon
        if self.epsilon_increment:
            self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max

        self.learn_step_counter += 1
        #print('learning times: ', self.learn_step_counter)

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()



