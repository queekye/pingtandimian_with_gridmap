"""
Note: This is a updated version from my previous code,
for the target network, I use moving average to soft replace target parameters instead using assign function.
By doing this, it has 20% speed up on my machine (CPU).
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
Pendulum example.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
tensorflow 1.0
gym 0.8.0
"""

import tensorflow as tf
import numpy as np
from grid_map import grid_map
from ou_noise import OUNoise
from replay_buffer import ReplayBuffer

#####################  hyper parameters  ####################

MAX_EPISODES = 10000
MAX_EP_STEPS = 200
LR_A = 0.0001  # learning rate for actor
LR_C = 0.001  # learning rate for critic
GAMMA = 0.99  # reward discount
TAU = 0.001  # soft replacement
MEMORY_CAPACITY = 1000000
REPLAY_START = 1000
BATCH_SIZE = 32

RENDER = False

###############################  DDPG  ####################################


class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound, m_dim, pixel_meter, att_dim):
        self.time_step = 1
        self.memory = ReplayBuffer(MEMORY_CAPACITY)
        self.exploration_noise = OUNoise(a_dim)
        self.pointer = 0
        self.sess = tf.Session()
        writer = tf.summary.FileWriter("logs/", self.sess.graph)

        self.a_dim, self.s_dim, self.a_bound, self.m_dim, self.pixel_meter, self.att_dim = \
            a_dim, s_dim, a_bound, m_dim, pixel_meter, att_dim
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.GM = tf.placeholder(tf.float32, [None, m_dim, m_dim, 1], 'gm')
        self.LM = tf.placeholder(tf.int32, [None, att_dim*2+1, att_dim*2+1, 4], 'lm')
        self.LM_ = tf.placeholder(tf.int32, [None, att_dim*2+1, att_dim*2+1, 4], 'lm_')

        self.a = self._build_a(self.S, self.GM, self.LM, )
        q = self._build_c(self.S, self.GM, self.LM, self.a, )
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)  # soft replacement

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        target_update = [ema.apply(a_params), ema.apply(c_params)]  # soft update operation
        a_ = self._build_a(self.S_, self.GM, self.LM_, reuse=True, custom_getter=ema_getter)  # replaced target parameters
        q_ = self._build_c(self.S_, self.GM, self.LM_, a_, reuse=True, custom_getter=ema_getter)

        a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=a_params)

        with tf.control_dependencies(target_update):  # soft replacement happened at here
            q_target = self.R + GAMMA * q_
            td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
            self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=c_params)

        self.sess.run(tf.global_variables_initializer())

    def noise_action(self, s1, gm1, loc1):
        locm = np.zeros([1, self.att_dim*2+1, self.att_dim*2+1, 4])
        for j in range(self.att_dim * 2 + 1):
            for k in range(self.att_dim * 2 + 1):
                locm[0, j, k, :] = np.array([0, loc1[0] - self.att_dim + j, loc1[1] - self.att_dim + k, 0])
        return self.sess.run(self.a, {self.S: s1[np.newaxis, :], self.GM: gm1[np.newaxis, :, :, np.newaxis],
                                      self.LM: locm})[0] + self.exploration_noise.noise()

    def action(self, s1, gm1, loc1):
        locm = np.zeros([1, self.att_dim * 2 + 1, self.att_dim * 2 + 1, 4])
        for j in range(self.att_dim * 2 + 1):
            for k in range(self.att_dim * 2 + 1):
                locm[0, j, k, :] = np.array([0, loc1[0] - self.att_dim + j, loc1[1] - self.att_dim + k, 0])
        return self.sess.run(self.a, {self.S: s1[np.newaxis, :], self.GM: gm1[np.newaxis, :, :, np.newaxis],
                                      self.LM: locm})[0]

    def perceive(self, sd, p, loc, s, a_store, r, s_, loc_, done):
        self.memory.add(sd, p, loc, s, a_store, r, s_, loc_, done)
        if self.memory.count() > REPLAY_START:
            self.learn()
        if self.time_step % 500000 == 0:
            self.save_network()

    def learn(self):
        self.time_step += 1
        replay = self.memory.get_batch(BATCH_SIZE)
        bm_sd = np.asarray([data[0] for data in replay])
        bp = np.asarray([data[1] for data in replay])
        bloc = np.asarray([data[2] for data in replay])
        bs = np.asarray([data[3] for data in replay])
        ba = np.asarray([data[4] for data in replay])
        br = np.reshape(np.asarray([data[5] for data in replay]), [-1, 1])
        bs_ = np.asarray([data[6] for data in replay])
        bloc_ = np.asarray([data[7] for data in replay])
        bgm = np.zeros([BATCH_SIZE, self.m_dim, self.m_dim, 1])
        for batch in range(BATCH_SIZE):
            sd1 = bm_sd[batch]
            terrian_map = grid_map(sd1, self.m_dim, self.pixel_meter, bp[batch])
            bgm[batch, :, :, 0] = terrian_map.map_matrix
        blocm = np.zeros([BATCH_SIZE, self.att_dim*2+1, self.att_dim*2+1, 4])
        blocm_ = np.zeros([BATCH_SIZE, self.att_dim * 2 + 1, self.att_dim * 2 + 1, 4])
        for i in range(BATCH_SIZE):
            for j in range(self.att_dim*2+1):
                for k in range(self.att_dim*2+1):
                    blocm[i, j, k, :] = np.array([i, bloc[i, 0]-self.att_dim+j, bloc[i, 1]-self.att_dim+k, 0])
                    blocm_[i, j, k, :] = np.array([i, bloc_[i, 0] - self.att_dim + j, bloc_[i, 1] - self.att_dim + k, 0])

        self.sess.run(self.atrain, {self.S: bs, self.GM: bgm, self.LM: blocm})
        self.sess.run(self.ctrain, {self.GM: bgm, self.S: bs, self.LM: blocm, self.a: ba, self.R: br, self.S_: bs_, self.LM_: blocm_})

    def _build_a(self, s, gm, locm, reuse=None, custom_getter=None):

        def _conv2d_keep_size(x, y, kernel_size, name, use_bias=False, reuse_conv=None, trainable_conv=True):
            return tf.layers.conv2d(inputs=x,
                                    filters=y,
                                    kernel_size=kernel_size,
                                    padding="same",
                                    use_bias=use_bias,
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                    bias_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                    reuse=reuse_conv,
                                    name=name,
                                    trainable=trainable_conv)

        def _build_vin(mat, name, reuse, trainable_vin):
            h1 = _conv2d_keep_size(mat, 150, 3, name+"_h1", use_bias=True, reuse_conv=reuse, trainable_conv=trainable_vin)
            r = _conv2d_keep_size(h1, 1, 1, name+"_r", reuse_conv=reuse, trainable_conv=trainable_vin)
            q0 = _conv2d_keep_size(r, 10, 9, name+"_q0", reuse_conv=reuse, trainable_conv=trainable_vin)
            v = tf.reduce_max(q0, axis=3, keep_dims=True, name=name+"_v")
            rv = tf.concat([r, v], axis=3)
            q = _conv2d_keep_size(rv, 10, 9, name + "_q", reuse_conv=False, trainable_conv=trainable_vin)
            v = tf.reduce_max(q, axis=3, keep_dims=True, name=name + "_v")
            for k in range(30):
                rv = tf.concat([r, v], axis=3)
                q = _conv2d_keep_size(rv, 10, 9, name+"_q", reuse_conv=True, trainable_conv=trainable_vin)
                v = tf.reduce_max(q, axis=3, keep_dims=True, name=name+"_v")
            return v

        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            gv = _build_vin(gm, name="global_map_vin", reuse=reuse, trainable_vin=trainable)
            att = tf.reshape(tf.gather_nd(gv, locm), [-1, (self.att_dim*2+1)**2])
            layer_1 = tf.layers.dense(s, 300, activation=tf.nn.relu, name='l1', trainable=trainable)
            layer_2a = tf.layers.dense(layer_1, 600, name='l2a', trainable=trainable)
            layer_2att = tf.layers.dense(att, 600, name='l2att', trainable=trainable)
            layer_2 = tf.add(layer_2a, layer_2att, name="l2")
            layer_3 = tf.layers.dense(layer_2, 600, activation=tf.nn.relu, name='l3', trainable=trainable)
            a = tf.layers.dense(layer_3, 7, activation=tf.nn.tanh, name='a1', trainable=trainable)
            return a

    def _build_c(self, s, gm, loc, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            gm_flat = tf.reshape(gm, [-1, self.m_dim**2])
            layer_gm = tf.layers.dense(gm_flat, self.s_dim, activation=tf.nn.relu, name='lgm', trainable=trainable)
            s_all = tf.concat([layer_gm, s], axis=1)
            layer_1 = tf.layers.dense(s_all, 300, activation=tf.nn.relu, name='l1', trainable=trainable)
            layer_2s = tf.layers.dense(layer_1, 600, activation=None, name='l2s', trainable=trainable)
            layer_2a = tf.layers.dense(a, 600, activation=None, name='l2a', trainable=trainable)
            layer_2 = tf.add(layer_2s, layer_2a, name="l2")
            layer_3 = tf.layers.dense(layer_2, 600, activation=tf.nn.relu, name='l3', trainable=trainable)
            return tf.layers.dense(layer_3, 1, trainable=trainable)  # Q(s,a)

    def save_network(self):
        self.saver = tf.train.Saver()
        print("save ddpg-network...", self.time_step)
        self.saver.save(self.sess, 'saved_ddpg_networks/' + "ddpg-network", global_step=self.time_step)

    def load_network(self):
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("saved_ddpg_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")


if __name__ == '__main__':
    agent = DDPG(7, 6, np.ones(7), 64, 20, 3)
