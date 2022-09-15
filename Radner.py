

import numpy as np
import tensorflow as tf
import time
from abc import ABC, abstractmethod


class Radner_equilibrium(ABC):  # Forward-Backward Stochastic Neural Network
    def __init__(self, Xi, T,
                 M, N, d,
                 layers):

        self.Xi = Xi  # initial point
        self.T = T  # terminal time
        #self.I = I  # I agents

        self.M = M  # number of trajectories
        self.N = N  # number of time snapshots
        self.d = d  # number of dimensions

        # layers
        self.layers = layers  # (d+1) --> 2

        # initialize NN
        self.weights, self.biases = self.initialize_NN(layers)

        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        # tf placeholders and graph (training)
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.t_tf = tf.placeholder(tf.float32, shape=[M, self.N + 1, 1])  # M x (N+1) x 1
        self.W_tf = tf.placeholder(tf.float32, shape=[M, self.N + 1, self.d])  # M x (N+1) x d (this is the Brownian Motion part)
        self.Xi_tf = tf.placeholder(tf.float32, shape=[1, d])  # 1 x d

        self.loss, self.X_pred, self.S_pred, self.S0_pred, self.R_pred, self.R0_pred = self.loss_function(self.t_tf, self.W_tf, self.Xi_tf)

        # optimizers
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)

        # initialize session and variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim],
                                               stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1

        H = X
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.sin(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_u(self, t, X):  # M x 1, M x d

        M = self.M

        u = self.neural_net(tf.concat([t, X], 1), self.weights, self.biases)  # M x 2
        #Du = tf.gradients(u, X)[0]  # M x D

        ST = u[:, 0]
        S = tf.reshape(ST,[M,1])
        RT = u[:, 1]
        R = tf.reshape(RT, [M, 1])

        DS = tf.gradients(ST,X)[0]   # M * d
        DS1 = tf.expand_dims(DS, 1) # M * 1 * d

        DR = tf.gradients(RT,X)[0]   # M * d
        DR1 = tf.expand_dims(DR, 1) # M * 1 * d

        Z = tf.concat([DS1,DR1],1)  # M * 2 * d

        return S, R, Z

    def Dg_tf(self, X):  # M x d
        S1 = tf.gradients(self.g_tf1(X), X)[0]  # M x d
        S2 = tf.expand_dims(S1, 1)  # M x 1 x d

        R1 = tf.gradients(self.g_tf2(X), X)[0]
        R2 = tf.expand_dims(R1, 1)

        Z = tf.concat([S2,R2],1)

        return Z    # M x 2 x d

    def loss_function(self, t, W, Xi):  # M x (N+1) x 1, M x (N+1) x D, 1 x D
        loss = 0
        X_list = []
        S_list = []
        R_list = []

        t0 = t[:, 0, :] # M x 1
        W0 = W[:, 0, :]  # M x d
        X0 = tf.tile(Xi, [self.M, 1])  # M x d
        S0, R0, Z0 = self.net_u(t0, X0)  # M x 1, M x 1, M x 2 x d

        #Inter0 = tf.matmul(Z0, self.sigma_tf(t0, X0, S0, R0))   # M x 2 x d
        #zeta0 = tf.expand_dims(Inter0[:, 0, :],1)  # M x 1 x d
        #gamma0 = tf.expand_dims(Inter0[:, 1, :],1)    # M x 1 x d

        X_list.append(X0)
        S_list.append(S0)
        R_list.append(R0)

        for n in range(0, self.N):

            Inter0 = tf.matmul(Z0, self.sigma_tf(t0, X0, S0, R0))  # M x 2 x d
            zeta0 = tf.expand_dims(Inter0[:, 0, :], 1)  # M x 1 x d
            gamma0 = tf.expand_dims(Inter0[:, 1, :], 1)  # M x 1 x d

            t1 = t[:, n + 1, :]
            W1 = W[:, n + 1, :]
            X1 = X0 + self.b_tf(t0,X0,S0,R0,Z0) * (t1-t0) + tf.squeeze(tf.matmul(self.sigma_tf(t0,X0,S0,R0),tf.expand_dims(W1-W0,-1)), axis=[-1])
            S1_tilde = S0 + self.getr_S(zeta0, gamma0) * (t1-t0) + tf.squeeze(tf.matmul(zeta0,tf.expand_dims(W1-W0,-1)), axis=[-1])
            R1_tilde = R0 + self.getr_R(zeta0, gamma0) * (t1-t0) + tf.squeeze(tf.matmul(gamma0,tf.expand_dims(W1-W0,-1)), axis=[-1])
            S1, R1, Z1 = self.net_u(t1,X1)

            loss += tf.reduce_sum(tf.square(S1-S1_tilde))
            loss += tf.reduce_sum(tf.square(R1-R1_tilde))

            t0 = t1
            W0 = W1
            X0 = X1
            S0 = S1
            R0 = R1
            Z0 = Z1

            X_list.append(X0)
            S_list.append(S0)
            R_list.append(R0)

        loss += tf.reduce_sum(tf.square(S1 - self.g_tf1(X1)))
        loss += tf.reduce_sum(tf.square(R1 - self.g_tf2(X1)))
        loss += tf.reduce_sum(tf.square(Z1 - self.Dg_tf(X1)))

        X = tf.stack(X_list, axis=1)
        S = tf.stack(S_list, axis=1)
        R = tf.stack(R_list, axis=1)

        return loss, X, S, S[0, 0, 0], R, R[0, 0, 0]

    def fetch_minibatch(self):
        T = self.T

        M = self.M
        N = self.N
        d = self.d

        Dt = np.zeros((M, N + 1, 1))  # M x (N+1) x 1
        DW = np.zeros((M, N + 1, d))  # M x (N+1) x d

        dt = T / N

        Dt[:, 1:, :] = dt
        DW[:, 1:, :] = np.sqrt(dt) * np.random.normal(size=(M, N, d))

        t = np.cumsum(Dt, axis=1)  # M x (N+1) x 1
        W = np.cumsum(DW, axis=1)  # M x (N+1) x d

        return t, W

    def train(self, N_Iter, learning_rate):

        start_time = time.time()
        for it in range(N_Iter):

            t_batch, W_batch = self.fetch_minibatch()  # M x (N+1) x 1, M x (N+1) x d

            tf_dict = {self.Xi_tf: self.Xi, self.t_tf: t_batch, self.W_tf: W_batch, self.learning_rate: learning_rate}

            self.sess.run(self.train_op, tf_dict)

            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value, S0_value, R0_value, learning_rate_value = self.sess.run([self.loss, self.S0_pred, self.R0_pred, self.learning_rate], tf_dict)
                print('It: %d, Loss: %.3e, S0: %.3f, R0: %.3f, Time: %.2f, Learning Rate: %.3e' %
                      (it, loss_value, S0_value, R0_value, elapsed, learning_rate_value))
                start_time = time.time()


    def predict(self, Xi_star, t_star, W_star):

        tf_dict = {self.Xi_tf: Xi_star, self.t_tf: t_star, self.W_tf: W_star}

        X_star = self.sess.run(self.X_pred, tf_dict)
        S_star = self.sess.run(self.S_pred, tf_dict)
        R_star = self.sess.run(self.R_pred, tf_dict)

        return X_star, S_star, R_star

    ###########################################################################
    ############################# Change Here! ################################
    ###########################################################################
    #@abstractmethod
    #def phi_tf(self, t, X, Y, Z):  # M x 1, M x D, M x 1, M x D
       # pass  # M x1

    @abstractmethod
    def g_tf1(self, X):  # M x d
        pass  # M x 1

    @abstractmethod
    def g_tf2(self, X):  # M x d
        pass  # M x 1

    @abstractmethod
    def b_tf(self, t, X, S, R, Z):  # M x 1, M x d, M x 1, M x d
        M = self.M
        d = self.d
        return np.zeros([M, d])  # M x d

    @abstractmethod
    def sigma_tf(self, t, X, S, R):  # M x 1, M x d, M x 1
        M = self.M
        d = self.d
        return tf.matrix_diag(tf.ones([M, d]))  # M x d x d
    ###########################################################################


    def getr_S(self, zeta, gamma):
        M = self.M

        Inter = zeta + gamma
        zetaT = tf.transpose(zeta, [0,2,1])
        gteR = tf.matmul(Inter,zetaT)
        gter = tf.squeeze(gteR,axis=[-1])

        return gter

    def getr_R(self, zeta, gamma):
        M = self.M

        gama = tf.reduce_sum(gamma**2, 2, keepdims = True)
        zetA = tf.reduce_sum(zeta**2, 2, keepdims = True)

        zetaT = tf.transpose(zeta, [0, 2, 1])
        Inter = tf.matmul(zeta, zetaT)
        InteR = tf.square(Inter)

        gteR = (-1/2) * ((InteR / zetA) - gama)
        gter = tf.squeeze(gteR,axis=[-1])

        return gter