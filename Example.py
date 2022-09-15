import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 不显示等级2以下的提示信息
import numpy as np
import tensorflow as tf
from Radner import Radner_equilibrium
import matplotlib.pyplot as plt


# from plotting import newfig, savefig

class Example1(Radner_equilibrium):
    def __init__(self, Xi, T,
                 M, N, d,
                 layers):
        super().__init__(Xi, T,
                         M, N, d,
                         layers)

    #def phi_tf(self, t, X, Y, Z):  # M x 1, M x D, M x 1, M x D
        #return 0.05 * (Y - tf.reduce_sum(X * Z, 1, keepdims=True))  # M x 1

    def g_tf1(self, X):  # M x d
        M = self.M
        X1 = tf.expand_dims(X, 1)   # M x 1 x d
        X2 = tf.transpose(X1, [0,2,1])  # M x d x 1
        b0a = tf.constant([[1.,2.,1.,2.,1.,2.,1.,2.,1.,2.]], dtype= tf.float32)    # 1 x d
        b0b = tf.tile(b0a, [M, 1])  # M x d
        b0c = tf.expand_dims(b0b, 1) # M x 1 x d
        zetaT = tf.matmul(b0c, X2)  # M x 1 x 1
        zeta = tf.squeeze(zetaT, axis=[-1]) # M x 1

        return zeta  # M x 1

    def g_tf2(self, X):  # M x D
        M = self.M
        X1 = tf.expand_dims(X, 1)  # M x 1 x d
        X2 = tf.transpose(X1, [0, 2, 1])  # M x d x 1
        b1a = tf.constant([[2., 1., 1., 1., 2., 2., 1., 1., 1., 2.]], dtype= tf.float32)  # 1 x d
        b1b = tf.tile(b1a, [M, 1])  # M x d
        b1c = tf.expand_dims(b1b, 1)  # M x 1 x d
        gammaT = tf.matmul(b1c, X2)  # M x 1 x 1
        gamma = tf.squeeze(gammaT, axis=[-1])  # M x 1

        return gamma    # M x 1

    def b_tf(self, t, X, S, R, Z):  # M x 1, M x d, M x 1, M x d
        return super().b_tf(t, X, S, R, Z)  # M x d

    def sigma_tf(self, t, X, S, R):  # M x 1, M x d, M x 1
        return super().sigma_tf(t, X, S, R)  # M x d x d

    ###########################################################################


if __name__ == "__main__":
    M = 100  # number of trajectories (batch size)
    N = 50  # number of time snapshots
    d = 10  # number of dimensions

    layers = [d + 1] + 4 * [256] + [2]

    Xi = np.array([0.0, 0.0] * int(d / 2))[None, :]
    T = 1.0

    # Training
    model = Example1(Xi, T, M, N, d, layers)

    model.train(N_Iter=4 * 10 ** 4, learning_rate=1e-3)
    model.train(N_Iter=5 * 10 ** 4, learning_rate=1e-4)
    model.train(N_Iter=5 * 10 ** 4, learning_rate=1e-5)
    model.train(N_Iter=2 * 10 ** 4, learning_rate=1e-6)

    ##### PLOT RESULTS

    t_test, W_test = model.fetch_minibatch()

    X_pred, S_pred, R_pred = model.predict(Xi, t_test, W_test)


    def S_exact(t, X):  # (N+1) x 1, (N+1) x d
        b0 = np.array([[1., 2., 1., 2., 1., 2., 1., 2., 1., 2.]])
        b1 = np.array([[2., 1., 1., 1., 2., 2., 1., 1., 1., 2.]])
        b0T = np.transpose(b0)

        return (t - 1) * np.matmul((b0 + b1), b0T) + np.sum(b0 * X, 1, keepdims=True)  # (N+1) x 1



    def R_exact(t, X):  # (N+1) x 1, (N+1) x d
        b0 = np.array([[1., 2., 1., 2., 1., 2., 1., 2., 1., 2.]])
        b1 = np.array([[2., 1., 1., 1., 2., 2., 1., 1., 1., 2.]])
        b0T = np.transpose(b0)
        b1sum = np.sum(b1 * b1)
        b0sum = np.sum(b0 * b0)
        return (t - 1) * ((-1/2) * b0sum + (1/2) * b1sum) + np.sum(b1 * X, 1, keepdims=True)   # (N+1) x 1


    S_test = np.reshape(S_exact(np.reshape(t_test[0:M, :, :], [-1, 1]), np.reshape(X_pred[0:M, :, :], [-1, d])), [M, -1, 1])
    R_test = np.reshape(R_exact(np.reshape(t_test[0:M, :, :], [-1, 1]), np.reshape(X_pred[0:M, :, :], [-1, d])), [M, -1, 1])

    samples = 5

    plt.figure()
    plt.plot(t_test[0:1, :, 0].T, S_pred[0:1, :, 0].T, 'b', label='Learned $u(t,X_t)$')
    plt.plot(t_test[0:1, :, 0].T, S_test[0:1, :, 0].T, 'r--', label='Exact $u(t,X_t)$')
    plt.plot(t_test[0:1, -1, 0], S_test[0:1, -1, 0], 'ko', label='$S_T = S(T,X_T)$')    # this is the terminal value

    plt.plot(t_test[1:samples, :, 0].T, S_pred[1:samples, :, 0].T, 'b')
    plt.plot(t_test[1:samples, :, 0].T, S_test[1:samples, :, 0].T, 'r--')
    plt.plot(t_test[1:samples, -1, 0], S_test[1:samples, -1, 0], 'ko')

    plt.plot([0], S_test[0, 0, 0], 'ks', label='$S_0 = S(0,X_0)$')

    plt.xlabel('$t$')
    plt.ylabel('$S_t = S(t,X_t)$')
    plt.title('the stock price process')
    plt.legend()
    plt.show()
    plt.close()

    # savefig('./figures/stock_price_process', crop = False)

    ###################################################################################################################1

    errors = np.sqrt((S_test - S_pred) ** 2 / S_test ** 2)
    mean_errors = np.mean(errors, 0)
    std_errors = np.std(errors, 0)

    plt.figure()
    plt.plot(t_test[0, :, 0], mean_errors, 'b', label='mean')
    plt.plot(t_test[0, :, 0], mean_errors + 2 * std_errors, 'r--', label='mean + two standard deviations')
    plt.xlabel('$t$')
    plt.ylabel('relative error of the stock price process')
    plt.title('Error of the stock price process')
    plt.legend()
    plt.show()
    plt.close()

    # savefig('./figures/stock_price_errors', crop = False)"""

    ###################################################################################################################2

    plt.figure()
    plt.plot(t_test[0:1, :, 0].T, R_pred[0:1, :, 0].T, 'b', label='Learned $u(t,X_t)$')
    plt.plot(t_test[0:1, :, 0].T, R_test[0:1, :, 0].T, 'r--', label='Exact $u(t,X_t)$')
    plt.plot(t_test[0:1, -1, 0], R_test[0:1, -1, 0], 'ko', label='$R_T = R(T,X_T)$')    # this is the terminal value

    plt.plot(t_test[1:samples, :, 0].T, R_pred[1:samples, :, 0].T, 'b')
    plt.plot(t_test[1:samples, :, 0].T, R_test[1:samples, :, 0].T, 'r--')
    plt.plot(t_test[1:samples, -1, 0], R_test[1:samples, -1, 0], 'ko')

    plt.plot([0], R_test[0, 0, 0], 'ks', label='$R_0 = R(0,X_0)$')

    plt.xlabel('$t$')
    plt.ylabel('$R_t = R(t,X_t)$')
    plt.title('the certainty equivalent process')
    plt.legend()
    plt.show()
    plt.close()

    ###################################################################################################################3


    errors = np.sqrt((R_test - R_pred) ** 2 / R_test ** 2)
    mean_errors = np.mean(errors, 0)
    std_errors = np.std(errors, 0)

    plt.figure()
    plt.plot(t_test[0, :, 0], mean_errors, 'b', label='mean')
    plt.plot(t_test[0, :, 0], mean_errors + 2 * std_errors, 'r--', label='mean + two standard deviations')
    plt.xlabel('$t$')
    plt.ylabel('relative error of the certainty equivalent process')
    plt.title('Error of the certainty equivalent process')
    plt.legend()
    plt.show()
    plt.close()

    # savefig('./figures/BSB_Apr18_50_errors', crop = False)"""