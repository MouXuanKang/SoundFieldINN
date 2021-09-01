import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

np.random.seed(1234)
tf.set_random_seed(1234)


class PINN():
    def __init__(self, x, y,
                 p, layers):
        X = np.concatenate([x, y], 1)

        self.lb = X.min(0)
        self.ub = X.max(0)

        self.X = X

        self.x = X[:, 0:1]
        self.y = X[:, 1:2]

        self.p = p

        self.layers = layers

        # Initialize NN
        self.weights, self.biases = self.initialize_NN(layers)

        # Initialize parameters
        self.lambda_1 = tf.Variable([0.0], dtype=tf.float32)

        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y.shape[1]])

        self.p_tf = tf.placeholder(tf.float32, shape=[None, self.p.shape[1]])

        self.p_pred, self.f_p_pred = self.net_NS(self.x_tf, self.y_tf)

        self.loss = tf.reduce_sum(tf.square(self.p_tf - self.p_pred)) + \
                    tf.reduce_sum(tf.square(self.f_p_pred))

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 50000,
                                                                         'maxfun': 5e5,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps})
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

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
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1

        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_NS(self, x, y):
        lambda_1 = self.lambda_1

        p = self.neural_net(tf.concat([x, y], 1), self.weights, self.biases)

        p_x = tf.gradients(p, x)[0]
        p_y = tf.gradients(p, y)[0]

        p_xx = tf.gradients(p_x, x)[0]
        p_yy = tf.gradients(p_y, y)[0]

        f_p = p_xx + p_yy + lambda_1 * p
        return p, f_p

    def callback(self, loss, lambda_1, lambda_2):
        print('Loss: %.3e, l1: %.3f' % (loss, lambda_1))

    def train(self, nIter):

        tf_dict = {self.x_tf: self.x, self.y_tf: self.y,
                   self.p_tf: self.p}

        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)

            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                lambda_1_value = self.sess.run(self.lambda_1)
                print('It: %d, Loss: %.3e, l1: %.3f, Time: %.2f' %
                      (it, loss_value, lambda_1_value, elapsed))
                start_time = time.time()

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss, self.lambda_1],
                                loss_callback=self.callback)

    def predict(self, x_star, y_star):

        tf_dict = {self.x_tf:x_star, self.y_tf:y_star}
        p_star = self.sess.run(self.p_pred, tf_dict)
        return p_star

