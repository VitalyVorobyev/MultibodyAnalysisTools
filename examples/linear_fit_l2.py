#! /usr/bin/python2
"""
 #  L2 regularization tutorial
 #  Based on tutorial 8.1 from
 #  http://ipython-books.github.io/cookbook/
"""

import numpy as np
import scipy.stats as st
import sklearn.linear_model as lm
import matplotlib.pyplot as plt

plt.style.use('seaborn-white')
plt.rc('font', size=26)
plt.rc('text', usetex=True)

f = lambda x: np.exp(3 * x)

x_tr = np.linspace(0., 2, 200)
y_tr = f(x_tr)

x = np.array([0, .1, .2, .5, .8, .9, 1])
y = f(x) + np.random.randn(len(x))

def plot_data(x_tr=x_tr, y_tr=y_tr, x=x, y=y):
    plt.figure(figsize=(9,5))
    plt.plot(x_tr, y_tr, '--k', lw=2)
    plt.plot(x, y, 'bo', ms=15)
    plt.xlim((-0.1, 1.1))
    plt.ylim((-4, 24))
    plt.grid()

def linear_model(x_tr=x_tr, y_tr=y_tr, x=x, y=y):
    # We create the model.
    lr = lm.LinearRegression()
    # We train the model on our training dataset.
    lr.fit(x[:, np.newaxis], y);
    # Now, we predict points with our trained model.
    y_lr = lr.predict(x_tr[:, np.newaxis])
    return y_lr

def plot_linear_model(y_lr, x_tr=x_tr, y_tr=y_tr, x=x, y=y):
    plot_data(x_tr, y_tr, x, y)
    plt.plot(x_tr, y_lr, 'g', lw=2)
    plt.title("Linear regression")

def without_regularization(x_tr=x_tr, y_tr=y_tr, x=x, y=y):
    lrp = lm.LinearRegression()
    plot_data(x_tr, y_tr, x, y)

    for deg, s in zip([2, 5], ['-', '-']):
        lrp.fit(np.vander(x, deg + 1), y);
        y_lrp = lrp.predict(np.vander(x_tr, deg + 1))
        plt.plot(x_tr, y_lrp, s, label='degree ' + str(deg), lw=2);
        plt.legend(loc=2);
        plt.xlim(-0.1, 1.4);
        plt.ylim(-10, 40);
        # Print the model's coefficients.
        print(' '.join(['%.2f' % c for c in lrp.coef_]))
    plt.title("Linear regression");

def with_regularization(x_tr=x_tr, y_tr=y_tr, x=x, y=y):
    ridge = lm.RidgeCV()
    plot_data(x_tr, y_tr, x, y)

    for deg, s in zip([2, 5], ['-', '-']):
        ridge.fit(np.vander(x, deg + 1), y);
        y_ridge = ridge.predict(np.vander(x_tr, deg + 1))
        plt.plot(x_tr, y_ridge, s, label='degree ' + str(deg), lw=2);
        plt.legend(loc=2);
        plt.xlim(-0.1, 1.5);
        plt.ylim(-5, 80);
        # Print the model's coefficients.
        print(' '.join(['%.2f' % c for c in ridge.coef_]))
    plt.title("Ridge regression");

# plot_data()
y_lr = linear_model()
plot_linear_model(y_lr)
# without_regularization()
# with_regularization()
plt.show()