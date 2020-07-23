from sklearn.datasets import load_boston
from sklearn.utils import shuffle, resample
import matplotlib.pyplot as plt
import numpy as np

from nn import Linear, Sigmoid, ReLU, MSE, Placeholder
from utils import topological_sort_feed_dict, forward_and_backward, optimize

if __name__ == '__main__':
    data = load_boston()
    losses = []

    # Load data
    data = load_boston()
    X_ = data['data']
    y_ = data['target']

    # Normalize data
    X_ = (X_ - np.mean(X_, axis=0)) / np.std(X_, axis=0)

    n_features = X_.shape[1]
    n_hidden = 10
    W1_ = np.random.randn(n_features, n_hidden)
    b1_ = np.zeros(n_hidden)
    W2_ = np.random.randn(n_hidden, 1)
    b2_ = np.zeros(1)

    # Neural network
    X, y = Placeholder(), Placeholder()
    W1, b1 = Placeholder(), Placeholder()
    W2, b2 = Placeholder(), Placeholder()

    l1 = Linear(X, W1, b1)
    s1 = Sigmoid(l1)
    #s1 = ReLU(l1)
    l2 = Linear(s1, W2, b2)
    cost = MSE(y, l2)

    feed_dict = {
        X: X_,
        y: y_,
        W1: W1_,
        b1: b1_,
        W2: W2_,
        b2: b2_
    }

    epochs = 5000
    # Total number of examples
    m = X_.shape[0]
    batch_size = 16
    steps_per_epoch = m // batch_size

    graph = topological_sort_feed_dict(feed_dict)
    trainables = [W1, b1, W2, b2]

    print("Total number of examples = {}".format(m))

    for i in range(epochs):
        loss = 0
        for j in range(steps_per_epoch):
            # Step 1
            # Randomly sample a batch of examples
            X_batch, y_batch = resample(X_, y_, n_samples=batch_size)

            # Reset value of X and y Inputs
            X.value = X_batch
            y.value = y_batch

            # Step 2
            _ = None
            forward_and_backward(_, graph)  # set output node not important.

            # Step 3
            rate = 1e-2

            optimize(trainables, rate)

            loss += graph[-1].value

        if i % 100 == 0:
            print("Epoch: {}, Loss: {:.3f}".format(i + 1, loss / steps_per_epoch))
            losses.append(loss / steps_per_epoch)

    plt.plot(losses)
    plt.show()
