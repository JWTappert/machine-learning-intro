import random
import numpy as np

# setup out little AND logic training set
train_data = [
    (np.array([0,0,1]), 0),
    (np.array([0,1,1]), 0),
    (np.array([1,0,1]), 0),
    (np.array([1,1,1]), 1)
]

# vanilla step function
step_fn = lambda x: 0 if x < 0 else 1

# start with random weight
weight = np.random.rand(3)

# set our learn rate
learn = 0.01

# set our number of iterations
iterations = 1000

# train our little perceptron
for i in xrange(iterations):
    # grab out x and expected result
    x, expected = random.choice(train_data)
    # get the product of the value and the random weight
    result = np.dot(weight, x)
    # set the error based on the result of the step function
    # if result is larger than expected we decrease the weight
    # and if the result is smaller than the expected we incease
    # the weight
    error = expected - step_fn(result)
    # adjust the weight
    weight += learn * error * x

# test our perceptron
for x, _ in train_data:
    result = np.dot(x, weight)
    print("{} => {}".format(x[:2], step_fn(result)))