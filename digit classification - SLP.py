import numpy as np
import csv
from sklearn.model_selection import train_test_split

with open('C:/Users/Sanjay/Desktop/IDL/Assignment1/data/train_in.csv', 'r') as f:
    data = list(csv.reader(f))
with open('C:/Users/Sanjay/Desktop/IDL/Assignment1/data/train_out.csv', 'r') as d:
    label = list(csv.reader(d))

with open('C:/Users/Sanjay/Desktop/IDL/Assignment1/data/train_in.csv', 'r') as e:
    test_data = list(csv.reader(e))
with open('C:/Users/Sanjay/Desktop/IDL/Assignment1/data/train_out.csv', 'r') as g:
    test_label = list(csv.reader(g))

X = np.asarray(data[0:], dtype='float')
Y = np.asarray(label[0:], dtype='float')

test_X = np.asarray(test_data[0:], dtype='float')
test_Y = np.asarray(test_label[0:], dtype='float')


def add_bias(input_pixels):
    q, r = np.shape(input_pixels)
    s = np.ones((q, 1))
    t = np.hstack((s, input_pixels))
    return t


def initialize_weights(input_pixels):
    q, r = np.shape(input_pixels)
    weights = np.random.rand(r, 1)
    return weights


def prediction(input_pixels_point, weights):
    r = np.dot(input_pixels_point, weights)
    q = r > 0
    return (q * 1)


def label(labels, number):
    return ((labels == number) * 1)


def update_weights(weights, input_pixels_point, labels, l_r=.1):
    predicted = prediction(input_pixels_point, weights)
    error = labels - predicted
    weight_temp = np.zeros(np.shape(weights))
    weight_temp[:, 0] = error * input_pixels_point*l_r
    weight_temp = weight_temp + weights
    return weight_temp


def train(input_pixels, labels, weights, l_r = .1, iterations = 100):
    #print(l_r)
    for j in range(0, iterations):
        #print(j)
        for i in range(0, len(input_pixels)):
            weights = update_weights(weights, input_pixels[i], labels[i], l_r)
    return weights


def create_nodes(input_pixels, labels):
    w = initialize_weights(input_pixels)
    weights = []
    for i in range(0, len(np.unique(labels))):
        z = label(labels, i)
        a = train(input_pixels, z, w, 0.1, 50)
        weights.append(a[:, 0])
    return np.asarray(weights)


def test_all(input_pixels, weights):
    q = np.dot(input_pixels, np.transpose(weights))
    r = len(np.shape(input_pixels))
    if r == 1:
        return np.argmax(q)
    return np.argmax(q, axis=1)


def test(input_pixels, labels, weights):
    q, r = np.shape(labels)
    predicted = test_all(input_pixels, weights)
    correct = predicted == labels[:, 0]
    accuracy_score = np.sum(correct) / float(q)
    return accuracy_score


X = add_bias(X)
test_X = add_bias(test_X)


# Validate by splitting training set
'''X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4)
W = create_nodes(X_train, Y_train)
print(np.shape(W))

accuracy = test(X_test, Y_test, W)
print(accuracy)'''

# Evaluate with separate test and train
W = create_nodes(X, Y)
print(np.shape(W))

accuracy = test(test_X, test_Y, W)
print(accuracy)