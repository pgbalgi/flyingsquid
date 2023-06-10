'''
This example code shows a bare-minimum example of how to get FlyingSquid up and
running.

It generates synthetic data from the tutorials folder, and trains up a label
model.

You can run this file from the examples folder.
'''

from flyingsquid.label_model import LabelModel
from examples.tutorials.tutorial_helpers import *

L_train, L_dev, Y_dev = synthetic_data_basics()

def map_val_to_vector(val, num_classes=2, is_lambda = True):
    if is_lambda:
        if val == 1:
            vec = np.ones(num_classes) * -1
            vec[0] = 1
        elif val == -1:
            vec = np.ones(num_classes) * -1
            vec[1] = 1
        else: # val = 0
            vec = np.zeros(num_classes)
        return vec
    else:
        y = np.zeros(num_classes)
        if val == 1:
            y[0] = 1
        else:
            y[1] = 1
        return y
    
num_classes = 2

def map_L_to_vectors(L, num_classes):
    L_multiclass = np.zeros((L.shape[0], L.shape[1], num_classes))
    for i, row in enumerate(L):
        for j, val in enumerate(row):
            L_multiclass[i][j] = map_val_to_vector(val, num_classes=num_classes)

    return L_multiclass

def map_Y_to_vectors(Y, num_classes):
    Y_multiclass = np.zeros((Y.shape[0], num_classes))
    for i in range(len(Y)):
        Y_multiclass[i] = map_val_to_vector(Y[i], num_classes=num_classes, is_lambda=False)

    return Y_multiclass

# L_train_multiclass = np.zeros((L_train.shape[0], 5, 10))
# for i, row in enumerate(L_train):
#     for j, val in enumerate(row):
#         L_train_multiclass[i][j] = map_val_to_vector(val)

L_train_multiclass = map_L_to_vectors(L_train, num_classes=10)
L_dev_multiclass = map_L_to_vectors(L_dev, num_classes=10)
Y_dev_multiclass = map_Y_to_vectors(Y_dev, num_classes=10)
print(L_train_multiclass.shape)

d = L_train_multiclass.shape[2]
cb = np.ones(d)/d
label_model = LabelModel(cb)

# label_model.fit(L_train)
label_model.fit(L_train_multiclass)


print(Y_dev_multiclass.shape)
preds = label_model.predict(L_dev_multiclass).reshape(Y_dev_multiclass.shape)
accuracy = np.sum(preds * Y_dev_multiclass) / Y_dev_multiclass.shape[0]

print('Label model accuracy: {}%'.format(int(100 * accuracy)))
