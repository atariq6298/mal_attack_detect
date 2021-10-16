from rnn_1 import Rnn
from cnn import Cnn
from dt import Dt
import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, roc_curve
from matplotlib import pyplot
dataset = load_dataset.dataset()
obj_rnn = Rnn(load_dataset.matrix_dataset(p_dataset=dataset))
rnn_model = obj_rnn.model

obj_cnn = Cnn(load_dataset.matrix_dataset(p_dataset=dataset))
cnn_model = obj_rnn.model

obj_dt = Dt(dataset)
dt_model = obj_dt.model

threshold = 0.5
prediction = np.empty(obj_dt.prediction.shape)
weights = (0.4, 0.3, 0.3)
predictions = (obj_cnn.prediction, obj_dt.prediction, obj_rnn.prediction)
for i in range(obj_dt.prediction.shape[0]):
    avg = (predictions[0][i] * weights[0] + predictions[1][i] * weights[1] + predictions[2][i] * weights[2])

    val = 1 if avg > threshold else 0
    prediction[i] = val

X_train, X_test, y_train, y_test = dataset
y_test = np.argmax(y_test, axis=1)
print('\n Accuracy: ')
print(accuracy_score(y_test, prediction))
print('\n F1 score: ')
print(f1_score(y_test, prediction))
print('\n Recall: ')
print(recall_score(y_test, prediction))
print('\n Precision: ')
print(precision_score(y_test, prediction))
print('\n confusion matrix: \n')
print(confusion_matrix(y_test, prediction))

