import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from tensorflow import keras
from keras.models import sequential
from keras.layers import Dense
import load_dataset

X_train, X_test, y_train, y_test = load_dataset.dataset()

#X_train = X_train.to_numpy().reshape(-1, 1, 16)
#X_test = X_test.to_numpy().reshape(-1, 1, 16)
#y_train = y_train.reshape(-1, 1, 2)
#y_test = y_test.reshape(-1, 1, 2)



model = keras.Sequential()
model.add(Dense(16, activation = 'relu', input_shape=(X_test, y_test,1)))
model.add(Dense(2, activation = 'relu',))
model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)

#model.add(Dense(output_dim = 8, init = 'uniform', activation = 'relu', input_dim=16))
#model.add(Dense(output_dim = 8, init = 'uniform', activation = 'relu'))
#model.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
#model.add(Dense(output_dim=6, init='uniform', activation='relu'))
#model.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
#model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
#model.fit(X_train, y_train, batch_size=100, nb_epoch=150)

# Add a LSTM layer with 128 internal units.
# model.add(layers.LSTM(128))

# Add a Dense layer with 10 units.
# model.add(layers.Dense(2))

# model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])

# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1)

print(model.predict(X_test[:4]))
prediction = np.argmax(model.predict(X_test), axis=1)
y_test = np.argmax(y_test.reshape(2200, 2), axis=1)  # 2200 original
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
