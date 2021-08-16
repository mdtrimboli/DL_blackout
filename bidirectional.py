import pandas as pd
import numpy as np
import Split_Process as sp
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report, precision_score, f1_score, recall_score


# =========== Loading and Visualizing Data =============

print("Loading and Visualizing Data")
data = pd.read_csv("Data_for_UCI_named.csv")
random_list = data.sample(n=5)
print(random_list)
x_data = data.drop(['stab', 'stabf'], axis=1)
y_data = data.iloc[:, -1]


# =========== Transform Label =============

y_data = y_data.replace(['stable'], 1)
y_data = y_data.replace(['unstable'], 0)


# ================ Split sets ================

mmscaler = MinMaxScaler()
X_scaled = mmscaler.fit_transform(x_data)                               # Normalize database
XTrain, XTest, yTrain, yTest = sp.splitX(X_scaled, y_data)

XTrain = XTrain.reshape(len(XTrain), 1, XTrain.shape[1])
XTest = XTest.reshape(len(XTest), 1, XTest.shape[1])

#  ================ Creation of Neural Network ================

model = keras.Sequential()
model.add(keras.layers.Bidirectional(keras.layers.LSTM(120, return_sequences=True), input_shape=(XTrain.shape[1:])))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Bidirectional(keras.layers.LSTM(120, return_sequences=True)))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Bidirectional(keras.layers.LSTM(120, return_sequences=True)))
model.add(keras.layers.Dense(120, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(20, activation='relu'))
model.add(keras.layers.Dense(5, activation='relu'))
model.add(keras.layers.Dense(2, activation='softmax'))
model.summary()

model.compile(loss="sparse_categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])

log_dir = "C:/Users/maxit/Documents/Facultad/Carrera de Doctorado - Beca/Cursos/ML UNRC/Trabajo Final/Folder/logs/fit/" \
          + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

# Para activar: tensorboard --logdir logs/fit

#history = model.fit(x=XTrain, y=yTrain, epochs=150, batch_size=50, validation_data=(XTest, yTest), verbose=2)

cross_val_round = 1

for train_index, val_index in KFold(10, shuffle=True, random_state=10).split(XTrain):
    x_train, y_train = XTrain[train_index], yTrain.iloc[train_index]
    x_val, y_val = XTrain[val_index], yTrain.iloc[val_index]
    history = model.fit(x_train, y_train, epochs=50, callbacks=[tensorboard_callback])
    print(f'\nModel evaluation - Round {cross_val_round}: {model.evaluate(x_val, y_val)}\n')
    cross_val_round += 1


#  ================ Prediction ==============

yProb = model.predict(XTest)
evaluation = model.evaluate(XTest, yTest)

yPredict = np.zeros(1500)

for i in range(1500):
    max_index_predict = np.argmax(yProb[i, :])
    yPredict[i] = max_index_predict


print("Evaluation Accuracy")
print(evaluation[1])

matrixConfusion = confusion_matrix(yTest, yPredict)
print('Confusion Matrix')
print('TP,FP // FN, TN')
print(matrixConfusion)

matrix = classification_report(yTest, yPredict, labels=[1, 0])
print('Classification report : \n', matrix)

precision_model = precision_score(yTest, yPredict)
print('Precision')
print(precision_model)

recall_model = recall_score(yTest, yPredict)
print('Recall')
print(recall_model)

f1_model = f1_score(yTest, yPredict)
print('F1 Score')
print(recall_model)
