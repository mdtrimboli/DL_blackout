import pandas as pd
import numpy as np
import Split_Process as sp
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report


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
model.add(keras.layers.Bidirectional(keras.layers.LSTM(20, return_sequences=True), input_shape=(XTrain.shape[1:])))
model.add(keras.layers.Bidirectional(keras.layers.LSTM(20, return_sequences=True)))
model.add(keras.layers.Bidirectional(keras.layers.LSTM(20, return_sequences=True)))
model.add(keras.layers.Bidirectional(keras.layers.LSTM(20)))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(50, activation='relu'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(2, activation='softmax'))
model.summary()

model.compile(loss="sparse_categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])
history = model.fit(x=XTrain, y=yTrain, epochs=200, batch_size=50, validation_data=(XTest, yTest), verbose=2)

train_loss = history.history['loss'][-1]
val_acc = history.history['val_acc'][-1]
train_acc = history.history['acc'][-1]
val_loss = history.history['val_loss'][-1]
print("Training Accuracy")
print(train_acc)
print("Validation Accuracy")
print(val_acc)
print("Training Loss")
print(train_loss)
print("Validation Loss")
print(val_loss)

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)                                         # Rango de Y
plt.show()

#  ================ Prediction ==============

yProb = model.predict(XTest, batch_size=5)
evaluation = model.evaluate(XTest, yTest)

yPredict = np.zeros(2500)

for i in range(2500):
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
