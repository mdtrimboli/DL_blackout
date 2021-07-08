import pandas as pd
import numpy as np
import Split_Process as sp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from sklearn.metrics import confusion_matrix


# =========== Loading and Visualizing Data =============

print("Loading and Visualizing Data")
data = pd.read_csv("Data_for_UCI_named.csv")
print(data[:5])
x_data = data.drop(['stab', 'stabf'], axis=1)
y_data = data.iloc[:, -1]

x_data_tau = x_data.iloc[:, 0:3]
x_data_g = x_data[["g1", "g2", "g3", "g4"]]

data['tau_system'] = data['tau1'] * data['tau2'] * data['tau3'] * data['tau4']
data['g_system'] = data['g1'] * data['g2'] * data['g3'] * data['g4']
fig, ax = plt.subplots(1, 2)
sns.boxplot(x='tau_system', y='stabf', data=data, ax=ax[0])
sns.boxplot(x='g_system', y='stabf', data=data, ax=ax[1])
ax[0].set_title("Tau Data")
ax[1].set_title("G Data")


# =========== Transform Label =============

y_data = y_data.replace(['stable'], 1)
y_data = y_data.replace(['unstable'], 0)


# ================ Split sets ================

mmscaler = MinMaxScaler()
X_scaled = mmscaler.fit_transform(x_data)                               # Normalize database
XTrain, XTest, yTrain, yTest, XVal, yVal = sp.splitX(X_scaled, y_data)


#  ================ Creation of Neural Network ================

model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(12, )))
model.add(keras.layers.Dense(300, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(50, activation='relu'))
model.add(keras.layers.Dense(2, activation='softmax'))
model.summary()

model.compile(loss="sparse_categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])
history = model.fit(x=XTrain, y=yTrain, epochs=30, batch_size=10, validation_data=(XVal, yVal), verbose=2)

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

yProb = model.predict(XTest, batch_size=10)
evaluation = model.evaluate(XTest, yTest)

yPredict = np.zeros(2000)

for i in range(2000):
    max_index_predict = np.argmax(yProb[i, :])
    yPredict[i] = max_index_predict


print("Evaluation Accuracy")
print(evaluation[1])

matrixConfusion = confusion_matrix(yTest, yPredict)
print('Confusion Matrix')
print(matrixConfusion)
