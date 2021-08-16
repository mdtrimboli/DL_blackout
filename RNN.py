import pandas as pd
import numpy as np
import Split_Process as sp
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from tensorflow import keras
from sklearn.metrics import confusion_matrix, precision_score, f1_score, recall_score
from tensorboard import main as tb


# =========== Loading and Visualizing Data =============

print("Loading and Visualizing Data")
data = pd.read_csv("Data_for_UCI_named.csv")
random_list = data.sample(n=5)
print(random_list)
# create excel writer object
writer = pd.ExcelWriter('output.xlsx')
# write dataframe to excel
random_list.to_excel(writer)
# save the excel
writer.save()
print('DataFrame is written successfully to Excel File.')
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

def create_model():
    model = keras.Sequential()
    model.add(keras.layers.SimpleRNN(120, input_shape=(XTrain.shape[1:]),
                                     return_sequences=True, name="Input_RecLay"))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.SimpleRNN(120, return_sequences=True, name="HideRecLay_1"))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.SimpleRNN(120, name="HideRecLay_2"))
    model.add(keras.layers.Dense(120, activation='relu', name="HideDense_1"))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(20, activation='relu', name="HideDense_2"))
    model.add(keras.layers.Dense(5, activation='relu', name="HideDense_3"))
    model.add(keras.layers.Dense(2, activation='softmax', name="Output_Layer"))
    return model


model = create_model()
model.compile(loss="sparse_categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])

log_dir = "C:/Users/maxit/Documents/Facultad/Carrera de Doctorado - Beca/Cursos/ML UNRC/Trabajo Final/Folder/logs/fit/"\
          + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

#Para activar: tensorboard --logdir logs/fit

cross_val_round = 1

for train_index, val_index in KFold(10, shuffle=True, random_state=10).split(XTrain):
    x_train, y_train = XTrain[train_index], yTrain.iloc[train_index]
    x_val, y_val = XTrain[val_index], yTrain.iloc[val_index]
    history = model.fit(x_train, y_train, epochs=150, callbacks=[tensorboard_callback])
    print(f'\nModel evaluation - Round {cross_val_round}: {model.evaluate(x_val, y_val)}\n')
    cross_val_round += 1

#history = model.fit(x=XTrain, y=yTrain, validation_split=0.15, epochs=150, batch_size=10, verbose=0, callbacks=[tensorboard_callback])

# val_acc = history.history['val_accuracy'][-1]


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
print(matrixConfusion)

precision_model = precision_score(yTest, yPredict)
print('Precision')
print(precision_model)

recall_model = recall_score(yTest, yPredict)
print('Recall')
print(recall_model)

f1_model = f1_score(yTest, yPredict)
print('F1 Score')
print(recall_model)

