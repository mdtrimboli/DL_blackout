import pandas as pd
import numpy as np
import Split_Process as sp
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from sklearn.metrics import confusion_matrix
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


#  ================ Creation of Neural Network ================

def create_model():
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(12, ), name="Input_Layer"))
    model.add(keras.layers.Dense(300, activation='relu', name="Hidden_1"))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(50, activation='relu', name="Hidden_2"))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(20, activation='relu', name="Hidden_3"))
    model.add(keras.layers.Dense(2, activation='softmax', name="Output_Layer"))
    #model.summary()
    return model


model = create_model()
model.compile(loss="sparse_categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])

log_dir = "C:/Users/maxit/Documents/Facultad/Carrera de Doctorado - Beca/Cursos/ML UNRC/Trabajo Final/Folder/logs/fit/"\
          + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

history = model.fit(x=XTrain, y=yTrain,
                    validation_split=0.15, epochs=150, batch_size=10,
                    verbose=0, callbacks=[tensorboard_callback])

train_loss = history.history['loss'][-1]
val_acc = history.history['val_accuracy'][-1]
train_acc = history.history['accuracy'][-1]
val_loss = history.history['val_loss'][-1]
print("Training Accuracy")
print(train_acc)
print("Validation Accuracy")
print(val_acc)
print("Training Loss")
print(train_loss)
print("Validation Loss")
print(val_loss)

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
print(matrixConfusion)

