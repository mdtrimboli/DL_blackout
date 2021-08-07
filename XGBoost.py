import pandas as pd
import numpy as np
import Split_Process as sp
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
from xgboost import XGBClassifier


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
    model = XGBClassifier(learning_rate=0.009, min_child_weight=25, max_depth=5, gamma=0,
                          n_estimators=5000, colsample_bytree=0.8, colsample_bylevel=0.8,
                          colsample_bynode=0.8, subsample=0.5, reg_lambda=0.5, reg_alpha=0.01, nthread=6)
    return model


model = create_model()
model.fit(XTrain, yTrain)

scores = cross_val_score(model, XTrain, yTrain, cv=5)
print("Mean cross-validation score: %.2f" % scores.mean())

yTr_pred = model.predict(XTrain)
acc_train = accuracy_score(yTrain, yTr_pred)
print("Training Accuracy")
print(acc_train)

#  ================ Prediction ==============

yPred = model.predict(XTest)

acc_test = accuracy_score(yTest, yPred)
print("Evaluation Accuracy")
print(acc_test)

matrixConfusion = confusion_matrix(yTest, yPred)
print('Confusion Matrix')
print(matrixConfusion)

