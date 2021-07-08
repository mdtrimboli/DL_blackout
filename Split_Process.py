from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def splitX(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=0)
    fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)
    fig.suptitle('Histograma de entrenamiento y prueba')


    axs[0].hist(x=y_train, bins=2)
    axs[1].hist(x=y_test, bins=2)
    axs[2].hist(x=y_val, bins=2)
    #plt.show()
    return X_train, X_test, y_train, y_test, X_val, y_val
