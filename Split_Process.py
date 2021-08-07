from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def splitX(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    fig.suptitle('Histograma de entrenamiento y prueba')

    axs[0].hist(x=y_train, bins=2)
    axs[1].hist(x=y_test, bins=2)
    # plt.show()
    return X_train, X_test, y_train, y_test
