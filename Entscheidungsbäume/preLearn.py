# Importiert matplotlib zum Visualisieren der Daten.
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
# Importiert sklearn.
import sklearn
# Importiert das Datenset für die Blumen.
from sklearn import datasets
# Importiert NumPy.
import numpy as np

# Importiert die Datenselektion.
import sklearn.model_selection
# Importiert das Tree-Modul.
import sklearn.tree
# Importiert das Ensemble-Modul.
import sklearn.ensemble

def getIris(p, arg):
    """Lädt den Testdatensatz der Iris-Blumen herunter und spaltet den Datensatz in Test- und Trainingsdaten auf.
    p: float
    p gibt die Prozentzahl der Testdatensätze an.
    arg: list
    arg gibt die jeweiligen Features des Datensatzes an.
    """
    # Lädt das Datenset aus der Sci-Kit-Bibliothek herunter.
    iris = datasets.load_iris()
    # Nimmt sich die jeweiligen Features aus dem Datensatz.
    x = iris.data[:, arg]
    # Nimmt sich die Klassenbezeichnungen.
    y = iris.target

    # Spaltet den Datensatz in Testdaten und Trainingsdaten auf.
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        x, y, test_size=p, random_state=0)

    # Lädt die Normiereinheit der Sci-Kit-Bibliothek.
    sc = sklearn.preprocessing.StandardScaler()
    # Ermittelt den Wert des Erwartungswertes und der Standardabweichung.
    sc.fit(x_train)
    # Normiert die Test- und Trainingsdaten.
    x_train_std = sc.transform(x_train)
    x_test_std = sc.transform(x_test)

    # Kombiniert die Test- und Trainingsdaten zu einem Datensatz.
    x_combined_std = np.vstack((x_train_std, x_test_std))
    y_combined = np.hstack((y_train, y_test))

    return [x_train, y_train, x_train_std, x_test_std, x_combined_std, y_combined, x_test, y_test]


def std(x, y, p):
    """Lädt den Testdatensatz der Iris-Blumen herunter und spaltet den Datensatz in Test- und Trainingsdaten auf.
        x: list
        x ist die Liste der ausgewählten Daten.
        y: list
        y ist die Liste der Klassenbezeichnungen.
        p: float
        p gibt die Prozentzahl der Testdatensätze an.
        """
    # Spaltet den Datensatz in Testdaten und Trainingsdaten auf.
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        x, y, test_size=p, random_state=0)

    # Lädt die Normiereinheit der Sci-Kit-Bibliothek.
    sc = sklearn.preprocessing.StandardScaler()
    # Ermittelt den Wert des Erwartungswertes und der Standardabweichung.
    sc.fit(x_train)
    # Normiert die Test- und Trainingsdaten.
    x_train_std = sc.transform(x_train)
    x_test_std = sc.transform(x_test)

    # Kombiniert die Test- und Trainingsdaten zu einem Datensatz.
    x_combined_std = np.vstack((x_train_std, x_test_std))
    y_combined = np.hstack((y_train, y_test))

    return [x_train, y_train, x_train_std, x_test_std, x_combined_std, y_combined]


def stdSingle(x):
    """Lädt den Testdatensatz der Iris-Blumen herunter und spaltet den Datensatz in Test- und Trainingsdaten auf.
        x: list
        x ist die Liste der ausgewählten Daten.
        """
    # Spaltet den Datensatz in Testdaten und Trainingsdaten auf.

    # Lädt die Normiereinheit der Sci-Kit-Bibliothek.
    sc = sklearn.preprocessing.StandardScaler()
    # Ermittelt den Wert des Erwartungswertes und der Standardabweichung.
    sc.fit(x)
    # Normiert die Test- und Trainingsdaten.
    x_train_std = sc.transform(x)

    # Kombiniert die Test- und Trainingsdaten zu einem Datensatz.

    return x_train_std


def plot_decision_regions(X, Y, classifier, test_idx, resolution):
    """Plottet die Entscheidungsgrenzen."""
    # Initialisiert ein Tupel aus den Markierungen.
    markers = ("s", "x", "o", "^", "∨")

    # Initialisiert ein Tupel aus den verschiedenen Farben.
    colors = ("red", "blue", "lightgreen", "gray", "cyan")

    # Initialisiert die Colourmap.
    cmap = ListedColormap(colors[:len(np.unique(Y))])

    # Plotten der Entscheidungsgrenzen mithilfe der Maximal und Minimalwerte.
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # Initialisiert ein Skalarfeld mithilfe der Trainingsdaten.
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    # Trainiert das Neuron.
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)

    # Ordnet die Entscheidungsgrenze.
    Z = Z.reshape(xx1.shape)

    # Initialisiert einen neuen Konturen plot und setzen die Grenzen für den Plot.
    plt.contour(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Plotten aller Objekte.
    for idx, cl in enumerate(np.unique(Y)):
        plt.scatter(x=X[Y == cl, 0], y=X[Y == cl, 1], alpha=0.8, marker=markers[idx], label=cl)

    if test_idx:
        # plot all samples
        X_test, Y_test = X[test_idx, :], Y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    alpha=1.0,
                    linewidths=1,
                    marker='o',
                    s=55, label='test set')
