# Importiert Numpy für die Funktionen der linearen Algebra.
import numpy as np

# Importiert pandas für die Datenverarbeitung
import pandas as pd

# Importiert pyplot um die Daten graphisch darzustellen.
import matplotlib.pyplot as plt

# Importiert ListedColormap für die Visualisierung der Datengrenze
from matplotlib.colors import ListedColormap

# Importiert die Standardisierte Variante
import standardize

# Importiert die Testdaten
if True:
    # Importiert die Testdaten.
    df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)

    # Lädt die ersten 100 Zeilen und davon die Vierte Zelle aus, von dem sie dann die Werte abfragt.
    # Dies sind die richtigen Werte der Klassifizierung.
    y = df.iloc[0:100, 4].values

    # Lädt die ersten 100 Zeilen und davon das nullte und die zweite Zeile. Dies sind dann die Trainingsdaten.
    x = df.iloc[0:100, [0, 2]].values

    # Ersetzt die Bezeichnung der richtigen Blumenklasse durch die binären Werte der Heaviside-Funktion.
    y = np.where(y == "Iris-setosa", -1, 1)

    # Plottet die Daten der ersten 50 Datenpunkte, die als setosa markiert sind. Die beiden Argumente
    # sind dabei die Kelchblatt- und die Blütenblattlänge und stellt die als "o" in roter Farbe da.
    plt.scatter(x[0:50, 0], x[:50, 1], color="red", marker="o", label="setosa")

    # Plottet die Daten der Datenpunkte 50 bis 100, die als versicolor markiert sind. Die beiden Argumente
    # sind dabei die Kelchblatt- und die Blütenblattlänge und stellt die als "x" in blauer Farbe da.
    plt.scatter(x[50:100, 0], x[50:100, 1], color="blue", marker="x", label="versicolor")

    # Setzt das Label der X-Achse auf "Länge des Kelchblattes [cm]".
    plt.xlabel("Länge des Kelchblattes [cm]")

    # Setzt das Label der Y-Achse auf "Länge des Blütenblattes [cm]".
    plt.ylabel("Länge des Blüttenblattes [cm]")

    # Setzt die Legende in die Ecke oben links.
    plt.legend(loc="upper left")

    # Rendert den Plot.
    plt.show()

# Definiert eine Funktion zur Visualisierung der linearen Grenze.
if True:
    def plot_decision_regions(X, Y, classifier, resolution=0.02):
        # Markierungen und Farben werden eingestellt.

        # Initialisiert ein Tupel aus den Markierungen.
        markers = ("s", "x", "o", "^", "∨")

        # Initialisiert ein Tupel aus den verschiedenen Farben.
        colors = ("red", "blue", "lightgreen", "gray", "cyan")

        # Initialisiert die Colourmap.
        cmap = ListedColormap(colors[:len(np.unique(y))])

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
        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[Y == cl, 0], y=X[Y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)

# Die Funktionen werden zu Lernzwecken selber noch einmal programmiert und nachvollzogen.
if True:
    def dotP(X, Y):
        # Überprüft ob die Listen die gleichen Dimensionen haben.
        if len(X) == len(Y):
            dSum = 0
            for i in range(len(X)):
                dSum += X[i] * Y[i]
            return dSum
        return IndexError


    def Heaviside(X):
        # Gibt den Wert der Heaviside-Funktion für jede Zelle der Liste als Liste gleicher Dimension zurück.
        for i in range(len(X)):
            if X[i] >= 0:
                X[i] = 1
            else:
                X[i] = -1
        return X


class AdalineGD:
    """Perzeptron-Klassifizierer"""
    # Lernrate
    eta: float

    # Iterations
    n_iter: int

    # Gewichtungsarray nach Anpassungen
    w_: np.array

    # Anzahl der Fehler pro Epoche
    errors_: list

    eta = 0
    n_iter = 0

    def __init__(self, peta, pn_iter):
        self.eta = peta
        self.n_iter = pn_iter
        self.cost_ = []

    def fit(self, X, Y):
        """
        X: {array-like}, shape = [n_samples, n_features]
        Trainingsvektoren
        n_samples: Anzahl der Objekte
        n_features: Anzahl der Features

        y: shape = [n_samples]
        n_samples:
        """
        # Initialisiert einen Gewichtsarray der Länge von 1 + der.
        # Anzahl der Objekte.
        self.w_ = np.zeros(1 + X.shape[1])

        # Initialisiert die Liste für die Anzahl der Fehler pro Epoche.
        self.cost_ = []

        # Wiederholt so lange, wie Iterationen vorgegeben sind.
        for _ in range(self.n_iter):
            # Gibt den Term der Aktivierungsfunktion zurück
            output = self.net_input(X)

            # Initialisiert den Fehlerterm.
            errors = Y-output

            # Verändert die Gewichte durch Addition des Wertes der Lernregel.
            self.w_[1:] += self.eta * X.T.dot(errors)
            # Verändert den Schwellenwert.
            self.w_[0] += self.eta * errors.sum()

            cost = (errors**2).sum() / 2.0

            self.cost_.append(cost)

        return self

    def net_input(self, X):
        """Nettoeingabe berechnen"""

        # Berechnet das Skalarprodukt zwischen dem Vektor und den Gewichten. Die Gewichte sind alle, bis auf das
        # Erste nicht enthalten. Danach wird der Wert in Zelle 0 addiert um den ersten negativen Term des
        # Schwellenwertes Im Skalarprodukt wieder auszugleichen. Somit kann nun die Heaviside-Funktion genutzt werden,
        # Um die Differenz zum Schwellenwert zu bestimmen.

        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        """Klassenbezeichnung zurückgeben"""

        # Die Heaviside-Funktion; Gibt bei jedem positiven Wert oder Null eine Eins und bei jedem negativen Wert zurück.
        # Diese Resultate werden in einer Liste zurückgegeben.
        return np.where(self.activation(X) >= 0.0, 1, -1)


# Visualisiert den Fehler
if True:
    # Kreiert zwei Unterplots
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

    # Trainiert das Modell mit einer Lernrate von 0.01.
    ada1 = AdalineGD(pn_iter=10, peta=0.01).fit(x, y)
    # Setzt die Grenzen und das Symbol für diesen Plot.
    ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')

    # Setzt das Label der Y-Achse auf "Epochen".
    ax[0].set_xlabel('Epochen')
    # Setzt das Label der Y-Achse auf "logarithmierter Fehler".
    ax[0].set_ylabel("logarithmierter Fehler")
    # Setzt den Titel des Plots
    ax[0].set_title('Adaline - Lernrate von 0.01')

    # Trainiert das Modell mit einer Lernrate von 0.0001.
    ada2 = AdalineGD(pn_iter=10, peta=0.0001).fit(x, y)
    ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
    # Setzt das Label der Y-Achse auf "Epochen".
    ax[1].set_xlabel('Epochen')
    # Setzt das Label der Y-Achse auf "Fehler".
    ax[1].set_ylabel('Fehler')
    ax[1].set_title('Adaline - Lernrate von 0.0001')

    # Visualisiert den Plot.
    plt.show()

if True:
    # Standardisiert die Daten
    X_std = standardize.standardizeData(x)

    # Initialisiert das Modell
    ada = AdalineGD(pn_iter=15, peta=0.01)

    # Trainiert das Modell
    ada.fit(X_std, y)

    # Erstellt eine Colourmap
    plot_decision_regions(X_std, y, classifier=ada)
    plt.title('Adaline - Gradient Descent')
    # Setzt das Label der X-Achse auf "Kelchblattlänge [Standardisiert]".
    plt.xlabel('Kelchblattlänge [Standardisiert]')
    # Setzt das Label der Y-Achse auf "Blütenblattlänge [Standardisiert]".
    plt.ylabel('Blütenblattlänge [Standardisiert]')
    # Setzt die Legende in die Ecke oben links
    plt.legend(loc='upper left')
    # Zeigt den Plot
    plt.show()

    # Setzt die Grenzen des Plots
    plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
    # Setzt das Label der X-Achse auf "Epochen".
    plt.xlabel('Epochen')
    # Setzt das Label der Y-Achse auf "Fehler".
    plt.ylabel('Fehler')

    # Zeigt den Plot
    plt.show()
