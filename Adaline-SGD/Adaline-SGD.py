# Importiert Numpy für die Funktionen der linearen Algebra.
import numpy as np

# Importiert pandas für die Datenverarbeitung.
import pandas as pd

# Importiert pyplot um die Daten graphisch darzustellen.
import matplotlib.pyplot as plt

# Importiert ListedColormap für die Visualisierung der Datengrenze.
from matplotlib.colors import ListedColormap

# Importiert die Pre-Data-Datei.
import preLearn

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


class AdalineSGD:
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

    shuffle = True
    random_state = None

    def __init__(self, peta, pn_iter, shuffle, random_state):
        self.w_initialized = False
        self.shuffle = shuffle
        self.eta = peta
        self.n_iter = pn_iter

        self.cost_ = []
        if random_state:
            np.random.seed(random_state)

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
        self._initialize_weights(X.shape[1])

        # Initialisiert die Fehlerliste
        self.cost_ = []

        # Wiederholt so lange, wie Iterationen vorgegeben sind.
        for _ in range(self.n_iter):
            # Wenn das Shuffeln vorgegeben ist, wird der Datensatz geshuffelt.
            if self.shuffle:
                X, Y = self._shuffle(X, Y)
            cost = []
            # Aktualisiert die Gewichte mithilfe der Trainingsdaten.
            for xi, target in zip(X, Y):
                cost.append(self._update_weights(xi, target))
            # Berechnet den durchschnittlichen Fehlerterm und speichert ihn
            # für jede Epoche.
            avg_cost = sum(cost) / len(Y)
            self.cost_.append(avg_cost)

        return self

    def partial_fit(self, X, Y):
        """Trainiert das Modell ohne die Gewichte neu initialisieren zu müssen."""
        # Wenn die Gewichte des Modells nicht initialisiert sind, werden sie initialisiert.
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        # Wenn die Länge der ersten Liste im der zusammengefügten Liste Y größer als 1 ist
        if Y.ravel().shape[0] > 1:
            # Werden die Gewichte aus der Liste aktualisiert
            for xi, target in zip(X, Y):
                self._update_weights(xi, target)
        # Sind es nur einzelne Werte und keine Liste, mit der das Modell aktualisiert wird, werden
        # diese einfach zum aktualisieren genutzt.
        else:
            self._update_weights(X, Y)
        return self

    def _shuffle(self, X, Y):
        """Permutiert die Trainingsdaten."""
        r = np.random.permutation(len(Y))
        return X[r], Y[r]

    def _initialize_weights(self, m):
        """Initialisiert die Gewichte."""
        # Erschafft einen neuen Gewichtsarray und setzt die Werte auf initialisiert.
        self.w_ = np.zeros(1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """Aktualisiert die Gewichte."""
        # Berechnet die Aktivierungsfunktion
        output = self.net_input(xi)
        # Berechnet die Abweichung vom erwarteten Wert
        error = (target - output)
        # Aktualisiert jedes Gewicht mit der Lernrate multipliziert mit dem Fehler.
        self.w_[1:] += self.eta * xi.dot(error)
        # Aktualisiert den Schwellenwert
        self.w_[0] += self.eta * error
        # Speichert die Gewichtsänderung als Art der Varianz.
        cost = 0.5 * error**2
        # Gibt die Gewichtsänderung zurück.
        return cost

    def net_input(self, X):
        """Rechnet den Input aus."""
        # Gibt den Aktivierungsterm zurück.
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation"""
        # Gibt die Aktivierungsfunktion aus.
        return self.net_input(X)

    def predict(self, X):
        """Gibt die Vorhersage des Adaline-Modells aus."""
        # Entscheidet bei jedem Wert mithilfe der Heaviside-Funktion die Vorhersage.
        return np.where(self.activation(X) >= 0.0, 1, -1)


# Standardisiert die Daten.
X_std = preLearn.stdSingle(x)

# Initialisiert das Modell.
ada = AdalineSGD(pn_iter=30, peta=0.01, random_state=1, shuffle=True)
# Trainiert das Modell mit den Daten.
ada.fit(X_std, y)

# Initialisiert und speichert den Contourplot.
plot_decision_regions(X_std, y, classifier=ada)
# Setzt den Titel des Plots auf den Namen des Modells.
plt.title('Adaline - Stochastic Gradient Descent')
# Setzt das Label der X-Achse auf "Kelchlänge [standardisiert]".
plt.xlabel('Kelchlänge [standardisiert]')
# Setzt das Label der Y-Achse auf "Blütenblattlänge [standardisiert]".
plt.ylabel('Blütenblattlänge [standardisiert]')
# Platziert die Legende in der oberen linken Ecke.
plt.legend(loc='upper left')
# Visualisiert den Plot.
plt.show()

# Setzt die Grenzen des Plots auf die Grenzen der Kostenliste
plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
# Setzt das Label der X-Achse auf "Epochen".
plt.xlabel('Epochen')
# Setzt das Label der Y-Achse auf "Durchschnittlicher Korrekturterm".
plt.ylabel('Durchschnittlicher Korrekturterm')

# Visualisiert den Plot.
plt.show()

# Trainiert das Modell nachträglich mit weiteren Daten.
ada = ada.partial_fit(X_std[0, :], y[0])
