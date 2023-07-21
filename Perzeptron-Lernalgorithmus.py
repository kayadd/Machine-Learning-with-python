# Importiert Numpy für die Funktionen der linearen Algebra.
import numpy as np

# Die Funktionen werden zu Lernzwecken selber noch einmal programmiert und nachvollzogen.


def dotP(X, Y):
    # Überprüft ob die Listen die gleichen Dimensionen haben.
    if len(X) == len(Y):
        dSum = 0
        for i in range(len(X)):
            dSum += X[i]*Y[i]
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


class Perceptron:
    """Perzeptron-Klassifizierer"""
    # Lernrate
    eta: float

    # Iterations
    n_iter: int

    # Gewichtungsarray nach Anpassungen
    w_: np.array

    # Anzahl der Fehler pro Epoche
    errors_: list

    def __int__(self, peta, pn_iter):
        self.eta, self.n_iter = 0.1, 10

    def fit(self, X, y):
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
        self.errors_ = []

        # Wiederholt so lange, wie Iterationen vorgegeben sind.
        for _ in range(self.n_iter):
            # Initialisiert den Fehlerterm.
            errors = 0

            # xi ist der Trainingsvektor.
            # target ist der angepeilte Zielwert für den Vektor.
            for xi, target in zip(X, y):
                # Nutzt die Lernregel des Perzeptrons um die Gewichte zu aktualisieren.
                update = self.eta * (target - self.predict(xi))

                # Verändert die Gewichte durch Addition des Wertes der Lernregel.
                self.w_[1:] += update * xi
                # Lässt den Schwellenwart aber unangetastet
                self.w_[0] += update

                # Passt den Fehler an, wenn er vorhanden ist.
                errors += int(update != 0.0)

            # Fügt den Fehlerterm der Liste hinzu.
            self.errors_.append(errors)

        return self

    def net_input(self, X):
        """Nettoeingabe berechnen"""

        # Berechnet das Skalarprodukt zwischen dem Vektor und den Gewichten. Die Gewichte sind alle, bis auf das
        # Erste nicht enthalten. Danach wird der Wert in Zelle 0 addiert um den ersten negativen Term des
        # Schwellenwertes Im Skalarprodukt wieder auszugleichen. Somit kann nun die Heaviside-Funktion genutzt werden,
        # Um die Differenz zum Schwellenwert zu bestimmen.

        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Klassenbezeichnung zurückgeben"""

        # Die Heaviside-Funktion; Gibt bei jedem positiven Wert oder Null eine Eins und bei jedem negativen Wert zurück.
        # Diese Resultate werden in einer Liste zurückgegeben.
        return np.where(self.net_input(X) >= 0.0, 1, -1)
