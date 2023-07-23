# Importiert und fasst damit die Funktionen zur Datenvorbereitung.
import preLearn
# Importiert NumPy für die lineare Algebra.
import numpy as np
# Importiert Matplotlib für die Visualisierung der Daten.
import matplotlib.pyplot as plt
# Importiert Sci-Kit für die Modelle.
import sklearn

# Importiert die Iris-Daten
data = preLearn.getIris(0.3, [2, 3])

# Weißt die Daten ein
x_train = data[0]
y_train = data[1]
y_combined = data[5]
x_train_std = data[2]
x_combined_std = data[4]
x_test = data[6]
y_test = data[7]


# Errechnet den Gini-Koeffizienten.
def gini(p):
    return p * (1 - p) + (1 - p) * (1 - (1 - p))


# Errechnet die Entropie
def entropy(p):
    return - p * np.log2(p) - (1 - p) * np.log2((1 - p))


# Errechnet die Fehlklassifizierungen.
def error(p):
    return 1 - np.max([p, 1 - p])


# Initialisiert das Entscheidungsbaum-Modell.
tree = sklearn.tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
# Trainiert das Modell.
tree.fit(x_train, y_train)

# Kombiniert die Test- und Trainingsdaten.
X_combined = np.vstack((x_train, x_test))

# Speichert den Contourplot.
preLearn.plot_decision_regions(X_combined, y_combined,
                               classifier=tree, test_idx=range(105, 150), resolution=0.02)


# Setzt das Label der Y-Achse auf "Blütenblattlänge [standardisiert]".
plt.xlabel('Blütenblattlänge [standardisiert]')
# Setzt das Label der Y-Achse auf "Blütenblattbreite [standardisiert]".
plt.ylabel('Blütenblattbreite [standardisiert]')
# Platziert die Legende in der oberen linken Ecke.
plt.legend(loc='upper left')
# Visualisiert den Plot.
plt.show()

# Initialisiert das Forest-Modell.
forest = sklearn.ensemble.RandomForestClassifier(criterion='entropy',
                                                 n_estimators=10,
                                                 random_state=1,
                                                 n_jobs=2)
# Trainiert das Modell.
forest.fit(x_train, y_train)

# Speichert den Contourplot.
preLearn.plot_decision_regions(X_combined, y_combined,
                               classifier=forest, test_idx=range(105, 150), resolution=0.02)

# Setzt das Label der Y-Achse auf "Blütenblattlänge [standardisiert]".
plt.xlabel('Blütenblattlänge [standardisiert]')
# Setzt das Label der Y-Achse auf "Blütenblattbreite [standardisiert]".
plt.ylabel('Blütenblattbreite [standardisiert]')
# Platziert die Legende in der oberen linken Ecke.
plt.legend(loc='upper left')
# Visualisiert den Plot.
plt.show()
