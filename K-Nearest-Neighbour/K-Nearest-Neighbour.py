import sklearn.neighbors
import preLearn
import matplotlib.pyplot as plt

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

# Initialisiert das K-Nearest-Neighbour-Modell mit der Abstandsbestimmung der Minkowski-Metrik, der 5 nächsten Nachbarn
# p als Manhatten- oder euklidische Distanz.
knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
# Trainiert das Modell.
knn.fit(x_train_std, y_train)

# Speichert den Contourplot.
preLearn. plot_decision_regions(x_combined_std, y_combined,
                                classifier=knn, test_idx=range(105, 150), resolution=0.02)


# Setzt das Label der Y-Achse auf "Blütenblattlänge [standardisiert]".
plt.xlabel('Blütenblattlänge [standardisiert]')
# Setzt das Label der Y-Achse auf "Blütenblattbreite [standardisiert]".
plt.ylabel('Blütenblattbreite [standardisiert]')
# Platziert die Legende in der oberen linken Ecke.
plt.legend(loc='upper left')
# Visualisiert den Plot.
plt.show()
