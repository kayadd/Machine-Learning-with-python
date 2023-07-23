# Importiert matplotlib zum Visualisieren der Daten.
import matplotlib.pyplot as plt
# Importiert die logistische Regression von Python
from sklearn.linear_model import LogisticRegression
# Importiert alle Hilfsdateien um die Daten zu standardisieren
import preLearn

# Importiert die Iris-Daten
data = preLearn.getIris(0.3, [2, 3])

# Weißt die Daten ein.
y_train = data[1]
y_combined = data[5]
x_train_std = data[2]
x_combined_std = data[4]

# Initialisiert die logistische Regression mit dem
lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(x_train_std, y_train)

# Initialisiert den Plot.
preLearn.plot_decision_regions(x_combined_std, y_combined, classifier=lr, test_idx=range(105, 150), resolution=0.02)

#
plt.xlabel('Blütenblattbreite [standardisiert]')
# Setzt das Label der Y-Achse auf "Blütenblattlänge [standardisiert]".
plt.ylabel('Blütenblattlänge [standardisiert]')
# Platziert die Legende in der oberen linken Ecke.
plt.legend(loc='upper left')
# Visualisiert den Plot.
plt.show()
