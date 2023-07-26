# Importiert alle notwendigen Module.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_moons

# Generiert zufällige Punkte.
X, y = make_blobs(n_samples=150,
                  n_features=2,
                  centers=3,
                  cluster_std=0.5,
                  shuffle=True,
                  random_state=0)

# Initialisiert das K-Means-Modell mit einer vorgegebenen Anzahl von 3 Clustern, bei denen die Zentroiden zufällig
# ausgewählt werden. Die Grenze der Iterationen wird auf 300 gesetzt und der Fehler, ab dem der Fehler als 0 gilt
# wird auf 10^-4 gesetzt. Damit werden die Varianzdifferenzen begrenzt. n_init trainiert dabei 10 Modelle mit
# unterschiedlicher Clusterauswahl.
km = KMeans(n_clusters=3,
            init='random',
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0)
# Trainiert das Modell mit den Trainingsdaten.
y_km = km.fit_predict(X)

# Plottet die einzelnen Clusterpunkten, markiert mit der Clusterzugehörigkeit.
plt.scatter(X[y_km == 0, 0],
            X[y_km == 0, 1],
            s=50,
            c='lightgreen',
            marker='s',
            label='cluster 1')
plt.scatter(X[y_km == 1, 0],
            X[y_km == 1, 1],
            s=50,
            c='orange',
            marker='o',
            label='cluster 2')
plt.scatter(X[y_km == 2, 0],
            X[y_km == 2, 1],
            s=50,
            c='lightblue',
            marker='v',
            label='cluster 3')
plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1],
            s=250,
            marker='*',
            c='red',
            label='centroids')
# Fügt die Legende hinzu.
plt.legend()
# Fügt dem Plot ein Gitternetz hinzu.
plt.grid()
# Visualisiert den Plot.
plt.show()

# Erzeugt ein Halbmond-Datenset.
X, Y = make_moons(n_samples=200, noise=0.05, random_state=0)
# Fügt das Halbmond-Datenset dem Plot hinzu.
plt.scatter(X[:, 0], X[:, 1])
# Visualisiert den Plot.
plt.show()

# Initialisiert zwei Subplots.
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

# Initialisiert das Modell des k-Means-Clustering.
km = KMeans(n_clusters=2, random_state=0)
# Trainiert das Modell.
y_km = km.fit_predict(X)
# Fügt die Cluster dem Plot hinzu.
ax1.scatter(X[y_km == 0, 0], X[y_km == 0, 1],
            c='lightblue', marker='o', s=40, label='cluster 1')
ax1.scatter(X[y_km == 1, 0], X[y_km == 1, 1],
            c='red', marker='s', s=40, label='cluster 2')
# Setzt en Titel auf "k-Means-Cluster".
ax1.set_title('k-Means-Cluster')

# Initialisiert das Clustering mit jedem Objekt als Cluster mit zwei Clustern und der euklidischen Distanz als Maß.
ac = AgglomerativeClustering(n_clusters=2,
                             affinity='euclidean',
                             linkage='complete')
# Sortiert das Modell.
y_ac = ac.fit_predict(X)

# Fügt dem Plot die Punkte hinzu.
ax2.scatter(X[y_ac == 0, 0], X[y_ac == 0, 1], c='lightblue',
            marker='o', s=40, label='cluster 1')
ax2.scatter(X[y_ac == 1, 0], X[y_ac == 1, 1], c='red',
            marker='s', s=40, label='cluster 2')
# Setzt den Titel auf "Einzelcluster".
ax2.set_title('Einzelcluster')

# Fügt dem Plot die Legende hinzu.
plt.legend()
# Visualisiert den Plot.
plt.show()

# Initialisiert die Variablen als Labels.
variables = ['X', 'Y', 'Z']
labels = ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4']

# Generiert zufällige Daten.
X = np.random.random_sample([5, 3])*10

# Initialisiert einen Panda-Dataframe mit der Distanzmatrix, die für dem Algorithmus benötigt wird.
df = pd.DataFrame(X, columns=variables, index=labels)

# Initialisiert das Einzel-Clustering und führt den Algorithmus durch. Die Distanz wird als euklidisch gesetzt.
row_clusters = linkage(df.values, method='complete', metric='euclidean')

# Initialisiert das Dendrogramm.
row_dendr = dendrogram(row_clusters,
                       labels=labels,
                       )
# Setzt das Label der Y-Achse auf "Euklidische Distanz".
plt.ylabel('Euklidische Distanz')
plt.show()
