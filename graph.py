import pandas as pd
import matplotlib.pyplot as plt

# Chargement des données
episode_data = pd.read_csv('episode_performance.csv')

# # Graphique Évolution des Récompenses par Épisode :
# # Filtrer les données pour ne prendre que chaque 50ème épisode
# filtered_data = episode_data[episode_data['Episode'] % 250 == 0]
#
# # Création du graphique
# plt.figure(figsize=(10, 6))
#
# # Tracer la ligne pour les récompenses du policier
# plt.plot(filtered_data['Episode'], filtered_data['Police_Reward'], label='Récompense Policier', marker='o', color='blue')
#
# # Tracer la ligne pour les récompenses du voleur
# plt.plot(filtered_data['Episode'], filtered_data['Thief_Reward'], label='Récompense Voleur', marker='o', color='red')
#
# # Titre et légendes
# plt.title("Évolution des Récompenses par Épisode (tous les 250 épisodes)")
# plt.xlabel("Épisode")
# plt.ylabel("Récompense")
# plt.legend()

# Graphique pour la Fréquence des Gagnants
# Regrouper les données par intervalle de 250 épisodes et compter les victoires
win_counts = episode_data.groupby(episode_data['Episode'] // 100)['Winner'].value_counts().unstack().fillna(0)

# Création du graphique à barres
win_counts.plot(kind='bar', figsize=(10, 6))

# Titre et légendes
plt.title("Fréquence des Gagnants par Intervalle de 100 Épisodes")
plt.xlabel("Intervalle de 100 Épisodes")
plt.ylabel("Nombre de Victoires")
plt.xticks(rotation=0)  # Garder les étiquettes de l'axe des x horizontales
plt.legend(title="Gagnant")

# Afficher le graphique
plt.show()
