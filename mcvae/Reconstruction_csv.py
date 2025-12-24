import pandas as pd
import numpy as np

# -----------------------------
# 1. Charger les CSV
# -----------------------------
original_csv = "/chemin/vers/original.csv"
reconstructed_csv = "/chemin/vers/reconstructed.csv"

orig = pd.read_csv(original_csv, index_col=0)
rec = pd.read_csv(reconstructed_csv, index_col=0)

# Vérification des dimensions et index/colonnes
assert orig.shape == rec.shape, "Les fichiers n'ont pas la même taille !"
assert all(orig.index == rec.index), "Les indices (lignes) ne correspondent pas !"
assert all(orig.columns == rec.columns), "Les colonnes ne correspondent pas !"

# -----------------------------
# 2. Calculer la différence
# -----------------------------
diff = rec - orig  # différence élément par élément

# -----------------------------
# 3. Calculer l'erreur standard
# -----------------------------
# Par ligne
std_row = diff.std(axis=1)
# Par colonne
std_col = diff.std(axis=0)
# Global
std_global = diff.values.flatten().std()

print("Erreur standard globale:", std_global)

# -----------------------------
# 4. Sauvegarder la différence
# -----------------------------
diff.to_csv("/chemin/vers/difference.csv")

# Optionnel : sauvegarder les écarts types par colonne ou ligne
# std_row.to_csv("/chemin/vers/std_row.csv")
# std_col.to_csv("/chemin/vers/std_col.csv")
