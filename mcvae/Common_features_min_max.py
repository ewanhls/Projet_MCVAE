import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# Chargement des fichiers
# -----------------------------
file2 = "/data/projets/bio-int/Propre/Projet_MCVAE/Data_CPTAC/FeaturesCommunesPr/normal_transcriptomics_features_communes.tsv"
file1 = "/data/projets/bio-int/Propre/Projet_MCVAE/Data_CPTAC/FeaturesCommunesPr/Ccrcc_transcriptomics_features_communes.tsv"

df1 = pd.read_csv(file1, sep='\t', index_col=0)
df2 = pd.read_csv(file2, sep='\t', index_col=0)

print(f"Fichier 1 - Shape: {df1.shape}")
print(f"Fichier 2 - Shape: {df2.shape}")

# -----------------------------------
# Suppression colonnes parasites
# -----------------------------------
for df in [df1, df2]:
    cols_to_drop = [col for col in df.columns if 'ID' in str(col) or 'Original' in str(col)]
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
        print(f"Colonnes supprimées: {cols_to_drop}")

# -----------------------------------
# Trouver les gènes communs
# -----------------------------------
common_genes = sorted(set(df1.columns) & set(df2.columns))
print(f"\nNombre de gènes communs: {len(common_genes)}")

df1_common = df1[common_genes]
df2_common = df2[common_genes]

print(f"DF1 après filtrage: {df1_common.shape}")
print(f"DF2 après filtrage: {df2_common.shape}")

# -----------------------------------
# Min-Max scaling entre les 2 tableaux
# -----------------------------------
print("\n➡️ Applique MinMaxScaling sur les deux datasets combinés...")

# Concaténer verticalement
df_combined = pd.concat([df1_common, df2_common], axis=0)

# Appliquer min-max colonne par colonne
scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(df_combined)

# Recréer un dataframe avec le même index et colonnes
df_scaled = pd.DataFrame(scaled_values, index=df_combined.index, columns=df_combined.columns)

# Séparer comme avant
df1_scaled = df_scaled.loc[df1_common.index]
df2_scaled = df_scaled.loc[df2_common.index]

print("✔ Scaling terminé")
print(f"DF1 scaled: {df1_scaled.shape}")
print(f"DF2 scaled: {df2_scaled.shape}")

# -----------------------------------
# Sauvegarde
# -----------------------------------
output_dir = "/data/projets/bio-int/Propre/Projet_MCVAE/Data_CPTAC/Training/CCRCC/Data/"

df1_scaled.to_csv(output_dir + "Ccrcc_transcriptomics_features_communes_scaled.tsv", sep='\t')
df2_scaled.to_csv(output_dir + "normal_transcriptomics_features_communes_scaled.tsv", sep='\t')

print("\n✔ Fichiers sauvegardés :")
print(f"  - {output_dir}Ccrcc_proteomics_features_communes_scaled.tsv")
print(f"  - {output_dir}normal_proteomics_features_communes_scaled.tsv")

# Aperçu
print("\nAperçu DF1:")
print(df1_scaled.head())

print("\nAperçu DF2:")
print(df2_scaled.head())
