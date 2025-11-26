import torch
import pandas as pd
import numpy as np
from pathlib import Path

from mcvae.models import Mcvae
from mcvae.models.utils import DEVICE, load_or_fit
from mcvae.utilities import ltonumpy
from mcvae.diagnostics import plot_loss

print(f"Using device: {DEVICE}")


# -----------------------------------
# 1. Load your ALREADY preprocessed omics
# -----------------------------------
rna = pd.read_csv("/data/projets/bio-int/Propre/Projet_MCVAE/Data_CPTAC/FeaturesCommunes/Ccrcc_transcriptomics_features_communes.tsv", sep='\t', index_col=0)
rppa = pd.read_csv("/data/projets/bio-int/Propre/Projet_MCVAE/Data_CPTAC/FeaturesCommunes/Ccrcc_proteomics_features_communes.tsv", sep='\t', index_col=0)

rna = rna.drop(columns=['Original_ID'])
rppa = rppa.drop(columns=['Original_ID'])  # si tu as aussi cette colonne dans RPPA

print("RNA shape:", rna.shape)
print("RPPA shape:", rppa.shape)

print(rna.head())      # inspecte les valeurs
print(rna.dtypes)       # vérifie le type de chaque colonne
print(rna.values.dtype) # dtype global

assert all(rna.index == rppa.index), "Les lignes doivent représenter les mêmes échantillons !"

# -----------   ------------------------
# 2. Convert to tensors (no preprocessing!)
# -----------------------------------
x = [
    torch.tensor(rna.values, dtype=torch.float32).to(DEVICE),
    torch.tensor(rppa.values, dtype=torch.float32).to(DEVICE),
]

X = [c.to(DEVICE) for c in x] if torch.cuda.is_available() else x

# -----------------------------------
# 3. Create model
# -----------------------------------

#Hyperparameters

adam_lr = 1e-3
nb_epochs = 20000

fit_lat_dims = 5

# --- Créer le modèle
models = {}
models['smcvae'] = Mcvae(data=X, lat_dim=fit_lat_dims, sparse=True)
model = models['smcvae']
model.optimizer = torch.optim.Adam(params=model.parameters(), lr=adam_lr)
model.to(DEVICE)

# --- Train / load
ptfile = Path('/data/projets/bio-int/Propre/Projet_MCVAE/mcvae/Trained_models/first_try.pt')
load_or_fit(model, model.data, epochs=nb_epochs, ptfile=ptfile)

# --- Outputs
q = model.encode(X)
z = np.array([q[i].mean.detach().cpu().numpy() for i in range(len(X))]).reshape(-1)
X_hat = model.reconstruct(X, dropout_threshold=0.1)

plot_loss(model)