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
rna = pd.read_csv("/data/projets/bio-int/Data_TCGA/RNAseq_sains.csv", index_col=0)
rppa = pd.read_csv("/data/projets/bio-int/Data_TCGA/RPPA_sains.csv", index_col=0)

print("RNA shape:", rna.shape)
print("RPPA shape:", rppa.shape)

assert all(rna.index == rppa.index), "Les lignes doivent représenter les mêmes échantillons !"

# -----------------------------------
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

#Model

model = {}

model['smcvae'] = Mcvae(data=X, lat_dim=fit_lat_dims, sparse=True)
model.optimizer = torch.optim.Adam(params=model.parameters(), lr=adam_lr)
model.to(DEVICE)

# -----------------------------------
# 4. Train or load existing weights
# -----------------------------------

ptfile = Path('/data/projets/bio-int/MCVAE/mcvae2/Trained_models/first_try.pt')
load_or_fit(model, model.data, epochs=nb_epochs, ptfile=ptfile)



# -----------------------------------
# 5. Outputs
# -----------------------------------


#Récupération du latent
q = model.encode(X)
z = np.array([q[i].mean.detach().cpu().numpy() for i in range(len(X))]).reshape(-1)


# reconstructions
X_hat = model.reconstruct(X, dropout_threshold=0.1)

plot_loss(model)