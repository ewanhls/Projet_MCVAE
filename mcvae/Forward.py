import torch
import pandas as pd
from pathlib import Path
from mcvae.models import Mcvae
from mcvae.models.utils import DEVICE
from mcvae.models.vae import ThreeLayersVAE

# -----------------------------
# 1. Load preprocessed inputs
# -----------------------------
rppa = pd.read_csv("/data/projets/bio-int/Propre/Projet_MCVAE/Data_CPTAC/Training/CCRCC/11_12/test_dataset/test_rppa.tsv", sep="\t", index_col=0)
rna = pd.read_csv("/data/projets/bio-int/Propre/Projet_MCVAE/Data_CPTAC/Training/CCRCC/11_12/test_dataset/test_rna.tsv", sep="\t", index_col=0)

print("RNA shapes:", rna.shape)

print("\nRPPA shapes:", rppa.shape)

X = [
    torch.tensor(rna.values, dtype=torch.float32).to(DEVICE),
    torch.tensor(rppa.values, dtype=torch.float32).to(DEVICE),
]

# -----------------------------
# 2. Load trained model
# -----------------------------
ptfile = Path("/data/projets/bio-int/Propre/Projet_MCVAE/Data_CPTAC/Training/CCRCC/11_12/best_model.pt")

model = Mcvae(
    data=X,
    lat_dim=70,
    sparse=True,
    vae_class=ThreeLayersVAE
)
model.load_state_dict(torch.load(ptfile, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# -----------------------------
# 3. Forward pass
# -----------------------------
with torch.no_grad():
    out = model(X)

# -----------------------------
# 4. Extract latent space
# -----------------------------
mu = [q.loc.cpu().numpy() for q in out["q"]]

pd.DataFrame(mu[0]).to_csv("/data/projets/bio-int/Propre/Projet_MCVAE/Data_CPTAC/Training/CCRCC/11_12/Forward_CCRCC/latent_rna.csv", index=False)
pd.DataFrame(mu[1]).to_csv("/data/projets/bio-int/Propre/Projet_MCVAE/Data_CPTAC/Training/CCRCC/11_12/Forward_CCRCC/latent_rppa.csv", index=False)

# -----------------------------
# 5. Extract reconstructions
# -----------------------------
# p[x][z] : reconstruction of channel x using latent z
rna_recon = out["p"][0][0].loc.cpu().numpy()
rppa_recon = out["p"][1][1].loc.cpu().numpy()

pd.DataFrame(rna_recon).to_csv("/data/projets/bio-int/Propre/Projet_MCVAE/Data_CPTAC/Training/CCRCC/11_12/Forward_CCRCC/rna_normal_reconstructed.csv", index=False)
pd.DataFrame(rppa_recon).to_csv("/data/projets/bio-int/Propre/Projet_MCVAE/Data_CPTAC/Training/CCRCC/11_12/Forward_CCRCC/rppa_normal_reconstructed.csv", index=False)

print("DONE")
