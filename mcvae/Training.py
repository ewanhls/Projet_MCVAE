import torch
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

from mcvae.models import Mcvae
from mcvae.models.vae import ThreeLayersVAE
from mcvae.models.utils import DEVICE, load_or_fit
from mcvae.utilities import ltonumpy
from mcvae.diagnostics import plot_loss
import MultiChannelDataset
from torch.utils.data import Dataset, DataLoader


print(f"Using device: {DEVICE}")

class MultiChannelDataset(Dataset):
    def __init__(self, data_list):
        # Convertir chaque DataFrame en numpy, puis en tensor
        self.data_list = [torch.tensor(d.values, dtype=torch.float32) if isinstance(d, pd.DataFrame) else torch.tensor(d, dtype=torch.float32)
                          for d in data_list]
        self.n_samples = self.data_list[0].shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return [d[idx] for d in self.data_list]


# -----------------------------------
# 1. Load your raw omics
# -----------------------------------
rna = pd.read_csv(
    "/data/projets/bio-int/Propre/Projet_MCVAE/Data_CPTAC/Training/CCRCC/Data/normal_transcriptomics_features_communes_scaled.tsv",
    sep='\t',
    index_col=0
)
rppa = pd.read_csv(
    "/data/projets/bio-int/Propre/Projet_MCVAE/Data_CPTAC/Training/CCRCC/Data/normal_proteomics_features_communes_scaled.tsv",
    sep='\t',
    index_col=0
)

test_size = 0.15
val_size = 0.15


# 1) Séparation train + (val+test)
train_rna, temp_rna = train_test_split(rna, test_size=test_size + val_size, random_state=42)

# 2) Séparation val + test
relative_test_size = test_size / (test_size + val_size)

val_rna, test_rna = train_test_split(temp_rna, test_size=relative_test_size, random_state=42)

# 1) Séparation train + (val+test)
train_rppa, temp_rppa = train_test_split(rppa, test_size=test_size + val_size, random_state=42)

# 2) Séparation val + test
relative_test_size = test_size / (test_size + val_size)

val_rppa, test_rppa = train_test_split(temp_rppa, test_size=relative_test_size, random_state=42)



print("RNA shapes:", rna.shape)
print("Train:", train_rna.shape)
print("Validation:", val_rna.shape)
print("Test:", test_rna.shape)

print("\nRPPA shapes:", rppa.shape)
print("Train:", train_rppa.shape)
print("Validation:", val_rppa.shape)
print("Test:", test_rppa.shape)


# Vérification que les échantillons correspondent
assert all(rna.index == rppa.index), "Les lignes doivent représenter les mêmes échantillons !"



print("\n=== Vérification NaN ===")
print(f"RNA NaN count: {np.isnan(train_rna.values).sum()}")
print(f"RPPA NaN count: {np.isnan(train_rppa.values).sum()}")

# -----------------------------------
# 3. Convert to tensors
# -----------------------------------

train_set = MultiChannelDataset([train_rna, train_rppa])
val_set   = MultiChannelDataset([val_rna, val_rppa])
test_set  = MultiChannelDataset([test_rna, test_rppa])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_set, batch_size=32)
test_loader  = DataLoader(test_set, batch_size=32)


# -----------------------------------
# 4. Create model
# -----------------------------------

adam_lr = 1e-3
epochs = 3000
fit_lat_dims = 70

models = {}
# Séparer en test et en train
models['smcvae'] = Mcvae(data=[train_rna,train_rppa], lat_dim=fit_lat_dims, sparse=True, vae_class=ThreeLayersVAE)
model = models['smcvae']
model.optimizer = torch.optim.Adam(params=model.parameters(), lr=adam_lr)

# Initialisation de l'historique des pertes
model.loss = {key: [] for key in ["total", "kl", "ll"]}
# Optionnel si tu veux sauvegarder cet historique à la fin
model.init_loss = model.loss


# # Ajouter après la création de l'optimizer
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

model.to(DEVICE)

# -----------------------------------
# 5. Train / load
# -----------------------------------

best_val_loss = float("inf")
save_path = Path("/data/projets/bio-int/Propre/Projet_MCVAE/Data_CPTAC/Training/CCRCC/11_12")
save_path.mkdir(exist_ok=True)


for epoch in range(epochs):

    # --------------------
    # TRAINING
    # --------------------
    model.train()
    train_loss = 0

    for batch in train_loader:
        batch = [x.to("cuda") for x in batch]

        model.optimizer.zero_grad()
        forward_ret = model(batch)
        loss = model.loss_function(forward_ret)
        loss.backward()
        model.optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # --------------------
    # VALIDATION
    # --------------------
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = [x.to("cuda") for x in batch]
            forward_ret = model(batch)
            losses = model.loss_function(forward_ret)
            val_loss += losses["total"].item()

    val_loss /= len(val_loader)

    print(f"Epoch {epoch+1} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

    # --------------------
    # SAVE BEST MODEL
    # --------------------
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), save_path / "best_model.pt")
        print("  ✔ New best model saved.")

# -----------------------------------
# 6. Outputs
# -----------------------------------

model.load_state_dict(torch.load(save_path / "best_model.pt"))
model.eval()

all_mu = []

# with torch.no_grad():
#     for batch in train_loader:
#         batch = [x.to(DEVICE) for x in batch]
#         q = model.encode(batch)
#         mu = q[0].mean(dim=0)  # q[0] correspond au tenseur latent
#         all_mu.append(mu.cpu())

# z = torch.cat(all_mu, dim=0).numpy()
# np.save(save_path / "z_latent.npy", z)

test_save_path = save_path / "test_dataset"
test_save_path.mkdir(exist_ok=True)

# Sauvegarde RNA
test_rna.to_csv(test_save_path / "test_rna.tsv", sep='\t', index=True)

# Sauvegarde RPPA
test_rppa.to_csv(test_save_path / "test_rppa.tsv", sep='\t', index=True)

print(f"✔ Test datasets saved in: {test_save_path}")

# -----------------------------
# RECONSTRUCTIONS
# -----------------------------
all_recons = [ [] for _ in range(len(train_set.data_list)) ]

with torch.no_grad():
    for batch in train_loader:
        batch = [x.to(DEVICE) for x in batch]
        X_hat = model.reconstruct(batch, dropout_threshold=0.1)

        for i, Xi_hat in enumerate(X_hat):
            all_recons[i].append(Xi_hat.cpu())

# concat and save
for i in range(len(all_recons)):
    Xi = torch.cat(all_recons[i], dim=0).numpy()
    np.save(save_path / f"Xhat_view{i}.npy", Xi)

# -----------------------------
# LOSS HISTORY (si disponible)
# -----------------------------
if hasattr(model, "init_loss"):
    np.save(save_path / "loss.npy", np.array(model.init_loss))

# -----------------------------
# SAVE FINAL MODEL WEIGHTS
# -----------------------------
torch.save(model.state_dict(), save_path / "model_weights_final.pt")

print("✔ All outputs saved in:", save_path)


# q = model.encode(X)
# z = np.array([q[i].mean.detach().cpu().numpy() for i in range(len(X))]).reshape(-1)
# X_hat = model.reconstruct(X, dropout_threshold=0.1)

# # Save results
# output_dir = Path("/data/projets/bio-int/Propre/Projet_MCVAE/Data_CPTAC/Training/CCRCC/11_12")
# output_dir.mkdir(exist_ok=True)

# # Latent space
# np.save(output_dir / "z_latent.npy", z)

# # Reconstructions
# for i, Xi_hat in enumerate(X_hat):
#     np.save(output_dir / f"Xhat_view{i}.npy", Xi_hat.detach().cpu().numpy())

# # Loss
# # print([attr for attr in dir(model) if 'loss' in attr])
# np.save(output_dir / "loss.npy", np.array(model.init_loss))

# # Model weights
# torch.save(model.state_dict(), output_dir / "model_weights.pt")

# print("✔ All outputs saved in:", output_dir)

