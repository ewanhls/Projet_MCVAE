import pandas as pd
from pathlib import Path

def fix_csv_with_tsv(csv_path, tsv_path, output_path):
    """
    Drop first line of CSV, then prepend first row and first column of TSV.
    Save as TSV.
    """
    # Load files
    df_csv = pd.read_csv(csv_path, header=None)
    df_tsv = pd.read_csv(tsv_path, sep="\t", header=None)

    # Print initial shapes
    print(f"{csv_path} shape (before): {df_csv.shape}")
    print(f"{tsv_path} shape (before): {df_tsv.shape}")

    # Drop first line of CSV
    df_csv = df_csv.iloc[1:].reset_index(drop=True)

    # Extract first row and first column of TSV
    first_row = df_tsv.iloc[0:1, 1:]   # exclude TSV first column for header
    first_col = df_tsv.iloc[1:, 0].reset_index(drop=True)  # exclude TSV first row

    # Update CSV: prepend first row as header
    df_csv.columns = first_row.values[0]

    # Insert first column
    df_csv.insert(0, df_tsv.iloc[0,0], first_col)

    # Save as TSV
    df_csv.to_csv(output_path, sep="\t", index=False)

    # Print final shape
    print(f"{output_path} shape (after): {df_csv.shape}")
    print("-"*50)


if __name__ == "__main__":
    # ---- CHANGE PATHS ----
    rna_csv = Path("/data/projets/bio-int/Propre/Projet_MCVAE/Data_CPTAC/Training/CCRCC/11_12/Forward_CCRCC/rna_normal_reconstructed.csv")
    rna_orig_tsv = Path("/data/projets/bio-int/Propre/Projet_MCVAE/Data_CPTAC/Training/CCRCC/Data/normal_transcriptomics_features_communes_scaled.tsv")
    rna_out_tsv = Path("/data/projets/bio-int/Propre/Projet_MCVAE/Data_CPTAC/Training/CCRCC/11_12/Forward_CCRCC/rna_normal_reconstructed.tsv")

    rppa_csv = Path("/data/projets/bio-int/Propre/Projet_MCVAE/Data_CPTAC/Training/CCRCC/11_12/Forward_CCRCC/rppa_normal_reconstructed.csv")
    rppa_orig_tsv = Path("/data/projets/bio-int/Propre/Projet_MCVAE/Data_CPTAC/Training/CCRCC/Data/normal_proteomics_features_communes_scaled.tsv")
    rppa_out_tsv = Path("/data/projets/bio-int/Propre/Projet_MCVAE/Data_CPTAC/Training/CCRCC/11_12/Forward_CCRCC/rppa_normal_reconstructed.tsv")

    fix_csv_with_tsv(rna_csv, rna_orig_tsv, rna_out_tsv)
    fix_csv_with_tsv(rppa_csv, rppa_orig_tsv, rppa_out_tsv)
