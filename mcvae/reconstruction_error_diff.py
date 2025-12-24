import pandas as pd
from pathlib import Path

def compute_absolute_error(tsv1_path, tsv2_path, output_path):
    """
    Compute absolute error between two TSVs with patient rows and gene columns.
    Keeps patient names and gene names in the output TSV.
    
    :param tsv1_path: Path to first TSV
    :param tsv2_path: Path to second TSV
    :param output_path: Path to save the absolute error TSV
    """
    # Load TSVs with patient names as index
    df1 = pd.read_csv(tsv1_path, sep="\t", index_col=0)
    df2 = pd.read_csv(tsv2_path, sep="\t", index_col=0)

    # Find common patients
    common_patients = df1.index.intersection(df2.index)
    print(f"Number of common patients: {len(common_patients)}")

    # Keep only common patients
    df1_common = df1.loc[common_patients]
    df2_common = df2.loc[common_patients]

    # Check that number of columns match
    if df1_common.shape[1] != df2_common.shape[1]:
        raise ValueError(f"Number of genes mismatch: {df1_common.shape[1]} vs {df2_common.shape[1]}")

    # Compute absolute error
    df_error = (df1_common - df2_common).abs()

    # Re-add patient names as first column
    df_error.insert(0, 'Patient_ID', df_error.index)

    # Save to TSV
    df_error.to_csv(output_path, sep="\t", index=False)
    print(f"Absolute error saved to {output_path}, shape: {df_error.shape}")


# -----------------------------
# Exemple d'utilisation
# -----------------------------
if __name__ == "__main__":
    compute_absolute_error(
        "/data/projets/bio-int/Propre/Projet_MCVAE/Data_CPTAC/Training/CCRCC/04_12/Forward_sain/rna_normal_reconstruction_error.tsv",
        "/data/projets/bio-int/Propre/Projet_MCVAE/Data_CPTAC/Training/CCRCC/04_12/Forward_patho/rna_ccrcc_reconstruction_error.tsv",
        "/data/projets/bio-int/Propre/Projet_MCVAE/Data_CPTAC/Training/CCRCC/04_12/rna_reconstruction_error_diff.tsv"
    )
    compute_absolute_error(
        "/data/projets/bio-int/Propre/Projet_MCVAE/Data_CPTAC/Training/CCRCC/04_12/Forward_sain/rppa_normal_reconstruction_error.tsv",
        "/data/projets/bio-int/Propre/Projet_MCVAE/Data_CPTAC/Training/CCRCC/04_12/Forward_patho/rppa_ccrcc_reconstruction_error.tsv",
        "/data/projets/bio-int/Propre/Projet_MCVAE/Data_CPTAC/Training/CCRCC/04_12/rppa_reconstruction_error_diff.tsv"
    )
