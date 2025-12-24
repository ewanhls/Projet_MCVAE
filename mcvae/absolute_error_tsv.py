import pandas as pd
from pathlib import Path

def compute_absolute_error(tsv1_path, tsv2_path, output_path):
    """
    Compute absolute error between two TSVs, considering first column as patient IDs.
    
    :param tsv1_path: Path to first TSV (predicted / reconstructed)
    :param tsv2_path: Path to second TSV (ground truth / original)
    :param output_path: Path to save absolute error TSV
    """
    # Load TSVs with patient IDs as index
    df1 = pd.read_csv(tsv1_path, sep="\t", index_col=0)
    df2 = pd.read_csv(tsv2_path, sep="\t", index_col=0)

    # Check that shapes match
    if df1.shape != df2.shape:
        raise ValueError(f"Shape mismatch: {tsv1_path} is {df1.shape}, {tsv2_path} is {df2.shape}")

    # Compute absolute error
    df_error = (df1 - df2).abs()

    # Re-add patient IDs as first column
    df_error.insert(0, 'Patient_ID', df_error.index)

    # Save as TSV
    df_error.to_csv(output_path, sep="\t", index=False)
    print(f"Absolute error saved to {output_path}, shape: {df_error.shape}")


if __name__ == "__main__":
    # Exemple d'utilisation
    compute_absolute_error("/data/projets/bio-int/Propre/Projet_MCVAE/Data_CPTAC/Training/CCRCC/Data/normal_transcriptomics_features_communes.tsv",
                            "/data/projets/bio-int/Propre/Projet_MCVAE/Data_CPTAC/Training/CCRCC/04_12/Forward_sain/rna_normal_reconstructed.tsv", 
                            "/data/projets/bio-int/Propre/Projet_MCVAE/Data_CPTAC/Training/CCRCC/04_12/Forward_sain/rna_normal_reconstruction_error.tsv")
    compute_absolute_error("/data/projets/bio-int/Propre/Projet_MCVAE/Data_CPTAC/Training/CCRCC/Data/normal_proteomics_features_communes.tsv",
                            "/data/projets/bio-int/Propre/Projet_MCVAE/Data_CPTAC/Training/CCRCC/04_12/Forward_sain/rppa_normal_reconstructed.tsv", 
                            "/data/projets/bio-int/Propre/Projet_MCVAE/Data_CPTAC/Training/CCRCC/04_12/Forward_sain/rppa_normal_reconstruction_error.tsv")
