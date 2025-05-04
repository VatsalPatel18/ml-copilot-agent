import numpy as np
import pandas as pd

def generate_synthetic_data():
    # Parameters
    n_samples = 1000  # 800 cancer, 200 normal (approx.)
    n_genes = 1000    # Transcriptomic features
    n_mutations = 100 # Genomic features
    n_methylations = 100  # Epigenomic features
    n_cancer_types = 5
    n_subtypes_per_type = 3
    n_drugs = 3

    # Sample IDs
    sample_ids = [f"Sample_{i:04d}" for i in range(n_samples)]

    # Clinical Data
    # Cancer vs. Normal (80% cancer, 20% normal)
    is_cancer = np.random.choice([0, 1], n_samples, p=[0.2, 0.8])
    
    # Cancer types (for cancer samples only)
    cancer_types = np.where(is_cancer, 
                           np.random.choice([f"Type_{i}" for i in range(n_cancer_types)], n_samples), 
                           "Normal")
    
    # Subtypes (for cancer samples only)
    subtypes = np.where(is_cancer, 
                        [f"{ct}_Subtype_{np.random.randint(1, n_subtypes_per_type+1)}" 
                         if ct != "Normal" else "N/A" for ct in cancer_types], 
                        "N/A")
    
    # Survival time and censoring (for cancer samples only)
    survival_time = np.where(is_cancer, np.random.exponential(scale=100, size=n_samples), np.nan)
    censored = np.where(is_cancer, np.random.choice([0, 1], n_samples, p=[0.3, 0.7]), np.nan)
    
    # Drug responses (binary, for cancer samples only)
    drug_responses = {f"Drug_{i}": np.where(is_cancer, np.random.choice([0, 1], n_samples), np.nan) 
                      for i in range(n_drugs)}

    # Transcriptomic Data (Gene Expression)
    expression = np.random.normal(loc=0, scale=1, size=(n_samples, n_genes))
    
    # Patterns for Task 1: Cancer Diagnosis (Genes 0-99 upregulated in cancer)
    expression[is_cancer == 1, 0:100] += 2
    
    # Patterns for Task 1 & 6: Cancer Types (Genes 100-349, 50 genes per type)
    for i in range(n_cancer_types):
        ct = f"Type_{i}"
        mask = (cancer_types == ct)
        gene_start = 100 + i * 50
        gene_end = gene_start + 50
        expression[mask, gene_start:gene_end] += 2
    
    # Patterns for Task 1: Subtype Classification (Genes 350-649, 20 genes per subtype)
    unique_subtypes = [f"Type_{i}_Subtype_{j}" for i in range(n_cancer_types) 
                       for j in range(1, n_subtypes_per_type+1)]
    for k, st in enumerate(unique_subtypes):
        mask = (subtypes == st)
        gene_start = 350 + k * 20
        gene_end = gene_start + 20
        expression[mask, gene_start:gene_end] += 2
    
    # Patterns for Task 2: Survival Prediction (Genes 650-699 correlated with survival)
    survival_time_cancer = survival_time[is_cancer == 1]
    survival_time_norm = (survival_time_cancer - survival_time_cancer.mean()) / survival_time_cancer.std()
    a = np.random.uniform(-1, 1, size=50)
    for j in range(50):
        gene_idx = 650 + j
        expression[is_cancer == 1, gene_idx] += a[j] * survival_time_norm + np.random.normal(0, 0.5, sum(is_cancer))
    
    # Patterns for Task 4 & 10: Drug Response (Genes 700-849, 50 genes per drug)
    for i in range(n_drugs):
        drug_resp = drug_responses[f"Drug_{i}"]
        gene_start = 700 + i * 50
        gene_end = gene_start + 50
        mask = (is_cancer == 1) & (drug_resp == 1)
        expression[mask, gene_start:gene_end] += 2
    
    # Patterns for Task 8: TME Analysis (Genes 850-899 reflect immune infiltration)
    # Simulate immune-rich vs. immune-poor tumors
    immune_rich = np.random.choice([0, 1], n_samples, p=[0.5, 0.5])
    expression[immune_rich == 1, 850:900] += 1.5
    
    # Add Noise, Missing Values, and Outliers
    expression += np.random.normal(0, 0.5, expression.shape)  # Noise
    missing_mask = np.random.rand(*expression.shape) < 0.05  # 5% missing
    expression[missing_mask] = np.nan
    outlier_mask = np.random.rand(*expression.shape) < 0.01  # 1% outliers
    expression[outlier_mask] *= np.random.choice([5, 10], size=outlier_mask.sum())

    # Genomic Data (Mutations)
    mutation_data = np.zeros((n_samples, n_mutations), dtype=int)
    # Mutations specific to cancer types (0-99, 20 mutations per type)
    for i in range(n_cancer_types):
        ct = f"Type_{i}"
        mask = (cancer_types == ct)
        mut_start = i * 20
        mut_end = mut_start + 20
        mutation_data[mask, mut_start:mut_end] = np.random.binomial(1, 0.5, size=(sum(mask), 20))

    # Epigenomic Data (Methylation, 0-1 scale)
    methylation = np.random.beta(a=2, b=5, size=(n_samples, n_methylations))
    # Patterns for Task 5: Early Detection (Sites 0-19 hypermethylated in cancer)
    methylation[is_cancer == 1, 0:20] += 0.3
    methylation = np.clip(methylation, 0, 1)
    # Add imperfections
    methylation += np.random.normal(0, 0.1, methylation.shape)
    methylation[np.random.rand(*methylation.shape) < 0.05] = np.nan
    methylation = np.clip(methylation, 0, 1)

    # Feature Names
    gene_names = [f"Gene_{i:04d}" for i in range(n_genes)]
    mutation_names = [f"Mut_{i:03d}" for i in range(n_mutations)]
    methyl_names = [f"Methyl_{i:03d}" for i in range(n_methylations)]

    # DataFrames
    expression_df = pd.DataFrame(expression, index=sample_ids, columns=gene_names)
    mutation_df = pd.DataFrame(mutation_data, index=sample_ids, columns=mutation_names)
    methylation_df = pd.DataFrame(methylation, index=sample_ids, columns=methyl_names)
    
    clinical_data = {
        "is_cancer": is_cancer,
        "cancer_type": cancer_types,
        "subtype": subtypes,
        "survival_time": survival_time,
        "censored": censored,
        "immune_rich": immune_rich
    }
    for i in range(n_drugs):
        clinical_data[f"drug_response_{i}"] = drug_responses[f"Drug_{i}"]
    clinical_df = pd.DataFrame(clinical_data, index=sample_ids)

    # Save to CSV
    expression_df.to_csv("synthetic_expression.csv")
    mutation_df.to_csv("synthetic_mutations.csv")
    methylation_df.to_csv("synthetic_methylation.csv")
    clinical_df.to_csv("synthetic_clinical.csv")

    print("Data generated and saved to CSV files:")
    print("- synthetic_expression.csv")
    print("- synthetic_mutations.csv")
    print("- synthetic_methylation.csv")
    print("- synthetic_clinical.csv")

if __name__ == "__main__":
    generate_synthetic_data()