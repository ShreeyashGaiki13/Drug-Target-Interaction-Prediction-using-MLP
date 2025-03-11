import pandas as pd
import numpy as np
from encoder import encode

def preprocess_data(file_path):
    # Load the original dataset
    data = pd.read_csv(file_path)

    # Extract unique drugs (SMILES) and targets (protein sequences)
    unique_drugs = data['SMILES'].unique()
    unique_targets = data['Target_seq'].unique()

    # Create mappings for drug and target to unique IDs
    drug_to_id = {drug: idx for idx, drug in enumerate(unique_drugs)}
    target_to_id = {target: idx for idx, target in enumerate(unique_targets)}

    # Save the mappings to CSV files
    pd.DataFrame(list(drug_to_id.items()), columns=['SMILES', 'Drug_ID']).to_csv('D://Major Project//MajorDTA//Dataset//drug_ids.csv', index=False)
    pd.DataFrame(list(target_to_id.items()), columns=['Target_seq', 'Target_ID']).to_csv('D://Major Project//MajorDTA//Dataset//target_ids.csv', index=False)

    # Pre-calculate encoded vectors for each unique drug and target
    drug_encodings = []
    target_encodings = []

    for drug in unique_drugs:
        drug_vector = encode(drug, 'drug')
        drug_encodings.append(drug_vector)

    for target in unique_targets:
        target_vector = encode(target, 'target')
        target_encodings.append(target_vector)

    # Save the encoded vectors to CSV files
    np.save('D://Major Project//MajorDTA//Dataset//drug_encodings.npy', np.array(drug_encodings))
    np.save('D://Major Project//MajorDTA//Dataset//target_encodings.npy', np.array(target_encodings))

if __name__ == "__main__":
    preprocess_data("D://Major Project//MajorDTA//Dataset//Davis.csv")
