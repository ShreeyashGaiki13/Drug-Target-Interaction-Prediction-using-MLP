import os
import pandas as pd
import numpy as np
import json

# Function to load and process DAVIS dataset
def load_process_DAVIS(path='./data', binary=False, convert_to_log=True, threshold=30):
    print('Beginning Processing...')

    # Set path to the DAVIS dataset directory
    DAVIS_DIR = path
    
    # Check if the DAVIS folder exists
    if not os.path.exists(DAVIS_DIR):
        raise FileNotFoundError(f"Path {DAVIS_DIR} does not exist. Please check the dataset location.")

    # Load the affinity values
    affinity_file = os.path.join(DAVIS_DIR, 'affinity.txt')
    affinity = pd.read_csv(affinity_file, header=None, sep=' ')

    # Load the target sequences (proteins)
    target_seq_file = os.path.join(DAVIS_DIR, 'target_seq.txt')
    with open(target_seq_file) as f:
        target = json.load(f)

    # Load the drug SMILES strings
    smiles_file = os.path.join(DAVIS_DIR, 'SMILES.txt')
    with open(smiles_file) as f:
        drug = json.load(f)

    # Convert drug and target dictionaries to lists
    target = list(target.values())
    drug = list(drug.values())

    SMILES = []
    Target_seq = []
    y = []

    # Process the drug-target interactions
    for i in range(len(drug)):
        for j in range(len(target)):
            SMILES.append(drug[i])
            Target_seq.append(target[j])
            y.append(affinity.values[i, j])

    # Apply binary threshold if required
    if binary:
        print(f"Converting affinities to binary values with threshold: {threshold}")
        y = [1 if i < threshold else 0 for i in np.array(y)]
    else:
        # Convert binding affinities to log values if requested
        if convert_to_log:
            print('Converting binding affinities to log space (nM -> pM) for easier regression')
            y = np.log10(np.array(y))
        else:
            y = y
    
    print('Data processing complete.')
    return np.array(SMILES), np.array(Target_seq), np.array(y)

# Function to save the processed data to CSV
def save_to_csv(SMILES, Target_seq, y, output_file='Davis.csv'):
    # Create a DataFrame with the data
    data = {
        'SMILES': SMILES,
        'Target_seq': Target_seq,
        'Binding_Affinity': y
    }
    
    df = pd.DataFrame(data)
    
    # Save the DataFrame to CSV
    df.to_csv(output_file, index=False)
    print(f"CSV file saved to {output_file}")

# Main function to load, process, and save the dataset
def main():
    # Path to the DAVIS folder
    davis_path = r'D://Major Project//MajorDTA//Dataset//Davis.csv'
    
    # Load and process the DAVIS dataset
    SMILES, Target_seq, y = load_process_DAVIS(path=davis_path, binary=False, convert_to_log=True)

    # Save the processed data to CSV
    save_to_csv(SMILES, Target_seq, y, output_file='Davis.csv')

if __name__ == '__main__':
    main()