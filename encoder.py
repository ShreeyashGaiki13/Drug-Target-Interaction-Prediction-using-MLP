import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

# SMILES Encoder using Morgan Fingerprints
def encode_smiles(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)
    else:
        raise ValueError(f"Invalid SMILES string: {smiles}")

# Protein Sequence Encoder using ProtTrans (ESM2)
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
model = AutoModel.from_pretrained("facebook/esm2_t12_35M_UR50D")

def encode_protein(sequence):
    inputs = tokenizer(sequence, return_tensors="pt", truncation=True, padding=True, max_length=1000)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# General Encode Function
def encode(sequence, sequence_type):
    if sequence_type == 'drug':
        return encode_smiles(sequence)
    elif sequence_type == 'target':
        return encode_protein(sequence)
    else:
        raise ValueError("Invalid sequence_type. Must be 'drug' or 'target'.")
