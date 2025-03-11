import numpy as np
import torch
from keras.models import load_model
from encoder import encode  # Import the encode function from encoder.py

# Load pre-trained Keras model
model_path = "D://Major Project//MajorDTA//Model//MLP.keras"
model = load_model(model_path)

# Test with dummy input
smiles = "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4"  # Example SMILES string
protein = "MESLVFARRSGPTPSAAELARPLAEGLIKSPKPLMKKQAVKRHHHKHNLRHRYEFLETLGKGTYGKVKKARESSGRLVAIKSIRKDKIKDEQDLMHIRREIEIMSSLNHPHIIAIHEVFENSSKIVIVMEYASRGDLYDYISERQQLSEREARHFFRQIVSAVHYCHQNRVVHRDLKLENILLDANGNIKIADFGLSNLYHQGKFLQTFCGSPLYASPEIVNGKPYTGPEVDSWSLGVLLYILVHGTMPFDGHDHKILVKQISNGAYREPPKPSDACGLIRWLLMVNPTRRATLEDVASHWWVNWGYATRVGEQEAPHEGGHPGSDSARASMADWLRRSSRPLLENGAKVCSFFKQHAPGGGSTTPGLERQHSLKKSRKENDMAQSLHSDTADDTAHRPGKSNLKLPKGILKKKVSASAEGVQEDPPELSPIPASPGQAAPLLPKKGILKKPRQRESGYYSSPEPSESGELLDAGDVFVSGDPKEQKPPQASGLLLHRKGILKLNGKFSQTALELAAPTTFGSLDELAPPRPLARASRPSGAVSEDSILSSESFDQLDLPERLPEPPLRGCVSVDNLTGLEEPPSEGPGSCLRRWRQDPLGDSCFSLTDCQEVTATYRQALRVCSKLT"  # Example protein sequence

# Encode SMILES and protein sequence
drug_vector = encode(smiles, 'drug')  # Using the encode function from encoder.py for SMILES
protein_vector = encode(protein, 'target')  # Using the encode function from encoder.py for protein sequence

# Prepare input data for the model
input_data = np.concatenate([drug_vector, protein_vector])
input_data = input_data.reshape(1, -1)  # Reshape to (1, input_dim) for prediction

# Predict using the trained model
prediction = model.predict(input_data)

print(f"Predicted Binding Affinity: {prediction[0][0]:.4f}")