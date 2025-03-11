import tkinter as tk
from tkinter import ttk, messagebox
from keras.models import load_model
import numpy as np
from encoder import encode

# Load pre-trained model
model_path = "D://Major Project//MajorDTA//Model//MLP.keras"
model = load_model(model_path)

# Function to predict binding affinity
def predict_binding_affinity():
    drug_smiles = drug_input.get()
    protein_seq = protein_input.get()
    
    if not drug_smiles or not protein_seq:
        messagebox.showerror("Input Error", "Please provide both SMILES and Protein Sequence!")
        return
    
    try:
        drug_vector = encode(drug_smiles, 'drug')
        protein_vector = encode(protein_seq, 'target')
        input_data = np.concatenate([drug_vector, protein_vector]).reshape(1, -1)
        prediction = model.predict(input_data)
        result.set(f"Predicted Binding Affinity: {prediction[0][0]:.4f}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to predict binding affinity: {str(e)}")

# Function to reset all fields
def reset_fields():
    drug_input.set("")
    protein_input.set("")
    result.set("")

# Create main application window
app = tk.Tk()
app.title("Drug-Target Binding Affinity Predictor")
app.geometry("600x500")
app.resizable(True, True)

# Configure grid resizing
app.grid_rowconfigure(0, weight=1)
app.grid_columnconfigure(0, weight=1)

# Main content frame
content_frame = tk.Frame(app)
content_frame.grid(row=0, column=0, sticky="nsew")
content_frame.grid_rowconfigure(0, weight=1)
content_frame.grid_columnconfigure(0, weight=1)

# Heading with blue background
header = tk.Label(
    content_frame,
    text="Drug-Target Binding Affinity Predictor",
    font=("Arial", 18, "bold"),
    pady=10,
    bg="blue",
    fg="white"
)
header.pack(fill="x")

# Description box with border
description_frame = tk.Frame(content_frame, highlightbackground="black", highlightthickness=1, padx=10, pady=10)
description_frame.pack(padx=20, pady=10, fill="x")
description = tk.Label(
    description_frame,
    text=("This project predicts the binding affinity between drugs and protein targets using a "
          "deep learning-based model. Enter the SMILES representation of the drug and the amino acid "
          "sequence of the target protein to obtain the binding value."),
    wraplength=500, justify="center"
)
description.pack()

# Drug input
drug_input_label = tk.Label(content_frame, text="Drug SMILES:", font=("Arial", 12), anchor="w")
drug_input_label.pack(fill="x", padx=20)
drug_input = tk.StringVar()
drug_input_entry = ttk.Entry(content_frame, textvariable=drug_input, font=("Arial", 12))
drug_input_entry.pack(fill="x", padx=20, pady=5)

# Protein input
protein_input_label = tk.Label(content_frame, text="Protein Sequence:", font=("Arial", 12), anchor="w")
protein_input_label.pack(fill="x", padx=20)
protein_input = tk.StringVar()
protein_input_entry = ttk.Entry(content_frame, textvariable=protein_input, font=("Arial", 12))
protein_input_entry.pack(fill="x", padx=20, pady=5)

# Predict button
predict_button = ttk.Button(content_frame, text="Predict Binding Affinity", command=predict_binding_affinity)
predict_button.pack(pady=10)

# Output display
result = tk.StringVar()
result_label = tk.Label(content_frame, textvariable=result, font=("Arial", 14, "bold"), fg="blue")
result_label.pack(pady=10)

# Reset button
reset_button = ttk.Button(content_frame, text="Reset", command=reset_fields)
reset_button.pack(pady=10)

# Footer with team names
footer = tk.Label(
    content_frame,
    text="Team Members: Shreeyash Gaiki, Omkar Ingole, Chaitreya Shrawankar, Deep Khaut",
    font=("Arial", 10), pady=10, fg="gray"
)
footer.pack(side="bottom")

# Run the application
app.mainloop()