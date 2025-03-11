import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from sklearn.metrics import mean_squared_error
import time

# Load pre-calculated encodings
drug_encodings = np.load('D://Major Project//MajorDTA//Dataset//drug_encodings.npy')
target_encodings = np.load('D://Major Project//MajorDTA//Dataset//target_encodings.npy')

# Load mappings from unique IDs
drug_to_id = pd.read_csv('D://Major Project//MajorDTA//Dataset//drug_ids.csv', index_col=0).to_dict()['Drug_ID']
target_to_id = pd.read_csv('D://Major Project//MajorDTA//Dataset//target_ids.csv', index_col=0).to_dict()['Target_ID']

def load_data(file_path):
    data = pd.read_csv(file_path)
    smiles = data['SMILES'].values
    targets = data['Target_seq'].values
    binding_affinity = data['Binding_Affinity'].values
    return smiles, targets, binding_affinity

def prepare_data(smiles, targets):
    encoded_data = []
    for sm, tgt in zip(smiles, targets):
        try:
            # Fetch drug and target IDs
            drug_id = drug_to_id[sm]
            target_id = target_to_id[tgt]
            
            # Fetch the pre-calculated encoded vectors
            drug_vector = drug_encodings[drug_id]
            target_vector = target_encodings[target_id]
            
            # Concatenate drug and target vectors
            combined_vector = np.concatenate([drug_vector, target_vector])
            encoded_data.append(combined_vector)
        except Exception as e:
            print(f"Error encoding data: {e}")
    return np.array(encoded_data)

def build_mlp_model(input_dim):
    model = Sequential([
        Dense(2048, activation='relu', input_dim=input_dim, kernel_regularizer='l2'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1024, activation='relu', kernel_regularizer='l2'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(512, activation='relu', kernel_regularizer='l2'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(256, activation='relu', kernel_regularizer='l2'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(128, activation='relu', kernel_regularizer='l2'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse', metrics=['mse'])
    return model

def lr_schedule(epoch):
    initial_lr = 0.0005
    drop = 0.5
    epochs_drop = 10.0
    lr = initial_lr * np.power(drop, np.floor((1 + epoch) / epochs_drop))
    return lr
    

def train_model(train_file, test_file, model_path):
    start_time = time.time()
    
    # Load and encode training data
    print("Loading training data...")
    train_smiles, train_targets, train_binding_affinity = load_data(train_file)
    print("Encoding training data...")
    X_train = prepare_data(train_smiles, train_targets)
    y_train = np.array(train_binding_affinity[:len(X_train)])
    
    # Load and encode testing data
    print("Loading testing data...")
    test_smiles, test_targets, test_binding_affinity = load_data(test_file)
    print("Encoding testing data...")
    X_test = prepare_data(test_smiles, test_targets)
    y_test = np.array(test_binding_affinity[:len(X_test)])
    
    # Build the model
    model = build_mlp_model(X_train.shape[1])
    
    # Learning rate scheduler and callbacks
    lr_scheduler = LearningRateScheduler(lr_schedule)
    early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=1)

    # Train the model
    print("Training the model...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=64,
        callbacks=[early_stopping, checkpoint, lr_scheduler],
        verbose=1
    )
    
    # Load the best model
    model.load_weights(model_path)
    
    # Evaluate the model
    print("Evaluating the model...")
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Test Mean Squared Error: {mse}")
    
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")
    
    return history, mse

if __name__ == "__main__":
    train_file_path = "D://Major Project//MajorDTA//Dataset//train.csv"
    test_file_path = "D://Major Project//MajorDTA//Dataset//test.csv"
    model_save_path = "D://Major Project//MajorDTA//Model//MLP.keras"
    
    # Train the MLP model
    history, mse = train_model(train_file_path, test_file_path, model_save_path)