"""
Train a neural network model for circuit board component placement optimization.
"""
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import matplotlib.pyplot as plt
import random
import time
from tqdm import tqdm

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.optimization import optimize_placement, calculate_score

def create_dataset(num_samples=1000, board_size=(20, 20), 
                  min_components=5, max_components=15):
    """
    Create a synthetic dataset for training.
    
    Args:
        num_samples: Number of samples to generate
        board_size: (height, width) of the board
        min_components: Minimum number of components per board
        max_components: Maximum number of components per board
        
    Returns:
        Tuple of (input_data, output_data)
    """
    height, width = board_size
    input_data = np.zeros((num_samples, height, width, 4))
    output_data = np.zeros((num_samples, height, width, 4))
    
    print(f"Generating {num_samples} samples...")
    for i in tqdm(range(num_samples)):
        # Determine number of components
        num_components = random.randint(min_components, max_components)
        
        # Create random board
        board = np.zeros((height, width, 4))
        
        # Component types (1-5)
        component_types = list(range(1, 6))
        
        # Place components randomly
        for _ in range(num_components):
            # Random component type
            comp_type = random.choice(component_types)
            
            # Random size (1-3)
            width_comp = random.randint(1, 3)
            height_comp = random.randint(1, 3)
            
            # Random position
            x = random.randint(0, width - width_comp)
            y = random.randint(0, height - height_comp)
            
            # Check if position is empty
            if np.sum(board[y:y+height_comp, x:x+width_comp, 0]) == 0:
                # Place component
                board[y:y+height_comp, x:x+width_comp, 0] = comp_type
                
                # Add size information
                board[y:y+height_comp, x:x+width_comp, 1] = width_comp
                board[y:y+height_comp, x:x+width_comp, 2] = height_comp
                
                # Add power rating (random between 0.1 and 2.0)
                power = random.uniform(0.1, 2.0)
                board[y:y+height_comp, x:x+width_comp, 3] = power
        
        # Store input
        input_data[i] = board
        
        # Optimize placement
        optimized_board = optimize_placement(
            board, 
            method="simulated_annealing", 
            max_iterations=500
        )
        
        # Store output
        output_data[i] = optimized_board
    
    return input_data, output_data

def save_dataset_to_csv(input_data, output_data, output_dir="data"):
    """
    Save dataset to CSV files.
    
    Args:
        input_data: Input data as numpy array
        output_data: Output data as numpy array
        output_dir: Directory to save CSV files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get dimensions
    num_samples, height, width, features = input_data.shape
    
    # Create dataframes
    input_rows = []
    output_rows = []
    
    for i in range(num_samples):
        for y in range(height):
            for x in range(width):
                # Input data
                input_rows.append({
                    "sample_id": i,
                    "y": y,
                    "x": x,
                    "comp_type": input_data[i, y, x, 0],
                    "width": input_data[i, y, x, 1],
                    "height": input_data[i, y, x, 2],
                    "power": input_data[i, y, x, 3]
                })
                
                # Output data
                output_rows.append({
                    "sample_id": i,
                    "y": y,
                    "x": x,
                    "comp_type": output_data[i, y, x, 0],
                    "width": output_data[i, y, x, 1],
                    "height": output_data[i, y, x, 2],
                    "power": output_data[i, y, x, 3]
                })
    
    # Create dataframes
    input_df = pd.DataFrame(input_rows)
    output_df = pd.DataFrame(output_rows)
    
    # Save to CSV
    input_df.to_csv(os.path.join(output_dir, "input_data.csv"), index=False)
    output_df.to_csv(os.path.join(output_dir, "output_data.csv"), index=False)
    
    print(f"Saved dataset to {output_dir}")

def load_dataset_from_csv(input_file, output_file, board_size=(20, 20)):
    """
    Load dataset from CSV files.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        board_size: (height, width) of the board
        
    Returns:
        Tuple of (input_data, output_data)
    """
    # Load dataframes
    input_df = pd.read_csv(input_file)
    output_df = pd.read_csv(output_file)
    
    # Get dimensions
    height, width = board_size
    num_samples = input_df["sample_id"].max() + 1
    
    # Create numpy arrays
    input_data = np.zeros((num_samples, height, width, 4))
    output_data = np.zeros((num_samples, height, width, 4))
    
    # Fill arrays
    for _, row in input_df.iterrows():
        i, y, x = int(row["sample_id"]), int(row["y"]), int(row["x"])
        input_data[i, y, x, 0] = row["comp_type"]
        input_data[i, y, x, 1] = row["width"]
        input_data[i, y, x, 2] = row["height"]
        input_data[i, y, x, 3] = row["power"]
    
    for _, row in output_df.iterrows():
        i, y, x = int(row["sample_id"]), int(row["y"]), int(row["x"])
        output_data[i, y, x, 0] = row["comp_type"]
        output_data[i, y, x, 1] = row["width"]
        output_data[i, y, x, 2] = row["height"]
        output_data[i, y, x, 3] = row["power"]
    
    return input_data, output_data

def create_model(input_shape):
    """
    Create a U-Net model for component placement.
    
    Args:
        input_shape: Shape of input data (height, width, features)
        
    Returns:
        Keras model
    """
    # Input layer
    inputs = layers.Input(shape=input_shape)
    
    # Encoder
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    
    # Decoder
    up4 = layers.UpSampling2D(size=(2, 2))(conv3)
    up4 = layers.concatenate([up4, conv2], axis=-1)
    conv4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(up4)
    conv4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)
    
    up5 = layers.UpSampling2D(size=(2, 2))(conv4)
    up5 = layers.concatenate([up5, conv1], axis=-1)
    conv5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up5)
    conv5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv5)
    
    # Output layer
    outputs = layers.Conv2D(4, (1, 1), activation='linear')(conv5)
    
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model

def custom_loss(y_true, y_pred):
    """
    Custom loss function for component placement.
    
    Args:
        y_true: True output grid
        y_pred: Predicted output grid
        
    Returns:
        Loss value
    """
    # Mean squared error for component positions
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # Additional penalty for component overlap
    # Extract component type from feature vector (first channel)
    comp_type_true = y_true[..., 0]
    comp_type_pred = y_pred[..., 0]
    
    # Calculate overlap penalty
    # Horizontal overlap
    h_true_diff = tf.abs(comp_type_true[:, :, :-1] - comp_type_true[:, :, 1:])
    h_pred_diff = tf.abs(comp_type_pred[:, :, :-1] - comp_type_pred[:, :, 1:])
    h_overlap = tf.abs(h_true_diff - h_pred_diff)
    
    # Vertical overlap
    v_true_diff = tf.abs(comp_type_true[:, :-1, :] - comp_type_true[:, 1:, :])
    v_pred_diff = tf.abs(comp_type_pred[:, :-1, :] - comp_type_pred[:, 1:, :])
    v_overlap = tf.abs(v_true_diff - v_pred_diff)
    
    # Combine overlap penalties
    overlap_penalty = tf.reduce_mean(h_overlap) + tf.reduce_mean(v_overlap)
    
    # Combine losses
    total_loss = mse + 0.1 * overlap_penalty
    
    return total_loss

def train_model(input_data, output_data, model_path=None, 
               batch_size=16, epochs=50):
    """
    Train the model.
    
    Args:
        input_data: Input data as numpy array
        output_data: Output data as numpy array
        model_path: Path to save the trained model
        batch_size: Batch size for training
        epochs: Number of epochs to train
        
    Returns:
        Training history
    """
    # Set default model path if not provided
    if model_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "models", "circuit_model.h5")

    # Create model directory
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Split data into training and validation sets
    num_samples = input_data.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    train_idx = indices[:int(0.8 * num_samples)]
    val_idx = indices[int(0.8 * num_samples):]
    
    train_input = input_data[train_idx]
    train_output = output_data[train_idx]
    val_input = input_data[val_idx]
    val_output = output_data[val_idx]
    
    # Create model
    model = create_model(input_data.shape[1:])
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss=custom_loss,
        metrics=['mse']
    )
    
    # Create callbacks
    callbacks_list = [
        callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor='val_loss',
            save_best_only=True
        ),
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5
        )
    ]
    
    # Train model
    history = model.fit(
        train_input,
        train_output,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(val_input, val_output),
        callbacks=callbacks_list
    )
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mse'])
    plt.plot(history.history['val_mse'])
    plt.title('Model MSE')
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(model_path), "training_history.png"))
    
    return history

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train a circuit board placement model")
    parser.add_argument("--generate", action="store_true", help="Generate new dataset")
    parser.add_argument("--samples", type=int, default=200, help="Number of samples to generate")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    
    args = parser.parse_args()
    
    # Get the absolute path to the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set paths with absolute references
    data_dir = os.path.join(script_dir, "data")
    input_file = os.path.join(data_dir, "input_data.csv")
    output_file = os.path.join(data_dir, "output_data.csv")
    model_path = os.path.join(script_dir, "models", "circuit_model.h5")

    # Create directories if they don't exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Generate or load dataset
    if args.generate or not os.path.exists(input_file) or not os.path.exists(output_file):
        print("Generating new dataset...")
        input_data, output_data = create_dataset(num_samples=args.samples)
        save_dataset_to_csv(input_data, output_data, data_dir)
    else:
        print("Loading dataset from CSV...")
        input_data, output_data = load_dataset_from_csv(input_file, output_file)
    
    # Print dataset info
    print(f"Dataset shape: {input_data.shape}")
    
    # Train model
    print("Training model...")
    history = train_model(
        input_data, 
        output_data, 
        model_path=model_path,
        batch_size=args.batch_size,
        epochs=args.epochs
    )
    
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main() 
