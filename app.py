"""
Circuit Board Component Placement Optimizer

A Streamlit application for optimizing component placement on circuit boards.
"""
import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import random
from PIL import Image, ImageDraw
import io

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.components import Component, ComponentType, Board
from utils.optimization import optimize_placement, get_components, place_components, is_valid_placement

# Set page configuration
st.set_page_config(
    page_title="Circuit Board Optimizer",
    page_icon="🔌",
    layout="wide"
)

# Define color map for component types
COLOR_MAP = {
    1: "#FF5733",  # Resistor - Red
    2: "#33FF57",  # Capacitor - Green
    3: "#3357FF",  # Inductor - Blue
    4: "#FF33F5",  # Transistor - Pink
    5: "#F5FF33",  # IC - Yellow
    0: "#FFFFFF",  # Empty - White
}

@st.cache_data
def load_model(model_path):
    """Load the trained TensorFlow model."""
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def predict_placement(model, input_data):
    """Use the model to predict optimal component placement."""
    # Reshape input data for the model
    input_reshaped = input_data.reshape(1, *input_data.shape)
    
    # Make prediction
    prediction = model.predict(input_reshaped)
    
    # Return the first (and only) prediction
    return prediction[0]

def post_process_prediction(prediction, original_board):
    """
    Post-process the model prediction to ensure it's valid.
    
    Args:
        prediction: Raw model prediction
        original_board: Original board for reference
        
    Returns:
        Processed prediction that's valid for placement
    """
    # Get board dimensions
    height, width = prediction.shape[0], prediction.shape[1]
    
    # Check if prediction is empty (all zeros)
    if np.max(prediction) < 0.1:
        # If prediction is empty, fall back to simulated annealing
        return optimize_placement(original_board, method="simulated_annealing")
    
    # Extract components from original board
    original_components = get_components(original_board)
    
    # Extract components from prediction
    predicted_components = []
    visited = set()
    
    # First pass: identify components in the prediction
    for y in range(height):
        for x in range(width):
            comp_type = int(round(prediction[y, x, 0]))
            if comp_type > 0 and (y, x) not in visited:
                # Found a component
                comp_width = max(1, int(round(prediction[y, x, 1])))
                comp_height = max(1, int(round(prediction[y, x, 2])))
                power = max(0.1, prediction[y, x, 3])
                
                # Mark all cells of this component as visited
                for dy in range(comp_height):
                    for dx in range(comp_width):
                        if 0 <= y+dy < height and 0 <= x+dx < width:
                            visited.add((y+dy, x+dx))
                
                # Add component to list
                predicted_components.append({
                    "type": comp_type,
                    "width": comp_width,
                    "height": comp_height,
                    "power": power,
                    "position": (x, y)
                })
    
    # If no components were found or the prediction is invalid, use original components with new positions
    if not predicted_components or not is_valid_placement((height, width), predicted_components):
        # Use original components but with new positions
        predicted_components = []
        for i, comp in enumerate(original_components):
            # Try to find a valid position
            for attempt in range(100):
                new_x = random.randint(0, width - comp["width"])
                new_y = random.randint(0, height - comp["height"])
                
                # Create a new component with the new position
                new_comp = {
                    "type": comp["type"],
                    "width": comp["width"],
                    "height": comp["height"],
                    "power": comp["power"],
                    "position": (new_x, new_y)
                }
                
                # Check if this placement is valid with existing components
                temp_components = predicted_components + [new_comp]
                if is_valid_placement((height, width), temp_components):
                    predicted_components.append(new_comp)
                    break
            
            # If we couldn't find a valid position after 100 attempts, try a different approach
            if len(predicted_components) <= i:
                # Fall back to simulated annealing
                return optimize_placement(original_board, method="simulated_annealing")
    
    # Create a new board with the processed components
    processed_board = place_components((height, width), predicted_components)
    
    return processed_board

def create_random_board(width, height, num_components):
    """Create a random circuit board with components."""
    # Create an empty board
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
    
    return board

def visualize_board(board, title="Circuit Board"):
    """Visualize the circuit board."""
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Get board dimensions
    height, width = board.shape[0], board.shape[1]
    
    # Create RGB image
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Fill with component colors
    for y in range(height):
        for x in range(width):
            comp_type = int(round(board[y, x, 0]))
            if comp_type > 0:
                # Convert hex color to RGB
                color = COLOR_MAP.get(comp_type, "#FFFFFF")
                r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
                img[y, x] = [r, g, b]
            else:
                img[y, x] = [255, 255, 255]  # White for empty cells
    
    # Display the image
    ax.imshow(img)
    
    # Add grid
    ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
    
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add title
    ax.set_title(title)
    
    return fig

def calculate_score(board):
    """Calculate a score for the board layout."""
    # Get board dimensions
    height, width = board.shape[0], board.shape[1]
    
    # Initialize score
    score = 0
    
    # Penalty for components too close to the edge
    edge_penalty = 0
    for y in range(height):
        for x in range(width):
            if board[y, x, 0] > 0:  # If there's a component
                # Check if it's on the edge
                if x == 0 or x == width-1 or y == 0 or y == height-1:
                    edge_penalty += 1
    
    # Penalty for high power components close to each other
    power_penalty = 0
    for y in range(height):
        for x in range(width):
            if board[y, x, 0] > 0:  # If there's a component
                power = board[y, x, 3]
                
                # Check surrounding cells
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if 0 <= y+dy < height and 0 <= x+dx < width and (dx != 0 or dy != 0):
                            if board[y+dy, x+dx, 0] > 0:  # If there's another component
                                neighbor_power = board[y+dy, x+dx, 3]
                                power_penalty += power * neighbor_power
    
    # Calculate final score (lower is better)
    score = edge_penalty * 10 + power_penalty * 5
    
    return score

def main():
    """Main function for the Streamlit app."""
    st.title("Circuit Board Component Placement Optimizer")
    
    st.markdown("""
    This application helps optimize the placement of components on a circuit board.
    It uses a machine learning model to suggest optimal placements based on various constraints.
    """)
    
    # Check if model exists and show a prominent message if it doesn't
    model_path = os.path.join("models", "circuit_model.h5")
    if not os.path.exists(model_path):
        st.warning("""
        ⚠️ **AI Model Not Found**
        
        The AI optimization method will fall back to Simulated Annealing.
        
        To train the model, run:
        ```
        ./run.sh --train-model --generate-data
        ```
        
        Or select 'Simulated Annealing' or 'Genetic Algorithm' as your optimization method.
        """)
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Board size
    st.sidebar.subheader("Board Size")
    board_width = st.sidebar.slider("Width", 10, 30, 20)
    board_height = st.sidebar.slider("Height", 10, 30, 20)
    
    # Component count
    st.sidebar.subheader("Components")
    num_components = st.sidebar.slider("Number of Components", 3, 20, 8)
    
    # Optimization method
    st.sidebar.subheader("Optimization")
    optimization_method = st.sidebar.selectbox(
        "Method",
        ["AI Model", "Simulated Annealing", "Genetic Algorithm"]
    )
    
    # AI Model settings
    if optimization_method == "AI Model":
        model_path = os.path.join("models", "circuit_model.h5")
        if not os.path.exists(model_path):
            st.sidebar.warning("⚠️ Model not found. Using Simulated Annealing as fallback.")
            st.sidebar.info("Train the model first with: `./run.sh --train-model --generate-data`")
            st.sidebar.info("Or change the optimization method to 'Simulated Annealing' or 'Genetic Algorithm'")
        else:
            st.sidebar.success("✅ Model found and ready to use")
    
    # Main content area with two columns
    col1, col2 = st.columns(2)
    
    # Column 1: Generate random board
    with col1:
        st.header("Initial Board Layout")
        
        if st.button("Generate Random Board"):
            # Create random board
            random_board = create_random_board(board_width, board_height, num_components)
            
            # Store in session state
            st.session_state.random_board = random_board
            
            # Visualize
            fig = visualize_board(random_board, "Random Component Placement")
            st.pyplot(fig)
            
            # Calculate score
            score = calculate_score(random_board)
            st.metric("Layout Score", f"{score:.2f}", help="Lower is better")
            
            # Display component information
            st.subheader("Component Information")
            
            # Extract component data
            components = []
            for y in range(board_height):
                for x in range(board_width):
                    comp_type = int(round(random_board[y, x, 0]))
                    if comp_type > 0:
                        # Check if this is the top-left corner of the component
                        is_top_left = True
                        if x > 0 and random_board[y, x-1, 0] == comp_type:
                            is_top_left = False
                        if y > 0 and random_board[y-1, x, 0] == comp_type:
                            is_top_left = False
                        
                        if is_top_left:
                            width_comp = int(random_board[y, x, 1])
                            height_comp = int(random_board[y, x, 2])
                            power = random_board[y, x, 3]
                            
                            components.append({
                                "Type": {1: "Resistor", 2: "Capacitor", 3: "Inductor", 
                                         4: "Transistor", 5: "IC"}[comp_type],
                                "Size": f"{width_comp}x{height_comp}",
                                "Power": f"{power:.2f} W",
                                "Position": f"({x}, {y})"
                            })
            
            # Display as dataframe
            st.dataframe(components)
    
    # Column 2: Optimize board
    with col2:
        st.header("Optimized Board Layout")
        
        if st.button("Optimize Placement"):
            if "random_board" not in st.session_state:
                st.error("Please generate a random board first.")
            else:
                # Get random board
                random_board = st.session_state.random_board
                
                # Show spinner during optimization
                with st.spinner("Optimizing component placement..."):
                    # Start timer
                    start_time = time.time()
                    
                    # Optimize based on selected method
                    if optimization_method == "AI Model":
                        # Check if model exists
                        model_path = os.path.join("models", "circuit_model.h5")
                        if not os.path.exists(model_path):
                            st.warning("Model not found. Using Simulated Annealing as fallback.")
                            # Fallback to simulated annealing
                            optimized_board = optimize_placement(
                                random_board, 
                                method="simulated_annealing"
                            )
                        else:
                            # Load model
                            model = load_model(model_path)
                            
                            if model is None:
                                st.warning("Failed to load model. Using Simulated Annealing as fallback.")
                                # Fallback to simulated annealing
                                optimized_board = optimize_placement(
                                    random_board, 
                                    method="simulated_annealing"
                                )
                            else:
                                try:
                                    # Predict optimal placement
                                    raw_prediction = predict_placement(model, random_board)
                                    
                                    # Post-process prediction to ensure it's valid
                                    optimized_board = post_process_prediction(raw_prediction, random_board)
                                    
                                    # If the optimized board is empty, fall back to simulated annealing
                                    if np.max(optimized_board) < 0.1:
                                        st.warning("AI model produced an empty layout. Using Simulated Annealing as fallback.")
                                        optimized_board = optimize_placement(
                                            random_board, 
                                            method="simulated_annealing"
                                        )
                                except Exception as e:
                                    st.error(f"Error during AI optimization: {e}")
                                    st.warning("Using Simulated Annealing as fallback.")
                                    optimized_board = optimize_placement(
                                        random_board, 
                                        method="simulated_annealing"
                                    )
                    else:
                        # Use traditional optimization methods
                        optimized_board = optimize_placement(
                            random_board, 
                            method=optimization_method.lower().replace(" ", "_")
                        )
                    
                    # End timer
                    end_time = time.time()
                    
                    # Store optimized board
                    st.session_state.optimized_board = optimized_board
                    
                    # Visualize
                    fig = visualize_board(optimized_board, "Optimized Component Placement")
                    st.pyplot(fig)
                    
                    # Calculate scores
                    random_score = calculate_score(random_board)
                    optimized_score = calculate_score(optimized_board)
                    improvement = (random_score - optimized_score) / random_score * 100 if random_score > 0 else 0
                    
                    # Display metrics
                    st.subheader("Optimization Results")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Original Score", f"{random_score:.2f}")
                    
                    with col2:
                        st.metric("Optimized Score", f"{optimized_score:.2f}")
                    
                    with col3:
                        st.metric("Improvement", f"{improvement:.2f}%")
                    
                    st.info(f"Optimization completed in {end_time - start_time:.2f} seconds")
    
    # Comparison section (if both boards exist)
    if "random_board" in st.session_state and "optimized_board" in st.session_state:
        st.header("Side-by-Side Comparison")
        
        # Create comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Get boards
        random_board = st.session_state.random_board
        optimized_board = st.session_state.optimized_board
        
        # Get board dimensions
        height, width = random_board.shape[0], random_board.shape[1]
        
        # Create RGB images
        img1 = np.zeros((height, width, 3), dtype=np.uint8)
        img2 = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Fill with component colors
        for y in range(height):
            for x in range(width):
                # Random board
                comp_type1 = int(round(random_board[y, x, 0]))
                if comp_type1 > 0:
                    color = COLOR_MAP.get(comp_type1, "#FFFFFF")
                    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
                    img1[y, x] = [r, g, b]
                else:
                    img1[y, x] = [255, 255, 255]
                
                # Optimized board
                comp_type2 = int(round(optimized_board[y, x, 0]))
                if comp_type2 > 0:
                    color = COLOR_MAP.get(comp_type2, "#FFFFFF")
                    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
                    img2[y, x] = [r, g, b]
                else:
                    img2[y, x] = [255, 255, 255]
        
        # Display images
        ax1.imshow(img1)
        ax2.imshow(img2)
        
        # Add grid
        ax1.set_xticks(np.arange(-0.5, width, 1), minor=True)
        ax1.set_yticks(np.arange(-0.5, height, 1), minor=True)
        ax1.grid(which="minor", color="black", linestyle='-', linewidth=1)
        
        ax2.set_xticks(np.arange(-0.5, width, 1), minor=True)
        ax2.set_yticks(np.arange(-0.5, height, 1), minor=True)
        ax2.grid(which="minor", color="black", linestyle='-', linewidth=1)
        
        # Remove ticks
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])
        
        # Add titles
        ax1.set_title("Random Placement")
        ax2.set_title("Optimized Placement")
        
        # Display
        st.pyplot(fig)
        
        # Power distribution visualization
        st.header("Power Distribution")
        
        # Create power heatmaps
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Extract power data
        power1 = np.zeros((height, width))
        power2 = np.zeros((height, width))
        
        for y in range(height):
            for x in range(width):
                if random_board[y, x, 0] > 0:
                    power1[y, x] = random_board[y, x, 3]
                
                if optimized_board[y, x, 0] > 0:
                    power2[y, x] = optimized_board[y, x, 3]
        
        # Create heatmaps
        im1 = ax1.imshow(power1, cmap='hot')
        im2 = ax2.imshow(power2, cmap='hot')
        
        # Add colorbars
        plt.colorbar(im1, ax=ax1, label="Power (W)")
        plt.colorbar(im2, ax=ax2, label="Power (W)")
        
        # Add titles
        ax1.set_title("Power Distribution (Random)")
        ax2.set_title("Power Distribution (Optimized)")
        
        # Remove ticks
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])
        
        # Display
        st.pyplot(fig)
    
    # Footer
    st.markdown("---")
    st.markdown("Circuit Board Component Placement Optimizer | Built with Streamlit")

if __name__ == "__main__":
    main() 