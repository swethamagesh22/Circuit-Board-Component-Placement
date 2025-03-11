#!/bin/bash

# Circuit Board Component Placement Optimizer

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if Python is installed
if ! command_exists python3; then
    echo "Python 3 is not installed. Please install Python 3 and try again."
    exit 1
fi

# Check if pip is installed
if ! command_exists pip3; then
    echo "pip3 is not installed. Please install pip3 and try again."
    exit 1
fi

# Parse command line arguments
TRAIN_MODEL=false
RUN_APP=false
GENERATE_DATA=false
SAMPLES=200
EPOCHS=30
BATCH_SIZE=16

# If no arguments are provided, show help
if [ $# -eq 0 ]; then
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --train-model     Train the AI model"
    echo "  --run-app         Run the Streamlit app"
    echo "  --generate-data   Generate new dataset for training"
    echo "  --samples N       Number of samples to generate (default: 200)"
    echo "  --epochs N        Number of epochs to train (default: 30)"
    echo "  --batch-size N    Batch size for training (default: 16)"
    exit 0
fi

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --train-model)
            TRAIN_MODEL=true
            shift
            ;;
        --run-app)
            RUN_APP=true
            shift
            ;;
        --generate-data)
            GENERATE_DATA=true
            shift
            ;;
        --samples)
            SAMPLES="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create necessary directories
mkdir -p data
mkdir -p models

# Install dependencies
echo "Checking and installing dependencies..."
pip3 install -r requirements.txt

# Train model if requested
if [ "$TRAIN_MODEL" = true ]; then
    echo "========================================="
    echo "Training the AI model..."
    echo "========================================="
    
    # Check Python version
    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    if [[ "$PYTHON_VERSION" == "3.12" ]]; then
        echo "WARNING: You are using Python 3.12, which may have compatibility issues with TensorFlow."
        echo "If training fails, consider using Python 3.9-3.11 instead."
        echo ""
    fi
    
    if [ "$GENERATE_DATA" = true ]; then
        echo "Generating new dataset with $SAMPLES samples..."
        python3 train_model.py --generate --samples "$SAMPLES" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE"
    else
        # Check if dataset exists
        if [ ! -f "data/input_data.csv" ] || [ ! -f "data/output_data.csv" ]; then
            echo "Dataset not found. Generating new dataset with $SAMPLES samples..."
            python3 train_model.py --generate --samples "$SAMPLES" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE"
        else
            echo "Using existing dataset..."
            python3 train_model.py --epochs "$EPOCHS" --batch-size "$BATCH_SIZE"
        fi
    fi
    
    # Check if model was created successfully
    if [ -f "models/circuit_model.h5" ]; then
        echo "========================================="
        echo "Model trained successfully!"
        echo "Model saved to models/circuit_model.h5"
        echo "========================================="
    else
        echo "========================================="
        echo "WARNING: Model training may have failed."
        echo "Please check the error messages above."
        echo "========================================="
    fi
fi

# Run app if requested
if [ "$RUN_APP" = true ]; then
    echo "========================================="
    echo "Running the Streamlit app..."
    echo "========================================="
    
    # Check if streamlit is installed
    if ! command_exists streamlit; then
        echo "Streamlit is not installed. Installing streamlit..."
        pip3 install streamlit
    fi
    
    # Check if model exists
    if [ ! -f "models/circuit_model.h5" ]; then
        echo "WARNING: AI model not found at models/circuit_model.h5"
        echo "The AI optimization method will fall back to Simulated Annealing."
        echo "To train the model, run: ./run.sh --train-model --generate-data"
        echo ""
    fi
    
    streamlit run app.py
fi

echo "Done!" 