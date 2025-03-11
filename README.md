# Circuit Board Component Placement Optimizer

An interactive application for optimizing the placement of components on circuit boards using AI and traditional optimization algorithms.

## Features

- **Interactive UI**: Visualize and optimize circuit board layouts in real-time
- **Multiple Optimization Methods**:
  - AI Model (U-Net neural network)
  - Simulated Annealing
  - Genetic Algorithm
- **Visualization**: Compare layouts, view power distribution, and analyze component placement

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/Circuit-Board-Component-Placement.git
   cd Circuit-Board-Component-Placement
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Running the Application

Run the Streamlit app:
```
./run.sh --run-app
```

Or directly with Streamlit:
```
streamlit run app.py
```

### Training the AI Model

To train the AI model (required for AI optimization):
```
./run.sh --train-model --generate-data
```

Options:
- `--generate-data`: Generate a new synthetic dataset
- `--samples 200`: Number of samples to generate (default: 200)
- `--epochs 30`: Number of training epochs (default: 30)
- `--batch-size 16`: Batch size for training (default: 16)

## Project Structure

```
circuit_app/
├── app.py                # Main Streamlit application
├── train_model.py        # Script to train the AI model
├── run.sh                # Shell script to run the application
├── requirements.txt      # Dependencies
├── README.md             # Documentation
├── data/                 # Directory for CSV datasets
│   ├── input_data.csv    # Input board layouts
│   └── output_data.csv   # Optimized board layouts
├── models/               # Directory for trained models
│   └── circuit_model.h5  # Trained neural network model
└── utils/                # Utility modules
    ├── components.py     # Component definitions
    └── optimization.py   # Optimization algorithms
```

## How It Works

1. **Generate a random board**: Create a circuit board with random component placement
2. **Optimize placement**: Use AI or traditional algorithms to find optimal component positions
3. **Compare results**: View side-by-side comparisons and performance metrics

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
