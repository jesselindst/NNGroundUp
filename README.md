# Neural Network from Scratch for Image Classification

This project implements a simple feedforward neural network from scratch using NumPy for classifying images. It includes image preprocessing, layer implementation (with ReLU and Softmax activations), forward/backward propagation, cross-entropy loss calculation, and training/validation loops with plotting.

## Features

*   Neural network built purely with NumPy.
*   Customizable image dimensions (`IMAGEW`, `IMAGEH`).
*   Configurable network architecture (currently Input -> 10 (ReLU) -> 10 (ReLU) -> 10 (Softmax)).
*   Configurable learning rate (`LEARNINGRATE`) and epochs (`EPOCHS`).
*   ReLU activation for hidden layers.
*   Softmax activation for the output layer.
*   Cross-Entropy loss function.
*   Basic image encoding/flattening using Pillow.
*   Training and validation split (80/20).
*   Real-time plotting of training and validation loss using Matplotlib.

## Data

The training script expects images to be located in the `data/Img/Num/` directory relative to the project root. The `get_label` function in `model.py` currently expects filenames in the format `img<label_number>-<anything>.ext` (e.g., `img002-001.png` for label '2'). You might need to adjust the `PATH` constant or the `get_label` function based on your dataset structure and naming convention.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd NNGroundUp 
    ```
2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To train the model, simply run the `model.py` script:

```bash
python model.py
```

The script will:
1.  Load image filenames from `data/Img/Num/`.
2.  Split data into training and validation sets.
3.  Train the network for the specified number of `EPOCHS`.
4.  Print the training and validation loss for each epoch.
5.  Display a plot showing the loss curves, updated after each epoch.

## Configuration

Key parameters can be adjusted directly in the `model.py` script:

*   `PATH`: Path to the image directory.
*   `IMAGEW`, `IMAGEH`: Target width and height for image resizing (after division by 48).
*   `INPUTSHAPE`: Calculated automatically from `IMAGEW * IMAGEH`.
*   `OUTPUTSHAPE`: Number of output classes (e.g., 10 for digits 0-9).
*   `LEARNINGRATE`: Learning rate for gradient descent.
*   `EPOCHS`: Number of training epochs.

## Dependencies

*   [NumPy](https://numpy.org/)
*   [Pillow](https://python-pillow.org/) (PIL Fork)
*   [Matplotlib](https://matplotlib.org/) 