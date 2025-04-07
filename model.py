from PIL import Image
import numpy as np
import os
import random
import re
import matplotlib.pyplot as plt

# Constants
PATH = 'data/Img/Num/'
IMAGEW = 1200 // 48
IMAGEH = 900 // 48
INPUTSHAPE = IMAGEW * IMAGEH
OUTPUTSHAPE = 10
LEARNINGRATE = 0.001
EPOCHS = 1000

class encoder:
    def __init__(self, img_path, w=IMAGEW, h=IMAGEH):
        self.img = Image.open(img_path)
        self.w, self.h = w, h
        self.mat = None

    def encode(self):
        """Resize the image and store it as float32 np.array in [H, W], range [0..1]."""
        self.img = self.img.resize((self.w, self.h), Image.ANTIALIAS)  
        self.img = self.img.convert("L")  
        self.mat = np.array(self.img, dtype=np.float32) / 255.0
        return self.mat
    
    def get_flattened_pixels(self):
        if self.mat is None:
            raise ValueError("You must call encode() before flattening.")
        return self.mat.flatten()

class ml_ops:
    @staticmethod
    def relu(x):
        return np.maximum(x, 0)

    @staticmethod
    def softmax(x):
        """Numerically stable softmax."""
        shift_x = x - np.max(x)
        exps = np.exp(shift_x)
        return exps / np.sum(exps)

class Layer:
    """
    A general layer class that can do either:
      - ReLU for hidden layers
      - Softmax for output layer
    """
    def __init__(self, dim, previous=None, activation='ReLU'):
        self.dim = dim
        self.previous = previous
        self.activation = activation
        self.bias = np.random.randn(dim)
        self.output = np.zeros(dim)

        if previous:
            # He init (for ReLU): sqrt(2 / fan_in)
            self.weights = (np.random.randn(dim, previous.dim)
                            * np.sqrt(2.0 / previous.dim))
        else:
            self.weights = None  # input layer has no weights

    def forward(self, input_data):
        if not self.previous:
            # Input layer, just pass data
            self.output = input_data
            return self.output

        z = np.dot(self.weights, input_data) + self.bias
        if self.activation == 'ReLU':
            self.output = ml_ops.relu(z)
        elif self.activation == 'softmax':
            self.output = ml_ops.softmax(z)
        else:
            raise ValueError(f"Unknown activation '{self.activation}'")
        return self.output

    def backward(self, delta_next, next_weights, lr):
        """
        For hidden layers with ReLU:
          delta = (delta_next dot next_weights) * ReLU'(z)

        For output layer (softmax + CE):
          delta = (output - target)
        """
        if not self.previous:
            # Input layer: no previous to update
            return delta_next

        if self.activation == 'ReLU':
            delta_local = delta_next.dot(next_weights)
            # ReLU'(z) = 1 if output>0 else 0
            relu_grad = (self.output > 0).astype(float)
            delta = delta_local * relu_grad
        elif self.activation == 'softmax':
            # softmax + cross-entropy => gradient = (output - target)
            delta = delta_next
        else:
            raise ValueError(f"Unknown activation '{self.activation}'")

        # Update weights and bias
        for i in range(self.dim):
            for j in range(self.previous.dim):
                self.weights[i][j] -= lr * delta[i] * self.previous.output[j]
            self.bias[i] -= lr * delta[i]

        return delta

class model:
    def __init__(self, input_shape):
        # 1) Input layer
        self.input_layer = Layer(input_shape, previous=None, activation=None)
        # 2) Hidden layer 1 (ReLU)
        self.h1 = Layer(10, previous=self.input_layer, activation='ReLU')
        # 3) Hidden layer 2 (ReLU)
        self.h2 = Layer(10, previous=self.h1, activation='ReLU')
        # 4) Output layer (softmax for 10-class)
        self.output_layer = Layer(10, previous=self.h2, activation='softmax')

    def forward(self, input_data):
        x = self.input_layer.forward(input_data)
        x = self.h1.forward(x)
        x = self.h2.forward(x)
        x = self.output_layer.forward(x)
        return x

    def backward(self, target, lr):
        # output layer gradient: (prediction - target)
        output_error = self.output_layer.output - target
        # backprop through output layer
        h2_error = self.output_layer.backward(output_error, None, lr)
        # backprop hidden layer 2
        h1_error = self.h2.backward(h2_error, self.output_layer.weights, lr)
        # backprop hidden layer 1
        in_error = self.h1.backward(h1_error, self.h2.weights, lr)
        # input layer doesn't have a previous, but for consistency:
        self.input_layer.backward(in_error, self.h1.weights, lr)

def cross_entropy_loss(pred, label, eps=1e-15):
    pred = np.clip(pred, eps, 1 - eps)
    return -np.sum(label * np.log(pred))

def get_label(img_name):
    """
    Expects filenames like: img002-xxx => '2' => index=1
    Adjust to your naming scheme
    """
    matches = re.findall(r'img0*(\d+)-', img_name)
    if not matches:
        # default fallback
        return np.zeros(OUTPUTSHAPE)

    num = [int(num) - 1 for num in matches]
    label = np.zeros(OUTPUTSHAPE)
    label[num] = 1
    return label

def train(epochs, input_shape, lr):
    NN = model(input_shape)

    plt.ion()
    fig, ax = plt.subplots()
    train_losses = []
    val_losses = []

    # 1. Load all filenames, split into 80% train, 20% val
    all_files = os.listdir(PATH)
    if not all_files:
        raise ValueError(f"No images found in '{PATH}'")

    random.shuffle(all_files)
    split_idx = int(0.8 * len(all_files))
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]

    for epoch in range(epochs):
        # ========== 1) Training phase ==========
        random.shuffle(train_files)  # shuffle training data each epoch

        total_train_loss = 0.0
        train_count = 0

        for img_name in train_files:
            img_path = os.path.join(PATH, img_name)

            # 1) Encode
            enc = encoder(img_path)
            enc.encode()
            input_vec = enc.get_flattened_pixels()

            # 2) Forward
            output = NN.forward(input_vec)

            # 3) Label + Loss
            label = get_label(img_name)
            loss_val = cross_entropy_loss(output, label)
            total_train_loss += loss_val
            train_count += 1

            # 4) Backprop
            NN.backward(label, lr)

        avg_train_loss = total_train_loss / train_count
        train_losses.append(avg_train_loss)

        # ========== 2) Validation phase ==========
        # We do a forward pass on validation data, no weight updates
        total_val_loss = 0.0
        val_count = 0

        for img_name in val_files:
            img_path = os.path.join(PATH, img_name)

            # 1) Encode
            enc = encoder(img_path)
            enc.encode()
            input_vec = enc.get_flattened_pixels()

            # 2) Forward
            output = NN.forward(input_vec)

            # 3) Label + Loss (no backward)
            label = get_label(img_name)
            loss_val = cross_entropy_loss(output, label)
            total_val_loss += loss_val
            val_count += 1

        avg_val_loss = total_val_loss / val_count
        val_losses.append(avg_val_loss)

        # ========== 3) Plot both train + validation loss ==========
        ax.clear()
        ax.plot(train_losses, label="Training Loss")
        ax.plot(val_losses, label="Validation Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-Entropy Loss")
        ax.set_title("Training & Validation Loss Over Epochs (ReLU+Softmax)")
        ax.legend()
        plt.draw()
        plt.pause(0.001)

        print(f"Epoch {epoch+1}: "
              f"Train Loss = {avg_train_loss:.6f}, "
              f"Val Loss = {avg_val_loss:.6f}")

    # keep final plot open
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    train(EPOCHS, INPUTSHAPE, LEARNINGRATE)
