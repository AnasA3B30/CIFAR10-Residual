
# CIFAR-10 Residual Neural Network

This repository provides a complete workflow for image classification on the CIFAR-10 dataset using a custom Residual Convolutional Neural Network (CNN) implemented in TensorFlow/Keras. The project includes modular model code, training scripts, evaluation, and visualization.

## Repository Structure

- `CIFAR_10_residual.ipynb`: Jupyter notebook for data loading, preprocessing, model training, evaluation, and visualization of predictions.
- `model.py`: Contains modular Keras layers and the main model class (`ConvNN`) with residual convolutional blocks, dense blocks, and data augmentation.
- `cifar10_model_residual.keras`: Saved Keras model weights after training.

## Model Architecture

- **ResidualConvBlock**: Custom convolutional block with skip connections, batch normalization, dropout, and optional pooling.
- **DenseBlock & ResidualDenseBlock**: Modular dense layers with batch normalization, dropout, L2 regularization, and skip connections.
- **ConvNN**: Main model class combining data augmentation, stacked residual blocks, global average pooling, and a dense output layer.

## Notebook Workflow

1. **Data Preparation**: Loads CIFAR-10, normalizes images, and one-hot encodes labels.
2. **Model Instantiation**: Imports and builds the `ConvNN` model from `model.py`.
3. **Training**: Trains the model for 30 epochs with Adam optimizer and categorical cross-entropy loss.
4. **Evaluation**: Reports test loss and accuracy.
5. **Prediction & Visualization**: Displays sample predictions with true labels and color-coded accuracy.
6. **Model Saving**: Saves trained weights for future use.

## Getting Started

1. **Clone the repository:**
   ```powershell
   git clone https://github.com/AnasA3B30/cifar10-residual.git
   ```
2. **Install dependencies:**
   - Python 3.7+
   - TensorFlow
   - Keras
   - NumPy
   - Matplotlib
   - Jupyter Notebook
   
   Install with pip:
   ```powershell
   pip install tensorflow keras numpy matplotlib jupyter
   ```
3. **Run the notebook:**
   Open `CIFAR_10_residual.ipynb` in Jupyter and follow the workflow to train and evaluate the model.

## Usage Tips

- Modify `model.py` to experiment with different architectures or hyperparameters.
- Use the notebook to visualize predictions and model performance.
- Saved model weights can be loaded for inference or further training.

## License

This project is licensed under the MIT License.

## Author

- [Your Name](https://github.com/AnasA3B30)

---
For questions, suggestions, or contributions, please open an issue or pull request.
