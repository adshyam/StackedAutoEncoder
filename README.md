# StackedAutoEncoder
Developed a Stacked-Autoencoder using TensorFlow

Autoencoders are a type of neural network used for unsupervised learning of efficient codings. The primary application demonstrated in this notebook is image reconstruction, where the autoencoder learns to compress (encode) the input data into a lower-dimensional representation and then reconstruct (decode) it back to its original form.

## Key Components

# Imports and Data Loading:

TensorFlow, NumPy, Pandas, Matplotlib, Scikit-Learn, and Seaborn are imported for building the model, processing data, and visualizations.
The **KMNIST dataset**, a dataset of handwritten Japanese characters, is loaded for training and testing the autoencoder.

# Data Preprocessing:

1. The images are normalized to a range between 0 and 1.
2. Labels are converted to one-hot encoded vectors.
   
# Dataset Definition:

TensorFlow data pipelines are created for efficient loading and batching of the training and test data.

# Model Architecture - Stacked Autoencoder:

A custom TensorFlow model is defined, representing a Stacked Autoencoder.
The encoder consists of dense layers that progressively reduce the dimensionality.
The bottleneck layer represents the compressed representation.
The decoder consists of dense layers that progressively increase the dimensionality back to the original size.
An additional layer to store 'constellation targets' is defined, possibly for a classification task or to add a constraint during training.

# Training Step with Regularizer:

The training process involves computing the reconstruction loss (Mean Squared Error) between the input and its reconstruction.
A regularizer is added to the loss function, possibly to enforce a constraint on the latent space (e.g., keeping the representations of different classes separate).

# Hyperparameters and Model Compilation:

Layer sizes, number of classes, regularization weight, epochs, and batch size are defined.
An instance of the Stacked Autoencoder model is created and compiled with an Adam optimizer.

# Training Loop:

The model is trained for a specified number of epochs, tracking total loss, MSE loss, and regularizer loss.

# Visualization and Analysis:

Plots are generated to visualize the loss metrics over epochs.
A function for making predictions using the trained model is defined.
Predictions are made on the test set, and a confusion matrix is plotted to analyze the performance.

# Confusion Matrix:

The confusion matrix provides insights into the classification performance of the model, showing how well the latent space representations correlate with the actual labels.

# Conclusion

The Autoencoder.ipynb notebook showcases the development of a Stacked Autoencoder using TensorFlow. It focuses on training the autoencoder on the KMNIST dataset, with an emphasis on reconstruction accuracy and a regularized training approach. The notebook also includes an element of classification analysis, as seen in the confusion matrix visualization, suggesting that the encoded representations are used for a classification task. This makes the notebook a valuable resource for learning about autoencoders, especially for image reconstruction and classification tasks.
