import pandas as pd
import numpy as np
import random

import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap

from sklearn.preprocessing import MinMaxScaler

from keras.datasets import mnist
import umap.umap_ as umap


def plot_mnist_examples(grid_size=(5, 5)):
    """
    Plots a grid of 22 images with even labels and 3 with odd labels from the MNIST dataset.

    Parameters:
    - grid_size: Tuple indicating the size of the grid (rows, columns).
    """
    random.seed(2025)
    
    (images, labels), _ = mnist.load_data()

    # Normalize the images
    images = images / 255.0

    # Separate images based on even and odd labels
    even_images = [img for img, label in zip(images, labels) if label % 2 == 0]
    even_labels = [label for label in labels if label % 2 == 0]
    odd_images = [img for img, label in zip(images, labels) if label % 2 != 0]
    odd_labels = [label for label in labels if label % 2 != 0]

    # Select 22 even-numbered images and 3 odd-numbered images
    selected_images = even_images[:22] + odd_images[:3]
    selected_labels = even_labels[:22] + odd_labels[:3]

    # Shuffle the images and labels to randomize positions
    combined = list(zip(selected_images, selected_labels))
    random.shuffle(combined)
    selected_images[:], selected_labels[:] = zip(*combined)

    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(10, 10))


   
    
    for i, ax in enumerate(axes.flat):
        if i < len(selected_images):
            ax.imshow(selected_images[i], cmap='gray')
            ax.set_title(f'Image: {i}')
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()




def sample_and_reduce_mnist_data(n_samples_per_digit=500, n_features=4, random_state=42):
    """
    This function loads the MNIST dataset and applies UMAP for dimensionality reduction.
    It returns a DataFrame with the reduced features and the original labels.
    """
    # Load the MNIST dataset
    (df_train, labels), _ = mnist.load_data()

    # Convert to DataFrame for easier manipulation
    df_train = pd.DataFrame(df_train.reshape(-1, 28*28))
    df_train['label'] = labels

    # Select samples from each digit
    df_train_selected = pd.concat([df_train[df_train['label'] == digit].sample(n=n_samples_per_digit, random_state=random_state) for digit in range(10)])
    
    # Apply UMAP for dimensionality reduction
    reducer = umap.UMAP(n_components=n_features, random_state=random_state)
    embedding = reducer.fit_transform(df_train_selected.drop('label', axis=1))

    # Convert the embedding to a DataFrame
    scaler = MinMaxScaler()
    df_train_selected_umap = pd.DataFrame(scaler.fit_transform(embedding), columns=[f'UMAP{i+1}' for i in range(n_features)])
    df_train_selected_umap['label'] = df_train_selected['label'].values

    return df_train_selected_umap

def visualize_mnist_data(df, label, n_classes=10):
    """
    Used to visulaized the demensionality reduced data in a 3D plot. 
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
        
    if n_classes == 2:

        
        scatter = ax.scatter(df['UMAP1'], df['UMAP2'], df['UMAP3'], c=label, cmap=ListedColormap(['blue', 'red']))
        # Creating custom legend labels
        handles, _ = scatter.legend_elements()
        custom_labels = ["Even numbers - normal data", "Odd numbers - anomalous data"]
        legend1 = ax.legend(handles, custom_labels, title="")
        
    elif n_classes == 10:
        scatter = ax.scatter(df['UMAP1'], df['UMAP2'], df['UMAP3'], c=label, cmap='tab10')
        legend1 = ax.legend(*scatter.legend_elements(), title="Digits")
        
    else:
        print("Visualization is only supported for 2 or 10 classes.")

    ax.add_artist(legend1)
    plt.title('3D UMAP projection of MNIST dataset')
    plt.show()



def sample_mnist_data(data, sample_size=120, ratio_outliers = 0.2, random_state=42, number_of_features = 3):
    """
    This function produces the dataset used for the experiment. 
    It samples from even digits to produce the majority class, and then samples from the remaining digits (odd digits) to produce the minority class. 

    Parameters:
        number_of_features: Number of features for each sample (default is 3).
        random_state: Seed for random number generation to ensure reproducibility (default is 42).
        sample_size: Total number of samples to generate (default is 120).
        ratio_outliers: Proportion of outliers to normal points in the dataset (default is 0.2).
    Returns:
        X: Pandas DataFrame with the generated data
        y: corresponding 0-1 labels. 0 indicates normal points, 1 indicates outliers.
        original_label: contains the original 0-9 labels

    """
    # Calculate the number of majority and minority samples
    majority_samples = round(sample_size/(1+ratio_outliers))
    minority_samples = sample_size - majority_samples
    
    # Set even labels for the majority class
    even_labels = [0,2,4,6,8]
    
    # Define feature columns based on the number of features
    feature_columns = [f"UMAP{i+1}" for i in range(number_of_features)]

    sampled_data = pd.DataFrame()
    sample_size_per_digit = int(majority_samples / 5)
    
    # Sample data for each majority class digit
    for label in even_labels:
        label_data = data[data['label'] == label]
        sampled_data = pd.concat([sampled_data, label_data.sample(n=sample_size_per_digit, random_state=random_state)])
    
    # Get remaining digits for minority class
    other_labels = [digit for digit in range(10) if digit not in even_labels]
    other_data = data[data['label'].isin(other_labels)]
    
    # Sample data for minority class
    sampled_data = pd.concat([sampled_data, other_data.sample(n=minority_samples, random_state=random_state)])
    
    # Modify labels so that 0 indicates normal points and 1 indicates outliers
    sampled_data['original_label'] = sampled_data['label'] 
    sampled_data['label'] = sampled_data['label'].apply(lambda x: 0 if x in even_labels else 1)
    
    # Extract features and labels
    X = sampled_data[feature_columns]
    y = sampled_data["label"]
    original_label = sampled_data["original_label"]
    
    return X, y, original_label

def compare_predictions_and_labels(df, labels, predictions):
    """
    Visualizes the true labels and the predicted labels side by side.
    """
    fig = plt.figure(figsize=(20, 8))

    
    # Plot with true labels
    ax1 = fig.add_subplot(121, projection='3d')
    scatter1 = ax1.scatter(df['UMAP1'], df['UMAP2'], df['UMAP3'], c=labels, cmap=ListedColormap(['blue', 'red']))
    # Creating custom legend labels
    handles1, _ = scatter1.legend_elements()
    custom_labels1 = ["Even numbers - normal data", "Odd numbers - anomalous data"]
    legend1 = ax1.legend(handles1, custom_labels1, title="True Labels")
    ax1.add_artist(legend1)
    ax1.set_title('3D UMAP projection of MNIST dataset (True Labels)')
    
    # Plot with predicted labels
    ax2 = fig.add_subplot(122, projection='3d')
    scatter2 = ax2.scatter(df['UMAP1'], df['UMAP2'], df['UMAP3'], c=predictions, cmap=ListedColormap(['blue', 'red']))
    handles2, _ = scatter1.legend_elements()
    custom_labels2 = ["Predicted even numbers - normal data", "Predicted odd numbers - anomalous data"]
    legend2 = ax2.legend(handles2, custom_labels2, title="Predictions")
    ax2.add_artist(legend2)
    ax2.set_title('3D UMAP projection of MNIST dataset (Predicted Labels)')
    
    plt.show()