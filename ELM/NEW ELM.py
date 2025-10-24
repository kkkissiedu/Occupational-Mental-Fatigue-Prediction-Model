import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

class ELM(nn.Module):
    """
    Extreme Learning Machine implementation in PyTorch for binary classification
    Adapted for MEFAR fatigue prediction dataset
    """
    def __init__(self, input_size, hidden_size, output_size=1, activation='relu'):
        super(ELM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize input weights and biases randomly (these remain fixed)
        self.input_weights = nn.Parameter(torch.randn(input_size, hidden_size), requires_grad=False)
        self.biases = nn.Parameter(torch.randn(hidden_size), requires_grad=False)
        
        # Output weights (beta) will be calculated analytically
        self.beta = None
        
        # Set activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        else:
            raise ValueError("Supported activations: 'relu', 'tanh', 'sigmoid'")
    
    def hidden_layer(self, X):
        """Compute hidden layer output"""
        # Linear transformation
        G = torch.mm(X, self.input_weights) + self.biases
        # Apply activation function
        H = self.activation(G)
        return H
    
    def fit(self, X, y):
        """
        Train the ELM by calculating output weights analytically
        X: input data [batch_size, input_size]
        y: target labels [batch_size, 1] for binary classification
        """
        # Convert to torch tensors if needed
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)
        
        # Ensure y is the right shape
        if y.dim() == 1:
            y = y.unsqueeze(1)
        
        # Move to same device as model
        X = X.to(next(self.parameters()).device)
        y = y.to(next(self.parameters()).device)
        
        # Calculate hidden layer output
        H = self.hidden_layer(X)
        
        # Calculate output weights using Moore-Penrose pseudoinverse
        # beta = (H^T H)^-1 H^T y
        try:
            self.beta = torch.mm(torch.pinverse(H), y)
        except:
            # Fallback to SVD-based pseudoinverse if regular fails
            H_pinv = torch.linalg.pinv(H)
            self.beta = torch.mm(H_pinv, y)
        
        return self
    
    def forward(self, X):
        """Forward pass"""
        if self.beta is None:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Convert to torch tensor if needed
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        
        # Move to same device
        X = X.to(next(self.parameters()).device)
        
        # Calculate hidden layer output
        H = self.hidden_layer(X)
        
        # Calculate final output
        output = torch.mm(H, self.beta)
        
        return output
    
    def predict(self, X):
        """Make predictions (binary classification)"""
        with torch.no_grad():
            output = self.forward(X)
            # For binary classification, apply sigmoid and threshold at 0.5
            predictions = torch.sigmoid(output) > 0.5
            return predictions.cpu().numpy().astype(int)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        with torch.no_grad():
            output = self.forward(X)
            probabilities = torch.sigmoid(output)
            return probabilities.cpu().numpy()

class MEFARDataLoader:
    """Data loader for MEFAR dataset"""
    
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.scaler = StandardScaler()
    
    def load_data(self, X_data, y_data):
        """
        Load and preprocess MEFAR data
        X_data: physiological features
        y_data: fatigue scores (will be binarized using threshold 12)
        """
        # Convert to numpy arrays if needed
        if isinstance(X_data, pd.DataFrame):
            X = X_data.values
        else:
            X = np.array(X_data)
        
        if isinstance(y_data, pd.Series):
            y = y_data.values
        else:
            y = np.array(y_data)
        
        # Binarize labels: 1 if fatigue score >= 12, 0 otherwise
        y_binary = (y >= 12).astype(int)
        
        return X, y_binary
    
    def prepare_data(self, X, y, test_size=0.2, random_state=42):
        """Split and normalize data"""
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Normalize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test

def evaluate_model(model, X_test, y_test):
    """Evaluate the model performance"""
    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Fatigue', 'Fatigue']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Fatigue', 'Fatigue'],
                yticklabels=['No Fatigue', 'Fatigue'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    
    return accuracy, y_pred, y_proba

def experiment_with_hidden_sizes(X_train, X_test, y_train, y_test, hidden_sizes):
    """Experiment with different hidden layer sizes"""
    results = []
    
    for hidden_size in hidden_sizes:
        print(f"\nTesting with {hidden_size} hidden nodes...")
        
        # Create and train model
        model = ELM(input_size=X_train.shape[1], hidden_size=hidden_size)
        
        # Use GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Convert data to tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        
        # Fit the model
        model.fit(X_train_tensor, y_train_tensor)
        
        # Evaluate
        accuracy, _, _ = evaluate_model(model, X_test_tensor, y_test)
        results.append({'hidden_size': hidden_size, 'accuracy': accuracy})
        
        print(f"Accuracy with {hidden_size} hidden nodes: {accuracy:.4f}")
    
    return results

# Example usage and training script
if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load your MEFAR dataset
    print("Loading MEFAR dataset...")
    
    # Option 1: Load from CSV file directly

    df = pd.read_csv('./Datasets/MEFAR_DOWN.csv')
    if df.isnull().sum().sum() > 0:
        df.fillna(df.mean(), inplace=True)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    data_loader = MEFARDataLoader()
    
    # Load and prepare data
    X, y = data_loader.load_data(X, y)
    X_train, X_test, y_train, y_test = data_loader.prepare_data(X, y)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {', '.join(data_loader.feature_names)}")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples") 
    print(f"Class distribution - No Fatigue (0): {np.sum(y == 0)} ({np.mean(y == 0)*100:.1f}%)")
    print(f"Class distribution - Fatigue (1): {np.sum(y == 1)} ({np.mean(y == 1)*100:.1f}%)")
    
    # Single model training
    print("\n" + "="*60)
    print("Training ELM with 1000 hidden nodes...")
    
    model = ELM(input_size=X_train.shape[1], hidden_size=1000, activation='relu')
    model = model.to(device)
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    # Train model
    model.fit(X_train_tensor, y_train_tensor)
    
    # Evaluate
    print("\nEvaluation Results:")
    accuracy, predictions, probabilities = evaluate_model(model, X_test_tensor, y_test)
    
    # Experiment with different hidden sizes
    print("\n" + "="*60)
    print("Experimenting with different hidden layer sizes...")
    hidden_sizes = [100, 500, 1000, 2000, 3000]
    results = experiment_with_hidden_sizes(X_train, X_test, y_train, y_test, hidden_sizes)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    accuracies = [r['accuracy'] for r in results]
    hidden_sizes_plot = [r['hidden_size'] for r in results]
    
    plt.plot(hidden_sizes_plot, accuracies, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Hidden Nodes')
    plt.ylabel('Test Accuracy')
    plt.title('ELM Performance vs Hidden Layer Size\n(MEFAR Fatigue Classification)')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Print best configuration
    best_result = max(results, key=lambda x: x['accuracy'])
    print(f"\nBest configuration: {best_result['hidden_size']} hidden nodes")
    print(f"Best accuracy: {best_result['accuracy']:.4f}")
    
    # Feature importance analysis (approximate)
    print("\n" + "="*60)
    print("Analyzing feature contributions...")
    
    # Simple feature importance based on weight magnitudes
    with torch.no_grad():
        feature_importance = torch.mean(torch.abs(model.input_weights), dim=1).cpu().numpy()
        feature_ranking = np.argsort(feature_importance)[::-1]
        
        print("Top 10 most important features:")
        for i, feat_idx in enumerate(feature_ranking[:10]):
            print(f"{i+1:2d}. {data_loader.feature_names[feat_idx]:10s}: {feature_importance[feat_idx]:.4f}")
    
    print("\n" + "="*60)
    print("Training completed! To use with your actual MEFAR data:")
    print("1. Replace the synthetic data generation with:")
    print("   data_loader = MEFARDataLoader()")
    print("   X, y = data_loader.load_data_from_csv('your_dataset.csv')")
    print("2. Or load from your existing arrays:")
    print("   X, y = data_loader.load_data(your_X_data, your_y_data)")