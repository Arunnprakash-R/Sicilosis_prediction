"""
Quantum Hybrid Model using PennyLane
4-qubit variational quantum circuit for enhanced scoliosis prediction
"""

import os
import sys
import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
from pathlib import Path
import logging

sys.path.append(str(Path(__file__).parent.parent))
from src.config import QUANTUM_CONFIG, TRAINING_CONFIG, LOGGING_CONFIG
from src.utils import set_seed, setup_logging, save_checkpoint


class QuantumLayer(nn.Module):
    """Quantum layer using PennyLane variational circuit"""
    
    def __init__(self, n_qubits=4, n_layers=3):
        """
        Args:
            n_qubits: Number of qubits in the circuit
            n_layers: Number of variational layers
        """
        super().__init__()
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Create quantum device
        self.dev = qml.device(QUANTUM_CONFIG['device'], wires=n_qubits)
        
        # Initialize quantum circuit parameters
        self.n_params = n_qubits * n_layers * 3  # 3 params per qubit per layer
        self.params = nn.Parameter(torch.randn(self.n_params) * 0.1)
        
        # Create quantum node
        self.qnode = qml.QNode(
            self._quantum_circuit,
            self.dev,
            interface=QUANTUM_CONFIG['interface'],
            diff_method="backprop"
        )
    
    def _quantum_circuit(self, inputs, params):
        """Define the quantum circuit
        
        Args:
            inputs: Classical input features
            params: Variational parameters
        
        Returns:
            Expectation values
        """
        # Encode classical inputs
        for i in range(self.n_qubits):
            qml.RY(inputs[i % len(inputs)], wires=i)
        
        # Variational layers
        params = params.reshape(self.n_layers, self.n_qubits, 3)
        for layer in range(self.n_layers):
            # Rotation gates
            for i in range(self.n_qubits):
                qml.Rot(params[layer, i, 0], params[layer, i, 1], params[layer, i, 2], wires=i)
            
            # Entangling gates
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i+1])
            qml.CNOT(wires=[self.n_qubits-1, 0])  # Ring connectivity
        
        # Measurements
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
    
    def forward(self, x):
        """Forward pass through quantum layer
        
        Args:
            x: Input tensor [batch_size, features]
        
        Returns:
            Quantum layer output [batch_size, n_qubits]
        """
        batch_size = x.size(0)
        outputs = []
        
        for i in range(batch_size):
            # Get quantum circuit output for each sample
            input_features = x[i].float()  # Ensure float32
            qout = self.qnode(input_features, self.params)
            outputs.append(torch.stack(qout))
        
        return torch.stack(outputs).float()


class HybridQuantumClassicalModel(nn.Module):
    """Hybrid quantum-classical model for scoliosis prediction"""
    
    def __init__(self, input_dim=512, n_qubits=4, n_layers=3, num_classes=4):
        """
        Args:
            input_dim: Dimension of input features (from ViT)
            n_qubits: Number of qubits
            n_layers: Number of variational layers
            num_classes: Number of output classes
        """
        super().__init__()
        
        # Classical pre-processing
        self.pre_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_qubits),
            nn.Tanh()  # Normalize to [-1, 1] for quantum encoding
        )
        
        # Quantum layer
        self.quantum_layer = QuantumLayer(n_qubits, n_layers)
        
        # Classical post-processing
        self.post_net = nn.Sequential(
            nn.Linear(n_qubits, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        """Forward pass
        
        Args:
            x: Input features [batch_size, input_dim]
        
        Returns:
            Class logits [batch_size, num_classes]
        """
        # Classical pre-processing
        x = self.pre_net(x)
        
        # Quantum processing
        x = self.quantum_layer(x)
        
        # Classical post-processing
        x = self.post_net(x)
        
        return x


class QuantumTrainer:
    """Trainer for hybrid quantum-classical model"""
    
    def __init__(self, config=None):
        self.config = config or QUANTUM_CONFIG
        self.logger = setup_logging(LOGGING_CONFIG['log_file'])
        set_seed(TRAINING_CONFIG['seed'])
        
        self.device = torch.device('cpu')
        
        os.makedirs(self.config['save_dir'], exist_ok=True)
        
        self.logger.info("Quantum Hybrid Trainer initialized")
        self.logger.info(f"Qubits: {self.config['n_qubits']}")
        self.logger.info(f"Layers: {self.config['n_layers']}")
    
    def train(self, train_loader, val_loader, feature_dim=512, num_classes=4):
        """Train quantum hybrid model
        
        Args:
            train_loader: Training data loader (features, labels)
            val_loader: Validation data loader
            feature_dim: Input feature dimension
            num_classes: Number of output classes
        
        Returns:
            Training history
        """
        # Initialize model
        model = HybridQuantumClassicalModel(
            input_dim=feature_dim,
            n_qubits=self.config['n_qubits'],
            n_layers=self.config['n_layers'],
            num_classes=num_classes
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config['learning_rate']
        )
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        best_val_acc = 0.0
        
        self.logger.info("Starting quantum hybrid training...")
        
        for epoch in range(self.config['epochs']):
            # Train epoch
            train_loss, train_acc = self._train_epoch(model, train_loader, criterion, optimizer)
            
            # Validate epoch
            val_loss, val_acc = self._validate_epoch(model, val_loader, criterion)
            
            # Save history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            # Log progress
            self.logger.info(
                f"Epoch {epoch+1}/{self.config['epochs']} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_path = self.config['save_dir'] / 'best_quantum.pt'
                save_checkpoint(model, optimizer, epoch, val_loss, best_path)
                self.logger.info(f"Best quantum model saved (Val Acc: {val_acc:.4f})")
        
        self.logger.info("Quantum training completed!")
        return history
    
    def _train_epoch(self, model, loader, criterion, optimizer):
        """Train one epoch"""
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for features, labels in loader:
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _validate_epoch(self, model, loader, criterion):
        """Validate one epoch"""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(loader)
        accuracy = correct / total
        
        return avg_loss, accuracy


def visualize_quantum_circuit():
    """Visualize the quantum circuit structure"""
    n_qubits = 4
    n_layers = 3
    
    dev = qml.device('default.qubit', wires=n_qubits)
    
    @qml.qnode(dev)
    def circuit(inputs, params):
        # Encoding
        for i in range(n_qubits):
            qml.RY(inputs[i], wires=i)
        
        # Variational layers
        params = params.reshape(n_layers, n_qubits, 3)
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.Rot(params[layer, i, 0], params[layer, i, 1], params[layer, i, 2], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i+1])
            qml.CNOT(wires=[n_qubits-1, 0])
        
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    # Dummy inputs
    inputs = np.array([0.1, 0.2, 0.3, 0.4])
    params = np.random.randn(n_qubits * n_layers * 3)
    
    # Draw circuit
    print(qml.draw(circuit)(inputs, params))


def main():
    """Test quantum model"""
    # Visualize circuit
    print("Quantum Circuit Structure:")
    visualize_quantum_circuit()
    
    # Test model
    model = HybridQuantumClassicalModel(input_dim=512, n_qubits=4, n_layers=3, num_classes=4)
    
    # Dummy input
    x = torch.randn(2, 512)  # Batch of 2 samples
    
    print("\nTesting model forward pass...")
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output:\n{output}")


if __name__ == "__main__":
    main()
