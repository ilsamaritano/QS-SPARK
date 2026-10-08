"""
Variational Quantum Circuit (VQC) layer for PLASyR-Q.

Architecture: AngleEmbedding (RY) → StronglyEntanglingLayers → ⟨Z⟩ per qubit.
Implemented as a PennyLane TorchLayer so gradients flow through the VQC
during joint classical+quantum training.

Device choice: default.qubit + backprop. This was benchmarked against
lightning.qubit (adjoint, parameter-shift) and found to be 15–100× faster
for batch-32 forward+backward at n_qubits=6, n_layers=3:
  default.qubit  + backprop      : 77–130 ms / batch
  lightning.qubit + adjoint      : 1923 ms / batch
  lightning.qubit + param-shift  : 7583 ms / batch
"""
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import pennylane as qml
from pennylane.qnn import TorchLayer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg


def build_vqc_layer(n_qubits: int = cfg.N_QUBITS,
                    n_layers: int = cfg.VQC_LAYERS,
                    device_name: str = cfg.QUANTUM_DEVICE,
                    diff_method: str = cfg.QUANTUM_DIFF_METHOD) -> TorchLayer:
    """
    Build a PennyLane TorchLayer that maps (B, n_qubits) input angles to
    (B, n_qubits) expectation values ⟨Z_i⟩ ∈ [-1, 1].

    The caller is responsible for normalising the inputs to [-π, π] before
    passing them to this layer (typically via tanh(proj(features)) * π).
    """
    dev = qml.device(device_name, wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": (n_layers, n_qubits, 3)}
    return TorchLayer(circuit, weight_shapes)


class VQCBlock(nn.Module):
    """
    Self-contained quantum block: projects backbone features to rotation angles,
    runs the VQC, and exposes ⟨Z⟩ outputs for fusion / Mahalanobis scoring.

    Parameters
    ----------
    in_dim    : backbone feature dimension (512 for SE-ResNet-34, 1024 for DN-121)
    n_qubits  : number of qubits (and output dimension)
    n_layers  : number of StronglyEntanglingLayers
    """

    def __init__(self, in_dim: int, n_qubits: int = cfg.N_QUBITS,
                 n_layers: int = cfg.VQC_LAYERS):
        super().__init__()
        self.proj = nn.Linear(in_dim, n_qubits)
        self.vqc  = build_vqc_layer(n_qubits, n_layers)
        self.n_qubits = n_qubits

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (B, in_dim) → (B, n_qubits) expectation values."""
        angles = torch.tanh(self.proj(features)) * math.pi
        return self.vqc(angles)
