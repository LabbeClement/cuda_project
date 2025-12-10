#!/usr/bin/env python3
import torch
import torch.nn as nn
import sys
import os

def export_pytorch_to_onnx(model, input_shape, onnx_path):
    """
    Exporte un modèle PyTorch vers ONNX
    
    Args:
        model: modèle PyTorch (nn.Module)
        input_shape: tuple (batch_size, input_dim)
        onnx_path: chemin du fichier ONNX de sortie
    """
    model.eval()
    
    # Créer un input dummy
    dummy_input = torch.randn(input_shape)
    
    print(f"   Input shape: {input_shape}")
    
    # Exporter vers ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Modèle exporté vers '{onnx_path}'")

def create_test_mlp():
    """
    Créer un MLP de test avec des poids connus
    """
    layer_sizes = [4, 3, 2]
    
    # Créer le modèle
    layers = []
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        if i < len(layer_sizes) - 2:  # ReLU sauf dernière couche
            layers.append(nn.ReLU())
    
    model = nn.Sequential(*layers)
    
    # Initialiser avec des poids spécifiques (pour tester)
    with torch.no_grad():
        model[0].weight.data = torch.tensor([
            [0.1, 0.4, 0.7, 1.0],
            [0.2, 0.5, 0.8, 1.1],
            [0.3, 0.6, 0.9, 1.2]
        ])
        model[0].bias.data = torch.tensor([0.1, 0.2, 0.3])
        
        model[2].weight.data = torch.tensor([
            [0.5, 1.5, 2.5],
            [1.0, 2.0, 3.0]
        ])
        model[2].bias.data = torch.tensor([0.5, 1.0])
    
    return model, layer_sizes

def create_large_mlp():
    """
    Créer un MLP plus grand pour les benchmarks
    """
    layer_sizes = [784, 256, 128, 10]
    
    layers = []
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        if i < len(layer_sizes) - 2:
            layers.append(nn.ReLU())
    
    model = nn.Sequential(*layers)
    
    # Initialisation aléatoire
    return model, layer_sizes

if __name__ == "__main__":
    # Créer le répertoire de sortie
    os.makedirs("tests/data", exist_ok=True)
    
    print("╔════════════════════════════════════════╗")
    print("║   Export PyTorch → ONNX                ║")
    print("╚════════════════════════════════════════╝\n")
    
    # 1. Modèle de test (petit)
    print("1. Création du modèle de test [4→3→2]...")
    model_small, layer_sizes_small = create_test_mlp()
    onnx_path_small = "tests/data/mlp_model.onnx"
    export_pytorch_to_onnx(model_small, (1, layer_sizes_small[0]), onnx_path_small)
    torch.save(model_small.state_dict(), "tests/data/mlp_model.pth")
    
    # Tester avec PyTorch
    input_tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    output = model_small(input_tensor)
    print(f"   PyTorch output (test): {output.detach().numpy()}")
    print()
    
    # 2. Modèle plus grand
    print("2. Création du modèle large [784→256→128→10]...")
    model_large, layer_sizes_large = create_large_mlp()
    onnx_path_large = "tests/data/mlp_model_large.onnx"
    export_pytorch_to_onnx(model_large, (1, layer_sizes_large[0]), onnx_path_large)
    torch.save(model_large.state_dict(), "tests/data/mlp_model_large.pth")
    print()
    
    print("Tous les modèles ONNX ont été créés!")
    print(f"   - {onnx_path_small}")
    print(f"   - {onnx_path_large}")