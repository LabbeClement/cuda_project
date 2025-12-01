import torch
import torch.nn as nn

def export_mlp_to_txt(model, filename, layer_sizes):
    """
    Exporte un modèle PyTorch MLP vers un fichier texte
    
    Args:
        model: nn.Sequential avec Linear + ReLU
        filename: nom du fichier de sortie
        layer_sizes: [input_dim, hidden1, hidden2, ..., output_dim]
    """
    with open(filename, 'w') as f:
        num_layers = len(layer_sizes) - 1
        
        # Écrire l'architecture
        f.write(f"{num_layers}\n")
        f.write(" ".join(map(str, layer_sizes)) + "\n\n")
        
        # Extraire et écrire les poids/biais de chaque couche
        layer_idx = 0
        for module in model:
            if isinstance(module, nn.Linear):
                # Weights [output_dim, input_dim] en PyTorch
                # On doit transposer pour avoir [input_dim, output_dim]
                weights = module.weight.data.cpu().numpy().T
                bias = module.bias.data.cpu().numpy()
                
                input_dim, output_dim = weights.shape
                
                f.write(f"# Layer {layer_idx} weights [{input_dim} x {output_dim}]\n")
                for i in range(input_dim):
                    for j in range(output_dim):
                        f.write(f"{weights[i, j]:.6f} ")
                    f.write("\n")
                
                f.write(f"\n# Layer {layer_idx} bias [{output_dim}]\n")
                for b in bias:
                    f.write(f"{b:.6f} ")
                f.write("\n\n")
                
                layer_idx += 1
    
    print(f"Modèle exporté vers '{filename}'")

# Exemple d'utilisation
if __name__ == "__main__":
    # Créer un petit MLP
    layer_sizes = [4, 3, 2]
    model = nn.Sequential(
        nn.Linear(4, 3),
        nn.ReLU(),
        nn.Linear(3, 2)
    )
    
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
    
    # Exporter
    export_mlp_to_txt(model, "tests/data/mlp_from_pytorch.txt", layer_sizes)
    
    # Test forward pass
    input_tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    output = model(input_tensor)
    print(f"PyTorch output: {output}")