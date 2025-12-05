import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import os

# --- ARCHITECTURES ---

class MnistSmall(nn.Module):
    def __init__(self):
        super(MnistSmall, self).__init__()
        # Input: 1x28x28
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8 * 5 * 5, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        # Note: ReLU est géré implicitement dans l'export
        return x

class VGGLike(nn.Module):
    def __init__(self):
        super(VGGLike, self).__init__()
        # Input: 3x64x64
        # Conv1: 3->32, 3x3, pad 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # -> 32x32
        
        # Conv2: 32->64, 3x3, pad 1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # -> 16x16
        
        # Conv3: 64->128, 3x3, pad 1
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # -> 8x8
        
        self.flatten = nn.Flatten()
        # Flatten size: 128 * 8 * 8 = 8192
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)

# --- EXPORT FUNCTION ---

def export_model_to_txt(model, filename, input_channels):
    print(f"Exporting model to {filename}...")
    with open(filename, 'w') as f:
        
        # Helper to retrieve layers
        # On suppose une structure plate pour simplifier l'export ici
        # (Dans un vrai projet, on itérerait sur model.modules())
        
        if isinstance(model, MnistSmall):
            layers = [
                (model.conv1, 2, 2), # conv, pool_k, pool_s
                (model.conv2, 2, 2)
            ]
            fcs = [model.fc1, model.fc2]
            fc_shapes = [8*5*5, 64, 10]
        else: # VGG
            layers = [
                (model.conv1, 2, 2),
                (model.conv2, 2, 2),
                (model.conv3, 2, 2)
            ]
            fcs = [model.fc1, model.fc2]
            fc_shapes = [128*8*8, 512, 10]

        # WRITE CONV LAYERS
        for conv, pk, ps in layers:
            f.write("CONV\n")
            # Params: in_c, out_c, k, s, p, pool_k, pool_s
            f.write(f"{conv.in_channels} {conv.out_channels} {conv.kernel_size[0]} {conv.stride[0]} {conv.padding[0]} {pk} {ps}\n")
            
            # Weights (Flattened)
            w = conv.weight.data.numpy().flatten()
            b = conv.bias.data.numpy().flatten()
            np.savetxt(f, w, fmt='%.6f', newline=' ')
            f.write("\n")
            np.savetxt(f, b, fmt='%.6f', newline=' ')
            f.write("\n")

        # WRITE FC LAYERS
        f.write("FC\n")
        f.write(f"{len(fcs)}\n") # Num MLP layers
        
        # Write sizes [input, hidden..., output]
        for s in fc_shapes:
            f.write(f"{s} ")
        f.write("\n")
        
        for fc in fcs:
            # Transpose weights for C++ MLP (Input Major)
            w = fc.weight.data.numpy().T.flatten()
            b = fc.bias.data.numpy().flatten()
            np.savetxt(f, w, fmt='%.6f', newline=' ')
            f.write("\n")
            np.savetxt(f, b, fmt='%.6f', newline=' ')
            f.write("\n")

def main():
    os.makedirs('tests/data', exist_ok=True)

    # 1. MNIST (Entraîné rapidement)
    print("--- Generating MNIST Model ---")
    model_mnist = MnistSmall()
    # On entraîne un peu pour la forme (optionnel pour le bench de vitesse)
    optimizer = optim.Adam(model_mnist.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    dataset = datasets.MNIST('./data', train=True, download=True, 
                        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.13,), (0.3,))]))
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    
    model_mnist.train()
    print("Training MNIST (100 batches)...")
    for batch_idx, (data, target) in enumerate(loader):
        if batch_idx > 100: break
        optimizer.zero_grad()
        # Simulation forward (pooling manual calls in forward)
        x = model_mnist.pool1(torch.relu(model_mnist.conv1(data)))
        x = model_mnist.pool2(torch.relu(model_mnist.conv2(x)))
        x = model_mnist.flatten(x)
        output = model_mnist.fc2(torch.relu(model_mnist.fc1(x)))
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
    export_model_to_txt(model_mnist, 'tests/data/cnn_mnist.txt', 1)
    
    # 2. VGG-Like (Initialisé)
    print("\n--- Generating VGG-Like Model ---")
    model_vgg = VGGLike()
    # Pas d'entraînement, juste export de l'architecture et poids initiaux
    export_model_to_txt(model_vgg, 'tests/data/cnn_vgg.txt', 3)
    
    print("\nModels generated in tests/data/")

if __name__ == '__main__':
    main()