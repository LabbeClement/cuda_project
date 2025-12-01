import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import os

# 1. Définition du Modèle (Le même que nous allons implémenter en C)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Input: 1x28x28
        # Conv1: 1 -> 4 channels, Kernel 3x3, Stride 1, Padding 0 -> Output 4x26x26
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        # Pool1: 2x2, Stride 2 -> Output 4x13x13
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv2: 4 -> 8 channels, Kernel 3x3, Stride 1, Padding 0 -> Output 8x11x11
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        # Pool2: 2x2, Stride 2 -> Output 8x5x5
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Flatten: 8 * 5 * 5 = 200
        self.flatten = nn.Flatten()
        
        # FC1: 200 -> 64
        self.fc1 = nn.Linear(8 * 5 * 5, 64)
        self.relu3 = nn.ReLU()
        
        # FC2: 64 -> 10
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

def export_model_to_txt(model, filename):
    print(f"Exporting model to {filename}...")
    with open(filename, 'w') as f:
        # On écrit manuellement la structure pour notre parser C
        # Format par couche : [TYPE] [PARAMS...] [WEIGHTS] [BIAS]
        
        # --- LAYER 1: CONV ---
        # Params: in_c, out_c, k, s, p, pool_k, pool_s
        f.write("CONV\n")
        f.write(f"1 4 3 1 0 2 2\n") # 1->4 ch, k=3, s=1, p=0, pool=2, pool_s=2
        
        # Weights: [4, 1, 3, 3]
        w = model.conv1.weight.data.numpy().flatten()
        b = model.conv1.bias.data.numpy().flatten()
        np.savetxt(f, w, fmt='%.6f', newline=' ')
        f.write("\n")
        np.savetxt(f, b, fmt='%.6f', newline=' ')
        f.write("\n")
        
        # --- LAYER 2: CONV ---
        f.write("CONV\n")
        f.write(f"4 8 3 1 0 2 2\n") # 4->8 ch, k=3, s=1, p=0, pool=2, pool_s=2
        
        w = model.conv2.weight.data.numpy().flatten()
        b = model.conv2.bias.data.numpy().flatten()
        np.savetxt(f, w, fmt='%.6f', newline=' ')
        f.write("\n")
        np.savetxt(f, b, fmt='%.6f', newline=' ')
        f.write("\n")
        
        # --- MLP HEADER ---
        # Le code C attendra une suite de couches FC
        # Input size du MLP (8*5*5 = 200)
        # Layers: [200, 64, 10] -> 2 couches de poids
        f.write("FC\n")
        f.write("2 200 64 10\n") # num_layers, sizes...
        
        # FC 1
        w = model.fc1.weight.data.numpy().flatten() # PyTorch Linear is [out, in]
        # Attention: PyTorch stocke Linear weights comme [out_features, in_features]
        # Notre MLP C attend souvent [in_features x out_features] (Row Major) 
        # OU ALORS on fait W * x. 
        # Dans votre MLP.cu: MatMult(input [1, in], weights [in, out])
        # Donc on doit TRANSPOSE les poids PyTorch pour le MLP C.
        w = model.fc1.weight.data.numpy().T.flatten()
        b = model.fc1.bias.data.numpy().flatten()
        np.savetxt(f, w, fmt='%.6f', newline=' ')
        f.write("\n")
        np.savetxt(f, b, fmt='%.6f', newline=' ')
        f.write("\n")
        
        # FC 2
        w = model.fc2.weight.data.numpy().T.flatten()
        b = model.fc2.bias.data.numpy().flatten()
        np.savetxt(f, w, fmt='%.6f', newline=' ')
        f.write("\n")
        np.savetxt(f, b, fmt='%.6f', newline=' ')
        f.write("\n")

def main():
    # 1. Train
    device = torch.device("cpu")
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Load MNIST
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    
    print("Training simple CNN on MNIST (1 epoch)...")
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}/{len(train_loader)} Loss: {loss.item():.4f}")
            
    # 2. Export Model
    os.makedirs('tests/data', exist_ok=True)
    export_model_to_txt(model, 'tests/data/cnn_mnist.txt')
    
    # 3. Export One Sample (Input & Expected Output)
    model.eval()
    test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=1)
    data, target = next(iter(test_loader))
    
    output = model(data)
    pred = output.argmax(dim=1, keepdim=True)
    
    print(f"\nExporting test sample. True Label: {target.item()}, Prediction: {pred.item()}")
    
    # Save Input Image (1x28x28 flattened)
    with open('tests/data/mnist_sample_in.txt', 'w') as f:
        np.savetxt(f, data.numpy().flatten(), fmt='%.6f', newline=' ')
        
    # Save Expected Output (Logits)
    with open('tests/data/mnist_sample_out.txt', 'w') as f:
        np.savetxt(f, output.detach().numpy().flatten(), fmt='%.6f', newline=' ')

    print("Done!")

if __name__ == '__main__':
    main()