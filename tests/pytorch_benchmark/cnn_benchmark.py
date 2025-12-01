import torch
import torch.nn as nn
import time

class MnistSmall(nn.Module):
    def __init__(self):
        super(MnistSmall, self).__init__()
        # Mêmes params que votre C++
        self.features = nn.Sequential(
            nn.Conv2d(1, 4, 3, 1, 0), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(4, 8, 3, 1, 0), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8*5*5, 64), nn.ReLU(),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

class VGGLike(nn.Module):
    def __init__(self):
        super(VGGLike, self).__init__()
        # Mêmes params que votre C++
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*8*8, 512), nn.ReLU(),
            nn.Linear(512, 10)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

def benchmark(model, input_shape, batch_size=64, iterations=20, name="Model"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Random Input
    x = torch.randn(batch_size, *input_shape).to(device)
    
    # Warmup
    for _ in range(5):
        _ = model(x)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    for _ in range(iterations):
        _ = model(x)
    torch.cuda.synchronize()
    end = time.time()
    
    avg_time = (end - start) / iterations * 1000 # en ms
    print(f"[{name}] PyTorch Avg Time: {avg_time:.4f} ms")

if __name__ == "__main__":
    print("\n--- PyTorch Reference Benchmark ---")
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available for PyTorch. Benchmarking CPU (will be slow).")
    
    benchmark(MnistSmall(), (1, 28, 28), name="MNIST Small")
    benchmark(VGGLike(), (3, 64, 64), name="VGG-Like Medium")
    print("-----------------------------------\n")