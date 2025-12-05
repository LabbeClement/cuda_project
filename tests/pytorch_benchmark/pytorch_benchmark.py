import torch
import torch.nn as nn
import time
import sys

# --- CONFIGURATION (Mêmes tailles que le benchmark CUDA) ---
N = 4096
M = 4096
K = 4096 

def run_pytorch_benchmark():
    if not torch.cuda.is_available():
        print(" PyTorch ERROR: GPU non disponible. Vérifiez l'activation de l'environnement Conda et du driver.")
        return

    device = torch.device("cuda")
    # print(f"PyTorch GPU: {torch.cuda.get_device_name(0)}")

    # Création des tenseurs sur le GPU (équivalent de cudaMalloc et init)
    A = torch.randn(N, K, device=device)
    B = torch.randn(K, M, device=device)

    # -----------------------------------------------
    # TEST 1 : ADDITION DE MATRICE (A + A)
    # -----------------------------------------------
    
    # Warm-up (Chauffer le GPU)
    _ = A + A 
    torch.cuda.synchronize() 

    # Mesure du temps
    start = time.time()
    C_add = A + A
    torch.cuda.synchronize() 
    end = time.time()
    time_add = (end - start) * 1000 # Temps en millisecondes

    # -----------------------------------------------
    # TEST 2 : MULTIPLICATION DE MATRICE (A x B)
    # -----------------------------------------------

    # Warm-up
    _ = torch.matmul(A, B)
    torch.cuda.synchronize()

    # Mesure
    start = time.time()
    C_mult = torch.matmul(A, B)
    torch.cuda.synchronize()
    end = time.time()
    time_mult = (end - start) * 1000

    # Affichage des résultats pour le Makefile
    # On affiche les résultats dans un format clair pour la comparaison
    print(f"[Source: PyTorch] [Op: MatAdd] [Time: {time_add:.4f} ms]")
    print(f"[Source: PyTorch] [Op: MatMult] [Time: {time_mult:.4f} ms]")

BATCH_SIZE = 4096
LAYER_SIZES = [1024, 512, 256, 10]
NUM_LAYERS_MLP = len(LAYER_SIZES) - 1 # 3 couches cachées/output

def run_pytorch_mlp_benchmark():
    if not torch.cuda.is_available():
        print(" PyTorch ERROR: GPU non disponible.")
        return

    device = torch.device("cuda")
    
    # 2. Définition du modèle (Architecture identique)
    layers = []
    for i in range(NUM_LAYERS_MLP):
        input_dim = LAYER_SIZES[i]
        output_dim = LAYER_SIZES[i+1]
        
        # Couche Linéaire
        layers.append(nn.Linear(input_dim, output_dim))
        
        # ReLU, sauf pour la dernière couche (sortie)
        if i < NUM_LAYERS_MLP - 1:
            layers.append(nn.ReLU())
            
    model = nn.Sequential(*layers).to(device)
    
    # 3. Préparation de l'Input
    input_tensor = torch.randn(BATCH_SIZE, LAYER_SIZES[0], device=device)

    # 4. Warm-up (Exécution à vide)
    with torch.no_grad():
        _ = model(input_tensor)
    torch.cuda.synchronize()

    # 5. Mesure
    start = time.time()
    
    with torch.no_grad():
        output = model(input_tensor)
        
    torch.cuda.synchronize()
    end = time.time()
    time_mlp = (end - start) * 1000 # Temps en millisecondes
    
    # Affichage Standardisé
    print(f"[Source: PyTorch Ref] [Op: MLP_Forward ({NUM_LAYERS_MLP} L)] [Time: {time_mlp:.4f} ms]")

if __name__ == "__main__":
    run_pytorch_benchmark()
    run_pytorch_mlp_benchmark()