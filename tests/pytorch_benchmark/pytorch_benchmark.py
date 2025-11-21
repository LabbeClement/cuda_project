import torch
import time
import sys

# --- CONFIGURATION (Mêmes tailles que le benchmark CUDA) ---
N = 4096
M = 4096
K = 4096 

def run_pytorch_benchmark():
    # 1. SETUP : Vérification GPU et Tenseurs
    if not torch.cuda.is_available():
        print("❌ PyTorch ERROR: GPU non disponible. Vérifiez l'activation de l'environnement Conda et du driver.")
        return

    device = torch.device("cuda")
    # print(f"✅ PyTorch GPU: {torch.cuda.get_device_name(0)}")

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
    print("-------------------------------------------------------------------\n")


if __name__ == "__main__":
    run_pytorch_benchmark()