import torch
import torch.nn as nn
import time
import sys
import onnx
import collections

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
    print(f"[PyTorch MLP Synthetic] PyTorch Avg Time: {time_mlp:.4f} ms")


# --- ARCHITECTURES & TAILLES ---

# Architecture SMALL (MLP de test)
# [4 (input) → 3 (ReLU) → 2 (output)]
LAYER_SIZES_SMALL = [4, 3, 2]
BATCH_SIZE_SMALL = 256 # Taille raisonnable pour un petit test

# Architecture LARGE (Benchmark)
# [784 (input) → 256 (ReLU) → 128 (ReLU) → 10 (output)]
LAYER_SIZES_LARGE = [784, 256, 128, 10]
BATCH_SIZE_LARGE = 4096 # Grande taille pour le benchmark de débit

# ----------------------------------------------------
# 1. CLASSE GÉNÉRIQUE (Pour correspondre au nn.Sequential)
# ----------------------------------------------------

class SimpleMLP(nn.Module):
    """
    Crée un MLP dont les clés de poids correspondent au format simple 
    (0.weight, 2.bias, etc.) créé par nn.Sequential.
    """
    def __init__(self, layer_sizes):
        super(SimpleMLP, self).__init__()
        layers = []
        num_layers = len(layer_sizes) - 1

        for i in range(num_layers):
            input_dim = layer_sizes[i]
            output_dim = layer_sizes[i+1]
            
            layers.append(nn.Linear(input_dim, output_dim))
            
            if i < num_layers - 1:
                layers.append(nn.ReLU())
                
        # Le format Sequential génère les clés simples "0.weight", "2.bias", etc.
        self.net = nn.Sequential(*layers) 

    def forward(self, x):
        return self.net(x)

# ----------------------------------------------------
# 2. FONCTION DE BENCHMARK ET CHARGEMENT (FIXÉE)
# ----------------------------------------------------

def run_pytorch_mlp_benchmark_from_pth(path, arch_name, layer_sizes, batch_size):
    
    # SETUP : Vérification GPU
    if not torch.cuda.is_available():
        print(f"❌ PyTorch ERROR: GPU non disponible.")
        return

    device = torch.device("cuda")
    
    # A. Créer une instance vide du modèle (nn.Module)
    model = SimpleMLP(layer_sizes=layer_sizes)
    
    # B. Charger l'état (les poids) du fichier .pth
    try:
        # map_location=device charge directement les poids sur le GPU
        state_dict = torch.load(path, map_location=device) 
    except Exception as e:
        print(f"❌ Erreur lors du chargement des poids de {path}: {e}")
        return

    # C. Adapter les clés pour le chargement
    # Votre state_dict a été sauvegardé avec nn.Sequential, donc il contient des clés simples (0.weight, etc.)
    # Mon SimpleMLP a un conteneur 'net', donc il attend 'net.0.weight'. 
    # Solution la plus simple et sécurisée: Modifier le dictionnaire pour ajouter le préfixe 'net.'
    
    # Si le state_dict est un dictionnaire
    if isinstance(state_dict, collections.OrderedDict):
        
        new_state_dict = collections.OrderedDict()
        
        for k, v in state_dict.items():
            # Ajout du préfixe 'net.' aux clés qui n'en ont pas (comme '0.weight')
            if not k.startswith('net.'):
                name = 'net.' + k
            else:
                name = k
            
            new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict)
    else:
        # Si le fichier .pth contenait le modèle complet
        model = state_dict

    # D. Finalisation du modèle
    model.eval()
    model = model.to(device) # Déplacer le modèle sur le GPU

    
    # E. Préparation de l'Input sur le GPU
    input_tensor = torch.randn(batch_size, layer_sizes[0], device=device)

    # 3. Mesure du temps
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Warm-up (Exécution à vide)
    with torch.no_grad():
        _ = model(input_tensor)
    torch.cuda.synchronize()

    # Mesure
    start_event.record() 
    with torch.no_grad():
        output = model(input_tensor)
    end_event.record()
    
    torch.cuda.synchronize()
    time_mlp = start_event.elapsed_time(end_event) # Temps en millisecondes
    
    # Affichage Standardisé
    print(f"[PyTorch MLP {arch_name}] PyTorch Avg Time: {time_mlp:.4f} ms")


if __name__ == "__main__":
    run_pytorch_benchmark()
    run_pytorch_mlp_benchmark_from_pth(
        path="tests/data/mlp_model.pth", 
        arch_name="Small", 
        layer_sizes=LAYER_SIZES_SMALL, 
        batch_size=BATCH_SIZE_SMALL
    )
    run_pytorch_mlp_benchmark_from_pth(
        path="tests/data/mlp_model_large.pth", 
        arch_name="Large", 
        layer_sizes=LAYER_SIZES_LARGE, 
        batch_size=BATCH_SIZE_LARGE
    )
    run_pytorch_mlp_benchmark()