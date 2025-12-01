import onnx
import numpy as np
import sys
import os

def load_onnx_weights(onnx_path):
    """
    Charge un modèle ONNX et extrait les poids/biais
    
    Returns:
        layer_sizes: [input_dim, hidden1, ..., output_dim]
        weights: liste de matrices numpy [input_dim × output_dim]
        biases: liste de vecteurs numpy [output_dim]
    """
    print(f"Lecture du fichier ONNX: {onnx_path}")
    model = onnx.load(onnx_path)
    
    # Extraire les initializers (poids et biais)
    initializers = {}
    for tensor in model.graph.initializer:
        initializers[tensor.name] = onnx.numpy_helper.to_array(tensor)
    
    print(f"   Nombre de tenseurs: {len(initializers)}")
    
    # Extraire l'architecture en parcourant les nodes
    layers_info = []
    
    for node in model.graph.node:
        if node.op_type in ["Gemm", "MatMul"]:
            # Récupérer les poids
            weight_name = node.input[1]
            
            if weight_name not in initializers:
                print(f"   Attention: Poids '{weight_name}' non trouvé")
                continue
            
            weights = initializers[weight_name]
            
            # Récupérer le biais (optionnel)
            bias = None
            if len(node.input) > 2:
                bias_name = node.input[2]
                if bias_name in initializers:
                    bias = initializers[bias_name]
            
            # ONNX stocke les poids en [output_dim × input_dim]
            # On transpose pour avoir [input_dim × output_dim]
            weights_T = weights.T
            
            layers_info.append({
                'weights': weights_T,
                'bias': bias,
                'input_dim': weights_T.shape[0],
                'output_dim': weights_T.shape[1],
                'name': weight_name
            })
            
            print(f"   Couche trouvée: [{weights_T.shape[0]} × {weights_T.shape[1]}]")
    
    # Construire layer_sizes
    if len(layers_info) == 0:
        raise ValueError("Aucune couche linéaire trouvée dans le modèle ONNX")
    
    layer_sizes = [layers_info[0]['input_dim']]
    for layer in layers_info:
        layer_sizes.append(layer['output_dim'])
    
    weights = [layer['weights'] for layer in layers_info]
    biases = [layer['bias'] for layer in layers_info]
    
    return layer_sizes, weights, biases

def export_to_txt(layer_sizes, weights, biases, output_path):
    """
    Exporte vers le format texte SANS COMMENTAIRES
    Compatible avec le parser C de load_MLP_from_file
    """
    with open(output_path, 'w') as f:
        num_layers = len(weights)
        
        # Ligne 1 : nombre de couches
        f.write(f"{num_layers}\n")
        
        # Ligne 2 : architecture
        f.write(" ".join(map(str, layer_sizes)) + "\n")
        f.write("\n")  # Ligne vide importante
        
        # Pour chaque couche
        for i, (W, b) in enumerate(zip(weights, biases)):
            input_dim, output_dim = W.shape
            
            # Écrire les weights (SANS COMMENTAIRE)
            for row_idx in range(input_dim):
                for col_idx in range(output_dim):
                    f.write(f"{W[row_idx, col_idx]:.6f}")
                    if col_idx < output_dim - 1:
                        f.write(" ")
                f.write("\n")
            
            f.write("\n")  # Ligne vide après weights
            
            # Écrire les bias (SANS COMMENTAIRE)
            if b is not None:
                for idx, val in enumerate(b):
                    f.write(f"{val:.6f}")
                    if idx < len(b) - 1:
                        f.write(" ")
            else:
                for idx in range(output_dim):
                    f.write("0.000000")
                    if idx < output_dim - 1:
                        f.write(" ")
            f.write("\n")
            f.write("\n")  # Ligne vide après bias
    
    print(f"Conversion terminée: '{output_path}'")
    print(f"   Architecture: {layer_sizes}")

def onnx_to_txt(onnx_path, txt_path):
    """
    Convertit un fichier ONNX en fichier texte
    """
    try:
        layer_sizes, weights, biases = load_onnx_weights(onnx_path)
        
        print(f"   Nombre de couches: {len(weights)}")
        print(f"   Architecture: {layer_sizes}")
        
        export_to_txt(layer_sizes, weights, biases, txt_path)
        return 0
        
    except Exception as e:
        print(f"Erreur lors de la conversion: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 onnx_to_txt.py <input.onnx> <output.txt>")
        print("\nExemple:")
        print("  python3 onnx_to_txt.py tests/data/mlp_model.onnx tests/data/mlp_from_onnx.txt")
        sys.exit(1)
    
    onnx_path = sys.argv[1]
    txt_path = sys.argv[2]
    
    if not os.path.exists(onnx_path):
        print(f"Fichier non trouvé: {onnx_path}")
        sys.exit(1)
    
    exit_code = onnx_to_txt(onnx_path, txt_path)
    sys.exit(exit_code)