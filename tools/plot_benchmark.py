import sys
import re
import matplotlib.pyplot as plt
import numpy as np
import os

def parse_benchmark_log(filename):
    print(f"Lecture du fichier de log : {filename}")
    try:
        with open(filename, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print("Erreur : Fichier non trouvé.")
        sys.exit(1)

    data = {
        'mlp_large_synthetic': {}, # Optimization Benchmark (1024->...->10, Batch 4096)
        'mlp_small_4096': {},      # File Benchmark (4->3->2, Batch 4096)
        'mlp_large_file': {},      # File Benchmark (784->...->10, Batch 4096)
        'cnn_vgg': {},             # VGG Real
        'cnn_mnist_4096': {}       # CNN MNIST Batch 4096 (New)
    }

    # --- 1. PARSING MLP LARGE SYNTHETIC (Batch 4096) ---
    mlp_matches = re.findall(r'\[Source: CUDA (.*?)\] .*? \[Time: ([\d\.]+) ms\]', content)
    if len(mlp_matches) >= 4:
        for name, time in mlp_matches[:4]:
            key = name.replace("Optimisé (Fused)", "Fused").replace("Bibliothèque NVIDIA Optimisée", "cuBLAS").strip()
            if "Modulaire" in key: key = "Naive"
            if "Tiled" in key: key = "Tiled"
            if "Fused" in key: key = "Fused"
            if "cuBLAS" in key: key = "cuBLAS"
            data['mlp_large_synthetic'][key] = float(time)

    # --- 2. PARSING MLP SMALL (Batch 4096) ---
    if "BENCHMARKING MODEL: tests/data/mlp_model.onnx" in content:
        parts = content.split("BENCHMARKING MODEL: tests/data/mlp_model.onnx")
        if len(parts) > 1:
            section_4096 = parts[1].split("BENCHMARKING MODEL")[0]
            if "Batch Size: 4096" in section_4096:
                matches = re.findall(r'\d+\.\s+(.*?)\s+\|\s+([\d\.]+)\s+ms', section_4096)
                for name, time in matches:
                    t = float(time)
                    if "Modular" in name: data['mlp_small_4096']['Naive'] = t
                    elif "Tiled" in name: data['mlp_small_4096']['Tiled'] = t
                    elif "Fused" in name: data['mlp_small_4096']['Fused'] = t
                    elif "cuBLAS" in name: data['mlp_small_4096']['cuBLAS'] = t

    # --- 3. PARSING MLP LARGE FILE (Batch 4096) ---
    if "BENCHMARKING MODEL: tests/data/mlp_model_large.onnx" in content:
        parts = content.split("BENCHMARKING MODEL: tests/data/mlp_model_large.onnx")
        if len(parts) > 1:
            section = parts[1].split("BENCHMARKING MODEL")[0]
            matches = re.findall(r'\d+\.\s+(.*?)\s+\|\s+([\d\.]+)\s+ms', section)
            for name, time in matches:
                t = float(time)
                if "Modular" in name: data['mlp_large_file']['Naive'] = t
                elif "Tiled" in name: data['mlp_large_file']['Tiled'] = t
                elif "Fused" in name: data['mlp_large_file']['Fused'] = t
                elif "cuBLAS" in name: data['mlp_large_file']['cuBLAS'] = t

    # --- 4. PARSING CNN VGG ---
    if "Chargement: VGG-Like" in content:
        section = content.split("Chargement: VGG-Like")[1]
        cnn_matches = re.findall(r'\|\s+(.*?)\s+\|\s+([\d\.]+)\s+ms', section)
        for name, time in cnn_matches:
            data['cnn_vgg'][name.strip()] = float(time)

    # --- 5. PARSING CNN MNIST (Batch 4096) ---
    if "Chargement: MNIST Small" in content:
        section = content.split("Chargement: MNIST Small")[1]
        if "Chargement:" in section:
            section = section.split("Chargement:")[0]
            
        if "Benchmarking (Batch: 4096)" in section:
            cnn_mnist_matches = re.findall(r'\|\s+(.*?)\s+\|\s+([\d\.]+)\s+ms', section)
            for name, time in cnn_mnist_matches:
                 data['cnn_mnist_4096'][name.strip()] = float(time)

    # --- 6. PARSING PYTORCH ---
    pytorch_matches = re.findall(r'\[(.*?)\] PyTorch Avg Time: ([\d\.]+) ms', content)
    for name, time in pytorch_matches:
        if "VGG" in name:
            data['cnn_vgg']['PyTorch'] = float(time)
        if "MNIST Small" in name:
            data['cnn_mnist_4096']['PyTorch'] = float(time)

    return data

def plot_graph(ax, data_dict, title, ylabel, color_map):
    if not data_dict:
        return
    
    order = ['Naive', 'Naive Conv2D', 'Tiled', 'Fused', 'Im2Col + cuBLAS', 'cuBLAS', 'PyTorch']
    
    items = []
    for k in order:
        if k in data_dict:
            items.append((k, data_dict[k]))
    for k, v in data_dict.items():
        if k not in [item[0] for item in items]:
            items.append((k, v))
            
    labels = [k for k, v in items]
    values = [v for k, v in items]
    
    colors = []
    for label in labels:
        if 'Naive' in label: colors.append('#e74c3c') # Rouge
        elif 'Tiled' in label: colors.append('#e67e22') # Orange
        elif 'Fused' in label: colors.append('#2ecc71') # Vert (Gagnant latence)
        elif 'cuBLAS' in label and 'Im2Col' not in label: colors.append('#27ae60') # Vert foncé (Gagnant throughput)
        elif 'Im2Col' in label: colors.append('#f1c40f') # Jaune
        elif 'PyTorch' in label: colors.append('#3498db') # Bleu
        else: colors.append('#95a5a6') # Gris

    bars = ax.bar(labels, values, color=colors)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(axis='x', labelsize=8)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')

def plot_results(data):
    output_dir = "plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.style.use('ggplot')
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Résultats du Benchmark : CUDA vs PyTorch', fontsize=16)
    
    ax_list = axes.flatten()

    plot_graph(ax_list[0], data['mlp_large_synthetic'], 
               'MLP Large (Synthétique) - Batch 4096\n(Débit / Throughput)', 
               'Temps (ms)', {})

    plot_graph(ax_list[1], data['mlp_small_4096'], 
               'MLP Small (Fichier) - Batch 4096\n(Petit modèle, Gros Batch)', 
               'Temps (ms)', {})

    plot_graph(ax_list[2], data['mlp_large_file'], 
               'MLP Large (Fichier) - Batch 4096\n(Grand modèle, Gros Batch)', 
               'Temps (ms)', {})

    plot_graph(ax_list[3], data['cnn_vgg'], 
               'CNN VGG-Like (Réel) - Batch 64', 
               'Temps (ms)', {})

    plot_graph(ax_list[4], data['cnn_mnist_4096'], 
               'CNN MNIST (Réel) - Batch 4096\n(Petit modèle, Gros Batch)', 
               'Temps (ms)', {})

    ax_list[5].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.90) 
    
    outfile = os.path.join(output_dir, 'benchmark_results.png')
    plt.savefig(outfile)
    print(f"\nGraphique généré : {outfile}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 tools/plot_benchmark.py <logfile>")
        sys.exit(1)
    
    data = parse_benchmark_log(sys.argv[1])
    plot_results(data)