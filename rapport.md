Rapport de Projet : Moteur d'Inférence Deep Learning en CUDA

Ce rapport résume le développement d'un moteur d'inférence de réseaux de neurones (MLP et CNN) écrit en C++/CUDA "from scratch". L'objectif était de comprendre les primitives bas niveau du calcul sur GPU, d'implémenter les couches fondamentales, et d'optimiser les performances pour rivaliser avec les frameworks standards comme PyTorch.

I. Partie 1 : Le Perceptron Multicouche (MLP)

1. Primitives et Architecture

Le cœur du MLP repose sur des opérations matricielles. Nous avons implémenté une structure MLP dynamique capable de gérer un nombre arbitraire de couches.

MatMult (Multiplication Matricielle) : C'est l'opération critique ($Y = W \cdot X + B$).

Implémentation Naïve : Chaque thread calcule un élément de la matrice de sortie en itérant sur la dimension commune.

Fonctions d'Activation : Implémentation de la ReLU (Rectified Linear Unit) via un kernel CUDA simple (fmaxf(0, x)).

Pipeline Forward : Les activations sont passées de couche en couche via des buffers alloués sur le GPU.

2. Optimisations Implémentées

Pour pallier la lenteur de l'approche naïve sur de grandes matrices, trois stratégies ont été explorées :

Shared Memory Tiling (Tuilage) :

Principe : Charger des blocs (tuiles) de la matrice d'entrée dans la mémoire partagée (rapide) pour réduire les accès à la mémoire globale (lente).

Résultat : Gain de performance notable sur les grandes matrices en augmentant la réutilisation des données.

Kernel Fusion (Fusion de noyaux) :

Principe : Combiner la multiplication matrice-vecteur, l'addition de biais et la ReLU dans un seul kernel CUDA.

Avantage : Réduit la latence de lancement des kernels et évite les écritures intermédiaires en mémoire globale.

Performance : Champion sur les petits réseaux (ex: 4 neurones). Il bat cuBLAS sur des tailles minuscules grâce à un overhead quasi-nul.

Intégration cuBLAS :

Utilisation de la librairie officielle NVIDIA pour la multiplication matricielle (cublasSgemm).

Performance : Champion sur les grands réseaux (taille MNIST ou plus). Offre un débit (throughput) maximal.

3. Gestion des Données (ONNX & PyTorch)

Pour rendre le moteur utilisable, nous avons développé une chaîne d'outils (scripts Python dans tools/) pour :

Lire des modèles standards (.onnx ou modèles PyTorch).

Les convertir en fichiers texte plats (.txt) lisibles par notre code C++.

Charger ces poids directement dans la mémoire GPU.

II. Partie 2 : Réseaux Convolutionnels (CNN)

1. Extension de l'Architecture

Le moteur a été étendu pour traiter des tenseurs 4D (Batch, Channels, Height, Width).

Conv2D (Convolution) : Implémentation d'une convolution "naïve" à 6 boucles imbriquées (Batch, OutChannel, InChannel, KernelY, KernelX).

MaxPool2D : Réduction de dimension spatiale en prenant le maximum local.

Flatten : "Aplatissement" du tenseur 3D final pour le connecter à l'entrée du MLP (Partie 1).

2. Validation Fonctionnelle (MNIST)

Pour valider la justesse mathématique des kernels :

Entraînement d'un petit CNN sur le dataset MNIST via PyTorch (98% de précision).

Export des poids et d'une image de test (le chiffre "7").

Résultat : Notre moteur CUDA prédit correctement le chiffre "7" avec les mêmes logits que PyTorch.

3. L'Optimisation Majeure : Im2Col

La convolution naïve étant extrêmement lente en mémoire, nous avons implémenté Im2Col (Image To Column).

Le Concept : Transformer une opération de convolution 3D complexe en une multiplication de matrice 2D.

Méthode : Un kernel CUDA réarrange les pixels de l'image ("patchs") en colonnes d'une grande matrice.

Exécution : Une fois transformée, la convolution est calculée via cuBLAS (matrice des poids $\times$ matrice im2col).

Gain : Accélération significative (x2.14 sur VGG dans nos tests) par rapport à la version naïve.

III. Analyse des Performances et Modèles

1. Méthodologie de Benchmarking et Chargement

Pour mesurer les performances de manière fiable, deux approches distinctes ont été utilisées pour instancier les réseaux (MLP et CNN) :

Méthode Synthétique (En Mémoire) :

Utilisée pour les benchmarks d'optimisation pure (ex: tests/mlp/optimizations_benchmark.cu et tests/cnn/cnn_benchmark.cu).

Le réseau est défini directement dans le code C++ via des tableaux d'entiers (ex: {1024, 512, 10}).

L'allocation et l'initialisation se font directement sur le GPU avec des valeurs constantes. Cela permet de mesurer la vitesse brute des kernels sans l'overhead des lectures disque.

Méthode "Réelle" (Fichier Texte Plat) :

Utilisée pour valider l'inférence sur des modèles entraînés (ex: tests/mlp/file_benchmark.cu et tests/cnn/file_benchmark.cu).

Pipeline d'export : Un script Python exporte les modèles entraînés en "aplatissant" (flatten) tous les tenseurs multidimensionnels en une longue suite de nombres dans un fichier .txt.

Chargement C++ :

MLP : Lecture séquentielle des tailles puis des poids via fscanf.

CNN : Lecture d'un fichier structuré par balises (CONV, FC) contenant les hyperparamètres (kernel size, stride, padding) suivis des poids aplatis. Cela permet de reconstruire dynamiquement l'architecture complexe du CNN.

2. Modèle "MNIST Small" (Petit)

Architecture : 2 couches Conv (3x3) + 2 couches FC. Entrée 28x28.

Observation :

Notre code (CUDA Naïf) : ~0.09 ms

PyTorch : ~0.31 ms

Analyse : Sur de très petits calculs, notre code est 3x plus rapide que PyTorch. PyTorch souffre de l'overhead du langage Python et de la gestion du graphe dynamique. Notre code C++ "nu" a une latence minimale.

3. Modèle "VGG-Like Medium" (Moyen)

Architecture : 3 blocs Conv (32, 64, 128 filtres) + FC. Entrée 64x64. Beaucoup plus de calculs.

Observation :

Notre code (Naïf) : ~17.0 ms

Notre code (Im2Col) : ~7.8 ms

PyTorch (cuDNN) : ~2.3 ms

Analyse : Ici, la puissance brute prime.

L'optimisation im2col double nos performances.

Cependant, PyTorch reste plus rapide car il utilise cuDNN. Contrairement à im2col qui nécessite beaucoup de mémoire pour dupliquer les pixels (matrice intermédiaire), cuDNN utilise des algorithmes comme Winograd ou Implicit GEMM qui sont plus économes en mémoire et en accès cache.

IV. Conclusion

Le projet a permis de construire un pipeline complet d'inférence GPU.

Ce qui a été accompli :

Indépendance : Capacité à charger et exécuter des modèles sans dépendre de Python au runtime.

Modularité : Une architecture flexible (MLP seul ou CNN+MLP).

Compréhension fine : La mise en évidence du compromis Latence (où notre code brille) vs Débit/Optimisation Algorithmique (où les librairies industrielles comme cuDNN brillent).

Le moteur est fonctionnel, validé numériquement, et dispose de benchmarks clairs démontrant l'efficacité des optimisations implémentées.