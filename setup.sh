#!/bin/bash

echo "Installation de l'environnement ASL Recognition..."

# Vérifier si Git est installé
if ! command -v git &> /dev/null; then
    echo "Git n'est pas installé. Veuillez installer Git."
    echo "Sur Ubuntu/Debian : sudo apt-get install git"
    echo "Sur MacOS : brew install git"
    exit 1
fi

# Vérifier si Python est installé
if ! command -v python3 &> /dev/null; then
    echo "Python n'est pas installé. Veuillez installer Python 3.7 ou supérieur."
    exit 1
fi

# Cloner le dépôt s'il n'existe pas
if [ ! -d ".git" ]; then
    echo "Clonage du dépôt..."
    git clone https://github.com/khayem487/asl-recognition.git temp
    cp -r temp/. .
    rm -rf temp
fi

# Créer l'environnement virtuel s'il n'existe pas
if [ ! -d ".venv" ]; then
    echo "Création de l'environnement virtuel..."
    python3 -m venv .venv
fi

# Activer l'environnement virtuel
source .venv/bin/activate

# Mettre à jour pip
python -m pip install --upgrade pip

# Installer les dépendances
echo "Installation des dépendances..."
pip install -r requirements.txt

echo ""
echo "Installation terminée !"
echo "Pour activer l'environnement virtuel, utilisez la commande : source .venv/bin/activate"
echo "" 