# ASL Recognition

Un système de reconnaissance de la langue des signes américaine (ASL) utilisant l'apprentissage automatique et la vision par ordinateur.

## Description

Ce projet utilise MediaPipe et scikit-learn pour détecter et classifier les gestes de la langue des signes américaine en temps réel. Le système capture les mouvements de la main via la webcam et utilise un modèle de machine learning pour identifier les signes.

## Fonctionnalités

- Détection en temps réel des gestes de la main
- Classification des signes de l'alphabet ASL
- Interface visuelle avec affichage des points de repère de la main
- Précision élevée grâce à l'utilisation de Random Forest Classifier
- Outil de capture d'images pour créer votre propre jeu de données

## Prérequis

- Python 3.7+
- Webcam
- Les bibliothèques Python suivantes :
  - OpenCV (cv2)
  - MediaPipe
  - NumPy
  - scikit-learn
  - matplotlib
  - seaborn

## Installation

### Installation Automatique (Recommandée)

#### Windows
1. Double-cliquez sur `setup.bat`
2. Suivez les instructions à l'écran

#### Linux/Mac
1. Ouvrez un terminal dans le dossier du projet
2. Rendez le script exécutable : `chmod +x setup.sh`
3. Exécutez le script : `./setup.sh`

### Installation Manuelle

1. Clonez ce dépôt :
```bash
git clone https://github.com/khayem487/asl-recognition.git
cd asl-recognition
```

2. Créez un environnement virtuel :
```bash
python -m venv .venv
```

3. Activez l'environnement virtuel :
- Windows : `.venv\Scripts\activate`
- Linux/Mac : `source .venv/bin/activate`

4. Installez les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

### Création du jeu de données
1. Utilisez l'outil de capture d'images :
```bash
python imagetaker.py
```
- Appuyez sur 'c' pour capturer une image
- Appuyez sur 'q' pour quitter

### Entraînement du modèle
1. Entraînez le modèle :
```bash
python train_classifier.py
```

### Test en temps réel
2. Testez le système en temps réel :
```bash
python test_classifier.py
```
- Appuyez sur 'q' pour quitter l'application

## Structure du Projet

- `train_classifier.py` : Script pour l'entraînement du modèle
- `test_classifier.py` : Script pour la détection en temps réel
- `create_dataset.py` : Script pour la création du jeu de données
- `imagetaker.py` : Outil de capture d'images pour créer votre propre jeu de données
- `data/` : Dossier contenant les données d'entraînement
- `model.p` : Modèle entraîné sauvegardé
- `setup.bat` : Script d'installation automatique pour Windows
- `setup.sh` : Script d'installation automatique pour Linux/Mac

## Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou à soumettre une pull request.

## Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails. 