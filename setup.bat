@echo off
echo Installation de l'environnement ASL Recognition...

:: Vérifier si Git est installé
git --version >nul 2>&1
if errorlevel 1 (
    echo Git n'est pas installé. Veuillez installer Git depuis https://git-scm.com/download/win
    pause
    exit /b 1
)

:: Vérifier si Python est installé
python --version >nul 2>&1
if errorlevel 1 (
    echo Python n'est pas installé. Veuillez installer Python 3.7 ou supérieur.
    pause
    exit /b 1
)

:: Cloner le dépôt s'il n'existe pas
if not exist .git (
    echo Clonage du dépôt...
    git clone https://github.com/khayem487/asl-recognition.git temp
    xcopy /E /I /Y temp\* .
    rmdir /S /Q temp
)

:: Créer l'environnement virtuel s'il n'existe pas
if not exist .venv (
    echo Création de l'environnement virtuel...
    python -m venv .venv
)

:: Activer l'environnement virtuel
call .venv\Scripts\activate.bat

:: Mettre à jour pip
python -m pip install --upgrade pip

:: Installer les dépendances
echo Installation des dépendances...
pip install -r requirements.txt

echo.
echo Installation terminée !
echo Pour activer l'environnement virtuel, utilisez la commande : .venv\Scripts\activate
echo.
pause 