# install_local_project.py
import subprocess
import sys
import os
import nltk

def install_requirements():
    """Install all required packages"""
    packages = [
        "torch",
        "transformers",
        "sentencepiece",
        "pdfplumber",
        "pdf2image",
        "pytesseract",
        "layoutparser",
        "arabic-reshaper",
        "python-bidi",
        "nltk",
        "pillow",
        "opencv-python",
        "requests"
    ]

    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def setup_project_structure():
    """Create project directory structure with model storage"""
    directories = [
        'data/raw',
        'data/processed',
        'outputs',
        'models/mbart-large-50',  # Specific model directory
        'models/checkpoints'
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created: {directory}")

def download_nltk_data():
    """Download required NLTK data"""
    try:
        # Try to load punkt first
        nltk.data.find('tokenizers/punkt')
        print("✅ NLTK punkt data already installed")
    except LookupError:
        print("📥 Downloading NLTK punkt data...")
        nltk.download('punkt', quiet=False)

    try:
        # Also download the English punkt models
        nltk.data.find('tokenizers/punkt_tab/english.pickle')
        print("✅ NLTK punkt_tab data already installed")
    except LookupError:
        print("📥 Downloading NLTK punkt_tab data...")
        nltk.download('punkt_tab', quiet=False)

if __name__ == "__main__":
    print("🚀 Setting up PDF Translator with local model storage...")
    install_requirements()
    setup_project_structure()
    download_nltk_data()
    print("✅ Setup complete! Model will be saved in 'models/' folder")