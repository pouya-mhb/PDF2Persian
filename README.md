# 📘 English-to-Persian PDF Translator (Offline)

This project provides a **local deep learning pipeline** for translating **English PDF documents into Persian (Farsi)** using **MBART multilingual models** from Hugging Face. It supports PDF text extraction, OCR for scanned pages, and right-to-left Persian text rendering.

---

## 🚀 Features

- ✅ Translate English → Persian using **MBART-50**  
- 🧠 Works fully **offline** (after initial model download)  
- 📄 Extracts text from both **digital and scanned PDFs**  
- 🔤 Handles **Persian text reshaping and BiDi correction**  
- 🧩 Automatically installs dependencies and prepares directories  
- 📦 Saves outputs (translated text files or PDFs) in `outputs/`

---

## 🛠️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/pouya-mhb/PDF2Persian.git
cd PDF2Persian
```

### 2️⃣ Run the setup script
This installs all required packages and prepares the project directories:
```
python install_local_project.py
```
It will automatically:
* Install required Python libraries
* Create an outputs/ folder for results
* Download NLTK tokenizer data
---
## 📚 Usage
### 1️⃣ Place your English PDF

Put your file (e.g., sample.pdf) in the project folder.

### 2️⃣ Run the translator
```
python pdf_translator_local_models.py
```
This script will:
* Extract and clean text from the PDF
* Translate content to Persian
* Fix Persian text order (using arabic_reshaper + python-bidi)
* Save the translated output to outputs/translated_sample.txt
---
## 🧩 Directory Structure
```
pdf-persian-translator/
│
├── install_local_project.py        # Installs dependencies and sets up environment
├── pdf_translator_local_models.py  # Main translation script
├── outputs/                        # Folder for translated outputs
├── models/                         # (Optional) for storing local model weights
└── README.md                       # Project documentation
```
---
## ⚙️ Dependencies

Installed automatically by install_local_project.py:
```
torch
transformers
sentencepiece
pdfplumber
pdf2image
pytesseract
layoutparser
arabic-reshaper
python-bidi
nltk
pillow
opencv-python
requests
```
---
## 📝 Notes

Make sure **Tesseract OCR** is installed on your system if you want to process scanned PDFs.

Persian text is reshaped and rendered correctly using:
```
from arabic_reshaper import reshape
from bidi. algorithm import get_display

final_text = get_display(reshape(translated_text))
```
* The model (facebook/mbart-large-50-many-to-many-mmt) will be automatically downloaded the first time you run the script.
---
## 🧠 Example

Input (English):
```
This study aimed to investigate the impact of using ChatGPT as a learning tool on students' motivation.
```

Output (Persian):
```
این مطالعه با هدف بررسی تأثیر استفاده از ChatGPT به عنوان ابزار یادگیری بر انگیزه دانشجویان انجام شد.
```
---
## 📄 License

This project is released under the MIT License.
Feel free to use, modify, and distribute with proper credit.
