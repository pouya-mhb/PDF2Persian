# pdf_translator_local_models.py
import torch
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
import re
import os
import logging
from arabic_reshaper import reshape
from bidi.algorithm import get_display
import nltk
from typing import List, Dict, Tuple
from tqdm import tqdm
import requests
import hashlib
import sys

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase import pdfmetrics


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelManager:
    """Manages model downloading and loading from project directory"""

    def __init__(self, project_models_dir: str = "models"):
        self.project_models_dir = project_models_dir
        self.model_configs = {
            "mbart-large-50": {
                "name": "facebook/mbart-large-50-many-to-many-mmt",
                "local_path": os.path.join(project_models_dir, "mbart-large-50"),
                "files": [
                    "config.json",
                    "pytorch_model.bin",
                    "sentencepiece.bpe.model",
                    "tokenizer.json",
                    "tokenizer_config.json"
                ]
            }
        }

    def is_model_downloaded(self, model_key: str) -> bool:
        """Check if model is already downloaded in project folder"""
        if model_key not in self.model_configs:
            return False

        local_path = self.model_configs[model_key]["local_path"]
        required_files = self.model_configs[model_key]["files"]

        # Check if all required files exist
        for file in required_files:
            if not os.path.exists(os.path.join(local_path, file)):
                return False
        return True

    def download_model(self, model_key: str = "mbart-large-50"):
        """Download model to project folder"""
        if model_key not in self.model_configs:
            raise ValueError(f"Unknown model: {model_key}")

        if self.is_model_downloaded(model_key):
            print(f"✅ Model '{model_key}' already downloaded in project folder")
            return True

        config = self.model_configs[model_key]
        local_path = config["local_path"]
        model_name = config["name"]

        print(f"📥 Downloading {model_name} to project folder...")
        print(f"📍 Location: {os.path.abspath(local_path)}")

        # Create directory
        os.makedirs(local_path, exist_ok=True)

        try:
            # Download using transformers with custom cache_dir
            tokenizer = MBart50TokenizerFast.from_pretrained(
                model_name,
                cache_dir=local_path,
                local_files_only=False
            )

            model = MBartForConditionalGeneration.from_pretrained(
                model_name,
                cache_dir=local_path,
                local_files_only=False
            )

            # Save explicitly to our project folder
            tokenizer.save_pretrained(local_path)
            model.save_pretrained(local_path)

            print(f"✅ Model successfully downloaded to: {local_path}")
            return True

        except Exception as e:
            print(f"❌ Failed to download model: {e}")
            return False

    def get_model_path(self, model_key: str = "mbart-large-50") -> str:
        """Get the local path for the model"""
        return self.model_configs[model_key]["local_path"]

class PDFProcessor:
    def __init__(self, use_ocr: bool = True):
        self.use_ocr = use_ocr
        self.logger = logging.getLogger(__name__)

    def extract_text_basic(self, pdf_path: str) -> str:
        """Extract text from simple text-based PDFs"""
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except Exception as e:
            self.logger.error(f"Basic extraction failed: {e}")
            return ""

    def extract_text_with_ocr(self, pdf_path: str) -> str:
        """Use OCR for scanned PDFs"""
        try:
            text = ""
            images = convert_from_path(pdf_path, dpi=200)

            for i, image in enumerate(images):
                page_text = pytesseract.image_to_string(image, lang='fas')
                text += f"Page {i+1}:\n{page_text}\n\n"

            return text
        except Exception as e:
            self.logger.error(f"OCR extraction failed: {e}")
            return self.extract_text_basic(pdf_path)

    def clean_text(self, text: str) -> str:
        """Clean extracted text"""
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
        return text.strip()

    def chunk_text(self, text: str, chunk_size: int = 400) -> List[str]:
        """Split text into chunks"""
        if not text:
            return []

        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def process_pdf(self, pdf_path: str) -> Tuple[str, List[str]]:
        """Process PDF and return text + chunks"""
        self.logger.info(f"Processing PDF: {pdf_path}")

        text = self.extract_text_basic(pdf_path)
        if not text or len(text.strip()) < 50:
            self.logger.info("Using OCR...")
            text = self.extract_text_with_ocr(pdf_path)

        cleaned_text = self.clean_text(text)
        chunks = self.chunk_text(cleaned_text)

        self.logger.info(f"Extracted {len(chunks)} chunks")
        return cleaned_text, chunks

class EnglishToPersianTranslator:
    def __init__(self, model_manager: ModelManager, model_key: str = "mbart-large-50"):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_manager = model_manager
        self.model_key = model_key

        print(f"🖥️  Using device: {self.device}")

        # Ensure model is downloaded
        if not self.model_manager.download_model(model_key):
            raise Exception("Failed to download model")

        # Load model from project folder
        model_path = self.model_manager.get_model_path(model_key)
        print(f"📁 Loading model from: {model_path}")

        self.tokenizer = MBart50TokenizerFast.from_pretrained(model_path)
        self.model = MBartForConditionalGeneration.from_pretrained(model_path)
        self.model = self.model.to(self.device)

        self.tokenizer.src_lang = "en_XX"
        print("✅ Model loaded successfully from project folder!")

    def translate_text(self, text: str) -> str:
        """Translate single text chunk"""
        if not text.strip():
            return ""

        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            translated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.lang_code_to_id["fa_IR"],
                max_length=512,
                num_beams=3,
                early_stopping=True
            )

            translation = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            return self._postprocess_persian_text(translation)

        except Exception as e:
            self.logger.error(f"Translation error: {e}")
            return f"[Error: {str(e)}]"

    def translate_batch(self, texts: List[str]) -> List[str]:
        """Translate multiple texts"""
        translations = []
        for text in tqdm(texts, desc="Translating"):
            translations.append(self.translate_text(text))
        return translations

    def _postprocess_persian_text(self, text: str) -> str:
        """Process Persian text for proper display"""
        try:
            reshaped_text = reshape(text)
            return get_display(reshaped_text)
        except:
            return text

class TranslationPipeline:
    def __init__(self):
        self.model_manager = ModelManager()
        self.pdf_processor = PDFProcessor()
        self.translator = EnglishToPersianTranslator(self.model_manager)

    def translate_pdf(self, pdf_path: str, output_path: str = None):
        """Main translation pipeline"""
        print(f"📖 Processing: {os.path.basename(pdf_path)}")

        # Extract text
        full_text, chunks = self.pdf_processor.process_pdf(pdf_path)
        if not chunks:
            print("❌ No text extracted")
            return None

        print(f"🔤 Extracted {len(chunks)} text chunks")

        # Translate
        print("🔄 Translating to Persian...")
        translated_chunks = self.translator.translate_batch(chunks)

        # Combine results
        result = {
            "original_text": full_text,
            "translated_text": "\n\n".join(translated_chunks),
            "chunks_original": chunks,
            "chunks_translated": translated_chunks,
            "stats": {
                "original_chars": len(full_text),
                "translated_chars": sum(len(chunk) for chunk in translated_chunks),
                "num_chunks": len(chunks)
            }
        }

        # Save output
        if output_path:
            self._save_result(result, output_path)
            print(f"💾 Saved to: {output_path}")

        return result

    def _save_result(self, result: Dict, output_path: str):
        """Save translation to text and PDF files with reshaped Persian text"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # ✅ Save as UTF-8 text file
        txt_path = output_path
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=== ENGLISH TO PERSIAN TRANSLATION ===\n\n")
            f.write("STATISTICS:\n")
            f.write(f"- Original characters: {result['stats']['original_chars']}\n")
            f.write(f"- Translated characters: {result['stats']['translated_chars']}\n")
            f.write(f"- Number of chunks: {result['stats']['num_chunks']}\n\n")

            f.write("ORIGINAL TEXT (first 2000 chars):\n")
            f.write(result["original_text"][:2000] + "...\n\n")
            f.write("="*50 + "\n\n")
            f.write("PERSIAN TRANSLATION:\n")

            # Apply reshaping for proper display
            reshaped_output = "\n\n".join([
                get_display(reshape(chunk)) for chunk in result["chunks_translated"]
            ])
            f.write(reshaped_output)

            f.write("\n\n" + "="*60 + "\n")
            f.write("DETAILED CHUNK COMPARISON:\n")
            for i, (orig, trans) in enumerate(zip(result["chunks_original"], result["chunks_translated"])):
                reshaped_trans = get_display(reshape(trans))
                f.write(f"\n--- Chunk {i+1} ---\n")
                f.write(f"EN: {orig}\n")
                f.write(f"FA: {reshaped_trans}\n")

        print(f"💾 Saved readable TXT: {txt_path}")

        # ✅ Save Persian translation as a PDF
        pdf_path = txt_path.replace(".txt", ".pdf")
        pdfmetrics.registerFont(UnicodeCIDFont('HYSMyeongJo-Medium'))  # RTL-safe font

        c = canvas.Canvas(pdf_path, pagesize=A4)
        width, height = A4
        c.setFont("HYSMyeongJo-Medium", 12)

        # Reshape text for PDF rendering
        persian_text = "\n\n".join([
            get_display(reshape(chunk)) for chunk in result["chunks_translated"]
        ])

        # Write text with right alignment (RTL)
        y = height - 50
        for line in persian_text.split('\n'):
            if not line.strip():
                y -= 20
                continue
            c.drawRightString(width - 40, y, line)
            y -= 20
            if y < 50:  # New page if out of space
                c.showPage()
                c.setFont("HYSMyeongJo-Medium", 12)
                y = height - 50

        c.save()
        print(f"📘 Saved Persian PDF: {pdf_path}")

def check_system_requirements():
    """Check if system has required dependencies"""
    print("🔍 Checking system requirements...")

    # Check Tesseract
    try:
        pytesseract.get_tesseract_version()
        print("✅ Tesseract OCR: Available")
    except:
        print("❌ Tesseract OCR: Not found - please install tesseract-ocr")

    # Check Python
    print(f"✅ Python: {sys.version}")

    # Check PyTorch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"✅ CUDA Available: {torch.cuda.is_available()}")

def main():
    """Main function to run the translator"""
    print("=" * 60)
    print("🔄 PDF English to Persian Translator")
    print("📍 Models stored in project folder: 'models/'")
    print("=" * 60)

    # Check system requirements
    check_system_requirements()
    print()

    # Download NLTK data if needed
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("📥 Downloading NLTK data...")
        nltk.download('punkt')

    # Create pipeline
    pipeline = TranslationPipeline()

    # Check for PDFs in data/raw
    pdf_files = [f for f in os.listdir('data/raw') if f.endswith('.pdf')]

    if not pdf_files:
        print("❌ No PDF files found in 'data/raw' directory")
        print("💡 Please add PDF files to the 'data/raw' folder and run again.")
        print("📁 Project structure:")
        print("   data/raw/          <-- Put your PDF files here")
        print("   models/            <-- Models are stored here")
        print("   outputs/           <-- Translations are saved here")
        return

    # Process each PDF
    for pdf_file in pdf_files:
        pdf_path = os.path.join('data/raw', pdf_file)
        output_path = os.path.join('outputs', f'translated_{pdf_file}.txt')

        print(f"\n🎯 Processing: {pdf_file}")
        result = pipeline.translate_pdf(pdf_path, output_path)

        if result:
            # Show sample
            print("\n📝 Translation Sample:")
            if result['chunks_translated']:
                print(f"EN: {result['chunks_original'][0][:100]}...")
                print(f"FA: {result['chunks_translated'][0][:100]}...")

            print(f"📊 Stats: {result['stats']}")
        print("✅ Completed!")

if __name__ == "__main__":
    main()