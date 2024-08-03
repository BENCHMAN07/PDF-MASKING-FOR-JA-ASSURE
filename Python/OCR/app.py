from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
from flask_cors import CORS
import fitz  # PyMuPDF
import pandas as pd
import pytesseract
from PIL import Image, ImageFilter, ImageDraw
import os
import numpy as np
import cv2
import re
import spacy
import logging
import io
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load spaCy model for NLP
nlp = spacy.load('en_core_web_sm')

# Configure logging
logging.basicConfig(level=logging.DEBUG)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        logging.error('No file part in the request')
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        logging.error('No selected file')
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        output_filename = f"masked_{filename}"
        output_file_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

        try:
            # Process the file (PDF)
            if filename.rsplit('.', 1)[1].lower() == 'pdf':
                data, text_data = extract_data_from_pdf(file_path)
                create_masked_pdf(file_path, output_file_path, data)
                save_to_csv(text_data, filename)
            else:
                text, marks = extract_special_marks(file_path)
                data = {'text': text, 'marks': marks}
                masked_image = mask_sensitive_info_image(Image.open(file_path))
                masked_image.save(output_file_path)
        except Exception as e:
            logging.error(f'Error processing file: {e}')
            return jsonify({'error': 'Error processing file', 'details': str(e)})

        file_url = url_for('uploaded_file', filename=filename)
        masked_file_url = url_for('output_file', filename=output_filename)
        
        logging.debug(f'File URL: {file_url}')
        logging.debug(f'Masked File URL: {masked_file_url}')

        return jsonify({
            'message': 'File processed successfully',
            'file_url': file_url,
            'masked_file_url': masked_file_url
        })
    else:
        logging.error('File type not allowed')
        return jsonify({'error': 'File type not allowed'})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    logging.debug(f'Serving uploaded file: {filename}')
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/output/<filename>')
def output_file(filename):
    logging.debug(f'Serving output file: {filename}')
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

def extract_data_from_pdf(pdf_path):
    structured_data = []
    text_data = []

    try:
        document = fitz.open(pdf_path)
    except Exception as e:
        logging.error(f"Error opening PDF: {e}")
        return structured_data, text_data

    try:
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            
            # Extract text and preserve layout
            blocks = page.get_text("blocks")
            for block in blocks:
                block_text = block[4].strip()
                if block_text:
                    masked_text = mask_sensitive_info(block_text)
                    structured_data.append({
                        'page_num': page_num,
                        'bbox': block[:4],  # Bounding box of the block
                        'original_text': block_text,
                        'masked_text': masked_text
                    })
                    text_data.append({
                        'page_num': page_num,
                        'original_text': block_text,
                        'masked_text': masked_text
                    })
    except Exception as e:
        logging.error(f"Error processing PDF content: {e}")

    return structured_data, text_data

def extract_special_marks(image_path):
    try:
        image = Image.open(image_path)
        text = extract_text_from_image(image)
        marks = detect_marks(image)
        return text, marks
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return "", []

def extract_text_from_image(image):
    try:
        # Convert image to grayscale
        gray_image = image.convert('L')
        
        # Resize the image to enhance text recognition
        width, height = gray_image.size
        new_size = (width * 2, height * 2)
        resized_image = gray_image.resize(new_size, Image.LANCZOS)  # Use LANCZOS filter for high quality
        
        # Apply sharpening filter
        enhanced_image = resized_image.filter(ImageFilter.SHARPEN)
        
        # Binarize the image (Thresholding)
        binary_image = enhanced_image.point(lambda x: 0 if x < 128 else 255, '1')
        
        # Perform OCR
        text = pytesseract.image_to_string(binary_image, config='--psm 6')
        return text
    except Exception as e:
        logging.error(f"Error during OCR: {e}")
        return ""

def detect_marks(image):
    try:
        # Convert PIL image to numpy array for OpenCV processing
        img_array = np.array(image)

        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

        # Thresholding to identify potential tick marks
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

        # Find contours which might represent ticks or shaded areas
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        marks = []
        for contour in contours:
            # Get bounding box of the contour
            x, y, w, h = cv2.boundingRect(contour)

            # Filter based on contour size to detect relevant marks
            if w > 10 and h > 10:  # Adjust these values as needed
                mark_image = crop_image(img_array, (x, y, w, h))
                text = extract_text_from_image(Image.fromarray(mark_image))
                marks.append({
                    'text': text
                })
                
        return marks
    except Exception as e:
        logging.error(f"Error during mark detection: {e}")
        return []

def crop_image(image_array, bbox):
    x, y, w, h = bbox
    return image_array[y:y+h, x:x+w]

def mask_sensitive_info(text):
    masked_text = text

    # Mask names following titles
    masked_text = re.sub(r'\b(Mr|Ms|Mrs)\.?\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b', '[MASKED]', masked_text, flags=re.IGNORECASE)

    # Mask first and last names
    masked_text = re.sub(r'\b[A-Z][a-z]+\s[A-Z][a-z]+\b', '[MASKED]', masked_text)

    # Mask email addresses
    masked_text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[MASKED]', masked_text)
    
    # Mask phone numbers (Indian format - 10 digits)
    masked_text = re.sub(r'\b\d{10}\b', '[MASKED]', masked_text)

    # Mask bank account numbers (assuming 9-18 digits)
    masked_text = re.sub(r'\b\d{9,18}\b', '[MASKED]', masked_text)
    
    # Mask clinic names (specific names)
    clinic_names = ['clinic', 'hospital', 'medical center']  # Add specific clinic names as needed
    for name in clinic_names:
        masked_text = re.sub(r'\b{}\b'.format(re.escape(name)), '[MASKED]', masked_text, flags=re.IGNORECASE)

    return masked_text

def mask_sensitive_info_image(image):
    try:
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

        # Threshold to create a binary image
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        # Find contours of the text regions
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            text_region = image.crop((x, y, x + w, y + h))
            text = extract_text_from_image(text_region)

            # Mask the sensitive information in the text
            masked_text = mask_sensitive_info(text)

            # Draw a rectangle over the sensitive information
            draw = ImageDraw.Draw(image)
            draw.rectangle([x, y, x + w, y + h], fill="white")
            draw.text((x, y), masked_text, fill="black")

        return image
    except Exception as e:
        logging.error(f"Error during image masking: {e}")
        return image

def create_masked_pdf(input_pdf_path, output_pdf_path, data):
    try:
        document = fitz.open(input_pdf_path)
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            
            # Detect and mask images
            for img in page.get_images(full=True):
                xref = img[0]
                base_image = document.extract_image(xref)
                img_data = base_image['image']

                # Load the image using PIL
                image = Image.open(io.BytesIO(img_data))

                # Get image rectangle
                img_rect = page.get_image_rects(xref)[0]
                logging.debug(f"Masking image at {img_rect}")

                # Mask the image with a black rectangle using Matplotlib
                fig, ax = plt.subplots(figsize=(img_rect.width / 72, img_rect.height / 72), dpi=72)
                ax.add_patch(plt.Rectangle((0, 0), img_rect.width, img_rect.height, color='black'))
                ax.axis('off')

                # Save the masked image to a buffer
                buf = io.BytesIO()
                plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
                plt.close(fig)
                buf.seek(0)

                # Replace the image in the document with the black box
                page.insert_image(img_rect, stream=buf, keep_proportion=False)
                buf.close()

            # Redact the text regions and apply redactions
            for item in data:
                if item['page_num'] == page_num:
                    bbox = item['bbox']
                    masked_text = item['masked_text']

                    page.add_redact_annot(fitz.Rect(bbox), fill=(1, 1, 1))
                    page.apply_redactions()

                    page.insert_text(fitz.Point(bbox[0], bbox[1]), masked_text, fontsize=12, color=(0, 0, 0))

        document.save(output_pdf_path)
        logging.info(f"Masked PDF saved at {output_pdf_path}")
    except Exception as e:
        logging.error(f"Error creating masked PDF: {e}")

def save_to_csv(data, filename):
    try:
        original_texts = []
        masked_texts = []
        for item in data:
            original_texts.append({'page_num': item['page_num'], 'original_text': item['original_text']})
            masked_texts.append({'page_num': item['page_num'], 'masked_text': item['masked_text']})

        original_csv_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{filename}_original.csv")
        masked_csv_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{filename}_masked.csv")

        pd.DataFrame(original_texts).to_csv(original_csv_path, index=False)
        pd.DataFrame(masked_texts).to_csv(masked_csv_path, index=False)

        logging.info(f"CSV files saved at {original_csv_path} and {masked_csv_path}")
    except Exception as e:
        logging.error(f"Error saving CSV files: {e}")

if __name__ == '__main__':
    app.run(debug=True)
