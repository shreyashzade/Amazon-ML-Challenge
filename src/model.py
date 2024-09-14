import pytesseract
from PIL import Image
import re

# Ensure tesseract is installed and path is set up correctly
# You may need to specify the path to your Tesseract installation
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_from_image(image_path):
    """
    Extracts text from an image using Tesseract OCR.
    
    Parameters:
    - image_path: str, the path to the image.
    
    Returns:
    - text: str, the extracted text from the image.
    """
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text

# Example usage:
# extracted_text = extract_text_from_image('path_to_image.jpg')



import spacy
import re

# Load the pre-trained spaCy model
nlp = spacy.load('en_core_web_sm')

# Entity extraction function
def extract_entity(text, entity_type):
    """
    Extract specific entity (e.g., weight, dimensions) from the text using regex and NLP.
    
    Parameters:
    - text: str, the input text.
    - entity_type: str, type of entity to extract (e.g., "item_weight").
    
    Returns:
    - entity_value: str, the extracted value with the unit (e.g., '34 gram').
    """
    
    # Define basic regex patterns for entities
    patterns = {
        "item_weight": r"(\d+(\.\d+)?\s?(gram|kg|kilogram|mg|pound|ounce|g))",
        "width": r"(\d+(\.\d+)?\s?(cm|centimetre|mm|metre|inch))",
        "height": r"(\d+(\.\d+)?\s?(cm|centimetre|mm|metre|inch))",
        "voltage": r"(\d+(\.\d+)?\s?(volt|V|kV|mV))",
        "wattage": r"(\d+(\.\d+)?\s?(watt|kW|mW))"
    }
    
    # Check if the entity_type has a corresponding pattern
    if entity_type in patterns:
        pattern = patterns[entity_type]
        match = re.search(pattern, text)
        
        if match:
            return match.group(0)
    
    return ""  # If no match is found

# Example usage:
# entity_value = extract_entity(extracted_text, 'item_weight')


import os
import pandas as pd

def ocr_nlp_pipeline(image_dir, test_csv, entity_type):
    """
    Hybrid OCR + NLP pipeline that extracts entity values from images and processes them.
    
    Parameters:
    - image_dir: str, directory where images are stored.
    - test_csv: str, path to test CSV file.
    - entity_type: str, the type of entity to extract (e.g., "item_weight", "width").
    
    Returns:
    - output: DataFrame, the predictions in the required format (index, prediction).
    """
    
    # Load the test file
    test_data = pd.read_csv(test_csv)
    results = []
    
    # Loop through each image in the test data
    for _, row in test_data.iterrows():
        image_path = os.path.join(image_dir, row['image_link'].split('/')[-1])  # Assuming image_link contains file name
        
        # Step 1: OCR
        extracted_text = extract_text_from_image(image_path)
        
        # Step 2: NLP Entity Extraction
        entity_value = extract_entity(extracted_text, entity_type)
        
        # Append result
        results.append([row['index'], entity_value])
    
    # Create DataFrame for output
    output_df = pd.DataFrame(results, columns=['index', 'prediction'])
    return output_df

# Example usage:
# ocr_nlp_pipeline('image_directory/', 'dataset/test.csv', 'item_weight')



def save_predictions_to_csv(output_df, output_path):
    """
    Save the predictions to a CSV file.
    
    Parameters:
    - output_df: DataFrame, containing the index and predictions.
    - output_path: str, path to save the output CSV file.
    """
    output_df.to_csv(output_path, index=False)

# Example usage:
# save_predictions_to_csv(output_df, 'output/test_out.csv')
