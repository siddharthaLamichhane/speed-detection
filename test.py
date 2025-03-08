import pytesseract

# Set the correct path for Tesseract-OCR
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Test if Tesseract works
print(pytesseract.get_tesseract_version())
