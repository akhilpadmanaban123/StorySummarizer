import cv2
import pytesseract
import openai
from flask import Flask, request, render_template

app = Flask(__name__)

openai.api_key="YOUR_API KEYYYYYYYYYYYYYYYYY"

def Image_to_Text(img_path):
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    # Load the image using OpenCV
    img = cv2.imread(img_path)

    # Preprocess the image (convert to grayscale and apply threshold)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Perform OCR using Tesseract
    text = pytesseract.image_to_string(thresh, lang='eng')

    # Print the recognized text
    return text

def generate_Summary(text):
    response = openai.Completion.create(
        model='text-davinci-002',
        prompt=f'{text}',
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.4
    )
    return response.choices[0].text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' and 'image' in request.files:
        image = request.files['image']
        image_path = f"static/{image.filename}"
        image.save(image_path)
        text = Image_to_Text(image_path)
        summary = generate_Summary('Summarize the content '+text)
        return render_template('index.html', summary=summary, text=text)
    else:
        return render_template('upload.html')


if __name__ == '__main__':
    app.run(debug=True)
