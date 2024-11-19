from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from tensorflow.lite.python.interpreter import Interpreter
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

app = FastAPI()

# Load the TensorFlow Lite model
tflite_model_path = "fake_currency_detector.tflite"
interpreter = Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to preprocess the image
def preprocess_image(image: Image.Image, target_size=(256, 256)):
    image = image.convert('L')  # Convert to grayscale
    image = image.resize(target_size)  # Resize to model's expected input size
    img_array = img_to_array(image) / 255.0  # Normalize the image
    return np.expand_dims(img_array, axis=0).astype(np.float32)  # Add batch dimension and cast to float32

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Load and preprocess the uploaded image
        image = Image.open(file.file)
        processed_image = preprocess_image(image)

        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], processed_image)

        # Perform inference
        interpreter.invoke()

        # Get the output prediction
        prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
        label = "Real" if prediction > 0.5 else "Fake"  # Binary classification
        confidence = float(prediction)  # Probability for the 'Fake' class

        # Return the result
        return JSONResponse(content={
            "filename": file.filename,
            "prediction": label,
            "confidence": confidence
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Run this app with `uvicorn app:app --reload`.