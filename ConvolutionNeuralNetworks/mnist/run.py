import tkinter as tk
from tensorflow import keras
import numpy as np
import io
from PIL import Image, ImageDraw

# Load the pre-trained MNIST model
model = keras.models.load_model('model.h5')

def xy(event):
    "Takes the coordinates of the mouse when you click the mouse"
    global lastx, lasty
    lastx, lasty = event.x, event.y

def addLine(event):
    """Creates a line when you drag the mouse
    from the point where you clicked the mouse to where the mouse is now"""
    global lastx, lasty
    canvas.create_line((lastx, lasty, event.x, event.y))
    # this makes the new starting point of the drawing
    lastx, lasty = event.x, event.y

def clear_canvas():
    canvas.delete("all")

def classify_digit():
    # Get the content of the canvas as a 28x28 image
    image = canvas.postscript(colormode='mono')
    img = Image.open(io.BytesIO(image.encode('utf-8')))
    img = img.resize((28, 28))
    img_data = np.array(img)

    # Preprocess the image for prediction
    img_data = img_data / 255.0  # Normalize pixel values (0-1)
    img_data = img_data.reshape(1, 28, 28, 1)  # Reshape for model input

    # Get the model prediction
    prediction = model.predict(img_data)
    digit = np.argmax(prediction)

    # Update the prediction label
    prediction_label.config(text=f"Predicted Digit: {digit}")

root = tk.Tk()
root.geometry("300x300")  # Adjust the window size to 28x28 pixels
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

canvas = tk.Canvas(root, width=28, height=28, bg='white')  # Set the canvas size to 28x28 pixels
canvas.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
canvas.bind("<Button-1>", xy)
canvas.bind("<B1-Motion>", addLine)

clear_button = tk.Button(root, text="Clear", command=clear_canvas)
classify_button = tk.Button(root, text="Classify", command=classify_digit)
clear_button.grid(row=1, column=0)
classify_button.grid(row=2, column=0)

prediction_label = tk.Label(root, text="Predicted Digit: ")
prediction_label.grid(row=3, column=0)

root.mainloop()

