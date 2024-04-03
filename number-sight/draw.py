from tkinter import Tk, Canvas, Button, Label, Entry
import numpy as np
import tensorflow as tf
from tensorflow import keras
from parse_training_data import parse_line

# Define grid dimensions (change these for a different size)
width = 16
height = 16
square_size = 15

# Create the main window
window = Tk()
window.title("Draw and Store Pixels")

# Create a 2D list to store pixel colors (white initially)
pixel_colors = [[0 for _ in range(width)] for _ in range(height)]

model = None
fromFile = False
modelFileName = "./models/model-0.keras"
try:
    with open(modelFileName, 'r') as file:
        fromFile = True
        model = keras.models.load_model(modelFileName)
except FileNotFoundError:
    fromFile = False
    
if not fromFile:
    # Define the model
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(16, 16)),  # Assuming input data is 28x28 images
        keras.layers.Dense(128, activation="relu"),  # First hidden layer with 128 neurons and ReLU activation
        keras.layers.Dense(10, activation="softmax")  # Output layer with 10 neurons (for 10 class classification)
    ])

    # Compile the model
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

def get_coordinates(event):
    x = event.x // square_size
    y = event.y // square_size
    if 0 <= x < width and 0 <= y < height:
        return x, y
    else:
        return None

def on_drag(event):
    coords = get_coordinates(event)
    if coords:
        x, y = coords
        canvas.itemconfig(pixels[x*width + y], fill = "white")
        pixel_colors[y][x] = 1

def get_likely_number(prediction):
    n, max_chance = 0, 0
    for i in range(len(prediction[0])):
        if prediction[0][i] > max_chance:
            n = i
            max_chance = prediction[0][i]
    return n, max_chance
   
def clear():
    global pixels
    global pixel_colors
    pixels = get_blank_canvas(width, height, square_size, canvas)
    pixel_colors = [[0 for _ in range(width)] for _ in range(height)]
    
def get_blank_canvas(width, height, square_size, canvas):
    pixels = []
    for i in range(width):
        for j in range(height):
            x0, y0 = i*square_size, j*square_size
            x1, y1 = x0+square_size, y0+square_size
            pixels.append(canvas.create_rectangle(x0, y0, x1, y1, fill="black"))
    return pixels

def store_pixels():
    global model
    model.save("./models/model-0.keras")
    answer = num_pixels_entry.get()
    # print(f"Storing pixels for answer {answer}")
    model.fit(np.array([pixel_colors]), np.array([convert_answer(answer)]), epochs=5, verbose=0)
    clear()

def convert_answer(answer):
    a = [0 for _ in range(10)]
    a[int(answer)] = 1
    return a

def predict(event):
    data = np.array(pixel_colors)
    data = tf.expand_dims(data, axis=0)
    n, p = get_likely_number(model.predict(data, verbose=0))
    prediction_label.config(text=f"Prediction: {n} - {p*100:.2f}% chance")
    
# Create the canvas for drawing
canvas = Canvas(window, width=width*square_size, height=height*square_size, bg="white")
pixels = get_blank_canvas(width, height, square_size, canvas)
canvas.pack()

# Define the selected drawing color (initially black)
selected_color = "black"

# Create a button to store pixel data
store_button = Button(window, text="Store Pixels", command=store_pixels)
store_button.pack(pady=10)

# Create a button to store pixel data
store_button = Button(window, text="Clear", command=clear)
store_button.pack(pady=10)

num_pixels_label = Label(window, text="Answer (0-9):")
num_pixels_label.pack(pady=5)
num_pixels_entry = Entry(window)
num_pixels_entry.pack()

prediction_label = Label(window, text="Prediction: None")
prediction_label.pack(pady=5)

# Bind events to the canvas
canvas.bind("<B1-Motion>", on_drag)  # Bind dragging motion
canvas.bind("<B1-ButtonRelease>", predict)  # Bind button release

# Run the main loop
window.mainloop()
