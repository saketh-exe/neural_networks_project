import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model

# ==========================================
# 1. LOAD AND PREP YOUR CUSTOM DATASET
# ==========================================
def load_training_data(folder_path, max_images=None):
    print(f"Scanning {folder_path} for images...")
    image_list = []
    
    # Get all the filenames in your folder
    filenames = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Optional: limit the number of images to speed up initial testing
    if max_images:
        filenames = filenames[:max_images]
        
    for filename in filenames:
        img_path = os.path.join(folder_path, filename)
        
        # Open image, ensure it is grayscale, and resize strictly to 64x64 just in case
        img = Image.open(img_path).convert('L').resize((64, 64))
        
        # Convert the image to a NumPy array of numbers
        img_array = np.array(img)
        image_list.append(img_array)
        
    # Convert the python list into a massive 4D NumPy block: (Num_Images, 64, 64, 1)
    dataset_array = np.array(image_list).reshape(-1, 64, 64, 1)
    
    # CRITICAL: Normalize the pixel values. 
    # Pixels are 0-255. Neural networks need them to be between 0.0 and 1.0.
    dataset_array = dataset_array.astype('float32') / 255.0
    
    print(f"Successfully loaded {dataset_array.shape[0]} images!")
    return dataset_array

# --- POINT THIS TO YOUR NEW FOLDER OF 54,000 PATCHES ---
# Note: For your very first test run, you might want to set max_images=5000 
# just to make sure the code works before waiting for all 54k to process.
TRAIN_FOLDER = "./images/train"
training_data = load_training_data(TRAIN_FOLDER, max_images=10000) 

# Make sure you have this import at the top of your file if you don't already:

# ==========================================
# 2 & 3. LOAD OR TRAIN THE AUTOENCODER
# ==========================================
MODEL_PATH = "wall_autoencoder.keras"
FORCE_RETRAIN = False  # Change to True if you add new training images and want to start over

if os.path.exists(MODEL_PATH) and not FORCE_RETRAIN:
    print(f"\nFound existing model! Loading weights from {MODEL_PATH}...")
    # This instantly loads the architecture, the learned weights, and the optimizer
    autoencoder = load_model(MODEL_PATH)
    print("Model loaded successfully. Ready for testing.")

else:
    print("\nNo saved model found (or FORCE_RETRAIN is True).")
    print("Building and training a new Autoencoder...")
    
    # Build the architecture from scratch
    autoencoder = models.Sequential([
        layers.Input(shape=(64, 64, 1)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), padding='same'),
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), padding='same'),
        
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.UpSampling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.UpSampling2D((2, 2)),
        layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')
    ])

    autoencoder.compile(optimizer='adam', loss='mse')
    
    # Train the model
    autoencoder.fit(
        training_data, 
        training_data, 
        epochs=10,            
        batch_size=128,       
        shuffle=True,         
        verbose=1
    )
    
    # Save the trained model to your hard drive
    autoencoder.save(MODEL_PATH)
    print(f"Training Complete! Model saved locally as '{MODEL_PATH}'.")

# ==========================================
# 4. TEST THE MODEL & FIND THE ANOMALY
# ==========================================
# Point this to a folder that contains your defect/anomaly patches
TEST_FOLDER = "path/to/your/defect_wall_patches"

def testing_anomalies(folder_path, model):
    print("\n--- Testing Phase Started ---")
    filenames = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if not filenames:
        print("Error: No test images found in the folder!")
        return

    for i in range(len(filenames)):

        # Let's just grab the first defect image to test
        test_img_path = os.path.join(folder_path, filenames[i])
        
        # Prep the image exactly like we did for training
        img = Image.open(test_img_path).convert('L').resize((64, 64))
        test_array = np.array(img).astype('float32') / 255.0
        
        # Reshape for the model: (1 image, 64 width, 64 height, 1 color channel)
        test_input = test_array.reshape(1, 64, 64, 1)

        # Force the model to try and reconstruct the defect
        print("Reconstructing...")
        reconstructed = model.predict(test_input)

        # Calculate the error difference
        error_heatmap = np.abs(test_input - reconstructed)

        # Display the results
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.title("1. Original Defect Image")
        plt.imshow(test_input[0, :, :, 0], cmap='gray', vmin=0, vmax=1)

        plt.subplot(1, 3, 2)
        plt.title("2. AI's Attempt to Redraw")
        plt.imshow(reconstructed[0, :, :, 0], cmap='gray', vmin=0, vmax=1)

        plt.subplot(1, 3, 3)
        plt.title("3. Error Heatmap")
        # Using 'hot' colormap so anomalies glow yellow/white against a dark background
        plt.imshow(error_heatmap[0, :, :, 0], cmap='hot')
        plt.colorbar()

        plt.show()

# Run the test
testing_anomalies("./images/test", autoencoder)