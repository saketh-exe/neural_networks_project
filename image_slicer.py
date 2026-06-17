import os
from PIL import Image

def slice_image_into_patches(input_image_path, output_folder ,patch_size=64):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the image and convert to Grayscale
    name = input_image_path.split("/")[-1].split("-")[0]
    img = Image.open(input_image_path).convert('L')
    width, height = img.size

    # Calculate how many full 64x64 squares fit across and down
    cols = width // patch_size
    rows = height // patch_size

    patch_count = 0

    # Loop through the grid and cut out each square
    for row in range(rows):
        for col in range(cols):
            left = col * patch_size
            top = row * patch_size
            right = left + patch_size
            bottom = top + patch_size

            # Cut the patch
            patch = img.crop((left, top, right, bottom))
            
            # Save it with a unique name (e.g., patch_0_0.jpg, patch_0_1.jpg)
            patch_filename = f"Image_{name}_patch_{row}_{col}.jpg"
            patch.save(os.path.join(output_folder, patch_filename))
            patch_count += 1

    print(f"Successfully created {patch_count} patches from {input_image_path}")

# --- How to use it ---
# Point it at ONE high-res photo of your clean wall
# slice_image_into_patches("path/to/one_clean_wall_photo.jpg", "path/to/training_folder")


for i in range(1,19):
    slice_image_into_patches(f"./images/raw_train/{i}-Image.jpg","./images/train/")