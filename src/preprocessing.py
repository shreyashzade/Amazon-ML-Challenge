import os
from PIL import Image
import cv2
import torchvision.transforms as transforms

# Preprocessing Function
def preprocess_image(image_path, target_size=(224, 224), augment=False):
    """
    Preprocess a single image: resize, normalize, and augment (optional).
    
    Parameters:
    - image_path: str, path to the image file.
    - target_size: tuple, resize the image to this target size (width, height).
    - augment: bool, apply augmentation if True.
    
    Returns:
    - preprocessed_img: the preprocessed image tensor.
    """
    
    # Load the image
    img = Image.open(image_path).convert('RGB')  # Ensure it's RGB
    
    # Define basic transformations: resize and normalization
    transform_list = [
        transforms.Resize(target_size),  # Resize the image
        transforms.ToTensor(),  # Convert image to a Tensor (between 0 and 1)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize as per ImageNet pre-trained model
    ]
    
    # Add augmentations (only if specified)
    if augment:
        augmentation = transforms.RandomChoice([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),  # Random rotation up to 10 degrees
            transforms.RandomResizedCrop(size=target_size, scale=(0.8, 1.0)),
        ])
        transform_list.insert(0, augmentation)  # Apply augmentations before resizing
    
    # Compose the transformations
    transform = transforms.Compose(transform_list)
    
    # Apply the transformations
    preprocessed_img = transform(img)
    
    return preprocessed_img

# Batch Image Preprocessing Function
def preprocess_images_in_directory(directory_path, output_dir, target_size=(224, 224), augment=False):
    """
    Preprocess all images in a directory.
    
    Parameters:
    - directory_path: str, path to the directory containing images.
    - output_dir: str, path to save the preprocessed images.
    - target_size: tuple, resize the image to this target size (width, height).
    - augment: bool, apply augmentation if True.
    """
    
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Iterate through all images in the directory
    for image_file in os.listdir(directory_path):
        image_path = os.path.join(directory_path, image_file)
        
        # Skip non-image files
        if not image_file.endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        # Preprocess the image
        preprocessed_img = preprocess_image(image_path, target_size, augment)
        
        # Convert the tensor back to an image and save it to the output directory
        save_image(preprocessed_img, os.path.join(output_dir, image_file))

# Helper Function to Save Tensor as Image
def save_image(tensor, output_path):
    """
    Convert a tensor to an image and save it.
    
    Parameters:
    - tensor: Tensor, image data in tensor format.
    - output_path: str, path to save the image.
    """
    # Convert the tensor to PIL image
    img = transforms.ToPILImage()(tensor)
    img.save(output_path)

# Example usage: preprocess and save all images in a directory
input_dir = 'images/'  # Directory with your downloaded images
output_dir = 'preprocessed_images/'  # Directory to save preprocessed images
preprocess_images_in_directory(input_dir, output_dir, target_size=(224, 224), augment=True)

