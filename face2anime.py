import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np




def load_image(image_path, size=(256, 256)):
    """
    Load and preprocess the input image.
    """
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension





def load_model(model_path):
    """
    Load the entire trained model from a .pth file.
    """
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    return model





def add_noise(image, noise_level=0.1):
    """
    Add Gaussian noise to the image.
    """
    noise = torch.randn_like(image) * noise_level
    noisy_image = image + noise
    return torch.clamp(noisy_image, 0, 1)




def denoise_image(model, noisy_image, num_steps=100):
    """
    Gradually denoise the image using the trained model.
    """
    model.eval()
    with torch.no_grad():
        x = noisy_image
        for t in reversed(range(num_steps)):
            t_tensor = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
            predicted_noise = model(x, t_tensor)
            x = x - predicted_noise  # Remove predicted noise
            x = torch.clamp(x, 0, 1)
    return x



def human_to_anime(model, image_path, output_path, noise_level=0.1, num_steps=100):
    """
    Convert a human face image to an anime face.
    """
    # Load and preprocess the image
    image = load_image(image_path)
    
    # Add noise to the image
    noisy_image = add_noise(image, noise_level)
    
    # Denoise the image using the trained model
    anime_image = denoise_image(model, noisy_image, num_steps)
    
    # Convert the result to a PIL Image and save
    result = transforms.ToPILImage()(anime_image.squeeze(0))
    result.save(output_path)
    print(f"Anime-style image saved to {output_path}")


