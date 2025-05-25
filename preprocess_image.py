import torch
import torchvision.transforms as transforms
from PIL import Image

def preprocess_image(image_path, output_path):
    # Cargar imagen
    image = Image.open(image_path).convert("RGB")

    # Definir transformaciones
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Redimensionar a 224x224
        transforms.ToTensor(),  # Convertir a tensor [C, H, W]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalización ImageNet
    ])

    # Aplicar transformaciones
    tensor = transform(image)

    # Añadir dimensión de batch [1, C, H, W]
    tensor = tensor.unsqueeze(0)

    # Guardar tensor
    torch.save(tensor, output_path)
    print(f"Tensor guardado en {output_path}")
    print(f"Forma del tensor: {tensor.shape}")

if __name__ == "__main__":
    image_path = "test_cat3.jpg" 
    output_path = "preprocessed_tensor.pt"
    preprocess_image(image_path, output_path)