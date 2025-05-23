#include "TorchImageProcessor.hpp"
#include <stdexcept>

namespace adapters {

torch::Tensor TorchImageProcessor::preprocess(const std::string& image_path) {
    // Simulación: en la práctica, usar OpenCV para cargar la imagen
    // Crear un tensor de 3x224x224 (RGB, normalizado)
    torch::Tensor tensor = torch::rand({1, 3, 224, 224});
    
    // Normalización (ImageNet: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tensor = (tensor - torch::tensor({0.485, 0.456, 0.406}).view({1, 3, 1, 1})) / 
             torch::tensor({0.229, 0.224, 0.225}).view({1, 3, 1, 1});
    
    return tensor;
}

} // namespace adapters