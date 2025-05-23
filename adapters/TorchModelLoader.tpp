#include "TorchModelLoader.hpp"
#include <stdexcept>

namespace adapters {

torch::jit::script::Module TorchModelLoader::load(const std::string& model_path) {
    try {
        return torch::jit::load(model_path);
    } catch (const std::exception& e) {
        throw std::runtime_error("Error al cargar el modelo: " + std::string(e.what()));
    }
}

} // namespace adapters