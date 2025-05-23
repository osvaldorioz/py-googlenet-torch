#ifndef TORCH_MODEL_LOADER_HPP
#define TORCH_MODEL_LOADER_HPP

#include "../ports/ModelLoaderPort.hpp"
#include <torch/script.h>
#include <string>

namespace adapters {

class TorchModelLoader : public ports::ModelLoaderPort {
public:
    torch::jit::script::Module load(const std::string& model_path) override;
};

} // namespace adapters

#include "TorchModelLoader.tpp"

#endif // TORCH_MODEL_LOADER_HPP