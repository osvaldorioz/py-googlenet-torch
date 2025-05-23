#ifndef TORCH_INFERENCE_HPP
#define TORCH_INFERENCE_HPP

#include "../ports/InferencePort.hpp"
#include <torch/torch.h>
#include <vector>
#include <string>

namespace adapters {

class TorchInference : public ports::InferencePort {
public:
    std::vector<std::pair<std::string, float>> infer(
        torch::jit::script::Module& model, 
        const torch::Tensor& input, 
        const std::vector<std::string>& class_names
    ) override;
};

} // namespace adapters

#include "TorchInference.tpp"

#endif // TORCH_INFERENCE_HPP