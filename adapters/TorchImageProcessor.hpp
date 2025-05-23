#ifndef TORCH_IMAGE_PROCESSOR_HPP
#define TORCH_IMAGE_PROCESSOR_HPP

#include "../ports/ImageProcessorPort.hpp"
#include <torch/torch.h>
#include <string>

namespace adapters {

class TorchImageProcessor : public ports::ImageProcessorPort {
public:
    torch::Tensor preprocess(const std::string& image_path) override;
};

} // namespace adapters

#include "TorchImageProcessor.tpp"

#endif // TORCH_IMAGE_PROCESSOR_HPP