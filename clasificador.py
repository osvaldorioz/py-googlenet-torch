import torch
import torchvision.models as models

# Modelo preentrenado
model = models.googlenet(pretrained=True)
model.eval()

# Ejemplo de entrada
example_input = torch.rand(1, 3, 224, 224)
scripted_model = torch.jit.trace(model, example_input)
#scripted_model.save('/opt/app-root/src/models/googlenet.pt')
scripted_model.save('/home/hadoop/Documentos/cpp_programs/pybind/py-googlenet-torch/googlenet.pt')
