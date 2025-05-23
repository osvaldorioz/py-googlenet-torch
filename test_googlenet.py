import googlenet
import matplotlib.pyplot as plt
import json

def load_imagenet_labels():
    # Cargar etiquetas reales de ImageNet (descarga imagenet_classes.txt si no lo tienes)
    try:
        with open("imagenet_classes.txt", "r") as f:
            return [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print("Advertencia: imagenet_classes.txt no encontrado. Usando etiquetas simuladas.")
        return ["cat_" + str(i) for i in range(5)] + ["unknown"] * 995  # Simulación temporal

def main():
    # Parámetros
    # model_path = "/opt/app-root/src/models/googlenet.pt"
    model_path = "/home/hadoop/Documentos/cpp_programs/pybind/py-googlenet-torch/googlenet.pt"
    image_path = "sample_cat.jpg"  # Simulado;
    class_names = load_imagenet_labels()
    
    # Inicializar servicio
    service = googlenet.GoogLeNetService("dummy")
    
    try:
        # Ejecutar clasificación
        results = service.classify(model_path, image_path, class_names)
        
        # Mostrar resultados
        print("Top-5 predicciones:")
        for label, prob in results:
            print(f"{label}: {prob:.4f}")
        
        # Generar gráfica
        labels = [label for label, _ in results]
        probs = [prob for _, prob in results]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(labels)), probs, color='skyblue')
        plt.title("Top-5 Predicciones de GoogLeNet")
        plt.xlabel("Clase")
        plt.ylabel("Probabilidad")
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig("googlenet_results.png")
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()