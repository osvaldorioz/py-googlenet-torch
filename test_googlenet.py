import googlenet
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def load_imagenet_labels():
    try:
        with open("imagenet_classes.txt", "r") as f:
            return [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print("Advertencia: imagenet_classes.txt no encontrado. Usando etiquetas simuladas.")
        return [f"simulated_class_{i}" for i in range(1000)]

def main():
    #model_path = "/opt/app-root/src/models/googlenet.pt"
    model_path = "googlenet.pt"
    image_path = "preprocessed_tensor.pt"
    class_names = load_imagenet_labels()

    service = googlenet.GoogLeNetService("clasificador")
    
    try:
        results = service.classify(model_path, image_path, class_names)
        
        print("Top-5 predicciones:")
        for label, prob in results:
            print(f"{label}: {prob:.4f}")
        
        labels = [label for label, prob in results]
        probs = [prob for label, prob in results]
        
        # Usar un colormap para las barras
        colors = cm.viridis([i/len(probs) for i in range(len(probs))])
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels, probs, color=colors, edgecolor='black')
        plt.title("Top-5 Predicciones de GoogLeNet", fontsize=14, pad=15)
        plt.xlabel("Clase", fontsize=12)
        plt.ylabel("Probabilidad", fontsize=12)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval, f"{yval:.4f}", va='bottom', fontsize=10)
        plt.tight_layout()
        plt.savefig("googlenet_results.png")
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
