import matplotlib.pyplot as plt

def save_png(image, file_path):
    rows, cols = 4, 8

    fig, axes = plt.subplots(rows, cols, figsize=(16, 8))
    axes = axes.flatten()

    for i in range(len(axes)):
        axes[i].imshow(image[i], cmap='gray')
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()