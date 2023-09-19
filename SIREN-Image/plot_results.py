import matplotlib.pyplot as plt

def plot_results(img, mlp_psnr, mlp_output, siren_psnr, siren_output, resolution):
    fig, axes = plt.subplots(1, 4, figsize=(15, 3))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Ground Truth', fontsize=13)

    axes[1].imshow(mlp_output.cpu().view(resolution, resolution).detach().numpy(), cmap='gray')
    axes[1].set_title('ReLU', fontsize=13)

    axes[2].imshow(siren_output.cpu().view(resolution, resolution).detach().numpy(), cmap='gray')
    axes[2].set_title('SIREN', fontsize=13)

    axes[3].plot(mlp_psnr, label='ReLU')
    axes[3].plot(siren_psnr, label='SIREN')
    axes[3].set_xlabel('Iterations', fontsize=14)
    axes[3].set_title('PSNR', fontsize=14)
    axes[3].legend(fontsize=13)

    for i in range(3):
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    plt.savefig('./SIREN-Image/imgs/Results.png')
    plt.close()