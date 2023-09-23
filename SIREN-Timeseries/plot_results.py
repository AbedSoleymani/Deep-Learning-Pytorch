import matplotlib.pyplot as plt

def plot_results(values,
                 mlp_output,
                 siren_output,
                 n_epochs):
    
    plt.plot(values.detach().numpy(), color='green', label='Ground Truth')
    plt.plot(siren_output.detach().numpy(), color='blue', label='SIREN')
    plt.plot(mlp_output.detach().numpy(), color='red', label='ReLU')
    plt.legend(fontsize=13)
    plt.title('Number of training epochs = {}'.format(n_epochs))
    plt.xlabel('Timestamp')
    plt.ylabel('Amplitude')
    plt.grid()

    plt.savefig('./SIREN-Timeseries/imgs/Results.png')
    plt.show()
