import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_fedmix_naivemix():
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(range(100), result['naivemix beta5'], label=r'naivemix', color='C0')
    ax[0].plot(range(100), result['fedmix beta5'], label=r'fedmix', color='C1')
    ax[0].plot(range(100), result['fedavg beta5'], label=r'fedavg', color='C2')
    ax[0].set_title(r'$\beta=5$')
    ax[1].plot(range(100), result['naivemix beta0.5'], color='C0')
    ax[1].plot(range(100), result['fedmix beta0.5'], color='C1')
    ax[1].plot(range(100), result['fedavg beta0.5'], color='C2')
    ax[1].set_title(r'$\beta=0.5$')
    ax[2].plot(range(100), result['naivemix beta0.1'], color='C0')
    ax[2].plot(range(100), result['fedmix beta0.1'], color='C1')
    ax[2].plot(range(100), result['fedavg beta0.1'], color='C2')
    ax[2].set_title(r'$\beta=0.1$')
    ax[0].set_xlabel('Global rounds')
    ax[0].set_ylabel('Server acc.')
    ax[1].set_xlabel('Global rounds')
    ax[2].set_xlabel('Global rounds')
    ax[0].set_ylim(0.4, 0.6)
    ax[1].set_ylim(0.4, 0.6)
    ax[2].set_ylim(0.4, 0.6)
    ax[0].set_yticks(np.linspace(0.4, 0.6, 5))
    ax[1].set_yticks(np.linspace(0.4, 0.6, 5))
    ax[2].set_yticks(np.linspace(0.4, 0.6, 5))
    fig.legend()
    fig.savefig('./fedmix.png', bbox_inches='tight', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    result = pd.read_csv('./results.csv')
    plot_fedmix_naivemix()
