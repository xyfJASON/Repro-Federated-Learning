import pandas as pd
import matplotlib.pyplot as plt


def plot_fedavg():
    fig, ax = plt.subplots(1, 1)
    ax.plot(range(100), result['beta0.1'], label=r'$\beta=0.1$')
    ax.plot(range(100), result['beta0.5'], label=r'$\beta=0.5$')
    ax.plot(range(100), result['beta5'], label=r'$\beta=5$')
    ax.plot(range(100), result['iid'], label=r'IID')
    ax.set_ylim(0.2, 0.8)
    ax.set_xlabel('Global rounds')
    ax.set_ylabel('Server acc.')
    ax.set_title('Without augmentation')
    ax.legend()
    fig.savefig('./fedavg.png', bbox_inches='tight', dpi=150)
    plt.close(fig)


def plot_fedavg_aug():
    fig, ax = plt.subplots(1, 1)
    ax.plot(range(100), result['aug_beta0.1'], label=r'$\beta=0.1$')
    ax.plot(range(100), result['aug_beta0.5'], label=r'$\beta=0.5$')
    ax.plot(range(100), result['aug_beta5'], label=r'$\beta=5$')
    ax.plot(range(100), result['aug_iid'], label=r'IID')
    ax.set_ylim(0.2, 0.8)
    ax.set_xlabel('Global rounds')
    ax.set_ylabel('Server acc.')
    ax.set_title('With augmentation')
    ax.legend()
    fig.savefig('./fedavg_aug.png', bbox_inches='tight', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    result = pd.read_csv('./results.csv')
    plot_fedavg()
    # plot_fedavg_aug()
