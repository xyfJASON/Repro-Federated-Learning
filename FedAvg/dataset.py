import numpy as np

from torch.utils.data import Dataset
import torchvision.datasets as dset
import torchvision.transforms as T


class DatasetPartitioner:
    """ Partition the dataset into several parts. """
    def __init__(self, dataset: Dataset, n_classes: int, n_parties: int, beta: float = None, random_seed=None, label_name='targets'):
        """

        Args:
            dataset: the dataset to be partitioned
            n_classes: number of classes
            n_parties: number of parties
            beta: parameter of dirichlet distribution
            random_seed: random seed
            label_name: name of attribute label in class dataset
        """
        self.dataset = dataset
        self.n_parties = n_parties

        rng = np.random.default_rng(random_seed)
        if hasattr(dataset, label_name):
            targets = getattr(dataset, label_name)
            if not isinstance(targets, np.ndarray):
                targets = np.array(targets)
        else:  # this is much slower
            targets = np.array([label for _, label in dataset])

        cnt_party_class = np.zeros((n_parties, n_classes), dtype=int)
        self.idx_parties = [[] for _ in range(n_parties)]  # index in each party
        for c in range(n_classes):
            idx_c = np.where(targets == c)[0]
            rng.shuffle(idx_c)
            proportions = rng.dirichlet(alpha=[beta]*n_parties)  # (n_parties, )
            partition_pos = (np.cumsum(proportions) * len(idx_c)).astype(int).tolist()[:-1]
            for idx, idx_split in zip(self.idx_parties, np.split(idx_c, partition_pos)):
                idx.extend(idx_split)
            cnt_party_class[:, c] += np.diff([0] + partition_pos + [len(idx_c)])
        for p in range(n_parties):
            rng.shuffle(self.idx_parties[p])

        print('Data partition:')
        for p in range(n_parties):
            print(f'\tPart {p}: {cnt_party_class[p]}\ttotal={np.sum(cnt_party_class[p])}')

    def get_dataset(self, party: int or str):
        if isinstance(party, int):
            assert party < self.n_parties, f'{party} should be less than {self.n_parties}'
            return PartialDataset(self.dataset, self.idx_parties[party])
        elif isinstance(party, str):
            assert party == 'all'
            return self.dataset


class PartialDataset(Dataset):
    def __init__(self, dataset, idx):
        self.dataset = dataset
        self.idx = idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, item):
        return self.dataset[self.idx[item]]


if __name__ == '__main__':
    transform = T.Compose([T.Resize((32, 32)),
                           T.RandomCrop((32, 32), padding=4, padding_mode='reflect'),
                           T.RandomHorizontalFlip(),
                           T.ToTensor(),
                           T.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])])
    train_CIFAR10 = dset.CIFAR10(root='../../data', train=True, transform=transform, download=False)
    test_CIFAR10 = dset.CIFAR10(root='../../data', train=False, transform=transform, download=False)
    partitioner = DatasetPartitioner(dataset=train_CIFAR10, n_classes=10, n_parties=5, beta=0.5, random_seed=0)
    part1 = partitioner.get_dataset(0)
    print(len(part1), part1[100][1])
