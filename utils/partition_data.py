import numpy as np

from torch.utils.data import Dataset
import torchvision.datasets as dset
import torchvision.transforms as T


class DatasetPartitioner:
    """ Partition the dataset into several parts. """
    def __init__(self, dataset: Dataset, n_classes: int, n_parties: int, method: str = 'IID',
                 beta: float = None, n_class_each_client: int = None, file_path: str = None,
                 random_seed=None, label_name='targets'):
        """

        Args:
            dataset: the dataset to be partitioned
            n_classes: number of classes
            n_parties: number of parties
            method: partition method, options: 'IID'(default), 'Dirichlet', 'NonIID', 'read_from_file'
            beta: parameter of dirichlet distribution, only valid when method is 'Dirichlet'
            n_class_each_client: number of classes in each party, only valid when method is 'NonIID'
            file_path: path to file storing indices, only valid when method is 'read_from_file'
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
        if method == 'IID':
            idxs = rng.permutation(len(targets))
            self.idx_parties = np.array_split(idxs, indices_or_sections=n_parties)
            for p in range(n_parties):
                cnt_party_class[p] = np.bincount(targets[self.idx_parties[p]], minlength=n_classes)
        elif method == 'Dirichlet':
            assert beta
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
        elif method == 'NonIID':
            assert isinstance(n_class_each_client, int) and 1 <= n_class_each_client <= n_classes
            self.idx_parties = [[] for _ in range(n_parties)]
            # Allocate matrix A is a boolean matrix,
            # where A_ij indicates whether the i'th party has data from the j'th class
            allocate_matrix = np.zeros((n_parties, n_classes), dtype=bool)
            # First, ensure that every class will be allocated to at least one party if possible
            rand_parties = rng.permutation(n_parties)
            rand_classes = rng.permutation(n_classes)
            p = 0
            for c in rand_classes:
                allocate_matrix[rand_parties[p % n_parties], c] = True
                p += 1
                if p // n_parties == n_class_each_client:
                    break
                if p % n_parties == 0:
                    rng.shuffle(rand_parties)
            # Then, randomly choose from remaining classes for each party
            for p in range(n_parties):
                num = n_class_each_client - sum(allocate_matrix[p])
                if num > 0:
                    remain_c = np.argwhere(allocate_matrix[p] == 0).flatten()
                    choose_c = rng.permutation(remain_c)[:num]
                    allocate_matrix[p, choose_c] = True
            # Allocate data to parties according to allocate matrix
            for c in range(n_classes):
                idx_c = np.where(targets == c)[0]
                rng.shuffle(idx_c)
                idx_p = np.where(allocate_matrix[:, c])[0]
                if len(idx_p) == 0:
                    continue
                for p, idx_split in zip(idx_p, np.array_split(idx_c, sum(allocate_matrix[:, c]))):
                    self.idx_parties[p].extend(idx_split)
                    cnt_party_class[p, c] += len(idx_split)
            for p in range(n_parties):
                rng.shuffle(self.idx_parties[p])
        elif method == 'read_from_file':
            assert file_path
            with open(file_path, 'r') as f:
                idx = f.readlines()
            assert len(idx) == n_parties
            self.idx_parties = [list(map(int, p.strip().split(' '))) for p in idx]
            for i in range(len(idx)):
                cnt_party_class[i] = np.bincount(targets[np.array(self.idx_parties[i])], minlength=n_classes)
        else:
            raise ValueError

        print('Data partition:')
        for p in range(n_parties):
            print(f'\tPart {p}: {cnt_party_class[p]}\ttotal={np.sum(cnt_party_class[p])}')

    def get_idx_parites(self):
        return self.idx_parties

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
    partitioner = DatasetPartitioner(dataset=train_CIFAR10, n_classes=10, n_parties=5, method='NonIID',
                                     beta=0.5, n_class_each_client=3, random_seed=0)
    part1 = partitioner.get_dataset(0)
    print(len(part1), part1[100][1])
