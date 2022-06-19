import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as dset
import torchvision.transforms as T

from model import SimpleCNN, modVGG
from fed import Client, Server
from utils.partition_data import DatasetPartitioner
from utils.general_utils import parse_config


class Trainer:
    def __init__(self, config_path):
        self.config, self.device, self.log_root = parse_config(config_path)
        (self.train_dataset_list, self.train_loader_list,
         self.train_dataset, self.train_loader,
         self.test_dataset, self.test_loader) = self._get_data()

        self.Xg, self.Yg = self._calculate_mean_data()

        # ============= define clients ============= #
        self.client_list = []
        for i in range(self.config['n_parties']):
            model, optimizer = self._prepare_training()
            self.client_list.append(Client(model, optimizer, self.train_loader_list[i],
                                           self.config['lambda'], self.Xg, self.Yg, self.device))

        # ============= define server ============= #
        model, _ = self._prepare_training()
        self.server = Server(model, self.device)

        self.writer = SummaryWriter(os.path.join(self.log_root, 'tensorboard'))

    def _get_data(self):
        print(f'==> Getting data...')
        transform = T.Compose([T.Resize((32, 32)),
                               T.ToTensor(),
                               T.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])])
        train_dataset = dset.CIFAR10(root=self.config['dataroot'], train=True, transform=transform, download=False)
        test_dataset = dset.CIFAR10(root=self.config['dataroot'], train=False, transform=transform, download=False)

        if self.config['partition']['choice'] == 'IID':
            partitioner = DatasetPartitioner(dataset=train_dataset, n_classes=10, n_parties=self.config['n_parties'], method='IID')
        elif self.config['partition']['choice'] == 'Dirichlet':
            partitioner = DatasetPartitioner(dataset=train_dataset, n_classes=10, n_parties=self.config['n_parties'],
                                             method='Dirichlet', beta=self.config['partition']['Dirichlet']['beta'])
        elif self.config['partition']['choice'] == 'NonIID':
            partitioner = DatasetPartitioner(dataset=train_dataset, n_classes=10, n_parties=self.config['n_parties'],
                                             method='NonIID', n_class_each_client=self.config['partition']['NonIID']['n_class_each_client'])
        elif self.config['partition']['choice'] == 'read_from_file':
            partitioner = DatasetPartitioner(dataset=train_dataset, n_classes=10, n_parties=self.config['n_parties'],
                                             method='read_from_file', file_path=self.config['partition']['read_from_file']['file_path'])
        else:
            raise ValueError(f"{self.config['partition']['choice']} is not a valid partition method")

        with open(os.path.join(self.log_root, 'idx_parties.txt'), 'w') as f:
            for p, idx in enumerate(partitioner.get_idx_parites()):
                if isinstance(idx, np.ndarray):
                    idx = idx.tolist()
                f.write(' '.join(map(str, idx)))
                f.write('\n')

        train_dataset_list = [partitioner.get_dataset(i) for i in range(self.config['n_parties'])]
        train_loader_list = [DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
                             for dataset in train_dataset_list]
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config['batch_size'])
        return train_dataset_list, train_loader_list, train_dataset, train_loader, test_dataset, test_loader

    def _prepare_training(self):
        if self.config['model'] == 'SimpleCNN':
            model = SimpleCNN()
        elif self.config['model'] == 'modVGG':
            model = modVGG()
        else:
            raise ValueError(f"{self.config['model']} is not a valid model")
        model.to(device=self.device)
        optimizer = optim.SGD(model.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'], momentum=self.config['momentum'])
        return model, optimizer

    def _calculate_mean_data(self):
        print(f'==> Calculating mean...')
        Xg, Yg = [], []

        for train_dataset in self.train_dataset_list:
            data, label = [], []
            for X, y in train_dataset:
                data.append(X)
                label.append(torch.tensor(y))
            data = torch.stack(data, dim=0)
            label = torch.stack(label, dim=0)
            data = torch.split(data, self.config['mean_batch'])
            label = torch.split(label, self.config['mean_batch'])
            for d, l in zip(data, label):
                if len(d) != self.config['mean_batch']:
                    break
                Xg.append(torch.mean(d, dim=0))
                Yg.append(torch.mean(F.one_hot(l, num_classes=10).to(dtype=torch.float32), dim=0))

        Xg = torch.stack(Xg, dim=0)
        Yg = torch.stack(Yg, dim=0)
        return Xg, Yg

    def save_model(self, save_path):
        torch.save(self.server.model.state_dict(), save_path)

    def train(self):
        print('==> Training...')
        for comm_round in range(self.config['comm_rounds']):
            print(f'communication round: {comm_round}')
            n_select = max(1, int(self.config['n_parties'] * self.config['select_frac']))
            select_clients = sorted(np.random.choice(range(self.config['n_parties']), (n_select, ), replace=False))
            for i in select_clients:
                print(f'\tclient{i}:')
                client = self.client_list[i]
                client.load_param(self.server.model.state_dict())
                for local_epoch in range(self.config['local_epochs']):
                    losses = client.train_one_epoch(self.config['naive_mix'])
                    train_acc, _ = client.evaluate(self.train_loader_list[i])
                    if isinstance(losses, list):
                        print(f'\t\ttrain loss={losses[0]:.6f}, loss1={losses[1]:.6f}, loss2={losses[2]:.6f}, loss3={losses[3]:.6f}, train acc={train_acc:.6f}')
                    else:
                        print(f'\t\ttrain loss={losses:.6f}, train acc={train_acc:.6f}')

            self.server.aggregate_model([client.model for client in self.client_list],
                                        weight=[len(self.train_dataset_list[i]) / len(self.train_dataset) for i in range(self.config['n_parties'])])
            acc, f1 = self.server.evaluate(self.test_loader)
            print(f'server: acc={acc:.6f}, f1={f1:.6f}', end='\n\n')
            self.writer.add_scalar('Server/acc', acc.item(), comm_round)

            if self.config['save_per_rounds'] and (comm_round + 1) % self.config['save_per_rounds'] == 0:
                self.save_model(os.path.join(self.log_root, 'ckpt', f'round_{comm_round}.pt'))

        self.save_model(os.path.join(self.log_root, 'model.pt'))
        self.writer.close()
