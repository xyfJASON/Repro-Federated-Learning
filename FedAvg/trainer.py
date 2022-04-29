import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as dset
import torchvision.transforms as T

from model import SimpleCNN
from engine import Client, Server
from utils.partition_data import DatasetPartitioner
from utils.general_utils import parse_config


class Trainer:
    def __init__(self, config_path):
        self.config, self.device, self.log_root = parse_config(config_path)
        (self.train_dataset_list, self.train_loader_list,
         self.train_dataset, self.train_loader,
         self.test_dataset, self.test_loader) = self._get_data()

        # ============= define clients ============= #
        self.client_list = []
        for i in range(self.config['n_parties']):
            model, optimizer, loss = self._prepare_training()
            self.client_list.append(Client(model, optimizer, self.train_loader_list[i], loss, self.device))

        # ============= define server ============= #
        model, _, _ = self._prepare_training()
        self.server = Server(model, self.device)

        self.writer = SummaryWriter(os.path.join(self.log_root, 'tensorboard'))

    def _get_data(self):
        print(f'==> Getting data...')
        train_transform = T.Compose([T.Resize((32, 32)),
                                     T.RandomCrop((32, 32), padding=4, padding_mode='reflect'),
                                     T.RandomHorizontalFlip(),
                                     T.ToTensor(),
                                     T.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])])
        test_transform = T.Compose([T.Resize((32, 32)),
                                    T.ToTensor(),
                                    T.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])])
        train_dataset = dset.CIFAR10(root=self.config['dataroot'], train=True, transform=train_transform, download=False)
        test_dataset = dset.CIFAR10(root=self.config['dataroot'], train=False, transform=test_transform, download=False)

        if self.config['partition']['choice'] == 'IID':
            partitioner = DatasetPartitioner(dataset=train_dataset, n_classes=10, n_parties=self.config['n_parties'], method='IID')
        elif self.config['partition']['choice'] == 'Dirichlet':
            partitioner = DatasetPartitioner(dataset=train_dataset, n_classes=10, n_parties=self.config['n_parties'],
                                             method='Dirichlet', beta=self.config['partition']['Dirichlet']['beta'])
        else:
            raise ValueError(f"{self.config['partition']['choice']} is not a valid partition method")
        train_dataset_list = [partitioner.get_dataset(i) for i in range(self.config['n_parties'])]
        train_loader_list = [DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
                             for dataset in train_dataset_list]
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config['batch_size'])
        return train_dataset_list, train_loader_list, train_dataset, train_loader, test_dataset, test_loader

    def _prepare_training(self):
        model = SimpleCNN()
        model.to(device=self.device)
        optimizer = optim.SGD(model.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'], momentum=self.config['momentum'])
        CrossEntropy = nn.CrossEntropyLoss()
        return model, optimizer, CrossEntropy

    def save_model(self, save_path):
        torch.save(self.server.model.state_dict(), save_path)

    def train(self):
        print('==> Training...')
        for comm_round in range(self.config['comm_rounds']):
            print(f'communication round: {comm_round}')
            for i, client in enumerate(self.client_list):
                print(f'\tclient{i}:')
                client.load_param(self.server.model.state_dict())
                for local_epoch in range(self.config['local_epochs']):
                    train_loss = client.train_one_epoch()
                    train_acc, _ = client.evaluate(self.train_loader_list[i])
                    print(f'\t\ttrain loss={train_loss:.6f}, train acc={train_acc:.6f}')

            self.server.aggregate_model([client.model for client in self.client_list],
                                        weight=[len(self.train_dataset_list[i]) / len(self.train_dataset) for i in range(self.config['n_parties'])])
            acc, f1 = self.server.evaluate(self.test_loader)
            print(f'server: acc={acc:.6f}, f1={f1:.6f}', end='\n\n')
            self.writer.add_scalar('Server/acc', acc.item(), comm_round)

            if self.config['save_per_rounds'] and (comm_round + 1) % self.config['save_per_rounds'] == 0:
                self.save_model(os.path.join(self.log_root, 'ckpt', f'round_{comm_round}.pt'))

        self.save_model(os.path.join(self.log_root, 'model.pt'))
        self.writer.close()
