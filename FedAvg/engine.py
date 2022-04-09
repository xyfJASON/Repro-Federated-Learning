from sklearn.metrics import accuracy_score, f1_score

import torch


class Client:
    def __init__(self, model, optimizer, train_loader, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.criterion = criterion
        self.device = device

    def load_param(self, state_dict):
        self.model.load_state_dict(state_dict, strict=True)

    def train_one_epoch(self):
        self.model.train()
        mloss = 0.
        for X, y in self.train_loader:
            X = X.to(device=self.device, dtype=torch.float32)
            y = y.to(device=self.device, dtype=torch.long)
            scores = self.model(X)
            loss = self.criterion(scores, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            mloss += loss.item()
        return mloss / len(self.train_loader)

    @torch.no_grad()
    def evaluate(self, eval_loader):
        self.model.eval()
        y_pred, y_true = [], []
        for X, y in eval_loader:
            X = X.to(device=self.device, dtype=torch.float32)
            scores = self.model(X)
            pred = torch.argmax(scores, dim=1)
            y_pred.extend(pred.tolist())
            y_true.extend(y.tolist())
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        return acc, f1


class Server:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def aggregate_model(self, client_models, weight=None):
        if weight is None:
            weight = [1 / len(client_models)] * len(client_models)
        params = {}
        for i, client in enumerate(client_models):
            for key, value in client.state_dict().items():
                if i == 0:
                    params[key] = value * weight[i]
                else:
                    params[key] += value * weight[i]
        self.model.load_state_dict(params, strict=True)

    @torch.no_grad()
    def evaluate(self, eval_loader):
        self.model.eval()
        y_pred, y_true = [], []
        for X, y in eval_loader:
            X = X.to(device=self.device, dtype=torch.float32)
            scores = self.model(X)
            pred = torch.argmax(scores, dim=1)
            y_pred.extend(pred.tolist())
            y_true.extend(y.tolist())
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        return acc, f1
