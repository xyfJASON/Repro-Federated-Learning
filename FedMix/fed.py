from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
import torch.autograd as autograd


class Client:
    def __init__(self, model, optimizer, train_loader, lamb, Xg, Yg, device):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.lamb = lamb
        self.Xg, self.Yg = Xg, Yg
        self.device = device

        self.CrossEntropy = nn.CrossEntropyLoss()

    def load_param(self, state_dict):
        self.model.load_state_dict(state_dict, strict=True)

    def train_one_epoch(self, naive: bool = False):
        self.model.train()
        if not naive:
            mloss = [0., 0., 0., 0.]
            for X, y in self.train_loader:
                X = X.to(device=self.device, dtype=torch.float32)
                y = y.to(device=self.device, dtype=torch.long)
                inputX = (1 - self.lamb) * X
                inputX.requires_grad_()

                idg = torch.randint(len(self.Xg), (1, ))
                xg = self.Xg[idg:idg+1].to(device=self.device)
                yg = self.Yg[idg:idg+1].to(device=self.device)

                scores = self.model(inputX)
                loss1 = (1 - self.lamb) * self.CrossEntropy(scores, y)
                loss2 = self.lamb * self.CrossEntropy(scores, yg.expand_as(scores))

                gradients = autograd.grad(outputs=loss1, inputs=inputX,
                                          create_graph=True, retain_graph=True)[0]
                loss3 = self.lamb * torch.inner(gradients.flatten(start_dim=1), xg.flatten(start_dim=1))
                loss3 = torch.mean(loss3)

                loss = loss1 + loss2 + loss3
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                mloss[0] += loss.item()
                mloss[1] += loss1.item()
                mloss[2] += loss2.item()
                mloss[3] += loss3.item()
            return [mloss[i] / len(self.train_loader) for i in range(4)]
        else:
            mloss = 0.
            for X, y in self.train_loader:
                X = X.to(device=self.device, dtype=torch.float32)
                y = y.to(device=self.device, dtype=torch.long)

                idg = torch.randint(len(self.Xg), (len(X), ))
                xg = self.Xg[idg].to(device=self.device)
                yg = self.Yg[idg].to(device=self.device)

                scores = self.model((1 - self.lamb) * X + self.lamb * xg)
                loss = (1 - self.lamb) * self.CrossEntropy(scores, y) + self.lamb * self.CrossEntropy(scores, yg)

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
