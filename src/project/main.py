import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch_geometric.loader import DataLoader

from loader import NRLDataset
from metrics import RMSE, ROCAUC
from models import AttentiveFP, TrimNet
from utils import load_model, save_model


class MSELoss(torch.nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, pred, target, mask):
        pred, target = pred[mask], target[mask]
        return F.mse_loss(pred, target)


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, pred, target, mask):
        pred, target = pred[mask], target[mask]
        return torch.sqrt(F.mse_loss(pred, target))


def focal_loss(pred, target, alpha=1., gamma=2., reduction='mean'):
    assert reduction in ['mean', 'sum', 'none']
    loss = F.cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-loss)
    loss = (alpha * loss * torch.pow(1. - pt, gamma))
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self, num_targets=13,  num_classes=4):
        self.num_targets = num_targets
        self.num_classes = num_classes
        super(CrossEntropyLoss, self).__init__()

    def forward(self, pred, target, mask):
        mask = mask.float()
        pred = pred.reshape(-1, self.num_targets, self.num_classes)
        pred = pred.reshape(-1, self.num_classes)
        target = target.reshape(-1)
        loss = F.cross_entropy(pred, target, reduction='none')
        # loss = focal_loss(pred, target, reduction='none')
        loss = loss.reshape(-1, self.num_targets)
        loss = (loss * mask).sum(dim=0) / mask.sum(dim=0)
        loss = loss.mean()
        return loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
    if args.train and args.test:
        raise ValueError('Cannot train and test at the same time')
    if args.train:
        config['mode'] = 'train'
    if args.test:
        config['mode'] = 'test'
    return config


def train(model, train_loader, optimizer, loss_fn, device):
    model.train()
    total_loss = total_examples = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = loss_fn(out, data.y, data.y_mask)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
        total_examples += data.num_graphs
    return total_loss / total_examples


@torch.no_grad()
def test(model, test_loader, metric_fn, device):
    model.eval()
    metric_fn.reset()
    for data in test_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        metric_fn.update(out, data.y, data.y_mask)
    return metric_fn.compute()


if __name__ == '__main__':
    config = parse_args()
    # torch.manual_seed(2021)
    torch.manual_seed(42)
    dataset = NRLDataset(config['data_dir'], config['dataset'], config['mode'])

    if config['objective'] == 'regression':
        loss_fn = MSELoss()
        metric_fn = RMSE()
    elif config['objective'] == 'classification':
        loss_fn = CrossEntropyLoss()
        metric_fn = ROCAUC(
            num_targets=config['num_targets'],
            num_classes=config['num_classes'],
            multi_classes='ovo'
        )
    else:
        raise ValueError('Invalid objective')

    device = torch.device(config['device'])

    model = AttentiveFP(
        in_channels=dataset.num_node_features,
        hidden_channels=config['hidden_channels'],
        out_channels=config['num_outputs'],
        edge_dim=dataset.num_edge_features,
        num_layers=config['num_layers'],
        num_timesteps=config['num_timesteps'],
        dropout=config['dropout'],
    ).to(device)
    # model = TrimNet(in_channels=dataset.num_node_features,
    #                 out_channels=config['num_outputs'],
    #                 edge_dim=dataset.num_edge_features,)

    if config['mode'] == 'train':
        # dataset = dataset.shuffle()
        N = len(dataset) // 10
        test_dataset = dataset[:N]
        valid_dataset = dataset[N:2 * N]
        train_dataset = dataset[2 * N:]

        test_loader = DataLoader(
            test_dataset, batch_size=config['batch_size'], shuffle=False)
        valid_loader = DataLoader(
            valid_dataset, batch_size=config['batch_size'], shuffle=False)
        train_loader = DataLoader(
            train_dataset, batch_size=config['batch_size'], shuffle=True)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

        best_epoch = -1
        if config['metric_direction'] == 'min':
            best_valid_metric = float('inf')
            def compare_func(x, y): return x < y
        else:
            best_valid_metric = float('-inf')
            def compare_func(x, y): return x > y

        early_stop_count = 0
        for epoch in range(1, config['epochs'] + 1):
            loss = train(model, train_loader, optimizer, loss_fn, device)
            train_metric = test(model, train_loader, metric_fn, device)
            valid_metric = test(model, valid_loader, metric_fn, device)
            test_metric = test(model, test_loader, metric_fn, device)
            print('Epoch: {:03d}, Loss: {:.4f}, Train: {:.4f}, Valid: {:.4f}, Test: {:.4f}'.format(
                epoch, loss, train_metric, valid_metric, test_metric))
            early_stop_count += 1
            if compare_func(valid_metric, best_valid_metric):
                best_valid_metric = valid_metric
                early_stop_count = 0
                best_epoch = epoch
                save_model(model, config['model_path'])
            if early_stop_count >= config['early_stop_count']:
                print("Early stopping")
                break

        print("Best Epoch: {:03d}, Best Valid: {:.4f}".format(
            best_epoch, best_valid_metric))

    else:
        model = load_model(model, config['model_path'])
        loader = DataLoader(dataset, batch_size=1)

        target_columns = [f"target_{i}" for i in range(config['num_targets'])]

        submission = pd.DataFrame(columns=['smiles'] + target_columns)
        for i, data in enumerate(loader):
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            if config['objective'] == 'classification':
                out = out.reshape(-1,
                                  config['num_targets'], config['num_classes'])
                out = out.argmax(dim=-1)
            submission.loc[i] = [data[0].smiles] + \
                list(out[0].detach().cpu().numpy())
        submission.to_csv(config['submission_path'], index=False)
