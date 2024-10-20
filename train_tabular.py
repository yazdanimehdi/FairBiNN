import argparse
import math
import typing

from data_loading.load_health import pre_process_and_load_health
from models.tabular import AttributeClassifierAblation

import torch
import pickle
import random
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.preprocessing import MinMaxScaler
from metrics import demographic_parity_difference_soft
from data_loading.load_adult import pre_process_and_load_adult
from fairlearn.metrics import equalized_odds_difference
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F


class ConstraintLoss(nn.Module):
    def __init__(self, n_class=2, alpha=1, p_norm=2):
        super(ConstraintLoss, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.alpha = alpha
        self.p_norm = p_norm
        self.n_class = n_class
        self.n_constraints = 2
        self.dim_condition = self.n_class + 1
        self.M = torch.zeros((self.n_constraints, self.dim_condition))
        self.c = torch.zeros(self.n_constraints)

    def mu_f(self, X=None, y=None, sensitive=None):
        return torch.zeros(self.n_constraints)

    def forward(self, X, out, sensitive, y=None):
        sensitive = sensitive.reshape(out.shape)
        if isinstance(y, torch.Tensor):
            y = y.view(out.shape)
        out = torch.sigmoid(out)
        mu = self.mu_f(X=X, out=out, sensitive=sensitive, y=y)
        gap_constraint = F.relu(
            torch.mv(self.M.to(self.device), mu.to(self.device)) - self.c.to(self.device)
        )
        if self.p_norm == 2:
            cons = self.alpha * torch.mean(torch.pow(gap_constraint, 2))
        else:
            cons = self.alpha * torch.dot(gap_constraint.detach(), gap_constraint)
        return cons


class DemographicParityLoss(ConstraintLoss):
    def __init__(self, sensitive_classes=[0, 1], alpha=1, p_norm=2):
        """loss of demograpfhic parity

        Args:
            sensitive_classes (list, optional): list of unique values of sensitive attribute. Defaults to [0, 1].
            alpha (int, optional): [description]. Defaults to 1.
            p_norm (int, optional): [description]. Defaults to 2.
        """
        self.sensitive_classes = sensitive_classes
        self.n_class = len(sensitive_classes)
        super(DemographicParityLoss, self).__init__(
            n_class=self.n_class, alpha=alpha, p_norm=p_norm
        )
        self.n_constraints = 2 * self.n_class
        self.dim_condition = self.n_class + 1
        self.M = torch.zeros((self.n_constraints, self.dim_condition))
        for i in range(self.n_constraints):
            j = i % 2
            if j == 0:
                self.M[i, j] = 1.0
                self.M[i, -1] = -1.0
            else:
                self.M[i, j - 1] = -1.0
                self.M[i, -1] = 1.0
        self.c = torch.zeros(self.n_constraints)

    def mu_f(self, X, out, sensitive, y=None):
        expected_values_list = []
        for v in self.sensitive_classes:
            idx_true = sensitive == v  # torch.bool
            expected_values_list.append(out[idx_true].mean())
        expected_values_list.append(out.mean())
        return torch.stack(expected_values_list)

    def forward(self, X, out, sensitive, y=None):
        return super(DemographicParityLoss, self).forward(X, out, sensitive)



def domination(x1, x2):
    # determin if x1 dominate x2
    # breakpoint()
    # want greater acc in 0 dim and lower dp in 1 dim
    if x1[0] > x2[0] and x1[1] < x2[1]:
        return True
    if x1[0] >= x2[0] and x1[1] < x2[1]:
        return True
    if x1[0] > x2[0] and x1[1] <= x2[1]:
        return True
    return False


def get_pareto_front(arr, x_dominates_y: typing.Callable = domination):
    pareto_front = []
    for i in arr:
        for j in arr:
            # print(i,j, x_dominates_y(j, i))
            # if j dominate i, we don't want i
            if x_dominates_y(j, i):
                # print("i is dominated by j")
                break
        else:
            pareto_front.append(i)
    return pareto_front


def fwd_pass(x, y_l, s, criterion_dp, criterion_acc, optimizer_dp, optimizer_acc, model):
    model.zero_grad()
    x = torch.Tensor(x).to(torch.float)
    y_l = torch.Tensor(y_l).view(-1, 1).to(torch.float)
    y_l = y_l.to(device)
    if optimizer_dp is not None:
        out = model(x.to(device))
        out = out.to(torch.float)

        loss_dp = criterion_dp(y_l, out, s)
        loss_dp.backward()
        optimizer_dp.step()

    out = model(x.to(device))
    out = out.to(torch.float)
    loss_acc = criterion_acc(out, y_l)
    loss_dp = criterion_dp(y_l, out, s)
    if optimizer_dp is None:
        try:
            dp_order = math.floor(math.log(loss_dp, 10))
        except ValueError:
            dp_order = 0

        if dp_order == 0:
            dp_order = 0.0001

        # loss_dp_scale = 10 ** (abs((dp_order/math.floor(math.log(loss_acc, 10)))) - 1)

        # loss_dp = loss_dp_scale * loss_dp
        loss_acc = loss_acc + loss_dp

    loss_acc.backward()
    optimizer_acc.step()
    out = model(x.to(device))

    matches = [torch.round(i) == torch.round(j) for i, j in zip(out, y_l)]
    acc = matches.count(True) / len(matches)

    return acc, loss_acc, loss_dp.detach().numpy(), out


def sdp(x, y):
    male_and_high = [1 if (i == 1 and torch.round(j) == 1) else 0 for i, j in zip(x[:, 9], y)].count(1)
    male = [i for i in x[:, 9]].count(1)
    female_and_high = [1 if (i == 0 and torch.round(j) == 1) else 0 for i, j in zip(x[:, 9], y)].count(1)
    female = [j for j in x[:, 9]].count(0)

    p_male_high = male_and_high / male
    p_female_high = female_and_high / female

    return abs(p_male_high - p_female_high)


def test_func(model_f, y_label, X_test_f, s):
    y_pred = []
    y_label = torch.Tensor(y_label)
    print("Testing:")
    print("-------------------")
    with tqdm(range(0, len(X_test_f), 100)) as tepoch:
        for i in tepoch:
            with torch.no_grad():
                x = torch.Tensor(X_test_f[i: i + 100]).to(device)
                y_pred.append(model_f(x).cpu())

    y_pred = torch.cat(y_pred, dim=0)
    matches = [torch.round(i) == torch.round(j) for i, j in zip(y_label, y_pred)]
    acc = matches.count(True) / len(matches)
    return acc, demographic_parity_difference_soft(y_label, s, y_pred)


acc_dp = {}


class FairLossFunc(torch.nn.Module):
    def __init__(self, eta, protected):
        super(FairLossFunc, self).__init__()
        self.protected = protected
        self.eta = eta

    def forward(self, y_label, y_pred, protected):
        losses_max = torch.Tensor([0])
        for i in self.protected:
            for j in self.protected:
                index_c1 = protected == i
                index_c2 = protected == j
                p_1 = torch.mean(y_pred[index_c1])
                p_2 = torch.mean(y_pred[index_c2])
                l = ((p_1 - p_2) ** 2)
                if losses_max.item() < l.item():
                    losses_max = l

        return losses_max

losses_step = []


def train_model(eta, mode, data, f_layers, a_layers, f_position):
    MODEL_NAME = f"model-{int(time.time())}"
    if data == 'Adult':
        X, y, s = pre_process_and_load_adult(mode="onehot", train=True)
        X_test, y_test, s_test = pre_process_and_load_adult(mode="onehot", train=False)
    elif data == 'Health':
        X, y, s, X_test, y_test, s_test = pre_process_and_load_health()
    # elif data == 'Compass':
    #     X, y, X_test, y_test = pre_process_and_load_compass()
    else:
        raise NotImplementedError()

    model = AttributeClassifierAblation(dataset=data, fairness_layer_mode=mode, fairness_layers=f_layers,
                                        accuracy_layers=a_layers, fairness_layers_position=f_position)
    model.to(device)
    optimizer_acc = torch.optim.Adam(model.get_accuracy_parameters(), lr=alr)
    if mode != "reg":
        optimizer_dp = torch.optim.Adam(model.get_fairness_parameters(), lr=flr)
        scheduler_dp = ExponentialLR(optimizer_dp, gamma=0.9)
    else:
        optimizer_dp = None
        scheduler_dp = None

    criterion_acc = torch.nn.BCELoss()
    scheduler_acc = ExponentialLR(optimizer_acc, gamma=0.5)
    s_c = [0, 1, 2, 3, 4, 5, 6, 7, 8] if data == 'Health' else [0, 1]
    criterion_dp = DemographicParityLoss(sensitive_classes=s_c, alpha=eta)
    test_acc = []
    test_dp = []
    test_eo = []
    times = []
    with open("model.log", "a") as f:
        for epoch in range(EPOCHS):
            a1 = time.time()
            losses = []
            accs = []
            losses_dp = []
            with tqdm(range(0, len(X), BATCH_SIZE)) as tepoch:
                for i in tepoch:
                    tepoch.set_description(f"Eta {eta}, Epoch {epoch + 1}")

                    batch_X = X[i: i + BATCH_SIZE]
                    batch_y = y[i: i + BATCH_SIZE]
                    batch_s = s[i: i + BATCH_SIZE]

                    acc, loss, loss_dp, _ = fwd_pass(batch_X, batch_y, batch_s, criterion_dp, criterion_acc, optimizer_dp,
                                                     optimizer_acc, model)
                    losses.append(loss.item())
                    losses_dp.append(loss_dp)
                    accs.append(acc)
                    acc_mean = np.array(accs).mean()
                    loss_mean = np.array(losses).mean()
                    loss_dp_mean = np.array(losses_dp).mean()
                    tepoch.set_postfix(loss=loss_mean, accuracy=100. * acc_mean, loss_dp=loss_dp_mean)
                    if i == 0:
                        acc, sdp = test_func(model, y_test, X_test, s_test)
                        test_acc.append(acc)
                        test_dp.append(sdp[0])
                        print(f'ACC: {acc}')
                        print(f'SDP: {sdp}')
                        f.write(
                            f"{MODEL_NAME},{epoch},{round(float(acc_mean), 2)},{round(float(loss_mean), 4)},{acc},{sdp}\n")

            if (epoch + 1) % 50 == 0:
                scheduler_acc.step()
                if mode != "reg":
                    scheduler_dp.step()
            dt = time.strftime("%Y_%m_%d-%H_%M_%S")
            losses_step.append(np.array(losses).mean())
            a2 = time.time()
            if epoch > 10:
                times.append(a2 - a1)

        acc_dp[eta] = (test_acc, test_dp, test_eo)
    for item in acc_dp.keys():
        a = [(i, j) for i, j in zip(acc_dp[item][0], acc_dp[item][1])]
        a.sort(key=lambda x: -x[0])
        acc_dp[item] = a

    pareto_set = [[] for item in acc_dp.keys()]
    for idx, item in enumerate(acc_dp.keys()):
        pareto_set[idx].append(get_pareto_front(acc_dp[item]))
    print(pareto_set)
    # with open(f'acc_dps_{data}_{len(fairness_layers)}_{mode}.pkl', 'wb') as fp:
    #     pickle.dump(acc_dp, fp)
    print(np.mean(times))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", help='mode of the fairness layer', default='reg')
    parser.add_argument("-e", "--eta", help="eta", default=10000)
    parser.add_argument("-d", "--data", help="dataset name", default='Adult')
    parser.add_argument("-fl", "--fairness_layers", nargs="+", help="Fairness Layers")
    parser.add_argument("-al", "--acc_layers", nargs="+", help="Accuracy Layers")
    parser.add_argument("-fp", "--fairness_position", help="fairness layer position", default=2)
    parser.add_argument("-dv", "--device", default="cpu")
    parser.add_argument("-ep", "--epochs", default=200)
    parser.add_argument("-flr", "--fairness_learning_rate", default=1e-5)
    parser.add_argument("-nlr", "--network_learning_rate", default=1e-3)
    parser.add_argument("-bs", "--batch_size", default=200)
    args = parser.parse_args()

    alr = args.network_learning_rate
    flr = args.fairness_learning_rate
    BATCH_SIZE = int(args.batch_size)
    EPOCHS = int(args.epochs)
    if args.device == "mps":
        assert torch.backends.mps.is_available()
    elif args.device != "cpu":
        assert torch.cuda.is_available()

    device = torch.device(args.device)
    if args.data == "Adult":
        acc_layers = (101, 101, 101, 101, 1)
        fairness_layers = (101, 101)
        protected = 9
    elif args.data == "Health":
        acc_layers = (125, 125, 125, 125, 125, 1)
        fairness_layers = (125, 125)
        protected = 123

    else:
        raise NotImplementedError()
    if args.acc_layers is not None:
        acc_layers = tuple(map(lambda x: int(x), args.acc_layers))

    if args.fairness_layers is not None:
        fairness_layers = tuple(map(lambda x: int(x), args.fairness_layers))

    train_model(args.eta, args.mode, args.data, fairness_layers, acc_layers, int(args.fairness_position))



