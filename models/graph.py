import torch.nn as nn
from graph_models.GCN import GCN, GCN_Body
from graph_models.GAT import GAT, GAT_body
import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import ExponentialLR


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
        sensitive = sensitive.view(out.shape)
        if isinstance(y, torch.Tensor):
            y = y.view(out.shape)
        out = torch.sigmoid(out)
        mu = self.mu_f(X=X, out=out, sensitive=sensitive, y=y)
        gap_constraint = F.relu(
            torch.mv(self.M.to(self.device), mu.to(self.device)) - self.c.to(self.device)
        )
        if self.p_norm == 2:
            cons = self.alpha * torch.dot(gap_constraint, gap_constraint)
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

def get_model(nfeat, args):
    if args.model == "GCN":
        model = GCN_Body(nfeat, args.num_hidden, args.dropout)
    elif args.model == "GAT":
        heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
        model = GAT_body(args.num_layers, nfeat, args.num_hidden, heads, args.dropout, args.attn_drop,
                         args.negative_slope, args.residual)
    else:
        print("Model not implement")
        return

    return model


class FairGNN(nn.Module):

    def __init__(self, nfeat, args):
        super(FairGNN, self).__init__()

        nhid = args.num_hidden
        dropout = args.dropout
        self.GNN = get_model(nfeat, args)
        self.fairness_layer = nn.Sequential(nn.Linear(nhid, nhid), nn.ReLU())
        self.classifier = nn.Linear(nhid, 1)

        G_params = list(self.GNN.parameters()) + list(self.classifier.parameters())
        F_params = list(self.fairness_layer.parameters())
        self.optimizer_G = torch.optim.Adam(G_params, lr=1e-3, weight_decay=1e-5)
        self.optimizer_F = torch.optim.Adam(F_params, lr=1e-9, weight_decay=1e-5)
        self.args = args
        self.criterion = nn.BCEWithLogitsLoss()
        self.criterion_fairness = DemographicParityLoss(sensitive_classes=[0, 1], alpha=100, p_norm=2)
        self.scheduler_dp = ExponentialLR(self.optimizer_G, gamma=0.99)
        self.scheduler_acc = ExponentialLR(self.optimizer_F, gamma=0.99)

        self.G_loss = 0
        self.F_loss = 0

    def forward(self, g, x):
        z = self.GNN(g, x)
        z = self.fairness_layer(z)
        y = self.classifier(z)
        return y, z

    def optimize(self, g, x, labels, idx_train, sens, idx_sens_train):
        self.train()
        self.zero_grad()

        h = self.GNN(g, x)
        h = self.fairness_layer(h)
        y = self.classifier(h)
        y = y.squeeze(dim=1)
        self.adv_loss = self.criterion_fairness(x[idx_train], F.sigmoid(y[idx_train]), sens[idx_train])
        self.adv_loss.backward()
        self.optimizer_F.step()

        h = self.GNN(g, x)
        h = self.fairness_layer(h)
        y = self.classifier(h)
        y = y.squeeze(dim=1)
        self.adv_loss = self.criterion_fairness(x[idx_train], y[idx_train], sens[idx_train])
        self.cls_loss = self.criterion(y[idx_train], labels[idx_train].float())
        self.cls_loss.backward()
        self.optimizer_G.step()
        # self.scheduler_dp.step()
        # self.scheduler_acc.step()

