import itertools
import pickle
import time

import numpy as np
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Dataset
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.resnet import BasicBlock
from torch import nn, optim
import torch
from torchvision.transforms import transforms
from torch.nn import functional as F

import os
import pandas as pd
from torchvision.io import read_image
from tqdm import tqdm

train_split = 0.95
batchsize = 128


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


df = pd.read_table("CelebA/Anno/list_attr_celeba_train.txt", delim_whitespace=True)


def dataset_construction(feature_index):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(256),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    class CustomImageDataset(Dataset):
        def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
            self.img_labels = pd.read_table(annotations_file, delim_whitespace=True)

            self.img_dir = img_dir
            self.transform = transform
            self.target_transform = target_transform

        def __len__(self):
            return len(self.img_labels)

        def __getitem__(self, idx):
            img_path = os.path.join(self.img_dir, self.img_labels.index[idx])
            image = read_image(img_path)
            thefeature = 0 if self.img_labels.iloc[idx, feature_index] == -1 else 1
            gender_male = 0 if self.img_labels.iloc[idx, 20] == -1 else 1
            if self.transform:
                image = self.transform(image)
            return image, gender_male, thefeature

    dataset = CustomImageDataset("CelebA/Anno/list_attr_celeba_train.txt", "CelebA/img_align_celeba_train/",
                                 transform=preprocess)

    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size

    dataset_sizes = {'train': train_size, 'val': val_size}
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    from torch.utils.data import DataLoader
    dataloaders = {'train': DataLoader(train_dataset, batch_size=batchsize, shuffle=True),
                   'val': DataLoader(test_dataset, batch_size=batchsize, shuffle=True)}

    return dataset_sizes, dataloaders


device = torch.device('cpu')
model_res = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model_res.fc = nn.Sequential(
    nn.Linear(512, 1, bias=True),
    nn.Sigmoid()
)

acc_params = model_res.fc.parameters()
fair_params = model_res.layer4.parameters()
model_res = model_res.to(device)

criterion = nn.BCELoss()
criterion_fairness = DemographicParityLoss(sensitive_classes=[0, 1], alpha=1000, p_norm=2)

optimizer_ft = optim.SGD(model_res.parameters(), lr=0.001, momentum=0.9)

dataset_sizes, dataloaders = dataset_construction(list(df.columns).index("Attractive"))


def test_func(model_f):
    y_pred = []
    sens = []
    y_label = []
    print("Testing:")
    print("-------------------")
    for inputs, gender_males, attractives, in dataloaders['val']:
        with torch.no_grad():
            y_pred.append(model_f(inputs).cpu())
            sens.append(gender_males)
            y_label.append(attractives)

    y_pred = torch.cat(y_pred, dim=0)
    sens = torch.cat(sens, dim=0)
    y_label = torch.cat(y_label, dim=0)
    matches = [torch.round(i) == torch.round(j) for i, j in zip(y_label, y_pred)]
    acc = matches.count(True) / len(matches)
    return acc, criterion_fairness(y_label, y_pred, sens)


def fwd_pass(x, sens, y_l, criterion_dp, criterion_acc, optimizer_dp, optimizer_acc, model):
    model.zero_grad()
    x = torch.Tensor(x).to(torch.float)
    y_l = torch.Tensor(y_l).view(-1, 1).to(torch.float)
    y_l = y_l.to(device)
    out = model(x.to(device))
    out = out.to(torch.float)

    loss_dp = criterion_dp(y_l, out, sens)
    loss_dp.backward()
    optimizer_dp.step()

    out = model(x.to(device))
    out = out.to(torch.float)
    loss_acc = criterion_acc(out, y_l)
    loss_acc.backward()
    optimizer_acc.step()
    out = model(x.to(device))

    matches = [torch.round(i) == torch.round(j) for i, j in zip(out, y_l)]
    acc = matches.count(True) / len(matches)

    return acc, loss_acc, loss_dp.detach().numpy(), out


acc_dp = {}
def train_model(eta, model, alr, flr, epochs):
    MODEL_NAME = f"model-{int(time.time())}"

    optimizer_acc = torch.optim.Adam(acc_params, lr=alr)
    optimizer_dp = torch.optim.Adam(fair_params, lr=flr)
    criterion_acc = torch.nn.BCELoss()
    scheduler_dp = ExponentialLR(optimizer_dp, gamma=0.9)
    scheduler_acc = ExponentialLR(optimizer_acc, gamma=0.9)
    s_c = [0, 1]
    criterion_dp = DemographicParityLoss(sensitive_classes=s_c, alpha=eta)

    test_acc = []
    test_dp = []
    with open("model.log", "a") as f:
        for epoch in range(epochs):
            losses = []
            accs = []
            losses_dp = []
            with tqdm(dataloaders['train']) as tepoch:
                for inputs, gender_males, attractives in tepoch:
                    tepoch.set_description(f"Eta {eta}, Epoch {epoch + 1}")

                    acc, loss, loss_dp, _ = fwd_pass(inputs, gender_males, attractives, criterion_dp, criterion_acc, optimizer_dp,
                                                     optimizer_acc, model)

                    losses.append(loss.item())
                    losses_dp.append(loss_dp)
                    accs.append(acc)
                    acc_mean = np.array(accs).mean()
                    loss_mean = np.array(losses).mean()
                    loss_dp_mean = np.array(losses_dp).mean()
                    tepoch.set_postfix(loss=loss_mean, accuracy=100. * acc_mean, loss_dp=loss_dp_mean)
            acc, sdp = test_func(model)
            test_acc.append(acc)
            test_dp.append(sdp[0])
            print(f'ACC: {acc}')
            print(f'SDP: {sdp}')
            f.write(
                f"{MODEL_NAME},{epoch},{round(float(acc_mean), 2)},{round(float(loss_mean), 4)},{acc},{sdp}\n")
            if (epoch + 1) % 20 == 0:
                scheduler_acc.step()
                scheduler_dp.step()
            dt = time.strftime("%Y_%m_%d-%H_%M_%S")

        acc_dp[eta] = (test_acc, test_dp)

    with open(f'acc_dps_.pkl', 'wb') as fp:
        pickle.dump(acc_dp, fp)

train_model(1000, model_res, 1e-3, 1e-5, 20)

torch.save(model_res.state_dict(), "saved_model_weights/other_classifiers/Attractive")
