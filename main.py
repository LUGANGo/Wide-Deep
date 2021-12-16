import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader
from model import WideDeep


class GetData(Dataset):
    def __init__(self, d_f_num, ori_data_path):
        super().__init__()

        self.data = pd.read_csv(ori_data_path, sep=',')
        self.Len = self.data.shape[0]
        self.label = self.data.Label.values
        self.label = torch.from_numpy(self.label).type(torch.float32)
        del self.data['Label']

        self.dense_part = self.data.iloc[:, :d_f_num].values
        self.sparse_part = self.data.iloc[:, d_f_num:].values

    def __getitem__(self, idx):
        dense_part = torch.from_numpy(self.dense_part[idx]).type(torch.float32)
        sparse_part = torch.from_numpy(self.sparse_part[idx]).type(torch.long)
        return dense_part, sparse_part, self.label[idx]

    def __len__(self):
        return self.Len


info_path = 'data/fea_col.npy'
info = np.load(info_path, allow_pickle=True)
dense_fea_num = len(info[0])
sparse_fea_num = len(info[1])
sparse_fea_dict = [x['feat_num'] for x in info[1]]
k = 16
hidden_layer_list = [256, 128, 64]

Model = WideDeep(dense_fea_num, sparse_fea_num, sparse_fea_dict, k, hidden_layer_list)
optimizer = torch.optim.Adam(Model.parameters(), lr=0.01)
loss_fn = torch.nn.BCELoss()

train_data = GetData(dense_fea_num, 'data/train_set.csv')
valid_data = GetData(dense_fea_num, 'data/val_set.csv')

valid_data_loader = DataLoader(
    dataset=valid_data,
    batch_size=valid_data.Len,
    shuffle=False
)

print('数据准备完成！！！')


def auc(y_pred, y_true):
    pred = y_pred.data
    y = y_true.data
    return roc_auc_score(y, pred)


def train():
    train_data_loader = DataLoader(
        dataset=train_data,
        batch_size=64,
        shuffle=True
    )

    loss_log = []
    for x_dense, x_sparse, label in train_data_loader:
        pred = Model(x_dense, x_sparse).squeeze()
        loss = loss_fn(pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_log.append(loss.item())
    # print('loss:{:.4f}'.format(sum(loss_log)))
    pass


def test():
    with torch.no_grad():
        for x_dense, x_sparse, label in valid_data_loader:
            pred = Model(x_dense, x_sparse)
            print('auc', auc(pred, label))


def run():
    epochs = 100
    for i in range(epochs):
        train()
        if (i + 1) % 10 == 0:
            test()


if __name__ == "__main__":
    run()
