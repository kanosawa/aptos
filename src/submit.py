import os
import pandas as pd
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import optim, nn
from efficientnet_pytorch import EfficientNet
from tqdm import tqdm
from sklearn.model_selection import train_test_split


class AptosDataset(Dataset):

    def __init__(self, df, img_dir, mode='train'):

        df['path'] = df['id_code'].map(lambda x: os.path.join(img_dir, '{}.png'.format(x)))
        self.df = df
        self.mode = mode

    def __len__(self):
        return len(self.df)
        #return 100

    def __getitem__(self, idx):

        img = cv2.imread(self.df['path'].iloc[idx])
        img = cv2.resize(img, (512, 512))
        img = img.astype('float32')
        img = img.transpose(2, 0, 1) / 255.0

        if self.mode == 'train':
            diagnosis = self.df['diagnosis'].iloc[idx]
            return img, diagnosis
        else:
            return img, self.df['id_code'].iloc[idx]


class AptosNet(nn.Module):
    def __init__(self):
        super(AptosNet, self).__init__()
        snapshot = torch.load('../input/efficientnet-pytorch/efficientnet-b0-08094119.pth')
        self.base_model = EfficientNet.from_name('efficientnet-b0')
        self.base_model.load_state_dict(snapshot)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 5)

    def forward(self, x):
        x = self.base_model.extract_features(x)
        x = torch.max(x, axis=1)[0]
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_epoch(data_loader, model, criterion, optimizer, epoch):

    model.train()
    loss_sum = 0
    for data, target in tqdm(data_loader):

        data = data.cuda()
        target = target.cuda()

        output = model(data)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()

    loss_epoch = loss_sum / len(data_loader)
    print('train_loss = ', loss_epoch)


def val_epoch(data_loader, model, criterion, epoch):

    model.eval()
    loss_sum = 0
    for data, target in tqdm(data_loader):

        data = data.cuda()
        target = target.cuda()

        output = model(data)
        loss = criterion(output, target)

        loss_sum += loss.item()

    loss_epoch = loss_sum / len(data_loader)
    print('val_loss = ', loss_epoch)


def train(data_dir):

    num_epoch = 2

    df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    df_train, df_val = train_test_split(df, test_size=0.1, random_state=42)

    train_dataset = AptosDataset(df=df_train, img_dir=os.path.join(data_dir, 'train_images/'))
    train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_dataset = AptosDataset(df=df_val, img_dir=os.path.join(data_dir, 'train_images/'))
    val_loader = DataLoader(dataset=val_dataset, batch_size=4, shuffle=False, num_workers=4)

    model = AptosNet().cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epoch):
        train_epoch(train_loader, model, criterion, optimizer, epoch)
        val_epoch(val_loader, model, criterion, epoch)

    return model


def test(data_dir, model):

    df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    df = df.assign(diagnosis=0)

    test_dataset = AptosDataset(df, img_dir=os.path.join(data_dir, 'test_images/'), mode='test')
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=1)

    df = df.set_index('id_code')

    for data, ids in tqdm(test_loader):
        data = data.cuda()

        output = model(data)
        _, preds = torch.max(output, axis=1)
        df.at[ids[0], 'diagnosis'] = preds.cpu().item()

    df = df.drop(columns=['path'])
    df.to_csv('submission.csv')


if __name__ == '__main__':

    data_dir = '../input/aptos2019-blindness-detection'
    model = train(data_dir)
    test(data_dir, model)