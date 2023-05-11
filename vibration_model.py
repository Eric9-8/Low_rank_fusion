# 上海工程技术大学
# 崔嘉亮
# 开发时间：2022/5/9 16:28
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import load_Rail, total
import argparse
import os
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from train_rail import display
import csv
import random


# class VibrationNet(nn.Module):
#     def __init__(self, in_size, hidden_size, num_layers=1, dropout=0.2, bidirectional=False):
#         super(VibrationNet, self).__init__()
#         self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout,
#                            bidirectional=bidirectional, batch_first=True)
#         self.dropout = nn.Dropout(dropout)
#         self.linear_1 = nn.Linear(hidden_size, 4)
#
#     def forward(self, x):
#         _, final_states = self.rnn(x)
#         # h = self.dropout(final_states[0].squeeze())
#         # y_1 = self.linear_1(h)
#         output = self.linear_1(_[:, -1, :]).squeeze(0)
#         return output

class VibrationNet(nn.Module):
    def __init__(self, in_size, hidden_size, num_layers=4, dropout=0.2, bidirectional=False):
        super(VibrationNet, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout,
                           bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # self.linear_1 = nn.Linear(hidden_size * 2, 4)
        self.linear_1 = nn.Linear(hidden_size, 4)

    def forward(self, x):
        batch_size, seq_len, embedding_dim = x.shape
        # h0 = torch.randn(self.num_layers * 2, batch_size, self.hidden_size).cuda()
        # c0 = torch.randn(self.num_layers * 2, batch_size, self.hidden_size).cuda()
        h0 = torch.randn(self.num_layers, batch_size, self.hidden_size).cuda()
        c0 = torch.randn(self.num_layers, batch_size, self.hidden_size).cuda()
        out, (_, _) = self.rnn(x, (h0, c0))
        output = self.linear_1(out[:, -1, :]).squeeze(0)
        return output


def main(options):
    run_id = options['run_id']
    epochs = options['epochs']
    data_path = options['data_path']
    model_path = options['model_path']
    output_path = options['output_path']
    signiture = options['signiture']
    patience = options['patience']
    output_dim = options['output_dim']

    print("Training initializing... Setup ID is: {}".format(run_id))

    # prepare the paths for storing models and outputs
    model_path = os.path.join(
        model_path, "model_{}_{}.pt".format(signiture, run_id))
    output_path = os.path.join(
        output_path, "results_{}_{}.csv".format(signiture, run_id))
    print("Temp location for models: {}".format(model_path))
    print("Grid search results are in: {}".format(output_path))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    train_set, valid_set, test_set, input_dims = load_Rail(data_path)

    params = dict()
    params['vibration_hidden'] = [16, 32, 64]
    params['vibration_dropout'] = [0, 0.1, 0.15, 0.2, 0.3, 0.5]
    params['factor_learning_rate'] = [0.0003, 0.0005, 0.001, 0.003]
    params['learning_rate'] = [0.0003, 0.0005, 0.001, 0.003]
    params['batch_size'] = [2, 4, 6, 8, 10, 12]
    params['weight_decay'] = [0, 0.001, 0.002, 0.01]

    total_settings = total(params)

    seen_settings = set(params)
    print("There are {} different hyper-parameter settings in total.".format(total_settings))

    with open(output_path, 'w+') as out:
        writer = csv.writer(out)
        writer.writerow(
            ["image_hidden", 'image_dropout', 'factor_learning_rate',
             'learning_rate', 'batch_size', 'weight_decay', 'Best Validation CrossEntropyLoss',
             'Test CrossEntropyLoss', 'Test MAE', 'Test F1-score', 'Test Accuracy Score', 'Test Accuracy'])

    for i in range(total_settings):

        vhid = random.choice(params['vibration_hidden'])
        vdr = random.choice(params['vibration_dropout'])
        factor_lr = random.choice(params['factor_learning_rate'])
        lr = random.choice(params['learning_rate'])
        batch_sz = random.choice(params['batch_size'])
        decay = random.choice(params['weight_decay'])

        # reject the setting if it has been tried
        current_setting = (vhid, vdr, factor_lr, lr, batch_sz, decay)
        if current_setting in seen_settings:
            continue
        else:
            seen_settings.add(current_setting)

        model = VibrationNet(input_dims[1], vhid, dropout=vdr)

        if options['cuda']:
            model = model.cuda()
            DTYPE = torch.cuda.FloatTensor
            LONG = torch.cuda.LongTensor

            print("Model initialized")
            # criterion = nn.L1Loss(size_average=False)
            criterion = nn.CrossEntropyLoss()

            factors = list(model.parameters())[:3]
            other = list(model.parameters())[3:]
            optimizer = optim.Adam([{"params": factors, "lr": factor_lr}, {"params": other, "lr": lr}],
                                   weight_decay=decay)

            complete = True
            min_valid_loss = float('Inf')
            train_iterator = DataLoader(train_set, batch_size=batch_sz, num_workers=0, shuffle=True)
            valid_iterator = DataLoader(valid_set, batch_size=len(valid_set), num_workers=0, shuffle=True)
            test_iterator = DataLoader(test_set, batch_size=len(test_set), num_workers=0, shuffle=True)
            curr_patience = patience

            for epoch in range(epochs):
                model.train()
                model.zero_grad()
                avg_train_loss = 0.0
                for batch in train_iterator:
                    model.zero_grad()

                    x = batch[:-1]
                    x_i = Variable(x[1].float().type(DTYPE), requires_grad=False).squeeze(1)
                    y = Variable(batch[-1].view(-1, output_dim).float().type(LONG), requires_grad=False)
                    output = model(x_i)
                    loss = criterion(output, y.data.squeeze())
                    # cc = torch.max(y, 1)[0].data.squeeze()
                    # loss = criterion(output, y)
                    loss.backward()
                    avg_loss = loss.item()
                    avg_train_loss += avg_loss / len(train_set)
                    optimizer.step()

                print("Epoch {} complete! Average Training loss: {}".format(epoch, avg_train_loss))

                # Terminate the training process if run into NaN
                if np.isnan(avg_train_loss):
                    print("Training got into NaN values...\n\n")
                    complete = False
                    break

                model.eval()
                torch.no_grad()
                for batch in valid_iterator:
                    x = batch[:-1]
                    x_i = Variable(x[1].float().type(DTYPE), requires_grad=False).squeeze(1)
                    y = Variable(batch[-1].view(-1, output_dim).float().type(LONG), requires_grad=False)
                    output = model(x_i)
                    valid_loss = criterion(output, y.data.squeeze())
                    # valid_loss = criterion(output, y)
                    avg_valid_loss = valid_loss.item()
                y = y.cpu().data.numpy().reshape(-1, output_dim)

                if np.isnan(avg_valid_loss):
                    print("Training got into NaN values...\n\n")
                    complete = False
                    break

                avg_valid_loss = avg_valid_loss / len(valid_set)
                print("Validation loss is: {}".format(avg_valid_loss))

                if (avg_valid_loss < min_valid_loss):
                    curr_patience = patience
                    min_valid_loss = avg_valid_loss
                    torch.save(model, model_path)
                    print("Found new best model, saving to disk...")
                else:
                    curr_patience -= 1

                if curr_patience <= 0:
                    break
                print("\n\n")

            if complete:

                best_model = torch.load(model_path)
                best_model.eval()
                torch.no_grad()
                for batch in test_iterator:
                    x = batch[:-1]
                    x_i = Variable(x[1].float().type(DTYPE), requires_grad=False).squeeze(1)
                    y = Variable(batch[-1].view(-1, output_dim).float().type(LONG), requires_grad=False)
                    output_test = best_model(x_i)
                    loss_test = criterion(output_test, y.data.squeeze())
                    # loss_test = criterion(output_test, y)
                    test_loss = loss_test.item()

                output_test = torch.max(output_test, 1)[1].data.squeeze()
                # output_test = output_test.cpu().data.numpy().reshape(-1, output_dim)
                y = y.cpu().data.numpy().reshape(-1, output_dim)

                # output_test = output_test.reshape((len(output_test),))
                output_test = output_test.cpu().data.numpy().reshape((len(output_test),))
                y = y.reshape((len(y),))

                test_loss = test_loss / len(test_set)

                # these are the needed metrics
                # all_true_label = np.argmax(y, axis=1)  # The index of max value
                # all_predicted_label = np.argmax(output_test, axis=1)
                all_true_label = y
                all_predicted_label = np.round(output_test)

                mae = np.mean(np.absolute(output_test - y), axis=0)
                f1 = f1_score(all_true_label, all_predicted_label, average='weighted')
                acc_score = accuracy_score(all_true_label, all_predicted_label)
                mult_acc = [round(sum(np.round(output_test[:]) == np.round(y[:])) / float(len(y)), 3)]

                display(mae, f1, acc_score)

                with open(output_path, 'a+') as out:
                    writer = csv.writer(out)
                    writer.writerow([vhid, vdr, factor_lr, lr, batch_sz, decay,
                                     min_valid_loss, test_loss, mae, f1, acc_score, mult_acc])


if __name__ == "__main__":
    OPTIONS = argparse.ArgumentParser()
    OPTIONS.add_argument('--run_id', dest='run_id', type=str, default='3661-vb')
    OPTIONS.add_argument('--epochs', dest='epochs', type=int, default=500)
    OPTIONS.add_argument('--patience', dest='patience', type=int, default=20)
    OPTIONS.add_argument('--output_dim', dest='output_dim', type=int, default=1)
    OPTIONS.add_argument('--signiture', dest='signiture', type=str, default='')
    OPTIONS.add_argument('--cuda', dest='cuda', type=bool, default=True)
    OPTIONS.add_argument('--data_path', dest='data_path',
                         type=str, default='D:/Pytorch/Fusion_lowrank/data/')
    OPTIONS.add_argument('--model_path', dest='model_path',
                         type=str, default='models')
    OPTIONS.add_argument('--output_path', dest='output_path',
                         type=str, default='results')
    PARAMS = vars(OPTIONS.parse_args())
    main(PARAMS)
