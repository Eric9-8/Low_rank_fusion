# 上海工程技术大学
# 崔嘉亮
# 开发时间：2022/4/27 21:30
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal_


class ImageNet(nn.Module):
    def __init__(self, in_size, hidden_size, dropout):
        super(ImageNet, self).__init__()
        self.norm = nn.BatchNorm2d(in_size)
        self.norm1 = nn.BatchNorm2d(32)
        self.norm2 = nn.BatchNorm2d(64)
        self.drop = nn.Dropout(p=dropout)
        self.pooling = nn.MaxPool2d(2)
        self.linear_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.linear_2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.linear_3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.linear_4 = nn.Linear(64 * 16 * 16, hidden_size)
        self.linear_5 = nn.Linear(hidden_size, hidden_size)
        self.flatten = nn.Flatten()

    def forward(self, x):
        # x = self.norm(x)
        x = F.relu(self.pooling((self.linear_1(x))))
        x = self.norm1(x)
        x = F.relu(self.pooling((self.linear_2(x))))
        x = self.norm2(x)
        x = F.relu(self.pooling((self.linear_3(x))))
        x = self.norm2(x)
        x = self.drop(x)
        x = self.flatten(x)
        x = F.relu((self.linear_4(x)))
        x = self.linear_5(x)
        return x


# class VibrationNet(nn.Module):
#     def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
#         super(VibrationNet, self).__init__()
#         self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout,
#                            bidirectional=bidirectional, batch_first=True)
#         self.dropout = nn.Dropout(dropout)
#         self.linear_1 = nn.Linear(hidden_size, out_size)
#
#     def forward(self, x):
#         _, final_states = self.rnn(x)
#         h = self.dropout(final_states[0].squeeze())
#         y_1 = self.linear_1(h)
#         return y_1

class VibrationNet(nn.Module):
    def __init__(self, in_size, hidden_size, dropout):
        super(VibrationNet, self).__init__()
        self.norm = nn.BatchNorm2d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.pooling = nn.MaxPool2d(2)
        self.linear_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.linear_2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.linear_3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.linear_4 = nn.Linear(64 * 16 * 16, hidden_size)
        self.linear_5 = nn.Linear(hidden_size, hidden_size)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.norm(x)
        x = F.relu(self.pooling((self.linear_1(x))))
        x = F.relu(self.pooling((self.linear_2(x))))
        x = F.relu(self.pooling((self.linear_3(x))))
        x = self.drop(x)
        x = self.flatten(x)
        x = F.relu((self.linear_4(x)))
        x = self.linear_5(x)
        return x


class LMF(nn.Module):

    def __init__(self, input_dims, hidden_dims, vibration_out, dropouts, output_dim, rank, use_softmax=False):

        super(LMF, self).__init__()

        self.image_in = input_dims[0]
        self.vibration_in = input_dims[1]

        self.image_hidden = hidden_dims[0]
        self.vibration_hidden = hidden_dims[1]

        self.vibration_out = vibration_out
        self.output_dim = output_dim
        self.rank = rank
        self.use_softmax = use_softmax

        self.image_prob = dropouts[0]
        self.vibration_prob = dropouts[1]
        self.post_fusion_prob = dropouts[2]

        self.image_subnet = ImageNet(self.image_in, self.image_hidden, self.image_prob)
        # self.vibration_subnet = VibrationNet(self.vibration_in, self.vibration_hidden, self.vibration_out,
        #                                      dropout=self.vibration_prob)
        self.vibration_subnet = VibrationNet(self.vibration_in, self.vibration_hidden, dropout=self.vibration_prob)

        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)

        self.image_factor = Parameter(torch.Tensor(self.rank, self.image_hidden + 1, self.output_dim))
        # self.vibration_factor = Parameter(torch.Tensor(self.rank, self.vibration_out + 1, self.output_dim))
        self.vibration_factor = Parameter(torch.Tensor(self.rank, self.image_hidden + 1, self.output_dim))

        self.fusion_weights = Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = Parameter(torch.Tensor(1, self.output_dim))

        xavier_normal_(self.image_factor)
        xavier_normal_(self.vibration_factor)
        xavier_normal_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

        self.rnn = nn.LSTM(1, self.vibration_hidden, num_layers=4, dropout=0.25, bidirectional=False, batch_first=True)
        self.dropout = nn.Dropout(0.25)
        # self.linear_1 = nn.Linear(self.vibration_hidden * 2, 4)
        self.linear_1 = nn.Linear(self.vibration_hidden, 4)

    def forward(self, image_x, vibration_x):

        image_h = self.image_subnet(image_x)
        vibration_h = self.vibration_subnet(vibration_x)

        batch_size = image_h.data.shape[0]

        if image_h.is_cuda:
            DTYPE = torch.cuda.FloatTensor
        else:
            DTYPE = torch.FloatTensor

        _image_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), image_h), dim=1)
        # _vibration_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), vibration_h),
        #                          dim=1)
        _vibration_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), image_h), dim=1)

        fusion_image = torch.matmul(_image_h, self.image_factor)
        fusion_vibration = torch.matmul(_vibration_h, self.vibration_factor)
        # fusion_vibration = torch.matmul(_vibration_h, self.image_factor)
        fusion_zy = fusion_image * fusion_vibration

        output = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
        output = output.view(-1, self.output_dim)
        output = output.unsqueeze(-1)
        batch_size, seq_len, embedding_dim = output.shape
        # h0 = torch.randn(2 * 2, batch_size, self.vibration_hidden).cuda()
        # c0 = torch.randn(2 * 2, batch_size, self.vibration_hidden).cuda()
        h0 = torch.randn(4, batch_size, self.vibration_hidden).cuda()
        c0 = torch.randn(4, batch_size, self.vibration_hidden).cuda()
        # h = self.dropout(final_states[0].squeeze())
        # y_1 = self.linear_1(h)
        out, (_, _) = self.rnn(output, (h0, c0))
        output = self.linear_1(out[:, -1, :]).squeeze(0)
        if self.use_softmax:
            output = F.softmax(output)
        return output
