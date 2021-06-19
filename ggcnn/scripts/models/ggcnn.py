import torch.nn as nn
import torch.nn.functional as F

filter_sizes = [32, 16, 8, 8, 16, 32]
kernel_sizes = [9, 5, 3, 3, 5, 9]
strides = [3, 2, 2, 2, 2, 3]

#===custom layer====
class ResidualBlock(nn.Module):
    """
    A residual block with dropout option
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x_in):
        x = self.bn1(self.conv1(x_in))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        return x + x_in

class GGCNN(nn.Module):

    def __init__(self, input_channels=1, dropout=False, prob=0.0, channel_size=32):
        super(GGCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.res1 = ResidualBlock(128, 128)
        self.res2 = ResidualBlock(128, 128)
        self.res3 = ResidualBlock(128, 128)
        self.res4 = ResidualBlock(128, 128)
        self.res5 = ResidualBlock(128, 128)

        self.conv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=2, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)

        self.conv6 = nn.ConvTranspose2d(32, 32, kernel_size=9, stride=1, padding=4)

        self.pos_output = nn.Conv2d(32, 1, kernel_size=2)
        self.cos_output = nn.Conv2d(32, 1, kernel_size=2)
        self.sin_output = nn.Conv2d(32, 1, kernel_size=2)
        self.width_output = nn.Conv2d(32, 1, kernel_size=2)

        self.dropout1 = nn.Dropout(p=prob)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x_in):
        x = F.relu(self.bn1(self.conv1(x_in)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)

        pos_output = self.pos_output(self.dropout1(x))
        cos_output = self.cos_output(self.dropout1(x))
        sin_output = self.sin_output(self.dropout1(x))
        width_output = self.width_output(self.dropout1(x))

        return pos_output, cos_output, sin_output, width_output

    def compute_loss(self, xc, yc):
        y_pos, y_cos, y_sin, y_width = yc
        pos_pred, cos_pred, sin_pred, width_pred = self(xc)

        p_loss = F.smooth_l1_loss(pos_pred, y_pos)
        cos_loss = F.smooth_l1_loss(cos_pred, y_cos)
        sin_loss = F.smooth_l1_loss(sin_pred, y_sin)
        width_loss = F.smooth_l1_loss(width_pred, y_width)

        return {
            'loss': p_loss + cos_loss + sin_loss + width_loss,
            'losses': {
                'p_loss': p_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss
            },
            'pred': {
                'pos': pos_pred,
                'cos': cos_pred,
                'sin': sin_pred,
                'width': width_pred
            }
        }
#===custom layer====



# class GGCNN(nn.Module):
#     """
#     GG-CNN
#     Equivalient to the Keras Model used in the RSS Paper (https://arxiv.org/abs/1804.05172)
#     """
#     def __init__(self, input_channels=1):
#         super().__init__()
#         self.conv1 = nn.Conv2d(input_channels, filter_sizes[0], kernel_sizes[0], stride=strides[0], padding=3)
#         self.conv2 = nn.Conv2d(filter_sizes[0], filter_sizes[1], kernel_sizes[1], stride=strides[1], padding=2)
#         self.conv3 = nn.Conv2d(filter_sizes[1], filter_sizes[2], kernel_sizes[2], stride=strides[2], padding=1)
#         self.convt1 = nn.ConvTranspose2d(filter_sizes[2], filter_sizes[3], kernel_sizes[3], stride=strides[3], padding=1, output_padding=1)
#         self.convt2 = nn.ConvTranspose2d(filter_sizes[3], filter_sizes[4], kernel_sizes[4], stride=strides[4], padding=2, output_padding=1)
#         self.convt3 = nn.ConvTranspose2d(filter_sizes[4], filter_sizes[5], kernel_sizes[5], stride=strides[5], padding=3, output_padding=1)

#         self.pos_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)
#         self.cos_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)
#         self.sin_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)
#         self.width_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)

#         for m in self.modules():
#             if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
#                 nn.init.xavier_uniform_(m.weight, gain=1)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = F.relu(self.convt1(x))
#         x = F.relu(self.convt2(x))
#         x = F.relu(self.convt3(x))

#         pos_output = self.pos_output(x)
#         cos_output = self.cos_output(x)
#         sin_output = self.sin_output(x)
#         width_output = self.width_output(x)

#         return pos_output, cos_output, sin_output, width_output

#     def compute_loss(self, xc, yc):
#         y_pos, y_cos, y_sin, y_width = yc
#         pos_pred, cos_pred, sin_pred, width_pred = self(xc)

#         p_loss = F.mse_loss(pos_pred, y_pos)
#         cos_loss = F.mse_loss(cos_pred, y_cos)
#         sin_loss = F.mse_loss(sin_pred, y_sin)
#         width_loss = F.mse_loss(width_pred, y_width)

#         return {
#             'loss': p_loss + cos_loss + sin_loss + width_loss,
#             'losses': {
#                 'p_loss': p_loss,
#                 'cos_loss': cos_loss,
#                 'sin_loss': sin_loss,
#                 'width_loss': width_loss
#             },
#             'pred': {
#                 'pos': pos_pred,
#                 'cos': cos_pred,
#                 'sin': sin_pred,
#                 'width': width_pred
#             }
#         }
