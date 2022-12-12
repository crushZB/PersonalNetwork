import torch.nn as nn
import torch
import torch.nn.functional as F
from loss.charbonnier import CharbonnierL1Loss
import torch.optim as optim

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class Teacher(nn.Module):
    """
    Takes hazy image as input and outputs hazy free image
    """

    def __init__(self, input_channels=3, inner_channels=64, block_count=1, mimicking_layers=[2, 4, 6]):
        super(Teacher, self).__init__()

        output_channels = input_channels

        self.mimicking_layers = mimicking_layers

        self.downsample = _make_downsample_layer(input_channels, inner_channels)

        self.res_blocks = nn.ModuleList(
            [ResidualInResiduals(inner_channels, block_count=3) for i in range(block_count)])

        self.upsample = _make_upsample_layer(inner_channels, inner_channels)

        self.reconstruction = _make_reconstruction_layer(inner_channels, output_channels)

    def forward(self, hazy_image):

        rec = self.downsample(hazy_image)

        for i, _ in enumerate(self.res_blocks):
            rec = self.res_blocks[i](rec)

        rec = self.upsample(rec)

        rec = self.reconstruction(rec)

        # Reshape the output image to match input image (odd shapes cause mismatch problem)
        return rec




class Student(nn.Module):
    """
    Takes hazy image as input and outputs hazy free image
    """

    def __init__(self, input_channels=3, inner_channels=64, block_count=2, mimicking_layers=[2, 4, 6]):
        super(Student, self).__init__()

        output_channels = input_channels

        self.mimicking_layers = mimicking_layers

        self.downsample = _make_downsample_layer(input_channels, inner_channels)

        self.res_blocks = nn.ModuleList(
            [ResidualInResiduals(inner_channels, block_count=3) for i in range(block_count)])

        self.upsample = _make_upsample_layer(inner_channels, inner_channels)

        self.reconstruction = _make_reconstruction_layer(inner_channels, output_channels)

    def forward(self, hazy_image):

        rec = self.downsample(hazy_image)

        for i, _ in enumerate(self.res_blocks):
            rec = self.res_blocks[i](rec)

        rec = self.upsample(rec)

        rec = self.reconstruction(rec)

        # Reshape the output image to match input image (odd shapes cause mismatch problem)
        return rec



class SAB(nn.Module):

    def __init__(self, inlayer=64, outlayer=64):
        super(SAB, self).__init__()

        self.conv3 = nn.Conv2d(inlayer,outlayer,kernel_size=1,stride=1)
        self.gelu = nn.GELU()
        self.conv1 = nn.Conv2d(inlayer,outlayer//16,kernel_size=1,stride=1)
        self.conv2 = nn.Conv2d(outlayer//16, 1, kernel_size=1, stride=1)
        self.restore = nn.Conv2d(1,outlayer,kernel_size=1,stride=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        out = x
        out = self.conv3(out)
        out = self.gelu(out)
        out = self.conv3(out)
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.conv2(x)
        x = self.sig(x)
        x = self.restore(x)
        x = x * out
        # Calculate attention
        return x


class ResidualInResiduals(nn.Module):

    def __init__(self, inner_channels=64, block_count=3):
        super(ResidualInResiduals, self).__init__()

        self.res_blocks = nn.ModuleList([SAB(inner_channels, inner_channels) for i in range(block_count)])

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        residual = x

        for i, _ in enumerate(self.res_blocks):
                x = self.res_blocks[i](x)

        x = self.conv_block1(x)

        x = x + residual

        return x


class  _make_reconstruction_layer(nn.Module):
    def __init__(self,inlayer, outlayer, stride=1):
        super(_make_reconstruction_layer, self).__init__()
        self.recon = nn.Sequential(
            nn.Conv2d(inlayer, outlayer, kernel_size=3, stride=stride, padding=1),
            nn.Tanh(),
        )
    def forward(self,x):
        x = self.recon(x)
        return x

class _make_downsample_layer(nn.Module):
    def __init__(self, inlayer, outlayer, stride=2):
        super(_make_downsample_layer, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(inlayer, outlayer, kernel_size=3, stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(outlayer, outlayer, kernel_size=3, stride=stride, padding=1),
            nn.ReLU()
        )
    def forward(self, x):
            x = self.downsample(x)
            return x

class _make_upsample_layer(nn.Module):
    def __init__(self, inlayer, outlayer, stride=1):
        super(_make_upsample_layer, self).__init__()
        self.upsample = nn.Sequential(
            nn.Conv2d(inlayer, inlayer, kernel_size=3, stride=stride, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(inlayer, outlayer, kernel_size=3, stride=stride, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear")
        )

    def forward(self, x):
        x = self.upsample(x)
        return x


# class Distilled(nn.Module):
#
#     def __init__(self):
#         super(Distilled, self).__init__()
#
#         self.teacher = Teacher()
#
#         self.student = Student()
#
#     def forward(self, x):
#         rec_gt = self.teacher.forward(x)
#
#         rec_hazy_free = self.student.forward(x)
#
#         results = {"rec_gt": rec_gt,
#                    "rec_hazy_free": rec_hazy_free}
#
#         return results



if __name__ == '__main__':
    x = torch.randn([1,3,256,256])
    x = torch.Tensor(x)
    model = Teacher()
    x = model(x)
    print(x.size())