import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

# 新的fuse “+ x” , "1,2(+),3" , 'lfe+' , 双cmc

up_kwargs = {'mode': 'bilinear', 'align_corners': True}

def conv1(in_chsnnels, out_channels):
    "1x1 convolution with padding"
    return nn.Conv2d(in_chsnnels, out_channels, kernel_size=1, stride=1, bias=False)


def conv3(in_chsnnels, out_channels):
    "3x3 convolution with padding"
    return nn.Conv2d(in_chsnnels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

class GMNet(nn.Module):

    def __init__(self, n_classes):
        super(GMNet, self).__init__()


        self.num_resnet_layers = 50

        if self.num_resnet_layers == 18:
            resnet_raw_model1 = models.resnet18(pretrained=True)
            resnet_raw_model2 = models.resnet18(pretrained=True)

        elif self.num_resnet_layers == 34:
            resnet_raw_model1 = models.resnet34(pretrained=True)
            resnet_raw_model2 = models.resnet34(pretrained=True)

        elif self.num_resnet_layers == 50:
            resnet_raw_model1 = models.resnet50(pretrained=True)
            resnet_raw_model2 = models.resnet50(pretrained=True)

        elif self.num_resnet_layers == 101:
            resnet_raw_model1 = models.resnet101(pretrained=True)
            resnet_raw_model2 = models.resnet101(pretrained=True)

        elif self.num_resnet_layers == 152:
            resnet_raw_model1 = models.resnet152(pretrained=True)
            resnet_raw_model2 = models.resnet152(pretrained=True)


        ########  Thermal ENCODER  ########

        self.encoder_thermal_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder_thermal_conv1.weight.data = torch.unsqueeze(torch.mean(resnet_raw_model1.conv1.weight.data, dim=1),
                                                                 dim=1)
        self.encoder_thermal_bn1 = resnet_raw_model1.bn1
        self.encoder_thermal_relu = resnet_raw_model1.relu
        self.encoder_thermal_maxpool = resnet_raw_model1.maxpool
        self.encoder_thermal_layer1 = resnet_raw_model1.layer1
        self.encoder_thermal_layer2 = resnet_raw_model1.layer2
        self.encoder_thermal_layer3 = resnet_raw_model1.layer3
        self.encoder_thermal_layer4 = resnet_raw_model1.layer4

        ########  RGB ENCODER  ########

        self.encoder_rgb_conv1 = resnet_raw_model2.conv1
        self.encoder_rgb_bn1 = resnet_raw_model2.bn1
        self.encoder_rgb_relu = resnet_raw_model2.relu
        self.encoder_rgb_maxpool = resnet_raw_model2.maxpool
        self.encoder_rgb_layer1 = resnet_raw_model2.layer1
        self.encoder_rgb_layer2 = resnet_raw_model2.layer2
        self.encoder_rgb_layer3 = resnet_raw_model2.layer3
        self.encoder_rgb_layer4 = resnet_raw_model2.layer4

        self.tp1 = nn.Conv2d(64, 64, kernel_size=1)
        self.tp2 = nn.Conv2d(256, 64, kernel_size=1)
        self.tp3 = nn.Conv2d(512, 64, kernel_size=1)
        self.tp4 = nn.Conv2d(1024, 64, kernel_size=1)
        self.tp5 = nn.Conv2d(2048, 64, kernel_size=1)

        self.tpmid = nn.Conv2d(128, 64, kernel_size=1)

        self.densefuse1 = Fuseblock(512)
        self.densefuse2 = Fuseblock(1024)
        self.densefuse3 = Fuseblock(2048)

        self.in1 = CMC(64, 64)
        self.in2 = CMC(64, 64)
        self.in3 = CMC(64, 64)
        self.gc1 = CMC(64, 64)
        self.gc2 = CMC(64, 64)
        self.gc3 = CMC(64, 64)

        self.convb = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.ss1 = SSnbt(64, 2)
        self.ss2 = SSnbt(64, 5)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # decoder_conv
        self.decoder5 = decoder(64)
        self.decoder4 = decoder(64)
        self.decoder3 = decoder(64)
        self.decoder2 = decoder(64)
        self.decoder2_withoutup = decoder_without(64)
        self.decoder1 = decoder(64)

        self.con1 = nn.Conv2d(64*3, 64, kernel_size=1, stride=1, bias=False)

        self.classfier1 = nn.Sequential(
            conv1(64, n_classes),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.classfier2 = nn.Sequential(
            conv1(64, 2),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.classfier3 = nn.Sequential(
            conv1(64, 2),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.projectlayer = nn.Conv2d(64, n_classes, 1)

        self.classify = Classifier(192, n_classes)

        self.fd = decoder_without(64)
        self.md = decoder_without(64)

        self.upmid = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.lfe1 = LFE(64)
        self.lfe2 = LFE(256)


    def forward(self, rgb, depth):
        rgb = rgb
        thermal = depth[:, :1, ...]

        vobose = False

        # encoder

        ######################################################################

        if vobose: print("rgb.size() original: ", rgb.size())  # (480, 640)
        if vobose: print("thermal.size() original: ", thermal.size())  # (480, 640)

        ######################################################################

        rgb = self.encoder_rgb_conv1(rgb)
        if vobose: print("rgb.size() after conv1: ", rgb.size())  # (240, 320)
        rgb = self.encoder_rgb_bn1(rgb)
        if vobose: print("rgb.size() after bn1: ", rgb.size())  # (240, 320)
        rgb = self.encoder_rgb_relu(rgb)
        if vobose: print("rgb.size() after relu: ", rgb.size())  # (240, 320)

        thermal = self.encoder_thermal_conv1(thermal)
        if vobose: print("thermal.size() after conv1: ", thermal.size())  # (240, 320)
        thermal = self.encoder_thermal_bn1(thermal)
        if vobose: print("thermal.size() after bn1: ", thermal.size())  # (240, 320)
        thermal = self.encoder_thermal_relu(thermal)
        if vobose: print("thermal.size() after relu: ", thermal.size())  # (240, 320)

        # print(rgb.size())

        r = rgb

        # rgb = rgb + thermal

        rgb1 = self.encoder_rgb_maxpool(rgb)
        if vobose: print("rgb.size() after maxpool: ", rgb.size())  # (120, 160)

        thermal1 = self.encoder_thermal_maxpool(thermal)
        if vobose: print("thermal.size() after maxpool: ", thermal.size())  # (120, 160)

        ######################################################################

        rgb1 = self.encoder_rgb_layer1(rgb1)
        if vobose: print("rgb.size() after layer1: ", rgb.size())  # (120, 160)
        thermal1 = self.encoder_thermal_layer1(thermal1)
        if vobose: print("thermal.size() after layer1: ", thermal.size())  # (120, 160)

        # print(rgb1.size())

        ######################################################################
        r1 = rgb1

        # rgb1 = rgb1 + thermal1

        rgb2 = self.encoder_rgb_layer2(rgb1)
        if vobose: print("rgb.size() after layer2: ", rgb.size())  # (60, 80)
        thermal2 = self.encoder_thermal_layer2(thermal1)
        if vobose: print("thermal.size() after layer2: ", thermal.size())  # (60, 80)

        ######################################################################

        # rgb2 = rgb2 + thermal2

        rgb3 = self.encoder_rgb_layer3(rgb2)
        if vobose: print("rgb.size() after layer3: ", rgb.size())  # (30, 40)
        thermal3 = self.encoder_thermal_layer3(thermal2)
        if vobose: print("thermal.size() after layer3: ", thermal.size())  # (30, 40)

        ######################################################################
        # rgb3 = rgb3 + thermal3

        rgb4 = self.encoder_rgb_layer4(rgb3)
        if vobose: print("rgb.size() after layer4: ", rgb.size())  # (15, 20)
        thermal4 = self.encoder_thermal_layer4(thermal3)
        if vobose: print("thermal.size() after layer4: ", thermal.size())  # (15, 20)

        # gc3
        df3 = self.densefuse3(rgb4, thermal4)
        in3 = self.in3(df3)
        sum3 = in3 + self.tp5(rgb4)
        gc3 = self.gc3(sum3)

        # gc2
        df2 = self.densefuse2(rgb3, thermal3)
        in2 = self.in2(df2)
        sum2 = in2 + self.tp4(rgb3)
        gc2 = self.gc2(sum2)

        # gc1
        df1 = self.densefuse1(rgb2, thermal2)
        # print(df1.size())
        mid = self.upmid(df1)
        in1 = self.in1(df1)
        sum1 = in1 + self.tp3(rgb2)
        gc1 = self.gc1(sum1)

        ss1 = self.lfe1(r, thermal)

        ss2 = self.lfe2(r1, thermal1)

        ss2 = ss2 + mid

        #decoder

        de5 = self.decoder5(gc3)
        gc2 = gc2 + de5
        de4 = self.decoder4(gc2)
        gc1 = gc1 + de4
        de3 = self.decoder3(gc1)
        out1 = self.decoder2(de3)
        # out1 = self.upsample(de3)

        semantic_out = self.classfier1(out1)
        attmap = self.decoder2_withoutup(de3)
        out2 = torch.mul(ss2, attmap)

        out2 = self.upsample(out2)

        out2 = self.md(out2)

        binary_out = self.classfier2(out2)

        out3 = torch.mul(ss1, out1)

        out3 = self.fd(out3)

        boundary_out = self.classfier3(out3)

        return semantic_out, binary_out, boundary_out

#########################################################################################################    DenseFuseLayer


class DenseLayer(nn.Module):
    def __init__(self, in_C, out_C, down_factor=4, k=4):
        """
        更像是DenseNet的Block，从而构造特征内的密集连接
        """
        super(DenseLayer, self).__init__()
        self.k = k
        self.down_factor = down_factor
        mid_C = out_C // self.down_factor

        self.down = nn.Conv2d(in_C, mid_C, 1)

        self.denseblock = nn.ModuleList()
        for i in range(1, self.k + 1):
            self.denseblock.append(BBasicConv2d(mid_C * i, mid_C, 3, 1, 1))

        self.fuse = BBasicConv2d(in_C + mid_C, out_C, kernel_size=3, stride=1, padding=1)

    def forward(self, in_feat):
        down_feats = self.down(in_feat)
        # print(down_feats.shape)
        # print(self.denseblock)
        out_feats = []
        for i in self.denseblock:
            # print(self.denseblock)
            feats = i(torch.cat((*out_feats, down_feats), dim=1))
            # print(feats.shape)
            out_feats.append(feats)

        feats = torch.cat((in_feat, feats), dim=1)
        return self.fuse(feats)

class CMC(nn.Module):
    def __init__(self, in_C, out_C, down_factor=4, k=3):
        """
        更像是DenseNet的Block，从而构造特征内的密集连接
        """
        super(CMC, self).__init__()
        self.k = k
        self.down_factor = down_factor
        mid_C = out_C // self.down_factor

        self.down = nn.Conv2d(in_C, mid_C, 1)

        self.denseblock = nn.ModuleList()
        for i in range(1, self.k + 1):
            self.denseblock.append(DilationConvB(mid_C * i, mid_C, 2*i+1, 2*i+1))

        self.fuse = BBasicConv2d(in_C + mid_C, out_C, kernel_size=3, stride=1, padding=1)

    def forward(self, in_feat):
        down_feats = self.down(in_feat)
        # print(down_feats.shape)
        # print(self.denseblock)
        out_feats = []
        for i in self.denseblock:
            # print(self.denseblock)
            feats = i(torch.cat((*out_feats, down_feats), dim=1))
            # print(feats.shape)
            out_feats.append(feats)

        feats = torch.cat((in_feat, feats), dim=1)
        return self.fuse(feats)


class DenseFuseLayer(nn.Module):
    def __init__(self, in_C, out_C):
        super(DenseFuseLayer, self).__init__()
        down_factor = in_C // out_C
        self.fuse_down_mul = BBasicConv2d(in_C, in_C, 3, 1, 1)
        self.res_main = DenseLayer(in_C, in_C, down_factor=down_factor)
        self.fuse_main = BBasicConv2d(in_C, out_C, kernel_size=3, stride=1, padding=1)

    def forward(self, rgb, depth):
        assert rgb.size() == depth.size()
        feat = self.fuse_down_mul(rgb + depth)
        return self.fuse_main(self.res_main(feat) + feat)


class BBasicConv2d(nn.Module):
    def __init__(
        self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
    ):
        super(BBasicConv2d, self).__init__()

        self.basicconv = nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.basicconv(x)

#########################################################################################################    Inception


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU6(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Inception(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, visual=1):
        super(Inception, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch0 = nn.Sequential(
                BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=stride),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual, dilation=visual, relu=False),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=1, stride=stride),

        )
        self.branch1 = nn.Sequential(
                BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=1),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=(3, 3), stride=stride, padding=visual),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=1, stride=1),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual+1, dilation=visual+1, relu=False)
                )
        self.branch2 = nn.Sequential(
                BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=1),
                BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=visual),
                BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=1, stride=1),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=2*visual+1, dilation=2*visual+1, relu=False)
                )

        self.ConvLinear = BasicConv(6*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=True)
        self.droupout = nn.Dropout2d(0.3)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        out = self.droupout(out)
        short = self.shortcut(x)
        out = out * short + short
        out = self.relu(out)

        return out

#########################################################################################################    ssnbt


class SSnbt(nn.Module):
    def __init__(self, in_channels, dilation=1, norm_layer=nn.BatchNorm2d, **kwargs):
        super(SSnbt, self).__init__()
        inter_channels = in_channels // 2
        self.branch1 = nn.Sequential(
            #branch1 非对称卷积
            nn.Conv2d(inter_channels, inter_channels, (3, 1), padding=(1, 0), bias=False),
            nn.ReLU(True),
            nn.Conv2d(inter_channels, inter_channels, (1, 3), padding=(0, 1), bias=False),
            norm_layer(inter_channels),
            nn.ReLU(True),
            nn.Conv2d(inter_channels, inter_channels, (3, 1), padding=(dilation, 0), dilation=(dilation, 1),
                      bias=False),
            nn.ReLU(True),
            nn.Conv2d(inter_channels, inter_channels, (1, 3), padding=(0, dilation), dilation=(1, dilation),
                      bias=False),
            norm_layer(inter_channels),
            nn.ReLU(True))

        self.branch2 = nn.Sequential(
            #branch2 非对称卷积
            nn.Conv2d(inter_channels, inter_channels, (1, 3), padding=(0, 1), bias=False),
            nn.ReLU(True),
            nn.Conv2d(inter_channels, inter_channels, (3, 1), padding=(1, 0), bias=False),
            norm_layer(inter_channels),
            nn.ReLU(True),
            nn.Conv2d(inter_channels, inter_channels, (1, 3), padding=(0, dilation), dilation=(1, dilation),
                      bias=False),
            nn.ReLU(True),
            nn.Conv2d(inter_channels, inter_channels, (3, 1), padding=(dilation, 0), dilation=(dilation, 1),
                      bias=False),
            norm_layer(inter_channels),
            nn.ReLU(True))

        self.relu = nn.ReLU(True)

    @staticmethod
    #channel shuffle(通道随机混合)可以看成“重塑-转置-重塑"
    #
    def channel_shuffle(x, groups):
        n, c, h, w = x.size()
        #分组
        channels_per_group = c // groups
        # reshape
        x = x.view(n, groups, channels_per_group, h, w)
        #转置
        x = torch.transpose(x, 1, 2).contiguous()
        #flatten为tensor
        x = x.view(n, -1, h, w)

        return x
    #SS-nbt(split-shuffle-non-bottleneck)
    def forward(self, x):
        #处理多通道图像时，有时需要对各个通道进行分离，分别处理
        # channels split
        x1, x2 = x.split(x.size(1) // 2, 1)

        x1 = self.branch1(x1)
        x2 = self.branch2(x2)

        out = torch.cat([x1, x2], dim=1)
        #残差
        out = self.relu(out + x)
        #channel shuffle
        out = self.channel_shuffle(out, groups=2)

        return out


class Fuseblock(nn.Module):
    def __init__(self, in_channels):
        super(Fuseblock, self).__init__()

        self.dense = DenseFuseLayer(in_channels, 64)

    def forward(self, rgb, depth):
        #处理多通道图像时，有时需要对各个通道进行分离，分别处理

        x1 = rgb + depth
        x2 = torch.mul(rgb, depth)

        # out = torch.cat([x1, x2], dim=1)
        # #残差
        # out = self.relu(out + rgb + depth)

        out = self.dense(x1, x2)
        #channel shuffle
        # out = self.channel_shuffle(out, groups=2)


        return out


class GCM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GCM, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class DilationConvB(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, d):
        super(DilationConvB, self).__init__()
        self.cov = nn.Sequential(
            BasicConv2d(in_channel, out_channel, kernel_size=(1, kernel), padding=(0, (kernel-1)//2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(kernel, 1), padding=((kernel-1)//2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=d, dilation=d)
        )
    def forward(self, x):
        return self.cov(x)




#########################################################################################################     decoder


class decoder(nn.Module):
    def __init__(self, channel=64):
        super(decoder, self).__init__()
        self.block1 = nn.Sequential(
            BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.block3 = nn.Sequential(
            BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        out = x3 + x
        out = self.up2(out)
        return out


class decoder_without(nn.Module):
    def __init__(self, channel=64):
        super(decoder_without, self).__init__()
        self.block1 = nn.Sequential(
            BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.block3 = nn.Sequential(
            BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        out = x3 + x

        return out

class Classifier(nn.Module):
    def __init__(self, feature, n_classes):
        super(Classifier, self).__init__()
        self.binary_conv1 = ConvBNReLU(feature, feature // 4, kernel_size=1)
        self.binary_conv2 = nn.Conv2d(feature // 4, 2, kernel_size=3, padding=1)

        self.semantic_conv1 = ConvBNReLU(feature, feature, kernel_size=1)
        self.semantic_conv2 = nn.Conv2d(feature, n_classes, kernel_size=3, padding=1)

        self.boundary_conv1 = ConvBNReLU(feature * 2, feature, kernel_size=1)
        self.boundary_conv2 = nn.Conv2d(feature, 2, kernel_size=3, padding=1)

        self.boundary_conv = nn.Sequential(
            nn.Conv2d(feature * 2, feature, kernel_size=1),
            nn.BatchNorm2d(feature),
            nn.ReLU6(inplace=True),
            nn.Conv2d(feature, 2, kernel_size=3, padding=1),
        )

        # self.up4x = nn.UpsamplingBilinear2d(scale_factor=4)
        self.up2x = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, feat):
        binary = self.binary_conv2(self.binary_conv1(feat))
        binary_out = self.up2x(binary)

        weight = torch.exp(binary)
        weight = weight[:, 1:2, :, :] / torch.sum(weight, dim=1, keepdim=True)

        feat_sematic = self.up2x(feat * weight)
        feat_sematic = self.semantic_conv1(feat_sematic)

        semantic_out = self.semantic_conv2(feat_sematic)

        feat_boundary = torch.cat([feat_sematic, self.up2x(feat)], dim=1)
        boundary_out = self.boundary_conv(feat_boundary)

        # print(binary_out.shape)
        # print(semantic_out.shape)
        # print(boundary_out.shape)

        return semantic_out, binary_out, boundary_out

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, dilation=1, groups=1, bn=True, relu=True):
        padding = ((kernel_size - 1) * dilation + 1) // 2
        # padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups,
                              bias=False if bn else True)
        self.bn = bn
        if bn:
            self.bnop = nn.BatchNorm2d(out_planes)
        self.relu = relu
        if relu:
            self.reluop = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bnop(x)
        if self.relu:
            x = self.reluop(x)
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class separable_conv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=2, padding=1, dilation=0, bias=False, norm_layer=nn.BatchNorm2d):
        super(separable_conv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation, groups=inplanes, bias=bias)
        self.bn = norm_layer(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class separable_deconv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=2, stride=2, padding=0, dilation=0, bias=False, norm_layer=nn.BatchNorm2d):
        super(separable_deconv2d, self).__init__()

        self.conv1 = nn.ConvTranspose2d(inplanes, inplanes, kernel_size, stride, padding, dilation, groups=inplanes, bias=bias)
        self.bn = norm_layer(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation, groups=inplanes, bias=bias)
        self.bn = norm_layer(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        att = self.sigmoid(out)
        out = torch.mul(x, att)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


class LFE(nn.Module):
    def __init__(self, in_dim, kernel_size=7):
        super(LFE, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.tp = nn.Conv2d(in_dim, 64, kernel_size=1)
        self.ca = ChannelAttention(64)

    def forward(self, x1, x2):
        max_out, _ = torch.max(x2, dim=1, keepdim=True)
        x2 = max_out
        x2 = self.conv1(x2)
        att2 = self.sigmoid(x2+x1)
        out = torch.mul(x1, att2) + x2
        tp = self.tp(out)
        fuseout = self.ca(tp)

        return fuseout


def unit_test():
    # from FLOP import CalParams2
    rgb = torch.randn((1, 3, 480, 640))
    depth = torch.randn((1, 1, 480, 640))
    net = GMNet(n_classes=9)
    out1, out2, out3 = net(rgb, depth)

    # print(out1.size())
    # print(out2.size())
    # print(out3.size())


if __name__ == '__main__':

    unit_test()