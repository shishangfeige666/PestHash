import torch.nn.functional as F
import torchvision.models as models
import torch
import torch.nn.modules as nn
import pdb



#多尺度特征+通道注意力+空间注意力
class MFE_CCRO(nn.Module):
    def __init__(self, dim=2048, kernel_sizes=(3, 5, 7)):
        super().__init__()
        self.dim = dim
        self.kernel_sizes = kernel_sizes

        # 多尺度关键特征提取 (key)
        self.key_embed = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=ks, padding=ks // 2, groups=8, bias=False),
                nn.BatchNorm2d(dim),
                nn.GELU()
            )
            for ks in kernel_sizes
        ])

        # 多尺度值特征提取 (value)
        self.value_embed = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=ks, padding=ks // 2, bias=False),
                nn.BatchNorm2d(dim),
                nn.GELU()
            )
            for ks in kernel_sizes
        ])

        # 注意力模块
        factor = 4
        self.attention_embed = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim // factor, kernel_size=1, bias=False),
            nn.BatchNorm2d(2 * dim // factor),
            nn.GELU(),
            nn.Conv2d(2 * dim // factor, sum(ks**2 for ks in kernel_sizes) * dim, kernel_size=1)
        )

        # 添加通道注意力机制
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // factor, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim // factor, dim, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # 通道正则优化
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

#同时具有通道注意力和空间注意力
    def forward(self, x):
        bs, c, h, w = x.shape

        # 多尺度关键特征提取
        k1_list = [key_layer(x) for key_layer in self.key_embed]
        k1 = sum(k1_list)  # [B, C, H, W]

        # 多尺度值特征提取
        v_list = [value_layer(x) for value_layer in self.value_embed]
        v = sum(v_list).view(bs, c, -1)  # [B, C, H*W]

        # 拼接特征生成注意力
        y = torch.cat([k1, x], dim=1)  # [B, 2C, H, W]
        att = self.attention_embed(y)  # [B, C*K*K, H, W]
        total_kernel_size = sum(ks**2 for ks in self.kernel_sizes)
        att = att.view(bs, c, total_kernel_size, h, w)
        att = att.mean(2).view(bs, c, -1)  # [B, C, H*W]

        # 计算注意力加权特征
        att = F.softmax(att, dim=-1) * v  # [B, C, H*W]
        k2 = att.view(bs, c, h, w)  # [B, C, H, W]

        # 通道注意力
        channel_att = self.channel_attention(k2)  # [B, C, 1, 1]
        k2 = k2 * channel_att

        # 通道正则优化
        avg_out = torch.mean(k2, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(k2, dim=1, keepdim=True)  # [B, 1, H, W]
        spatial_att = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))  # [B, 1, H, W]
        k2 = k2 * spatial_att

        # 合并关键特征与注意力增强特征
        out = k1 + k2  # [B, C, H, W]
        return out


class ResNet(nn.Module):
    def __init__(self, config, hash_bit, label_size, pretrained=True):
        super(ResNet, self).__init__()
        model_resnet = models.resnet50(pretrained=pretrained)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool

        self.cota = MFE_CCRO()

        self.feature_layer = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                                           self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)

        self.tanh = nn.Tanh()
        self.label_linear = nn.Linear(label_size, hash_bit)
        if config['without_BN']:
            self.hash_layer = nn.Linear(model_resnet.fc.in_features, hash_bit)
            self.hash_layer.weight.data.normal_(0, 0.01)
            self.hash_layer.bias.data.fill_(0.0)
        else:
            self.layer_hash = nn.Linear(model_resnet.fc.in_features, hash_bit)
            self.layer_hash.weight.data.normal_(0, 0.01)
            self.layer_hash.bias.data.fill_(0.0)
            self.hash_layer = nn.Sequential(self.layer_hash, nn.BatchNorm1d(hash_bit, momentum=0.1))

    def forward(self, x, T, label_vectors):
        feat = self.feature_layer(x)
        feat = self.cota(feat)
        feat = feat.view(feat.shape[0], -1)
        x = self.hash_layer(feat)
        x = self.tanh(x)

        return x, feat

class MoCo(nn.Module):
    def __init__(self, config, hash_bit, label_size, pretrained=True):
        super(MoCo, self).__init__()
        self.m = config['mome']
        self.encoder_q = ResNet(config, hash_bit, label_size, pretrained)
        self.encoder_k = ResNet(config, hash_bit, label_size, pretrained)
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def forward(self, x, T, label_vectors):
        encode_x, _ = self.encoder_q(x, T, label_vectors)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            encode_x2, _ = self.encoder_k(x, T, label_vectors)
        return encode_x, encode_x2

