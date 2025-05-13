from functools import partial
import torch
import torch.nn as nn
from timm.models.layers import DropPath
from einops import rearrange, repeat


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Gcn(nn.Module):
    def __init__(self, in_channels, out_channels, adj):
        super().__init__()
        self.adj = adj
        self.kernel_size = adj.size(0)
        self.conv = nn.Conv2d(in_channels, out_channels * self.kernel_size, kernel_size=(1, 1))

    def forward(self, x):
        x = self.conv(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        x = torch.einsum('nkctv, kvw->nctw', (x, self.adj))

        return x.contiguous()


class LRSC(nn.Module):
    def __init__(self, adj, dim, drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm_gcn1 = norm_layer(adj.size(1))
        self.norm_gcn2 = norm_layer(dim)

        self.gcn1 = Gcn(dim, dim, adj)
        self.gcn2 = Gcn(dim, dim, adj)

        self.gelu = nn.GELU()

        self.mlp1 = MLP(adj.size(1), dim, adj.size(1), drop=drop_path)
        self.mlp2 = MLP(dim, dim, dim, drop=drop_path)

    def forward(self, x):
        x = rearrange(x, f'b j c -> b c j')
        res = x
        x = self.norm_gcn1(x)

        x_gcn_1 = rearrange(x, 'b c j-> b c 1 j')  # 1024 128 17
        x_gcn_1 = self.gcn1(x_gcn_1)
        x_gcn_1 = rearrange(x_gcn_1, 'b c 1 j -> b c j')

        x = res + self.drop_path(self.mlp1(x) + x_gcn_1)  # 1024 128 17

        # 在此分开
        x = rearrange(x, f'b j c -> b c j')  # b, c, n  -> b, n , c
        x = self.norm_gcn2(x)
        x = rearrange(x, f'b j c -> b c j')

        res = x
        x_gcn_2 = rearrange(x, 'b c j-> b c 1 j')

        x_gcn_2 = self.gcn2(x_gcn_2)
        x_gcn_2 = rearrange(x_gcn_2, 'b c 1 j -> b c j')

        x = res + self.drop_path(self.mlp1(x) + x_gcn_2)

        x = rearrange(x, f'b j c -> b c j')

        return x



class LCM(nn.Module):
    def __init__(self, in_features, d_hid, out_features, in_fea, out_fea, kernel_size, stride, drop_path):
        super().__init__()
        self.mlp = MLP(in_features, d_hid, out_features)
        self.conv1 = nn.Conv1d(in_fea, out_fea, kernel_size, stride)
        self.conv2 = nn.Conv1d(in_fea, out_fea, kernel_size, stride)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()  # DropPath：将多分支结构的子路径 "随机删除“
        self.act_layer = nn.GELU()

    def forward(self, x):
        x_mlp = self.drop_path(self.act_layer(self.mlp(x)))

        t = torch.concat([x, x_mlp], -2)
        t = t.permute(0, 2, 1)
        t_conv = self.conv1(t)
        t_conv = t_conv.permute(0, 2, 1)
        
        return x


class GKPC(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, length=1):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm_attn = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, \
                              qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.attn2 = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, \
                               qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x_attn):
        x_attn = x_attn + self.drop_path(self.attn(self.norm_attn(x_attn)))
        x_attn2 = self.drop_path(self.attn2(self.norm_attn(x_attn)))
        return x_attn2


class Attention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.num_heads = num_heads
        head_dim = torch.div(dim, num_heads)
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, torch.div(C, self.num_heads, rounding_mode='floor')).permute(
            2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.drop(x)

        return x


class GG_ACM(nn.Module):
    def __init__(self, adj, dim, d_hid, drop_path, drop, norm_layer=nn.LayerNorm):
        super().__init__()
        self.index_1 = [0, 1, 4, 7, 8, 9, 10, 11, 14]
        self.index_r_Hip = [2, 3]
        self.index_l_Hip = [5, 6]
        self.index_l_Shoulder = [12, 13]
        self.index_r_Shoulder = [15, 16]

        self.mlp1 = MLP(dim, d_hid, dim, drop=drop).to(device)

        self.mlp2 = MLP(2 * dim, d_hid, dim, drop=drop).to(device)

        self.norm_layer = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()  # DropPath：将多分支结构的子路径 "随机删除“
        self.dim = dim

        self.gelu = nn.GELU()

        self.conv1 = nn.Conv1d(dim, dim, kernel_size=3, padding=4, stride=3)

        self.lrs = LRSC(adj, dim, drop_path=drop_path, norm_layer=norm_layer)


    def forward(self, x):
        x = self.norm_layer(x)  # （17， dim）
        x_pooling_r_Hip = x[:, self.index_r_Hip, :]
        x_pooling_l_Hip = x[:, self.index_l_Hip, :]
        x_pooling_l_Shoulder = x[:, self.index_l_Shoulder, :]
        x_pooling_r_Shoulder = x[:, self.index_r_Shoulder, :]

        y1 = x[:, self.index_1, :]  # (batch, 8, dim)
        # 
        for i in range(2):
            y1[:, 1, :] += x_pooling_r_Hip[:, i, :]
            y1[:, 2, :] += x_pooling_l_Hip[:, i, :]
            y1[:, 7, :] += x_pooling_l_Shoulder[:, i, :]
            y1[:, 8, :] += x_pooling_r_Shoulder[:, i, :]

        x1 = self.drop_path(self.gelu(self.mlp1(y1)))  # (9, dim)

        #
        B, N, C = x1.shape
        y2 = torch.zeros(B, 2 * N - 1, 2 * C).to(device)

        for i in range(0, 9):
            if i == 2:
                y2[:, 2 * i, :] = torch.concat([x[:, 2 * i, :], x[:, 0, :]], -1)
            elif i == 7:
                y2[:, 2 * i, :] = torch.concat([x[:, 2 * i, :], x[:, 8, :]], -1)
            else:
                y2[:, 2 * i, :] = torch.concat([x[:, 2 * i, :], x[:, 2 * i -1, :]], -1)
        for i in range(0, 8):
            if i == 3:
                y2[:, 2 * i + 1, :] = torch.concat([x[:, 2 * i + 1, :], x[:, 0, :]], -1)
            elif i == 5:
                y2[:, 2 * i + 1, :] = torch.concat([x[:, 2 * i + 1, :], x[:, 8, :]], -1)
            else:
                y2[:, 2 * i + 1, :] = torch.concat([x[:, 2 * i + 1, :], x[:, 8 + i, :]], -1)

        x2 = self.drop_path(self.gelu(self.mlp2(y2)))  # (17, dim)

        # 
        y3 = self.mlp1(torch.concat([x, x1, x2], -2))
        y3 = y3.permute(0, 2, 1)
        x3 = self.drop_path(self.gelu(self.conv1(y3)))
        x3 = x3.permute(0, 2, 1)

        x3 = self.drop_path(self.gelu(self.lrs(x3)))  # (17, dim)
        return x + x2 + x3


class RR_ACM(nn.Module):
    def __init__(self, adj, dim, d_hid, drop_path, ):
        super().__init__()
        self.layer_normal = nn.LayerNorm(dim)
        self.mlp1 = MLP(2 * dim, d_hid, dim)
        self.mlp2 = MLP(dim, d_hid, dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.gelu = nn.GELU()

        self.conv1 = nn.Conv1d(dim, dim, kernel_size=3, padding=1, stride=2)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=4, padding=1, stride=4)

        self.lrs = LRSC(adj, dim, drop_path=drop_path, norm_layer=nn.LayerNorm)

    def forward(self, x):
        x = self.layer_normal(x)

        # 扩大过程
        B, N, C = x.shape
        y1 = torch.zeros(B, 2 * N - 1, 2 * C).to(device)

        for i in range(0, 17):
            y1[:, 2 * i, :] = torch.concat([x[:, i, :], x[:, i, :]], -1)
        for i in range(0, 16):
            if i == 3 or i == 6:
                y1[:, 2 * i + 1, :] = torch.concat([x[:, 0, :], x[:, i + 1, :]], -1)
            elif i == 10 or i == 13:
                y1[:, 2 * i + 1, :] = torch.concat([x[:, 8, :], x[:, i + 1, :]], -1)
            else:
                y1[:, 2 * i + 1, :] = torch.concat([x[:, i, :], x[:, i + 1, :]], -1)  # 将相邻两行concat

        # print(y1.shape)

        x1 = self.drop_path(self.gelu(self.mlp1(y1)))  # (33, dim)

        # 
        x1_1 = x1.permute(0, 2, 1)
        y2 = self.drop_path(self.gelu(self.conv1(x1_1)))
        y2 = y2.permute(0, 2, 1)
        x2 = self.drop_path(self.gelu(self.mlp2(y2)))  # (17,dim)

        # 
        y3 = self.mlp2(torch.concat([x, x1, x2], -2))
        y3 = y3.permute(0, 2, 1)
        x3 = self.drop_path(self.gelu(self.conv2(y3)))
        x3 = x3.permute(0, 2, 1)
        x3 = self.drop_path(self.gelu(self.lrs(x3)))  # (17, dim)

        return x + x3 + x2


class Hiremixer(nn.Module):
    def __init__(self, adj, layers, channel, d_hid, length=9):
        super().__init__()
        h = 16
        qkv_bias = True
        qk_scale = None
        attn_drop_rate = 0.12
        norm_layer = partial(nn.LayerNorm, eps=1e-7)

        drop_path_rate = 0.27
        dpr = [x.item() for x in torch.linspace(0.1, drop_path_rate, layers)]

        drop_rate = 0.08  # 缺失率

        self.blocks = nn.ModuleList([
            Block(
                adj, dim=channel, d_hid=d_hid, drop_path=dpr[i], drop=drop_rate, norm_layer=norm_layer, length=length,
                num_heads=h, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop_rate)
            for i in range(layers)])

        self.Temporal_norm = norm_layer(channel)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.Temporal_norm(x)
        return x


class Block(nn.Module):
    def __init__(self, adj, dim, d_hid, drop_path, drop, norm_layer, length,
                 num_heads, qkv_bias=False, qk_scale=None, attn_drop=0.):
        super().__init__()

        self.dim = int(dim / 2)
        self.gelu = nn.GELU()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.lrs = LRSC(adj, dim, drop_path=drop_path, norm_layer=norm_layer)

        self.exp = RR_ACM(adj, self.dim, d_hid, drop_path=drop_path, )
        self.red = GG_ACM(adj, self.dim, d_hid, drop_path=drop_path, drop=drop, norm_layer=norm_layer)
        #
        self.gkp = GKPC(dim, num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=0.1, attn_drop=attn_drop,
                       drop_path=drop_path, norm_layer=norm_layer, length=length)

        self.lcm = LCM(dim, d_hid, dim, dim, dim, kernel_size=2, stride=2, drop_path=drop_path).to(device)

    def forward(self, x):
        # split
        x_1, x_2 = torch.chunk(x, 2, -1)

        # RR-ACM and GG-ACM
        x1 = self.drop_path(self.gelu(self.exp(x_1)))
        x2 = self.drop_path(self.gelu(self.red(x_1)))

        x3 = self.drop_path(self.gelu(self.exp(x_2)))
        x4 = self.drop_path(self.gelu(self.red(x_2)))

        # concat
        x_14 = torch.concat([x1, x4], -1)
        x_23 = torch.concat([x2, x3], -1)


        # LCM
        x_14_lcm = self.lcm(x_14)
        x_23_lcm = self.lcm(x_23)

        x5 = self.gkp(x + x_14_lcm * 0.3 + x_23_lcm * 0.7)
        x7 = self.lrs(x + x_14_lcm * 0.7 + x_23_lcm * 0.3)

        return x + x5 + x7
