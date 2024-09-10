import sys
from einops.einops import rearrange
sys.path.append("..")
import torch
import torch.nn as nn
from model.graph_frames import Graph
from model.Block import Hiremixer


class ICFNet(nn.Module):
    def __init__(self, args):
        super().__init__()

        if args == -1:
            layers, channel, d_hid, length = 3, 128, 1024, 27
            self.num_joints_in, self.num_joints_out = 17, 17
        else:
            layers, channel, d_hid, length  = args.layers, args.channel, args.d_hid, args.frames
            self.num_joints_in, self.num_joints_out = args.n_joints, args.out_joints

        self.graph = Graph('hm36_gt', 'spatial', pad=1)
        self.A = nn.Parameter(torch.tensor(self.graph.A, dtype=torch.float32), requires_grad=False)

        self.patch_embed = nn.Linear(2, channel)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_joints_in, channel))
        self.Hiremixer = Hiremixer(self.A, layers, channel, d_hid, length=length)
        self.fcn = nn.Linear(args.channel, 3)

    def forward(self, x):
        x = rearrange(x, 'b f j c -> (b f) j c').contiguous()  # 将前两个维度合并 (batch_size, 1, 17, 2) => (batch_size, 17, 2)
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.Hiremixer(x)
        x = self.fcn(x)
        x = x.view(x.shape[0], -1, self.num_joints_out, x.shape[2])
        return x


