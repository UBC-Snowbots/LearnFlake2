"""Causal spatio-temporal graph network for attention-seeking gestures.

Input: one tracked person's COCO-17 keypoints over the most recent T frames,
[N, T, 17, 3] with image-normalized x, y and confidence. Temporal convolutions
only look backwards, so the same weights can run on a rolling window.
"""
import torch
from torch import nn

COCO_EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 6), (5, 7), (7, 9),
              (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
FLIP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
CHANNELS = 7  # x, y, confidence, dx, dy, bone dx, bone dy
PARENT = [0, 0, 0, 1, 2, 0, 0, 5, 6, 7, 8, 5, 6, 11, 12, 13, 14]


def features(window):
    """[N,T,17,3] raw keypoints → [N,C,T,17] body-centred features.

    Centre on the latest frame's hip midpoint (shoulders if hips are missing) and
    scale by the window's largest torso length, so features use only past frames.
    """
    xy, conf = window[..., :2], window[..., 2:3]
    last = window[:, -1]
    hips = (last[:, 11, :2] + last[:, 12, :2]) / 2
    shoulders = (last[:, 5, :2] + last[:, 6, :2]) / 2
    hip_ok = (last[:, 11, 2] > 0.2) & (last[:, 12, 2] > 0.2)
    centre = torch.where(hip_ok[:, None], hips, shoulders)
    torso = ((window[..., 5, :2] + window[..., 6, :2]) / 2
             - (window[..., 11, :2] + window[..., 12, :2]) / 2).norm(dim=-1)
    scale = torso.amax(dim=1).clamp_min(1e-3)
    rel = (xy - centre[:, None, None]) / scale[:, None, None, None]
    rel = rel * (conf > 0.05)
    velocity = torch.diff(rel, dim=1, prepend=rel[:, :1])
    bone = rel - rel[:, :, PARENT]
    stacked = torch.cat([rel, conf, velocity, bone], dim=-1)
    return stacked.permute(0, 3, 1, 2).contiguous()


def adjacency():
    a = torch.eye(17)
    for i, j in COCO_EDGES:
        a[i, j] = a[j, i] = 1
    inward = a / a.sum(dim=1, keepdim=True)
    return torch.stack([torch.eye(17), inward, inward.T])


class Block(nn.Module):
    def __init__(self, cin, cout, stride, kernel=9):
        super().__init__()
        self.register_buffer("graph", adjacency())
        self.learned = nn.Parameter(torch.zeros_like(self.graph))
        self.spatial = nn.Conv2d(cin, cout * 3, 1)
        self.norm1 = nn.BatchNorm2d(cout)
        self.pad = kernel - 1
        self.temporal = nn.Conv2d(cout, cout, (kernel, 1), (stride, 1))
        self.norm2 = nn.BatchNorm2d(cout)
        self.residual = (nn.Identity() if cin == cout and stride == 1 else
                         nn.Sequential(nn.Conv2d(cin, cout, 1, (stride, 1)), nn.BatchNorm2d(cout)))
        self.act = nn.ReLU(inplace=True)
        self.cout = cout

    def forward(self, x):
        n, _, t, v = x.shape
        y = self.spatial(x).view(n, 3, self.cout, t, v)
        y = torch.einsum("nkctv,kvw->nctw", y, self.graph + self.learned)
        y = self.act(self.norm1(y))
        y = nn.functional.pad(y, (0, 0, self.pad, 0))  # causal: past frames only
        y = self.norm2(self.temporal(y))
        return self.act(y + self.residual(x))


class AttentionGCN(nn.Module):
    def __init__(self, classes, width=48, dropout=0.2):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(CHANNELS * 17)
        plan = [(CHANNELS, width, 1), (width, width, 1), (width, width * 2, 2),
                (width * 2, width * 2, 1), (width * 2, width * 4, 2), (width * 4, width * 4, 1)]
        self.blocks = nn.Sequential(*[Block(a, b, s) for a, b, s in plan])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(width * 4, classes)

    def forward(self, window):
        x = features(window)
        n, c, t, v = x.shape
        x = self.input_norm(x.permute(0, 1, 3, 2).reshape(n, c * v, t))
        x = x.view(n, c, v, t).permute(0, 1, 3, 2)
        x = self.blocks(x)
        return self.head(self.dropout(x.mean(dim=(2, 3))))
