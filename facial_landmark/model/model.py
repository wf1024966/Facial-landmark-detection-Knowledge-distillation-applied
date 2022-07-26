import torch 
import torch.nn as nn
from torchsummary import summary

torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
mobilenet = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)

class FaceNet(nn.Module):
    def __init__(self, num_keypoints, pretrained=False):
        super(FaceNet, self).__init__()
        self.backbone = mobilenet
        self.outLayer1 = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Dropout(0.1))
        self.outLayer2 = nn.Linear(512, num_keypoints*2)
    def forward(self, inputs):
        out = self.backbone(inputs)
        out = self.outLayer1(out)
        out = self.outLayer2(out)
        return out

