import torch
import torch.nn as nn
import torch.nn.functional as F

#mobilevit
class MobileViTV1Encoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        if hasattr(self.encoder, "pool"):
            del self.encoder.pool
        if hasattr(self.encoder, "fc"):
            del self.encoder.fc

    def forward(self, x):
        features = []
        x = self.encoder.conv_1(x)

        x = self.encoder.layer_1(x)
        x = self.encoder.layer_2(x)
        x = self.encoder.layer_3(x)
        x = self.encoder.layer_4(x)
        features.append(x)
        x = self.encoder.layer_5(x)
        features.append(x)
        # x = self.encoder.pool(x).view(-1, x.shape[1])
        # x = self.encoder.fc(x)
        return features

#mobilenetv2
class MobileNetV2WithProjector(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        if hasattr(self.encoder, "pool"):
            del self.encoder.pool
        if hasattr(self.encoder, "fc"):
            del self.encoder.fc

    def forward(self, x):
        features = []
        out = x
        for idx, layer in enumerate(self.encoder):
            out = layer(out)
            if idx in [11, 15, 17, 18]:
                features.append(out)
        return features[0], features[1], features[2], features[3]
    
#mobilenetv3
class MobileNetV3WithProjector(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        if hasattr(self.encoder, "pool"):
            del self.encoder.pool
        if hasattr(self.encoder, "fc"):
            del self.encoder.fc

    def forward(self, x):
        features = []
        out = x
        for idx, layer in enumerate(self.encoder):
            out = layer(out)
            if idx in [10, 12, 15, 16]:
                features.append(out)
        # print("features", len(features))
        return features


#mobilevit
class MobileViTWithProjector(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        if hasattr(self.encoder, "pool"):
            del self.encoder.pool
        if hasattr(self.encoder, "fc"):
            del self.encoder.fc

    def forward(self, x):
        features = []

        x = self.conv_1(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        features.append(x)
        x = self.layer_3(x)
        features.append(x)
        x = self.layer_4(x)
        features.append(x)
        x = self.layer_5(x)
        features.append(x)
        # x = self.conv_1x1_exp(x)
        # if self.pool_type == "mean":
        #     x = x.mean(dim=(2, 3))
        # elif self.pool_type == "max":
        #     x = x.amax(dim=(2, 3))
        # else:
        #     # default mean
        #     x = x.mean(dim=(2, 3))
        # x = self.classifier.fc(x)

        return features


#mobilevit
class MobileViTv1SWithProjector(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        if hasattr(self.encoder, "pool"):
            del self.encoder.pool
        if hasattr(self.encoder, "fc"):
            del self.encoder.fc

    def forward(self, x):
        features = []
        # Stem
        x = self.encoder.stem(x)
        # Body
        x = self.encoder.stage1(x)
        x = self.encoder.stage2(x)
        features.append(x)
        x = self.encoder.stage3(x)
        features.append(x)
        x = self.encoder.stage4[0](x)
        x = self.encoder.stage4[1](x)
        features.append(x)
        x = self.encoder.stage4[2](x)
        features.append(x)
        # x = self.encoder.pool(x).view(-1, x.shape[1])
        # x = self.encoder.fc(x)
        return features

#mobilevit
class MobileViTv1XXSWithProjector(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        if hasattr(self.encoder, "pool"):
            del self.encoder.pool
        if hasattr(self.encoder, "fc"):
            del self.encoder.fc

    def forward(self, x):
        features = []
        # Stem
        x = self.encoder.stem(x)
        # Body
        x = self.encoder.stage1(x)
        x = self.encoder.stage2(x)
        x = self.encoder.stage3(x)
        features.append(x)
        x = self.encoder.stage4[0](x)
        features.append(x)
        x = self.encoder.stage4[1](x)
        features.append(x)
        x = self.encoder.stage4[2](x)
        features.append(x)
        # x = self.encoder.pool(x).view(-1, x.shape[1])
        # x = self.encoder.fc(x)
        return features
