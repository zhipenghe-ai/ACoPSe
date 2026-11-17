"""Shared-code generator and modality-specific hash networks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, 1024)
        self.layer2 = nn.Linear(1024, 512)
        self.output_layer = nn.Linear(512, output_dim)

        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        output = self.output_layer(x)
        return output


class SelfAttention(nn.Module):
    def __init__(self, dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def forward(self, X):
        Q = self.query(X)
        K = self.key(X)
        V = self.value(X)

        attention_weights = F.softmax(Q @ K.transpose(-2, -1) / (K.shape[-1] ** 0.5), dim=-1)
        attention_output = attention_weights @ V

        return attention_output


class FuseTransEncoder(nn.Module):
    def __init__(self, num_layers, hidden_size, nhead):
        super(FuseTransEncoder, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, batch_first=False)
        self.transformerEncoder = TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        self.d_model = hidden_size
        self.sigal_d = int(self.d_model / 2)

    def forward(self, tokens):
        encoder_X = self.transformerEncoder(tokens)
        encoder_X_r = encoder_X.reshape(-1, self.d_model)
        return encoder_X_r


class MyNet(nn.Module):
    def __init__(self, code_len, ori_featI, ori_featT):
        super(MyNet, self).__init__()
        self.code_len = code_len

        self.attention = FuseTransEncoder(num_layers=2, hidden_size=ori_featI+ori_featT,nhead=2)

        self.encoder_fusion = MLP(input_dim=ori_featI+ori_featT,output_dim=512)
        self.encoderIMG = nn.Sequential(
            nn.Linear(ori_featI, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )

        self.gcnI1 = nn.Linear(512, 512)
        self.BNI1 = nn.BatchNorm1d(512)
        self.actI1 = nn.ReLU(inplace=True)

        self.decoderIMG = nn.Sequential(
            nn.Linear(code_len, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Linear(512, ori_featI),
            nn.BatchNorm1d(ori_featI)
            )

        self.encoderTXT = nn.Sequential(
            nn.Linear(ori_featT, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )

        self.gcnT1 = nn.Linear(512, 512)
        self.BNT1 = nn.BatchNorm1d(512)
        self.actT1 = nn.ReLU(inplace=True)

        self.decoderTXT = nn.Sequential(
            nn.Linear(code_len, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Linear(512, ori_featT),
            nn.BatchNorm1d(ori_featT)
            )

        self.gcnJ1 = nn.Linear(512, 512)
        self.BNJ1 = nn.BatchNorm1d(512)
        self.actJ1 = nn.ReLU(inplace=True)

        self.HJ = nn.Linear(512, code_len)
        self.img_fc = nn.Linear(512, code_len)
        self.txt_fc = nn.Linear(512, code_len)

        self.fuison_fc = nn.Linear(512, code_len)

        self.HC = nn.Linear(3 * code_len, code_len)

        self.HIBN = nn.BatchNorm1d(code_len)
        self.HTBN = nn.BatchNorm1d(code_len)
        self.HJBN = nn.BatchNorm1d(code_len)
        self.HBN = nn.BatchNorm1d(code_len)

    def forward(self, XI, XT, affinity_A):

        self.batch_num = XI.size(0)
        fusion = torch.cat((XI, XT), dim=1)
        fusion = self.attention(fusion)
        fusion = F.normalize(fusion, dim=1)
        fusion = self.encoder_fusion(fusion)
        VI = self.encoderIMG(XI)
        VI = F.normalize(VI, dim=1)

        VgcnI = self.gcnI1(VI)
        VgcnI = self.BNI1(VgcnI)
        VgcnI = self.actI1(VgcnI)

        VT = self.encoderTXT(XT)
        VT = F.normalize(VT, dim=1)

        VgcnT = self.gcnT1(VT)
        VgcnT = self.BNT1(VgcnT)
        VgcnT = self.actT1(VgcnT)


        HI = self.HIBN(self.img_fc(VgcnI))
        HT = self.HTBN(self.txt_fc(VgcnT))

        HJ = self.HJBN(self.fuison_fc(fusion))


        H = torch.tanh(self.HBN(self.HC(torch.cat((HI, HT,HJ), 1))))

        B = torch.sign(H)

        DeI_feat = self.decoderIMG(H + torch.tanh(HI))
        DeT_feat = self.decoderTXT(H + torch.tanh(HT))

        return HI, HT, H, B, DeI_feat, DeT_feat,VI,VT,VgcnI,VgcnT,fusion,HJ


class Img_Net(nn.Module):
    def __init__(self, code_len, img_feat_len):
        super(Img_Net, self).__init__()

        self.fc1 = nn.Linear(img_feat_len, 4096)
        self.relu = nn.ReLU(inplace=True)
        self.dp = nn.Dropout(0.3)
        self.tohash = nn.Linear(4096, code_len)
        self.tanh = nn.Tanh()

        torch.nn.init.normal_(self.tohash.weight, mean=0.0, std=1)

    def forward(self, x):
        feat = self.relu(self.fc1(x))
        hid = self.tohash(self.dp(feat))
        HI = self.tanh(hid)

        return HI

class Txt_Net(nn.Module):
    def __init__(self, code_len, txt_feat_len):
        super(Txt_Net, self).__init__()

        self.fc1 = nn.Linear(txt_feat_len, txt_feat_len)
        self.relu = nn.ReLU(inplace=True)
        self.dp = nn.Dropout(0.3)
        self.tohash = nn.Linear(txt_feat_len, code_len)
        self.tanh = nn.Tanh()

        torch.nn.init.normal_(self.tohash.weight, mean=0.0, std=1)

    def forward(self, x):
        feat = self.relu(self.fc1(x))
        hid = self.tohash(self.dp(feat))
        HT = self.tanh(hid)

        return HT