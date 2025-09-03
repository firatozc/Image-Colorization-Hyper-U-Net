import math
import numpy as np
from PIL import Image
import streamlit as st
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def rgb2lab2(r0, g0, b0):
    r = r0 / 255
    g = g0 / 255
    b = b0 / 255

    y = 0.299 * r + 0.587 * g + 0.114 * b
    x = 0.449 * r + 0.353 * g + 0.198 * b
    z = 0.012 * r + 0.089 * g + 0.899 * b

    l = y
    a = (x - y) / 0.234
    b = (y - z) / 0.785

    return l, a, b


def lab22rgb(l, a, b):
    a11 = 0.299
    a12 = 0.587
    a13 = 0.114
    a21 = (0.15 / 0.234)
    a22 = (-0.234 / 0.234)
    a23 = (0.084 / 0.234)
    a31 = (0.287 / 0.785)
    a32 = (0.498 / 0.785)
    a33 = (-0.785 / 0.785)

    aa = np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])
    c0 = np.zeros((l.shape[0], 3))
    c0[:, 0] = l[:, 0]
    c0[:, 1] = a[:, 0]
    c0[:, 2] = b[:, 0]
    c = np.transpose(c0)

    x = np.linalg.inv(aa).dot(c)
    x1_d = np.reshape(x, (x.shape[0] * x.shape[1], 1))
    p0 = np.where(x1_d < 0)
    x1_d[p0[0]] = 0
    p1 = np.where(x1_d > 1)
    x1_d[p1[0]] = 1
    xr = np.reshape(x1_d, (x.shape[0], x.shape[1]))

    Rr = xr[0][:]
    Gr = xr[1][:]
    Br = xr[2][:]

    R = np.uint8(np.round(Rr * 255))
    G = np.uint8(np.round(Gr * 255))
    B = np.uint8(np.round(Br * 255))
    return R, G, B


def psnr(img1, img2):
    mse = np.mean((img1.astype("float") - img2.astype("float")) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def mse(imageA, imageB, bands):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1] * bands)
    return err


def mae(imageA, imageB, bands):
    err = np.sum(np.abs((imageA.astype("float") - imageB.astype("float"))))
    err /= float(imageA.shape[0] * imageA.shape[1] * bands)
    return err


def rmse(imageA, imageB, bands):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1] * bands)
    err = np.sqrt(err)
    return err


class DoubleConv(nn.Module):
    """Double Convolution Block with optional Dropout"""

    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(DoubleConv, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(p=dropout))  # 2D veriler için
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class TripleConv(nn.Module):
    """Triple Convolution Block with optional Dropout"""

    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(TripleConv, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(p=dropout))
        self.triple_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.triple_conv(x)


class UNet1(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super(UNet1, self).__init__()

        # Encoder
        self.conv1 = DoubleConv(in_channels, 64, dropout=0.0)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = DoubleConv(64, 128, dropout=0.0)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = TripleConv(128, 256, dropout=0.1)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = TripleConv(256, 512, dropout=0.2)
        self.pool4 = nn.MaxPool2d(2)

        self.conv5 = TripleConv(512, 512, dropout=0.3)
        self.pool5 = nn.MaxPool2d(2)

        # Bottleneck
        self.conv55 = TripleConv(512, 512, dropout=0.3)

        # Decoder (daha düşük dropout koyabilirsin)
        self.up66 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.conv66 = DoubleConv(1024, 512, dropout=0.2)

        self.up6 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.conv6 = DoubleConv(1024, 512, dropout=0.2)

        self.up7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv7 = DoubleConv(512, 256, dropout=0.1)

        self.up8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv8 = DoubleConv(256, 128, dropout=0.1)

        self.up9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv9 = DoubleConv(128, 64, dropout=0.0)

        # Multi-scale feature fusion
        self.up_f02 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_f12 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Final layers (dropout koymaya gerek yok genelde)
        self.conv11 = nn.Conv2d(384, 128, kernel_size=3, padding=1)
        self.relu11 = nn.ReLU(inplace=True)

        self.conv12 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.relu12 = nn.ReLU(inplace=True)

        self.conv13 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu13 = nn.ReLU(inplace=True)

        self.conv14 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Encoder
        conv1 = self.conv1(x)
        x1 = self.pool1(conv1)

        conv2 = self.conv2(x1)
        x2 = self.pool2(conv2)

        conv3 = self.conv3(x2)
        x3 = self.pool3(conv3)

        conv4 = self.conv4(x3)
        x4 = self.pool4(conv4)

        conv5 = self.conv5(x4)
        x5 = self.pool5(conv5)

        # Bottleneck
        conv55 = self.conv55(x5)

        # Decoder
        up66 = self.up66(conv55)
        if up66.size()[2:] != conv5.size()[2:]:
            up66 = F.interpolate(up66, size=conv5.size()[2:], mode="bilinear", align_corners=True)
        merge66 = torch.cat([conv5, up66], dim=1)
        conv66 = self.conv66(merge66)

        up6 = self.up6(conv66)
        if up6.size()[2:] != conv4.size()[2:]:
            up6 = F.interpolate(up6, size=conv4.size()[2:], mode="bilinear", align_corners=True)
        merge6 = torch.cat([conv4, up6], dim=1)
        conv6 = self.conv6(merge6)

        up7 = self.up7(conv6)
        if up7.size()[2:] != conv3.size()[2:]:
            up7 = F.interpolate(up7, size=conv3.size()[2:], mode="bilinear", align_corners=True)
        merge7 = torch.cat([conv3, up7], dim=1)
        conv7 = self.conv7(merge7)

        up8 = self.up8(conv7)
        if up8.size()[2:] != conv2.size()[2:]:
            up8 = F.interpolate(up8, size=conv2.size()[2:], mode="bilinear", align_corners=True)
        merge8 = torch.cat([conv2, up8], dim=1)
        conv8 = self.conv8(merge8)

        up9 = self.up9(conv8)
        if up9.size()[2:] != conv1.size()[2:]:
            up9 = F.interpolate(up9, size=conv1.size()[2:], mode="bilinear", align_corners=True)
        merge9 = torch.cat([conv1, up9], dim=1)
        conv9 = self.conv9(merge9)

        # Multi-scale feature fusion
        up_f01 = conv1  # Original resolution
        up_f11 = conv9  # Decoded features
        up_f02 = self.up_f02(conv2)  # Upsampled encoder features
        up_f12 = self.up_f12(conv8)  # Upsampled decoder features

        # Concatenate multi-scale features
        merge11 = torch.cat([up_f01, up_f11, up_f02, up_f12], dim=1)

        # Final processing
        conv11 = self.relu11(self.conv11(merge11))
        conv12 = self.relu12(self.conv12(conv11))
        conv13 = self.relu13(self.conv13(conv12))
        output = self.tanh(self.conv14(conv13))

        return output


def load_vgg16_weights(model):
    """Load pretrained VGG16 weights to U-Net encoder"""
    vgg16 = models.vgg16(pretrained=True).to(device)
    vgg_features = vgg16.features

    # Adapt first layer from RGB to grayscale
    with torch.no_grad():
        # Get original RGB weights
        rgb_weights = vgg_features[0].weight  # Shape: (64, 3, 3, 3)
        # Average across RGB channels
        gray_weights = rgb_weights.mean(dim=1, keepdim=True)  # Shape: (64, 1, 3, 3)

        # Set weights for first layer
        model.conv1.double_conv[0].weight.data = gray_weights
        model.conv1.double_conv[0].bias.data = vgg_features[0].bias.data

        # Set weights for second conv in first block
        model.conv1.double_conv[3].weight.data = vgg_features[2].weight.data
        model.conv1.double_conv[3].bias.data = vgg_features[2].bias.data

        # Second block
        model.conv2.double_conv[0].weight.data = vgg_features[5].weight.data
        model.conv2.double_conv[0].bias.data = vgg_features[5].bias.data
        model.conv2.double_conv[3].weight.data = vgg_features[7].weight.data
        model.conv2.double_conv[3].bias.data = vgg_features[7].bias.data

        # Third block (first two convs)
        model.conv3.triple_conv[0].weight.data = vgg_features[10].weight.data
        model.conv3.triple_conv[0].bias.data = vgg_features[10].bias.data
        model.conv3.triple_conv[3].weight.data = vgg_features[12].weight.data
        model.conv3.triple_conv[3].bias.data = vgg_features[12].bias.data
        model.conv3.triple_conv[6].weight.data = vgg_features[14].weight.data
        model.conv3.triple_conv[6].bias.data = vgg_features[14].bias.data

        # Fourth block
        model.conv4.triple_conv[0].weight.data = vgg_features[17].weight.data
        model.conv4.triple_conv[0].bias.data = vgg_features[17].bias.data
        model.conv4.triple_conv[3].weight.data = vgg_features[19].weight.data
        model.conv4.triple_conv[3].bias.data = vgg_features[19].bias.data
        model.conv4.triple_conv[6].weight.data = vgg_features[21].weight.data
        model.conv4.triple_conv[6].bias.data = vgg_features[21].bias.data

        # Fifth block
        model.conv5.triple_conv[0].weight.data = vgg_features[24].weight.data
        model.conv5.triple_conv[0].bias.data = vgg_features[24].bias.data
        model.conv5.triple_conv[3].weight.data = vgg_features[26].weight.data
        model.conv5.triple_conv[3].bias.data = vgg_features[26].bias.data
        model.conv5.triple_conv[6].weight.data = vgg_features[28].weight.data
        model.conv5.triple_conv[6].bias.data = vgg_features[28].bias.data


def load_model_for_inference(model_path, device):
    model = UNet1(in_channels=1, out_channels=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def inference(model, l_channel):
    model.eval()
    with torch.no_grad():
        if len(l_channel.shape) == 3:
            l_channel = l_channel.unsqueeze(0)  # Add batch dimension

        l_tensor = torch.FloatTensor(l_channel).to(device)
        ab_pred = model(l_tensor)

        return ab_pred.cpu().numpy()


def prepare_test_image(img, dim=150):
    if isinstance(img, Image.Image):
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    img = cv2.resize(img, (dim, dim))

    sz0, sz1 = img.shape[:2]
    R1 = img[:, :, 2].reshape(-1, 1)
    G1 = img[:, :, 1].reshape(-1, 1)
    B1 = img[:, :, 0].reshape(-1, 1)

    L, A, B = rgb2lab2(R1, G1, B1)  # LAB2'ye çevir
    L = L.reshape(sz0, sz1, 1)

    L_tensor = torch.FloatTensor(L).permute(2, 0, 1)

    return L_tensor, A.reshape(sz0, sz1), B.reshape(sz0, sz1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = "Hyper_U_NET_pytorch-Huber_Loss-30Epoch_EarlyStopping_15Epoch.pth"

test_model = load_model_for_inference(model_path, device)

st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Image Colorization Demo</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: gray;'>Grayscale bir görüntü yükleyin, model sizin için renklendirsin.</p>",
    unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .css-18e3th9 {padding-top: 2rem;}
    div.stButton > button:first-child {

        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 16px;
        border: none;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background-color: #45a049;
        color: white;
    }
    div.stButton > button:active {
        background-color: #3e8e41 !important;
        color: white !important;
    }
    div.stButton > button:focus {
        box-shadow: none !important;
        outline: none !important;
        color: white !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)

with st.container():
    st.markdown("#### 📂 Grayscale Görüntü Yükle")
    uploaded_file = st.file_uploader("Yüklemek için sürükleyip bırakın", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

    l_tensor, A_true, B_true = prepare_test_image(img, dim=150)

    ab_pred = inference(test_model, l_tensor)
    ab_pred = ab_pred.squeeze(0)
    A_pred, B_pred = ab_pred[0], ab_pred[1]

    sz0, sz1 = A_pred.shape
    L = l_tensor.squeeze().numpy().reshape(-1, 1)
    A = A_pred.reshape(-1, 1)
    B = B_pred.reshape(-1, 1)

    R, G, B = lab22rgb(L, A, B)
    R = R.reshape(sz0, sz1)
    G = G.reshape(sz0, sz1)
    B = B.reshape(sz0, sz1)

    rgb_pred = cv2.merge([B, G, R])

    new_image = cv2.cvtColor(rgb_pred, cv2.COLOR_BGR2RGB)

    new_image2 = cv2.resize(new_image, (img.width, img.height), interpolation=cv2.INTER_LANCZOS4)

    if st.button("🎨 Renklendir"):
        with st.spinner("Model çalışıyor, lütfen bekleyin..."):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Girdi (Grayscale)**")
                st.image(img)

            with col2:
                st.markdown("**Model Çıkışı (Renkli)**")
                st.image(np.array(new_image2))

            st.success("Tamamlandı!")
