import math
import numpy as np
from PIL import Image
import streamlit as st
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # Önceden eğitilmiş VGG19 modelinin özellik çıkarma katmanlarını alıyoruz
        vgg = models.vgg19(pretrained=True).features

        # Sadece belirli katmanlardan sonraki özellikleri kullanacağız
        # Bu katmanlar genellikle doku ve içerik için iyi sonuçlar verir
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()

        # VGG19'un katmanları üzerinde döngü
        for x in range(2):
            self.slice1.add_module(str(x), vgg[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg[x])

        # VGG'nin ağırlıklarının eğitim sırasında güncellenmemesi için gradyanları kapatıyoruz
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, X, Y):
        # VGG modeli ImageNet üzerinde eğitildiği için normalize edilmiş 3 kanallı (RGB) girdi bekler
        # Girdileri normalize edelim
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        X = normalize(X)
        Y = normalize(Y)

        # Özellik haritalarını çıkaralım
        h_X = self.slice1(X)
        h_Y = self.slice1(Y)
        loss1 = F.l1_loss(h_X, h_Y)

        h_X = self.slice2(h_X)
        h_Y = self.slice2(h_Y)
        loss2 = F.l1_loss(h_X, h_Y)

        h_X = self.slice3(h_X)
        h_Y = self.slice3(h_Y)
        loss3 = F.l1_loss(h_X, h_Y)

        h_X = self.slice4(h_X)
        h_Y = self.slice4(h_Y)
        loss4 = F.l1_loss(h_X, h_Y)

        h_X = self.slice5(h_X)
        h_Y = self.slice5(h_Y)
        loss5 = F.l1_loss(h_X, h_Y)

        # Tüm katmanlardan gelen kayıpları toplayalım
        return loss1 + loss2 + loss3 + loss4 + loss5


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


def lab22rgb(l, a, b, device):
    a11, a12, a13 = 0.299, 0.587, 0.114
    a21, a22, a23 = (0.15 / 0.234), (-0.234 / 0.234), (0.084 / 0.234)
    a31, a32, a33 = (0.287 / 0.785), (0.498 / 0.785), (-0.785 / 0.785)

    aa_np = np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])
    aa_inv_tensor = torch.tensor(np.linalg.inv(aa_np), dtype=torch.float32).to(device)

    lab = torch.cat((l, a, b), dim=1)

    batch_size, _, h, w = lab.shape

    lab_reshaped = lab.view(batch_size, 3, -1)

    lab_permuted = lab_reshaped.permute(0, 2, 1)

    rgb_permuted = torch.matmul(lab_permuted, aa_inv_tensor.t())  # Matris çarpımı

    rgb_reshaped = rgb_permuted.permute(0, 2, 1)
    rgb_image = rgb_reshaped.view(batch_size, 3, h, w)

    rgb_image = torch.clamp(rgb_image, 0.0, 1.0) # Değerleri [0, 1] aralığına kırp

    return rgb_image


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
    """Double Convolution Block"""

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class TripleConv(nn.Module):
    """Triple Convolution Block"""

    def __init__(self, in_channels, out_channels):
        super(TripleConv, self).__init__()
        self.triple_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.triple_conv(x)


class UNet1(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super(UNet1, self).__init__()

        # Encoder
        self.conv1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = TripleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = TripleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.conv5 = TripleConv(512, 512)
        self.pool5 = nn.MaxPool2d(2)

        # Bottleneck
        self.conv55 = TripleConv(512, 512)

        # Decoder
        self.up66 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.conv66 = DoubleConv(1024, 512)  # 512 + 512 from skip connection

        self.up6 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.conv6 = DoubleConv(1024, 512)  # 512 + 512 from skip connection

        self.up7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv7 = DoubleConv(512, 256)  # 256 + 256 from skip connection

        self.up8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv8 = DoubleConv(256, 128)  # 128 + 128 from skip connection

        self.up9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv9 = DoubleConv(128, 64)  # 64 + 64 from skip connection

        # Multi-scale feature fusion
        self.up_f02 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_f12 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Final layers
        self.conv11 = nn.Conv2d(384, 128, kernel_size=3, padding=1)  # 64+64+128+128
        self.relu11 = nn.ReLU(inplace=True)

        self.conv12 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.relu12 = nn.ReLU(inplace=True)

        self.conv13 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu13 = nn.ReLU(inplace=True)

        self.conv14 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        self.tanh = nn.Tanh() # I've changed last activation to tanh because ab channels should be between -1 and 1. And tanh is used for that.

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
        up_f01 = conv1
        up_f11 = conv9
        up_f02 = self.up_f02(conv2)
        up_f12 = self.up_f12(conv8)

        merge11 = torch.cat([up_f01, up_f11, up_f02, up_f12], dim=1)  # Concatenate multi-scale features

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

    with torch.no_grad():
        rgb_weights = vgg_features[0].weight
        gray_weights = rgb_weights.mean(dim=1, keepdim=True)

        model.conv1.double_conv[0].weight.data = gray_weights
        model.conv1.double_conv[0].bias.data = vgg_features[0].bias.data

        model.conv1.double_conv[2].weight.data = vgg_features[2].weight.data
        model.conv1.double_conv[2].bias.data = vgg_features[2].bias.data

        model.conv2.double_conv[0].weight.data = vgg_features[5].weight.data
        model.conv2.double_conv[0].bias.data = vgg_features[5].bias.data
        model.conv2.double_conv[2].weight.data = vgg_features[7].weight.data
        model.conv2.double_conv[2].bias.data = vgg_features[7].bias.data

        model.conv3.triple_conv[0].weight.data = vgg_features[10].weight.data
        model.conv3.triple_conv[0].bias.data = vgg_features[10].bias.data
        model.conv3.triple_conv[2].weight.data = vgg_features[12].weight.data
        model.conv3.triple_conv[2].bias.data = vgg_features[12].bias.data
        model.conv3.triple_conv[4].weight.data = vgg_features[14].weight.data
        model.conv3.triple_conv[4].bias.data = vgg_features[14].bias.data

        model.conv4.triple_conv[0].weight.data = vgg_features[17].weight.data
        model.conv4.triple_conv[0].bias.data = vgg_features[17].bias.data
        model.conv4.triple_conv[2].weight.data = vgg_features[19].weight.data
        model.conv4.triple_conv[2].bias.data = vgg_features[19].bias.data
        model.conv4.triple_conv[4].weight.data = vgg_features[21].weight.data
        model.conv4.triple_conv[4].bias.data = vgg_features[21].bias.data

        model.conv5.triple_conv[0].weight.data = vgg_features[24].weight.data
        model.conv5.triple_conv[0].bias.data = vgg_features[24].bias.data
        model.conv5.triple_conv[2].weight.data = vgg_features[26].weight.data
        model.conv5.triple_conv[2].bias.data = vgg_features[26].bias.data
        model.conv5.triple_conv[4].weight.data = vgg_features[28].weight.data
        model.conv5.triple_conv[4].bias.data = vgg_features[28].bias.data


def load_model_for_inference(model_path, device):
    model = UNet1(in_channels=1, out_channels=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def inference(model, l_channel):
    """Perform inference on L channel to get ab channels"""
    model.eval()

    with torch.no_grad():
        ab_pred = model(l_channel)

    return ab_pred.detach().cpu().numpy()


def prepare_test_image(img, dim=150):
    if isinstance(img, Image.Image):
        img = np.array(img)  # PIL -> numpy
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # RGB -> BGR (cv2 uyumu için)

    img = cv2.resize(img, (dim, dim))

    sz0, sz1 = img.shape[:2]
    r1 = img[:, :, 2].reshape(-1, 1)  # OpenCV BGR olduğu için
    g1 = img[:, :, 1].reshape(-1, 1)
    b1 = img[:, :, 0].reshape(-1, 1)

    l, a, b = rgb2lab2(r1, g1, b1) # LAB2'ye çevir
    l = l.reshape(sz0, sz1, 1)

    l_tensor = torch.FloatTensor(l).permute(2, 0, 1) # Tensor formatına çevir

    return l_tensor, a.reshape(sz0, sz1), b.reshape(sz0, sz1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = "Hyper_U_NET_pytorch-Perceptual_Loss-30Loss.pth"
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
    l_tensor = l_tensor.to(device)
    l_tensor = l_tensor.unsqueeze(0).to(device)

    ab_pred_numpy = inference(test_model, l_tensor)

    ab_pred_tensor = torch.from_numpy(ab_pred_numpy).to(device)

    A_pred_tensor = ab_pred_tensor[:, 0:1, :, :]
    B_pred_tensor = ab_pred_tensor[:, 1:2, :, :]

    rgb_pred_tensor = lab22rgb(l_tensor, A_pred_tensor, B_pred_tensor, device)

    rgb_pred_numpy = rgb_pred_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

    rgb_pred_numpy = (rgb_pred_numpy * 255).astype(np.uint8)

    bgr_pred_image = cv2.cvtColor(rgb_pred_numpy, cv2.COLOR_RGB2BGR)
    bgr_pred_resized = cv2.resize(bgr_pred_image, (img.width, img.height))

    if st.button("🎨 Renklendir"):
        with st.spinner("Model çalışıyor, lütfen bekleyin..."):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Girdi (Grayscale)**")
                st.image(img)

            with col2:
                st.markdown("**Model Çıkışı (Renkli)**")
                st.image(np.array(bgr_pred_resized))

            st.success("Tamamlandı!")
