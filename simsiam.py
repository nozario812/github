import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class PulseDataset(Dataset):
    def __init__(self, pulses):
        self.pulses = pulses

    def __len__(self):
        return len(self.pulses)

    def __getitem__(self, idx):
        pulse = self.pulses[idx]
        return pulse, pulse  # 入力とターゲットは同じ


def augment(pulse, noise_std=0.01):
    """
    波形データのオーグメンテーション処理を多様化
    Args:
        pulse (numpy.ndarray): 入力波形データ（1次元の波形）
        noise_std (float): ノイズの標準偏差（デフォルト: 0.01）
    Returns:
        augmented (numpy.ndarray): オーグメンテーション後の波形データ
    """
    # ノイズの追加
    noise = np.random.normal(0, noise_std, pulse.shape)
    augmented = pulse + noise

    # ランダムシフト
    shift = np.random.randint(-10, 10) 
    augmented = np.roll(augmented, shift)
    if shift > 0:
        augmented[:shift] = 0  
    elif shift < 0:
        augmented[shift:] = 0

    # 振幅のランダムスケーリング
    scale_factor = np.random.uniform(0.8, 1.2) 
    augmented = augmented * scale_factor

    # ガウス平滑化（スムージング）
    window_size = np.random.randint(3, 7)  
    kernel = np.ones(window_size) / window_size
    augmented = np.convolve(augmented, kernel, mode='same')  #
    return augmented


#Machine Learning Parameter#
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        return self.net(x)

#Simple siam#
class SimSiam(nn.Module):
    def __init__(self, input_dim, feature_dim=128):
        super(SimSiam, self).__init__()
        self.encoder = MLP(input_dim, feature_dim)
        self.predictor = MLP(feature_dim, feature_dim)

    def forward(self, x1, x2):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        return p1, p2, z1.detach(), z2.detach()

#負のコサイン損失関数#
def negative_cosine_similarity(p, z):
    p = nn.functional.normalize(p, dim=1)#ベクトルの正規化#
    z = nn.functional.normalize(z, dim=1)
    return - (p * z).sum(dim=1).mean()

def train(model, dataloader, optimizer, epochs=100):
    model.train()
    for epoch in range(epochs):
        for x, _ in dataloader:
          x = x.numpy()  
          x1 = np.array([augment(p) for p in x]) 
          x2 = np.array([augment(p) for p in x]) 

          x1 = torch.tensor(x1, dtype=torch.float32)
          x2 = torch.tensor(x2, dtype=torch.float32)

          p1, p2, z1, z2 = model(x1, x2)
          loss = (negative_cosine_similarity(p1, z2) + negative_cosine_similarity(p2, z1)) * 0.5

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


def plot_embeddings(model, dataloader):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for x_batch, _ in dataloader:
            x_batch = torch.tensor(x_batch, dtype=torch.float32)
            z = model.encoder(x_batch)
            embeddings.append(z.numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    tsne = TSNE(n_components=2)
    embeddings_2d = tsne.fit_transform(embeddings)

    # x軸とy軸のデータを取得
    x = embeddings_2d[:, 0].tolist()
    y = embeddings_2d[:, 1].tolist()

    # 散布図をプロット
    plt.scatter(x, y, s=0.3)
    plt.title('Pulse Embeddings')
    plt.show()

    # xとyを返す
    return x, y

def fopen(filename):
    """
    Read FITS

    Parameters
    ==========
        filename:   file number to read

    Returns
    =======
        t:          time array
        wave:       waveform array ['pulse']
        wave:       waveform array ['noise']
    """

    import astropy.io.fits as pf  ##astropyは天文学のデータ解析によく使われて、座標変換をするときに使える##

    pulse = filename+'p.fits'   ##fitsファイルを使う時にはastropyを使う##
    noise = filename+'n.fits'

    keyname = np.asarray(['pulse', 'noise'])
    wave = {}
    # Open fits file and get pulse/noise data
    for i, j in enumerate(np.asarray([pulse, noise])):
        header = pf.open(j)
        wave[keyname[i]] = header[1].data.field(1).copy()
        dt = header[1].header['THSCL2']
        t = np.arange(wave[keyname[i]].shape[-1]) * dt
        header.close()

    return t, wave['pulse'], wave['noise']

#データの読み込みと整形
time,pulse,noise = fopen("/content/drive/MyDrive/b64/CH3_b64_190202")
pulse = pulse-np.max(pulse,axis = -1,keepdims=True)
pulse = -pulse
ph = np.max(pulse,axis = -1, keepdims=True)
pulse = pulse/ph

num_samples = len(pulse)
pulse_length = len(pulse[0])
pulses = pulse

# データセットとデータローダーの作成
dataset = PulseDataset(pulses)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# モデルとオプティマイザの作成
input_dim = pulse_length
model = SimSiam(input_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)
"""model.parametersはmodelという名前のモデルのパラメータを取得"""
"""lrは学習率"""

# 学習の実行
train(model, dataloader, optimizer, epochs=10)

torch.save(model.state_dict(), "simsiam_model.pth")
print("Model saved successfully!")

model = SimSiam(input_dim)

# 保存した重みをロード
model.load_state_dict(torch.load("simsiam_model.pth"))
model.eval()

x, y = plot_embeddings(model, dataloader)



