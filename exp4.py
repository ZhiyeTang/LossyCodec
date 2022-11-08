import os

from PIL import Image
import numpy as np

from LossyCodec import Decoder, Encoder

q = 2

for i in range(24):
    name = str(i+1).zfill(2)
    print("正在处理[{}/{}]：".format(name, 24))
    img = Image.open("dataset/kodim{}.png".format(name))

    encoder = Encoder()
    encoder.encode(img, "bin/kodim{}.bin".format(name), q)

    decoder = Decoder()
    img = decoder.decode("bin/kodim{}.bin".format(name), q)
    img.save("res/kodim{}.bmp".format(name))


ratio = 0.
psnr = 0.
for i in range(24):
    name = str(i+1).zfill(2)
    bmp_size = os.path.getsize("./bmp/kodim{}.bmp".format(name))
    bin_size = os.path.getsize("./bin/kodim{}.bin".format(name))
    bmp = np.asarray(Image.open("bmp/kodim{}.bmp".format(name))).astype(np.float16)
    res = np.asarray(Image.open("res/kodim{}.bmp".format(name))).astype(np.float16)
    ratio += bmp_size / bin_size / 24.
    psnr += 10 * np.log10(255**2/np.mean(np.square(res-bmp))) / 24.
print("最终压缩比：{:.4f}".format(ratio))
print("最终峰值信噪比（q={}）：{:.4f}dB".format(q, psnr))
