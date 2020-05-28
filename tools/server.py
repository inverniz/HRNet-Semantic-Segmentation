from starlette.applications import Starlette
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO
import base64

import cv2
from PIL import Image
import argparse
import os
import pprint
import sys

import timeit
from pathlib import Path

import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.nn import functional as F

import _init_paths
import models
from config import config
from config import update_config
from utils.modelsummary import get_model_summary
from utils.utils import create_logger, FullModel


def setup_model():
    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # build model
    if torch.__version__.startswith('1'):
        module = eval('models.seg_hrnet_ocr')
        module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
    model = eval('models.seg_hrnet_ocr.get_seg_model')(config)
    return model


model = setup_model()
model.eval()

path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='tools/static'))

PREDICTION_FILE_SRC = path / 'static' / 'predictions.txt'


@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    img_bytes = data["img"]
    bytes = base64.b64decode(img_bytes)
    return predict_from_bytes(bytes)


def get_palette(n):
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


def convert_label(label, inverse=False):
    ignore_label = -1
    label_mapping = {-1: ignore_label, 0: ignore_label,
                     1: ignore_label, 2: ignore_label,
                     3: ignore_label, 4: ignore_label,
                     5: ignore_label, 6: ignore_label,
                     7: 0, 8: 1, 9: ignore_label,
                     10: ignore_label, 11: 2, 12: 3,
                     13: 4, 14: ignore_label, 15: ignore_label,
                     16: ignore_label, 17: 5, 18: ignore_label,
                     19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                     25: 12, 26: 13, 27: 14, 28: 15,
                     29: ignore_label, 30: ignore_label,
                     31: 16, 32: 17, 33: 18}
    temp = label.copy()
    if inverse:
        for v, k in label_mapping.items():
            label[temp == k] = v
    else:
        for k, v in label_mapping.items():
            label[temp == k] = v
    return label

def input_transform(image):
    size = image.shape
    image = image.astype(np.float32)[:, :, ::-1]
    image = image / 255.0
    image -= [0.485, 0.456, 0.406]
    image /= [0.229, 0.224, 0.225]
    long_size = np.int(size[0] + 0.5)
    h, w = size[:2]
    if h > w:
        new_h = long_size
        new_w = np.int(w * long_size / h + 0.5)
    else:
        new_w = long_size
        new_h = np.int(h * long_size / w + 0.5)

    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    image = image.transpose((2, 0, 1))
    return image

def predict_from_bytes(bytes):
    frame = cv2.imdecode(np.frombuffer(bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    cv2.imwrite("tools/static/images/input.png", frame)
    size = frame.shape
    print("image shape: {}".format(size))

    frame = input_transform(frame)
    with torch.no_grad():
        logit = model(torch.as_tensor(np.expand_dims(frame, axis=0)))

        OUTPUT_INDEX = -2
        if config.MODEL.NUM_OUTPUTS > 1:
            logit = logit[OUTPUT_INDEX]

        print("logit shape right after prediction: {}".format(logit.shape))

        logit = F.interpolate(
            input=logit, size=size[:2],
            mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
        )

        print("logit shape after interpolation: {}".format(logit.shape))

        logit = logit.exp()
        logit = logit[0]

    prediction = np.asarray(np.argmax(logit, axis=0), dtype=np.uint8)
    print("prediction shape: {}".format(prediction.shape))
    print(np.unique(prediction))
    palette = get_palette(256)
    prediction = convert_label(prediction, inverse=True)
    print(np.unique(prediction))
    prediction = Image.fromarray(prediction)
    prediction.putpalette(palette)
    prediction.save("tools/static/images/prediction.png")

    result_html1 = path / 'static' / 'result1.html'
    result_html2 = path / 'static' / 'result2.html'

    result_html = str(result_html1.open().read() + str("") + result_html2.open().read())
    return HTMLResponse(result_html)


@app.route("/")
def form(request):
    index_html = path / 'static' / 'index.html'
    return HTMLResponse(index_html.open().read())


if __name__ == "__main__":
    if "serve" in sys.argv: uvicorn.run(app, host="0.0.0.0", port=8080)
