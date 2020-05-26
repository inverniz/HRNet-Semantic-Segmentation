from starlette.applications import Starlette
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO
import base64

import cv2
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

    dump_input = torch.rand(
        (1, 3, config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    )

    if config.MODEL.PRETRAINED:
        model_state_file = config.MODEL.PRETRAINED
    else:
        model_state_file = os.path.join(final_output_dir,
                                        'best.pth')

    pretrained_dict = torch.load(model_state_file, map_location='cpu')
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                       if k[6:] in model_dict.keys()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.eval()
    return model


model = setup_model()
classes = ['Blouse', 'Blazer', 'Button-Down', 'Bomber', 'Anorak', 'Tee', 'Tank', 'Top', 'Sweater', 'Flannel', 'Hoodie',
           'Cardigan', 'Jacket', 'Henley', 'Poncho', 'Jersey', 'Turtleneck', 'Parka', 'Peacoat', 'Halter', 'Skirt',
           'Shorts', 'Jeans', 'Joggers', 'Sweatpants', 'Jeggings', 'Cutoffs', 'Sweatshorts', 'Leggings', 'Culottes',
           'Chinos', 'Trunks', 'Sarong', 'Gauchos', 'Jodhpurs', 'Capris', 'Dress', 'Romper', 'Coat', 'Kimono',
           'Jumpsuit', 'Robe', 'Caftan', 'Kaftan', 'Coverup', 'Onesie']

path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='tools/static'))

PREDICTION_FILE_SRC = path / 'static' / 'predictions.txt'


@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    img_bytes = await (data["img"].read())
    bytes = base64.b64decode(img_bytes)
    return predict_from_bytes(bytes)


def predict_from_bytes(bytes):
    img = open_image(BytesIO(bytes))
    downsample_rate = 1 / config.TEST.DOWNSAMPLERATE
    frame = cv2.resize(
        img,
        None,
        fx=downsample_rate,
        fy=downsample_rate,
        interpolation=cv2.INTER_NEAREST
    )
    size = frame.shape

    input_frame = frame.copy()
    input_frame = input_frame.astype(np.float32)[:, :, ::-1]
    input_frame = input_frame / 255.0
    long_size = np.int(size[0] + 0.5)
    h, w = size[:2]
    if h > w:
        new_h = long_size
        new_w = np.int(w * long_size / h + 0.5)
    else:
        new_w = long_size
        new_h = np.int(h * long_size / w + 0.5)

    input_frame = cv2.resize(input_frame, (new_w, new_h),
                             interpolation=cv2.INTER_LINEAR)

    input_frame = input_frame.transpose((2, 0, 1))
    with torch.no_grad():
        logit = model(torch.as_tensor(np.expand_dims(input_frame, axis=0)))

        if config.MODEL.NUM_OUTPUTS > 1:
            logit = logit[config.TEST.OUTPUT_INDEX]

        # print("logit shape right after prediction: {}".format(logit.shape))

        logit = F.interpolate(
            input=logit, size=size[:2],
            mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
        )

        # print("logit shape after interpolation: {}".format(logit.shape))

        logit = logit.exp()
        logit = logit[0].cpu()

    prediction = np.asarray(np.argmax(logit, axis=0), dtype=np.uint8)
    result_html1 = path / 'static' / 'result1.html'
    result_html2 = path / 'static' / 'result2.html'

    result_html = str(result_html1.open().read() + str(predictions[0:3]) + result_html2.open().read())
    return HTMLResponse(result_html)


@app.route("/")
def form(request):
    index_html = path / 'static' / 'index.html'
    return HTMLResponse(index_html.open().read())


if __name__ == "__main__":
    if "serve" in sys.argv: uvicorn.run(app, host="0.0.0.0", port=8080)
