from pathlib import Path

from flask import Flask, request, jsonify

import numpy as np
from skimage import morphology, measure
import torch
from torch import nn
import torch.nn.functional as F

from models.tracking.resnet2 import ResNet2
from models.tracking.resnet2_utils import get_patch_path
from models.segmentation.unet import UNet
from models.segmentation.unet_utils import _preprocess_image, _postprocess_mask


app = Flask(__name__)

TRACK_MODEL_PATH = Path(__file__).parent / 'artifacts' / 'resnet2_grid_tracking.pt'
ROI_MODEL_PATH = Path(__file__).parent / 'artifacts' / 'model_cine_tag_only_myo_v0_finetuned_dmd_v3.pt'


@app.route('/')
def root():
    return 'Hi there. Please use /track for tracking and /segment for segmenting.'


@app.route('/track', methods=['POST'])
def track():

    if request.method == 'POST':

        payload = request.get_json(force=True)
        imt = np.array(payload['images'])
        r0 = np.array(payload['points'])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ResNet2([2, 2, 2, 2], do_coordconv=True, fc_shortcut=False)
        model.load_state_dict(torch.load(TRACK_MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()

        # Number of reference tracking points
        N = r0.shape[0]
        
        # Empty T x W x H for each reference tracking point
        X = np.empty((N, 1, 25, 32, 32), np.float32)
        
        for i, point in enumerate(r0):
            im_p, _ = get_patch_path(imt, point, is_scaled=True)
            X[i] = im_p.copy()

        X = X - (X.mean(axis=0) / X.std(axis=0))

        batch_size = 8
        N_batches = int(np.ceil(N / batch_size))

        _y1 = []

        with torch.no_grad():
            for i in range(N_batches):
                x = X[i * batch_size:(i + 1) * batch_size]
                x = torch.from_numpy(x).to(device)
                y_pred = model(x)
                _y1.append(y_pred.detach().cpu().numpy())

        y1: np.ndarray = np.vstack(_y1)
        y1 = y1.reshape(-1, 2, 25)
        y1 = y1 + r0[:, :, None]
    
        result = {
            'prediction': y1.tolist()
        }
        
        return jsonify(result)


@app.route('/segment', methods=['POST'])
def segment() -> torch.Tensor:
    
    if request.method == 'POST':
        
        payload = request.get_json(force=True)
        img = np.array(payload['image'])
    
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        inp = _preprocess_image()(img.astype(np.float64)).unsqueeze(0).to(device)

        model: nn.Module = UNet(n_channels=1, n_classes=2, bilinear=True).double()
        # Load old saved version of the model as a state dictionary
        saved_model_sd = torch.load(ROI_MODEL_PATH, map_location=device)
        # Extract UNet if saved model is parallelized
        model.load_state_dict(saved_model_sd)
        # model.eval()

        prediction = model(inp)
        pred = F.softmax(prediction, dim=1).argmax(dim=1).detach().cpu().numpy()
        
        pred = _postprocess_mask(img.shape)(pred[0])[0]
        
        pred = (pred == 1)  # Select MYO class
        pred = morphology.binary_closing(pred)  # Close segmentation mask
        blobs, num = measure.label(pred, background=0, return_num=True)  # Closed components
        sizes = [(blobs == i).sum() for i in range(1, num + 1)]  # Evaluate component size
        if len(sizes) > 0:
            blob_index = np.argmax(sizes) + 1  # Fetch index of largest blob
            out = (blobs == blob_index)
        else:
            out = pred
        
        result = {
            'prediction': out.tolist()
        }
        
        return jsonify(result)


if __name__ == '__main__':
    app.run(port=5000, debug=True)
