# Leyenda Project

TouNum · Zakarya Zitouni

This project takes a digitised image and decides whether it is a photograph. If it is, the project removes its noise and writes an English caption for it:

```
unidentified image → 1 · binary classification → 2 · pre-processing (denoising) → 3 · captioning
```

Each stage is a deliverable. Its notebook documents how the model was designed, trained and evaluated. A test website runs the whole workflow on any images you drop into it.

## Contents

| Folder | Content |
|---|---|
| `1 - Binary Classifier/` | `ClassifierNotebook.ipynb` and the trained classifier `modele_photo.keras`. It is a CNN that outputs P(photo) and accepts an image when P(photo) ≥ 0.5. |
| `2 - Denoiser/` | `DenoiserNotebook.ipynb` and the denoiser `denoiser_coco_v2.keras`. It is a residual U-Net trained on COCO with Gaussian, salt-and-pepper, speckle and Poisson noise, then fine-tuned on noise shared across the colour channels and on JPEG-compressed noise (Section 13 of the notebook). The fine-tuning curves and scores are in `denoiser_coco_v2_results.json`. |
| `3 - Captioner/` | `CaptionerNotebook.ipynb` and two captioners trained on COCO: `leyenda_captioner_gru/` (InceptionV3 with an attention GRU) and `leyenda_captioner/` (ConvNeXt-XL with a Transformer). `workflow_demo/` holds nine sample inputs (photographs, noisy photographs, a painting, a schematic, a sketch and text) and `captions.csv`, the output of the notebook's own chain on them (Section 12). |
| `website/` | The test page: `server.py`, `pipeline.py` (the workflow), and `index.html`, `style.css` and `app.js` (the interface). |

## Running the website

You need Python 3.10 to 3.13. Nothing else has to be installed beforehand.

- **Windows:** double-click `website/start.bat`.
- **Linux / macOS:** run `bash website/start.sh`.

Then open <http://localhost:8000> once the window shows `ready`.

- **First launch only:**
  - the launcher creates a Python environment and installs `requirements.txt` (TensorFlow, Keras, NumPy, Pillow), which takes a few minutes;
  - Keras then downloads the ImageNet weights of InceptionV3 and ConvNeXt-XL (about 1.4 GB, into `~/.keras/models`).
- **Later launches** take about a minute to load the models.
- **Stopping:** close the window to stop the page.
- **GPU:** none is needed. On a CPU a photograph takes about 20 s and a rejected image well under a second; with a GPU TensorFlow detects, a photograph takes about a second.

If you'd rather manage the environment yourself:

```
pip install -r requirements.txt
cd website
python server.py 8000
```

## Using the website

1. **Drop images** onto the page: files, several files, or a whole folder. Non-image files are skipped. The images in `3 - Captioner/workflow_demo/input/` make a quick check.
2. **Pick the order** with the switch:
   - **1 → 2 → 3** follows the brief: the classifier sees the file as received, and only photographs are denoised.
   - **2 → 1 → 3** denoises every file before classifying it, so more noisy photographs are kept (Extension E2 of the Captioner notebook).
3. **Read the results.** Each image gets one row in each of the two versions, which are stacked on the page. The two versions differ only in the captioner used at stage 3:
   - **A:** InceptionV3 with the GRU (CIDEr-D 1.056 on COCO `val2017`);
   - **B:** ConvNeXt-XL with the Transformer (CIDEr-D 1.395 on COCO `val2017`).

   A row shows the input, then P(photo), then the denoised image with how much it changed, then the caption. Click an image to compare the input and the denoised version at full size.

With the demo images, the order matters for the Gaussian-noise photograph (`3_photo_…_gaussian.png`): in 1 → 2 → 3 the classifier rejects it as received, while in 2 → 1 → 3 it is denoised, accepted and captioned. Version A in 2 → 1 → 3 gives the same captions as `captions.csv`.

## Reproducing the training

Run the notebooks in this order, each from its own folder and from top to bottom:

1. **Classifier:** `1 - Binary Classifier/ClassifierNotebook.ipynb` writes `modele_photo.keras`.
2. **Denoiser:** `2 - Denoiser/DenoiserNotebook.ipynb` trains the first-stage model (kept in `2 - Denoiser/cache/` as `denoiser_coco.keras`). Section 13 then fine-tunes it and writes `denoiser_coco_v2.keras`, the model used by the website. Set `COCO_DIR` to the folder that holds `train2017/`, `val2017/` and `test2017/`.
3. **Captioner:** `3 - Captioner/CaptionerNotebook.ipynb` trains and saves both captioners. Its workflow and extensions (Section 12, E1, E2) load the classifier from step 1 and the first-stage denoiser from step 2, the model they were measured with, so run it last.

Notes:

- **Data:** the notebooks were run with the versions in `requirements.txt`, on the COCO 2017 images and captions (<https://cocodataset.org>). The photo / non-photo dataset of Deliverable 1 was provided with the project. No dataset is included: each notebook sets the paths it expects in its first code cell, so change them to your own folders. The saved outputs still show the paths of the machine the notebooks were run on.
- **Reloading:** each notebook reloads a saved model instead of retraining it when the file is present. To retrain one, delete or move its file first.

## Note for the repository

- **Git LFS:** the trained models (`*.keras`, `*.weights.h5`) are stored with Git LFS, because the denoiser is larger than GitHub's 100 MB file limit. Run `git lfs install` once before cloning or committing.
- **CNN weights:** `3 - Captioner/cnn_weights/`, a local copy of the ImageNet weights, is not committed. Keras downloads the same files on first use.
