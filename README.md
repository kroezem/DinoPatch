# DinoPatch: Patch-Based Anomaly Detection with DINOv2

DinoPatch is a research notebook that detects anomalies in images by combining:
- DINOv2 Vision Transformer embeddings as patch features, and
- a PatchCore-style, spatially aligned memory bank built from nominal images.

It targets both texture anomalies (pasta) and shape anomalies (screws with washers) by enforcing consistent orientation, scale, and position before feature extraction, then scoring each patch by nearest neighbor distance.

> Notebook: [Google Collab](https://colab.research.google.com/drive/12ghImE8IA1DlnfT4H3bph1YCxx0I2AcG?usp=sharing)

> Read full paper: [PDF](https://makro.ca/assets/pdf/dinopatch.pdf)

---

## TL;DR
- Input: a set of nominal training images and a set of evaluation images.
- Preprocess: grayscale, blur, Canny, PCA-based alignment to normalize orientation, scale, position, and polarity.
- Features: extract DINOv2 patch embeddings on a dense grid.
- Memory bank: stack nominal patch features at each spatial location.
- Scoring: per-patch Euclidean distance to the nearest nominal patch at the same location. Build an anomaly map, then post-process and threshold.
- Output: anomaly heatmaps and image-level anomaly scores.

On the two small datasets used in this project, DinoPatch achieved Accuracy, Precision, Recall, F1, and AUROC of 1.00. See Limitations for caveats.

---

## Why this approach
- **ViT features without training:** DINOv2 gives strong, semantically rich patch embeddings.
- **Spatial awareness:** keeping patch positions in a memory bank helps with shape anomalies that global NN methods can miss.
- **Simple scoring:** Euclidean distance in feature space is fast and effective after alignment.

---

## Data requirements
Prepare two splits:
- **Nominal training set** to build the memory bank.
- **Evaluation set** with both nominal and anomalous images.

Images should include the object of interest with minimal clutter. The PCA step attempts to normalize pose and scale, but extreme backgrounds will add noise.

---

## Run it in Colab
1. Open `DinoPatch.ipynb`.
2. (Optional) Set the runtime to T4 GPU. CPU also works since the model runs inference only, but is very slow.
3. Install dependencies via the first cell if prompted:
   - `torch`, `timm` (for DINOv2), `opencv-python`, `scikit-learn`, `numpy`, `matplotlib`, `tqdm`.
4. Point the data loading cells to your image folders.
5. Execute sections in order:
   - Preprocessing and PCA alignment
   - Feature extraction with DINOv2
   - Build memory bank from nominal images
   - Evaluate images to produce anomaly maps
   - Hyperparameter sweep for post-processing and image-level scoring
6. Review outputs:
   - Per-image heatmaps
   - Quantitative metrics and AUROC curves

---

## Method details
- **PCA alignment:** Canny edges -> contours -> PCA to get center, principal axis angle, and a scale proxy. Apply wrap padding to avoid black borders, transform image to a canonical frame, and mirror if the polarity is reversed. The same transforms are applied to a mask to ignore padded regions during scoring.
- **DINOv2 features:** upscale inputs so the ViT produces a higher resolution feature grid (e.g., 56x56x768). Extract an orientation reference vector along the principal axis and use it to enforce consistent polarity across samples.
- **Memory bank:** for each spatial location, collect feature vectors from all nominal images.
- **Scoring:** for each evaluation patch, compute Euclidean distance to the closest nominal feature at that location. The minimum distance becomes the raw anomaly value at that pixel. Build an anomaly map, mask padded regions, then post-process.
- **Post-processing and image score:** thresholding and morphology to reduce noise. Image-level score options include max, mean, and 99th percentile. A small search picks the best combo on the evaluation split.

---

## Results
- On the included pasta and screw datasets, DinoPatch separated nominal vs anomalous cleanly and reported:
  - Accuracy = 1.00
  - Precision = 1.00
  - Recall = 1.00
  - F1 = 1.00
  - AUROC = 1.00
- Qualitative heatmaps localize anomalous regions clearly after post-processing.

---

## Limitations and what to watch for
- **Small data and overfitting risk:** Hyperparameters were tuned on the same evaluation pool. Without a disjoint test set, real performance may drop.
- **Background sensitivity:** The raw anomaly maps can be noisy when backgrounds vary. Post-processing is doing nontrivial lifting.
- **PCA assumptions:** Alignment relies on stable contours and a dominant principal axis. Failure cases will degrade both memory bank quality and polarity checks.
- **Memory footprint:** The memory bank scales with the number of nominal images. For larger datasets, consider N-shot subsampling, KNN-based distillation at each location, or PCA to reduce feature dimensions.

---

## Ideas for future work
- Combine local and global NN distances in one score to handle both shape and texture anomalies.
- Learn a small affine regressor to replace parts of the handcrafted PCA alignment.
- Add semantic masking to suppress irrelevant background.
- Reduce memory bank size with per-location clustering or dimensionality reduction.

---
