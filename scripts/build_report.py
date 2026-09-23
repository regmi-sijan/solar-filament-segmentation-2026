"""Fill the competition's acmart report format using measured experiment results."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
r=json.loads((ROOT/'results/validation.json').read_text());s=r['summary']
c=json.loads((ROOT/'results/calibration.json').read_text());config=s['config']
i=json.loads((ROOT/'results/inference-manifest.json').read_text());v=json.loads((ROOT/'results/submission-verification.json').read_text())
a=json.loads((ROOT/'configs/author.json').read_text())
lo,hi=s['pq_bootstrap_95ci'];sigma=[12,32,64][config['scale']]
text=r'''\documentclass[sigconf]{acmart}
\input{preamble}
\guideStylefalse
\begin{document}
\title{Local Contrast and Morphological Grouping for Solar Filament Segmentation}
\subtitle{A Solution to the Solar Filament Segmentation Challenge 2026}
\author{Sijan Regmi}
\affiliation{\country{Nepal}}
\email{sijanregmi419@gmail.com}
\begin{abstract}
This report describes our solution to the Solar Filament Segmentation Challenge~2026, a Kaggle competition on automatic segmentation of solar filaments in GONG H-$\alpha$ observations. We implement a full-resolution, CPU-based pipeline that estimates the solar disk, measures local intensity deficits, groups neighboring dark pixels, and exports separate filament instances. Parameters are selected using 111 development images, with all observations from the same date assigned to the same data partition. On 139 held-out images the method obtains micro Panoptic Quality (PQ) of \textbf{@@PQ@@}, with a date-cluster bootstrap 95\% interval of @@CI@@. The system requires no learned weights or external data. We report local validation, not a Kaggle leaderboard result, and provide source code, a complete notebook, deterministic parameters, and reproducibility checks.
\end{abstract}
\maketitle
\section{Introduction}
This report goes over the details of our submitted solution to the Solar Filament Segmentation Challenge~2026 \cite{filament-segmentation-2026}. The challenge, launched on July~10,~2026, is organized by the Earth-Space AI Research (ESAIR) Lab\footnote{\href{https://www.esairlab.com/home}{www.esairlab.com/}}. This challenge utilizes the MAGFiLO dataset for evaluation of the segmentation algorithms \cite{ahmadzadeh2024dataset}.

Our contribution is a reproducible classical baseline, not a claim of a new segmentation architecture. It favors transparent computation and native-resolution masks. Brightness normalization compensates for broad illumination variation; thresholding and morphology then identify connected dark structures. This simple representation can miss weak filaments and confuse other dark features with filaments. The evaluation below quantifies these limitations rather than inferring success from plausible-looking images.

\section{Methodology}\label{sec:methodology}
\subsection{Data and permitted supervision}
The supplied archive contains 707 training images, 180 test images, and one training annotation JSON file. Every image decodes as an 8-bit grayscale array of size $2048\times2048$. The training JSON contains 1,154 annotator--image records and 8,199 filament annotations. A single source image can have multiple independently annotated records: 411 images have one set, 145 have two, and 151 have three. We retain those sets separately during scoring; their polygons are not silently fused into a consensus mask.

Only segmentation polygons from the supplied training JSON are used for calibration and evaluation. Chirality labels, spines, annotation bounding boxes, external images, and external labels are not inputs to the predictor. Test images are used only for inference after configuration selection. Predicted masks depend on image pixels and the saved configuration, not on image identity or capture date. Capture dates extracted from filenames are used solely for partitioning and uncertainty estimation.

The file audit verifies archive CRC values, image decoding, dimensions, unique record identifiers, annotation references, and basic polygon structure. It finds no byte-identical image duplicates and no shared train/test filenames. These checks establish file integrity, but do not establish annotation correctness or exclude similar observations taken on nearby dates.

The median annotated filament area is 1,228 pixels, about 0.029\% of an image; the smallest annotation has area 9 pixels. We therefore preserve native image resolution. Nevertheless, our explicit minimum-area filter can reject real small structures. Its tradeoff is measured through calibration instead of being hidden in preprocessing.

\subsection{Pixel processing and instance generation}
\begin{figure*}[t]
\centering\includegraphics[width=\textwidth]{figures/pipeline.pdf}
\caption{Complete inference pipeline. Only the grayscale test image and saved scalar configuration are required. Calibration uses training polygons separately.}
\Description{Six-stage flow from grayscale image through solar disk masking, contrast normalization, thresholding, connected components, and RLE output.}
\end{figure*}
Let $I\in[0,1]^{2048\times2048}$ denote the grayscale input. Otsu thresholding produces a bright-region mask. We retain the largest connected component, fill its external contours, and erode it with a $25\times25$ elliptical structuring element. This disk mask suppresses sky background and the sharp solar-limb transition. The disk is inferred independently for each image, rather than imposed as a fixed circle. A failed or fragmented disk estimate is a potential source of error.

We first smooth $I$ with a Gaussian of standard deviation 1 pixel, producing $S$. A broader Gaussian estimates the local background $B_\sigma=G_\sigma*S$. The nonnegative relative intensity deficit is
\begin{equation}
D_\sigma=\operatorname{clip}\left(\frac{B_\sigma-S}{\max(B_\sigma,0.08)},0,1\right).
\end{equation}
The denominator floor avoids amplification in near-black regions. We compute maps at $\sigma\in\{12,32,64\}$ pixels. The selected model uses $\sigma=@@SIGMA@@$ and marks pixels where $D_\sigma>@@THRESH@@$ inside the disk. A fixed background scale is computationally convenient but may underrepresent very broad or low-contrast structures.

Optional morphological closing uses an elliptical element with radius $r$; closing is dilation followed by erosion \cite{opencv}. After closing, the disk mask is applied again. Eight-connected components define predicted instances. Components smaller than @@AREA@@ pixels or larger than 100,000 pixels are discarded; the selected closing radius is @@RADIUS@@ pixels; the selected minimum elongation is @@ELONG@@. The upper bound removes exceptionally large image artifacts. We do not impose a fixed instance count, fill missing detections with dummy masks, or modify predictions to exploit scoring behavior.

This construction partitions the accepted foreground into non-overlapping instances. It naturally separates disconnected dark regions, but cannot reliably decide whether nearby islands belong to the same physical filament. Closing can repair gaps and can also merge neighbors. Conversely, an elongated filament with a weak bridge can fragment into several predictions.

\subsection{Calibration and data partitioning}
Within each capture year, unique capture dates are sorted by the SHA-256 digest of the string \texttt{2026:YYYYMMDD}. The first approximately 20\% of dates form validation; the next approximately 15\% form calibration. All annotation records and source images from a given date remain together. Rounding is performed per year with at least one date in each selected group. The resulting manifest contains 568 development images, of which 111 are used for calibration, and 139 validation images. The remaining development images are not used to fit this parameter-only baseline.

The first search evaluates 32 configurations: two Gaussian scales, thresholds $\{0.08,0.12,0.16,0.20\}$, closing radii $\{0,2\}$, and minimum areas $\{100,300\}$. The maximum area is fixed. After observing excess false positives in calibration, a second 64-configuration grid tests scales $\{32,64\}$, thresholds $\{0.04,0.06,0.08,0.10\}$, closing radii $\{0,2\}$, minimum areas $\{300,800\}$, and minimum elongations $\{1,2\}$. Elongation is the square root of the ratio of coordinate-covariance eigenvalues after adding 1 to each; compact components below the limit are rejected. This refinement uses only calibration feedback, not validation. We select the highest pooled calibration PQ, resolving exact ties by the predefined configuration order. The winning calibration PQ is @@CALPQ@@. The saved configuration is then frozen before validation and test inference. No deep network or pixel classifier is trained, and no external model weights are necessary.

\section{Evaluation}\label{sec:evaluation}
\subsection{Metric implementation}
Ground-truth polygons are rasterized using \texttt{pycocotools}. Compressed COCO run-length encodings allow direct pairwise IoU calculation without materializing dense stacks of all masks. For each annotator--image record, every pair with IoU strictly greater than 0.5 is a match. Predictions without a qualifying overlap are false positives; unmatched ground-truth instances are false negatives. We sum counts and matched IoUs over all validation records, following the official self-evaluation notebook \cite{evaluation}:
\begin{equation}
\mathrm{PQ}=\frac{\sum_{(g,p):\mathrm{IoU}(g,p)>0.5}\mathrm{IoU}(g,p)}{\mathrm{TP}+0.5\mathrm{FP}+0.5\mathrm{FN}}.
\end{equation}
This is pooled micro PQ, not the arithmetic mean of per-image PQ. Each independent annotation set is evaluated against the same image prediction. Our implementation mirrors the reference overlap-matrix semantics, including its treatment of multiple qualifying matches; it does not add a different assignment algorithm. A zero denominator returns zero. Unit checks cover exact predictions, no predictions, no ground truth, strict rejection of IoU equal to 0.5, connected components, RLE round-tripping, and numerical agreement with the official notebook.

\begin{table}[h]
\caption{Frozen-configuration local validation and inference results.}
\centering\begin{tabular}{lr}\toprule
Quantity & Value\\\midrule
Held-out source images & 139\\
Annotator--image records & @@RECORDS@@\\
Capture-date clusters & @@CLUSTERS@@\\
Micro PQ & @@PQ@@\\
95\% cluster-bootstrap interval & @@CI@@\\
True positives / false positives & @@TP@@ / @@FP@@\\
False negatives & @@FN@@\\
Test images processed & 180\\
Predicted test instances & @@NROWS@@\\
Test inference wall time & @@TIME@@ s\\\bottomrule
\end{tabular}
\end{table}
Uncertainty uses 2,000 bootstrap resamples of capture-date clusters with replacement (seed 2026), preserving related source images and annotator sets within each resampled cluster. The interval describes variation within this validation partition; it does not account for all dataset shift, model-selection uncertainty, or temporal dependence between different dates. It is not a confidence interval for an unseen Kaggle leaderboard score.

\begin{figure*}[t]
\centering\includegraphics[width=\textwidth]{figures/metrics.pdf}
\caption{IoU and Dice distributions for pairs with nonzero overlap, plus the number of overlapping counterparts per GT or prediction. Zero-overlap objects appear in the count panel, not the pair distributions.}
\Description{Three panels show pairwise overlap histograms and fragmentation or merging counts.}
\end{figure*}
For overlapping pairs, Dice is computed as $2\mathrm{IoU}/(1+\mathrm{IoU})$. A ground-truth object overlapping more than one prediction indicates fragmentation; a prediction overlapping multiple ground-truth objects indicates merging. These diagnostic overlaps use IoU greater than zero, not the 0.5 threshold used for PQ matches. Pairwise histograms should not be interpreted as instance-averaged detection accuracy because they omit non-overlapping pairs.

\begin{figure*}[t]
\centering\includegraphics[width=.84\textwidth]{figures/examples.pdf}
\caption{Validation examples selected at approximately the 10th, 50th, and 90th percentiles of source-image mean PQ. Top: full image. Bottom: crop around the largest instance from the first annotation set. Red outlines show that annotator's ground truth; cyan outlines show predictions. Scores average over available annotation sets, so displayed outlines are not all the scored ground truth.}
\Description{Three solar images and corresponding detailed crops compare red annotation boundaries with cyan predicted boundaries.}
\end{figure*}
\subsection{Strengths, limitations, and reproduction}
The method has a small, inspectable configuration, preserves image resolution, and runs without an accelerator. It avoids external-label leakage and keeps independent annotation sets distinct. Its main limitations are the inability to recognize semantic context, sensitivity to disk estimation and local brightness, and an imperfect fragmentation/merging tradeoff. Small components are deliberately filtered, and limb erosion can remove genuine near-limb structures. No claim of competitive leaderboard placement is made.

A future comparison could train a supervised segmentation model on the development partition and evaluate it on a new untouched holdout, with explicit instance-separation supervision and controlled ablations. Such a model has not been trained for this report. The current validation result must not be reused indefinitely to guide architecture changes without acknowledging that it becomes development feedback.

The repository includes the exact split, selected parameters, complete calibration table, inference manifest, CSV checksum, pinned Python package versions, and an executed pipeline notebook. The README provides exact CLI commands for splitting, both calibration passes, evaluation, and prediction. Test output has two columns, \path{filament_id} and \path{segmentation_rle}; each nonempty instance uses the source image stem plus a unique integer suffix. We validate decoded dimensions, unique identifiers, nonempty masks, and canonical re-encoding. All 180 images are processed even if some yield no retained components.

Measurements were made on an ARM64 macOS laptop with 16~GiB of memory. OpenCV uses two threads. Timing includes image loading, mask generation, encoding, and CSV writing, but excludes installation and calibration. Reproduction requires the supplied competition archive, not third-party data or downloaded model weights. AI assistance was used to develop and document the code; the participant must independently understand and defend its operation under the competition rules.

\balance
\section{Acknowledgment}\label{sec:acknowledgment}
The organization of the Filament Segmentation Challenge 2026 is partially supported by the U.S. National Science Foundation (NSF) under Grant No. 2209912 and 2433781, directorate for Computer and Information Science and Engineering (CSE), and Office of Advanced Cyberinfrastructure (OAC), and Grant No. 2511630, AST Division Of Astronomical Sciences and MPS Directorate for Mathematical and Physical Sciences.

The Filament Segmentation Challenge 2026 is sponsored by the U.S. National Science Foundation (NSF) National Solar Observatory (NSO).

This work utilizes GONG data obtained by the NSO Integrated Synoptic Program, managed by the National Solar Observatory, which is operated by the Association of Universities for Research in Astronomy (AURA), Inc. under a cooperative agreement with the National Science Foundation and with contribution from the National Oceanic and Atmospheric Administration. The GONG network of instruments is hosted by the Big Bear Solar Observatory, High Altitude Observatory, Learmonth Solar Observatory, Udaipur Solar Observatory, Instituto de Astrofísica de Canarias, and Cerro Tololo Interamerican Observatory.
\bibliographystyle{ACM-Reference-Format}
\bibliography{main}
\end{document}
'''
values={'ELONG':config.get('min_elongation',1),'PQ':f"{s['pq']:.4f}",'CI':f'[{lo:.4f}, {hi:.4f}]','SIGMA':sigma,'THRESH':config['threshold'],'AREA':config['min_area'],'RADIUS':config['closing_radius'],'CALPQ':f"{c['trials'][0]['pq']:.4f}",'RECORDS':s['annotation_sets'],'CLUSTERS':s['date_clusters'],'TP':s['tp'],'FP':s['fp'],'FN':s['fn'],'NROWS':v['rows'],'TIME':f"{i['seconds']:.1f}"}
for key,value in values.items():text=text.replace('@@'+key+'@@',str(value))
assert '@@' not in text
(ROOT/'report/main.tex').write_text(text)
