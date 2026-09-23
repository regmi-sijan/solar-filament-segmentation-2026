# Understanding and defending the solution

This is a classical image-processing method with calibrated parameters. It is not a deep neural network, and it has no learned weight file. AI assistance was used to develop and document the pipeline. The competition requires you to understand and reproduce the method yourself.

## Explain the path from image to CSV

1. Read the JPEG as a single grayscale channel and scale intensities to [0,1].
2. Estimate the bright solar disk with Otsu thresholding, retain its largest connected region, fill holes, and erode the limb. This excludes most background and limb artifacts.
3. Smooth at sigma=1, estimate broad brightness with a larger Gaussian, and compute `(background - image) / max(background, 0.08)`. Dark features have positive deficits.
4. Threshold the deficit inside the disk. Optionally close small gaps with dilation followed by erosion.
5. Label eight-connected components and retain components within the selected area limits. Each retained component is a predicted filament instance.
6. Encode each full-resolution binary mask using COCO compressed RLE in column-major (Fortran) memory order. Write the source image ID plus a unique suffix, and the ASCII RLE counts, into the CSV.

## Explain how parameters were chosen

A reproducible manifest splits observations by capture date within each year. Independent annotation sets from the same image cannot cross partitions. Two documented grids totaling 96 configurations are scored on 111 development images. The highest pooled PQ chooses the configuration; 139 held-out images are then evaluated once for this development decision. The executed notebook can reproduce this evaluation without re-tuning.

The saved parameters alone are sufficient to run inference. The remaining development images are not used to fit this baseline. No external annotations, spines, chirality labels, or hidden test labels are used.

## Explain the score

A pair is a match only if its intersection-over-union is strictly greater than 0.5. Add matched IoUs across all annotator–image records and divide by TP + 0.5 FP + 0.5 FN. The implementation follows the organizers' notebook exactly; it does not swap in a different matching rule. Multiple independent annotations for one image are evaluated separately against the same prediction.

The bootstrap resamples capture dates, not individual annotations. This avoids treating several labels on one observation as independent uncertainty samples. It does not guarantee performance on unseen observatories, years, or the hidden leaderboard.

## Be ready to discuss limitations

- Dark sunspots, artifacts, and background structures can become false positives.
- Low-contrast and small filaments can disappear under thresholding or size filtering.
- Closing can merge neighbors; gaps can split a filament.
- Disk estimation and limb erosion can exclude real structures.
- Gaussian scale is fixed, so broad and narrow structures are not handled equally.
- This is an interpretable baseline, not evidence of state-of-the-art performance.

## Practice before final submission

Run one image through `core.contrast_maps` and `core.predict_from_maps`. Explain each saved parameter. Change a threshold on a development image and observe its effect without retuning against the held-out set. Decode one submission RLE and compare it with the component mask. Reproduce the reported PQ from the saved validation rows. Review the worst validation overlays and explain why they fail.
