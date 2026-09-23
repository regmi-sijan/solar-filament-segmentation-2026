# Competition deliverables

Checked September 23, 2026 against the competition Overview, Rules, official evaluation notebook, report template, and final Google Form.

- Account registered and competition rules accepted: verified in Kaggle UI.
- Data: 707 train and 180 test images; archive and decoded images verified.
- Approach: image-only processing; calibration uses supplied training segmentation polygons only.
- Reproduction: versioned requirements.txt, source modules, calibration configuration, saved date-grouped split, complete pipeline notebook, tests, and commands.
- Evaluation: official-notebook-compatible micro PQ at strict IoU > 0.5; overlapping-pair IoU/Dice distributions; fragmentation and merging counts; grouped uncertainty interval.
- Prediction: one row per predicted filament; filament_id and segmentation_rle; 2048 × 2048 COCO compressed RLE; all 180 test images processed.
- Report: supplied acmart/sigconf template, fixed subtitle/introduction/acknowledgment preserved; method pipeline graphic and segmentation examples; four-page limit excluding acknowledgments/references.
- Publication: publicly accessible Git repository required. Published at https://github.com/regmi-sijan/solar-filament-segmentation-2026. Do not redistribute competition images or labels with source.
- Kaggle: at most five submissions per day; up to two final selections. Verified Complete in the Kaggle UI, public score 0.15; selected for final score on September 23, 2026.
- Final form: email, team names/emails, countries, public Git URL, exact Kaggle submission CSV (max 10 MB), and report PDF (max 100 MB). Institution is optional.
- Participant understanding: rules allow AI assistance but require the participant to explain, justify, and reproduce the method.

External steps are not complete merely because files exist locally. Save Kaggle submission ID/score, public repository URL, and final form confirmation after those actions occur.

Sources:
- https://www.kaggle.com/competitions/filament-segmentation-2026
- https://www.kaggle.com/competitions/filament-segmentation-2026/rules
- https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook
- https://www.overleaf.com/read/vjztpvrnbrdh#43fbbe
- https://docs.google.com/forms/d/e/1FAIpQLSfyrM4dhc8QzxR90PtF6fb018KPf0S0Cbf5hPqMuUognmzQsA/viewform
