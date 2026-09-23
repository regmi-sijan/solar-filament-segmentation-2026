# Attribution and license boundaries

The MIT license applies to original project source code. It does not relicense competition data, the organizers' report template, the official evaluation notebook, or third-party software.

- MAGFiLO competition images and labels: competition data license CC BY-NC 4.0; not bundled in the source release. Cite Ahmadzadeh et al., 2024, DOI 10.1038/s41597-024-03876-y. Example figures derived from the data remain subject to its terms. The report preserves the organizers' GONG and NSF acknowledgments.
- Official self-evaluation notebook: Azim Ahmadzadeh, https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook (downloaded version 6). Retained locally as a reference; excluded from the public source package until its redistribution license is verified. The optional parity test skips if this reference notebook is absent. It is not required for inference or calibration.
- Report layout and fixed text: organizers' supplied template, https://www.overleaf.com/read/vjztpvrnbrdh#43fbbe. Provided for this competition. Use of acmart does not imply ACM publication, sponsorship, or endorsement. Do not assume suitability for another venue.
- NumPy, SciPy, pandas, Matplotlib, Pillow, OpenCV, pycocotools, PyTorch, and Jupyter components retain their respective upstream licenses. Pinned versions are recorded in requirements.txt and requirements-lock.txt.
