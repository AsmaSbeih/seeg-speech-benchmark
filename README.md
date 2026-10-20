# sEEG-to-speech spectrogram reconstruction: a controlled benchmark

Code and result files for the manuscript **"A Controlled Benchmark for sEEG-to-Speech Spectrogram Reconstruction with Template and Noise-Input Controls"** (A. Sbaih, J. García-Gutiérrez, D. Benavides; University of Seville).

Five systems (CNN-BiLSTM baseline, Hybrid with multi-receptive residual blocks, Verwoert-style linear decoder, and our re-implementations of NeuroIncept and ConvED-SR) are compared on the public SingleWordProductionDutch-iBIDS dataset under one protocol, with a time-in-trial template, noise-input networks and a constant-spectrum control.

* **Stage A**: one pooled model for all 10 participants (test blocks 3 and 8, validation blocks 1 and 6; seeds 42, 123, 456).
* **Stage B**: one model per participant, 10 contiguous folds (seed 42).

## Data (not included)
Download the dataset from OSF: https://doi.org/10.17605/OSF.IO/NRGX6 (Verwoert et al., Sci. Data 2022, https://doi.org/10.1038/s41597-022-01542-9). The raw recordings, the extracted feature arrays (`*_feat.npy`, `*_spec.npy`) and the saved predictions (`*_preds.npz`) are not in this repository.

## Vocoder
The audio metrics use the pre-trained SpeechBrain HiFi-GAN checkpoint `speechbrain/tts-hifigan-ljspeech` (22.05 kHz, 80 mel bins, n_fft 1024, hop 256, fmax 8000), downloaded on first use.

## Layout
```
code/        all Python scripts (flat: they import each other, keep them in one folder)
notebooks/   the notebooks used to run and inspect the experiments (audio players removed from the saved outputs)
results/     result files behind every table and figure of the paper, and make_numbers.py
```

## Environment
Python 3, with `pip install -r requirements.txt`. Set two environment variables before running the scripts:
```
SEEG_DATA_ROOT   folder with the downloaded dataset
SEEG_OUT_DIR     folder for extracted features and results
```
`cell_stageA_constant_control.py`, `export_for_paper.py` and `cell_verify_numbers.py` contain a default path `C:\seeg_v2`; edit it to your own folder.

## Order of execution (run from the `code/` folder)
1. **Features** (HG envelope 70-170 Hz, 80-bin log-mel on the HiFi-GAN grid):
   `python extract_continuous_80.py hifigan` (self-test: `python extract_continuous_80.py selftest`); checks: `python validate_extraction.py`.
2. **Stage A** (pooled model; learning-rate search and 3 seeds):
   `python stageA_pooled.py run baseline` / `hybrid` / `linear`, and the noise-input controls `baseline_null` / `hybrid_null`;
   `python stageA_extra.py run neuroincept` / `convedsr` / `neuroincept_null` / `convedsr_null`.
3. **Stage B** (per participant, 10 folds): `python stageB_verwoert_linear.py`, then `python stageB_subject.py run <linear|hybrid|baseline|neuroincept|convedsr>`.
   NeuroIncept uses the declared learning-rate override: set `STAGEB_LR_NEUROINCEPT=1e-4`.
4. **Template and residual PCC**: computed inside the scripts above with `template_floor_continuous.py` (time since the last word onset, up to 300 frames).
5. **Audio metrics**: `python stageA_stoi.py` (Stage A STOI/eSTOI), `python quality_metrics.py mcd`, `python quality_metrics.py stoiB folds=0` (Stage B STOI/eSTOI, fold 1 of the paper), and the constant-spectrum control `cell_stageA_constant_control.py`.
6. **Shift check** (preliminary, three participants): `python shift_scan.py` writes `shift_scan.csv`.
7. **Figures**: `fig_eval_matrix.py`, `fig_model_structures.py`, `fig_recon_pair.py`, `fig_word_variability.py`, `fig_graphical_abstract.py`.
8. **Numbers of the paper**: from `results/`, run `python make_numbers.py`. It reads the result files in that folder and regenerates `numbers.csv`, `numbers.tex` and the table bodies. `code/cell_verify_numbers.py` recomputes the headline numbers from the raw result files.

## Notes
* Fold numbering: the code is 0-indexed (Stage B fold k tests block k); the paper numbers blocks and folds from 1. Stage A test blocks (2, 7) in the code are blocks 3 and 8 in the paper.
* Stage B STOI, eSTOI and the constant-spectrum control use fold 1 of each participant; MCD uses all folds.
* NeuroIncept and ConvED-SR are our re-implementations from the published descriptions; their published numbers are not reproduced.

## Citation
See `CITATION.cff`.
