# Acoustic Reflector Based Forearm Ultrasound for Hand Gesture Classification — Code

![graphical_abstract](https://github.com/user-attachments/assets/f06ce2c3-a7fc-49da-8679-b959789bc728)

This repository contains reference implementations and runner scripts accompanying our IEEE Sensors Journal article on acoustic-reflector–assisted forearm ultrasound for hand-gesture classification.  

If you use this code (or any derived artifacts), **please cite both the paper and the dataset**.

## Paper citation

**Plain text**  
`K. Bimbraw, Y. Tang, and H. K. Zhang, "Acoustic Reflector Based Forearm Ultrasound for Hand Gesture Classification," IEEE Sensors Journal, 2025, doi: 10.1109/JSEN.2025.3621577.`

**BibTeX**
```bibtex
@ARTICLE{11208539,
  author  = {Bimbraw, Keshav and Tang, Yichuan and Zhang, Haichong K.},
  journal = {IEEE Sensors Journal},
  title   = {Acoustic Reflector Based Forearm Ultrasound for Hand Gesture Classification},
  year    = {2025},
  volume  = {},
  number  = {},
  pages   = {1--1},
  doi     = {10.1109/JSEN.2025.3621577}
}
```

## Dataset citation

If you use the dataset at https://zenodo.org/records/17386583, please cite it in addition to the code and paper:

**Plain text**  
`Bimbraw, K., Tang, Y., & Zhang, H. K. (2025). Acoustic Reflector Based Forearm Ultrasound for Hand Gesture Classification — Dataset (v1.0). Zenodo. https://doi.org/10.5281/zenodo.17386583`

**BibTeX**
```bibtex
@dataset{bimbraw2025_ultrasound_dataset,
  author    = {Bimbraw, Keshav and Tang, Yichuan and Zhang, Haichong K.},
  title     = {Acoustic Reflector Based Forearm Ultrasound for Hand Gesture Classification — Dataset},
  year      = {2025},
  version   = {v1.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.17386583}
}
```

## Related / precursor work

You may also cite the precursor SPIE paper:

**Plain text**  
`Bimbraw, K., & Zhang, H. K. (2024, April). Mirror-based ultrasound system for hand gesture classification through convolutional neural network and vision transformer. In *Medical Imaging 2024: Ultrasonic Imaging and Tomography* (Vol. 12932, pp. 218–222). SPIE.`

**BibTeX**
```bibtex
@inproceedings{bimbraw2024mirror,
  title     = {Mirror-based ultrasound system for hand gesture classification through convolutional neural network and vision transformer},
  author    = {Bimbraw, Keshav and Zhang, Haichong K},
  booktitle = {Medical Imaging 2024: Ultrasonic Imaging and Tomography},
  volume    = {12932},
  pages     = {218--222},
  year      = {2024},
  organization = {SPIE}
}
```
## Installation (Windows / PowerShell, Python 3.10)
```
# 1) Verify Python 3.10
py -3.10 -V

# 2) Create and activate a virtual environment
py -3.10 -m venv .venv
# If activation is blocked, bypass policy for THIS PowerShell session:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\activate

# 3) (Optional) Upgrade pip
python -m pip install --upgrade pip

# 4) Install the exact, known-good dependencies (you may also use the requirements.txt through pip install -r requirements.txt)
#    Pinning avoids the NumPy 2.x / TensorFlow 2.11.x incompatibility.
pip install --upgrade --force-reinstall "numpy==1.24.3" "tensorflow==2.12.1" "scikit-learn==1.7.2" "matplotlib==3.10.7"
```

### Troubleshooting
- If you previously installed packages that pulled in numpy>=2, you may see: “A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x …”. Fix by re-running the pinned install command above (step 4).
- Keep one-liner commands on a single line in PowerShell (avoid using ^ line continuations).

The script defaults to this root
```
C:\Users\bimbr\Documents\Mirror_Paper\Data_Upload\
├── mirror\Subject_1\{X_m_train.npy, X_m_test.npy, y_m_train.npy, y_m_test.npy}
└── perp\Subject_1\{X_m_train.npy, X_m_test.npy, y_m_train.npy, y_m_test.npy}
```
Override with `--root <path>` and set it to the folder where you've downloaded the files from Zenodo (https://zenodo.org/records/17386583).

## Running the baselines (Subject_1)

All runners share the same CLI style and output artifacts:
- `--save-model` saves the trained model
- `--out` saves a JSON with metrics
- `--cm` saves a confusion-matrix PNG
- `--mode` is `mirror` or `perp`

**Commands (single block, full lines, no line continuations):**
```powershell
# SVM (file: ultrasound_gesture_svc_classification.py) — Linear
python ultrasound_gesture_svc_classification.py --mode mirror --subject Subject_1 --kernel linear --C 1.0 --max-iter 2000 --save-model results\svc_linear_mirror_subject1.joblib --out results\subject1_svc_linear_mirror.json --cm results\figs\subject1_svc_linear_mirror_cm.png

# SVM (file: ultrasound_gesture_svc_classification.py) — RBF
python ultrasound_gesture_svc_classification.py --mode mirror --subject Subject_1 --kernel rbf --C 10 --gamma scale --max-iter 2000 --save-model results\svc_rbf_mirror_subject1.joblib --out results\subject1_svc_rbf_mirror.json --cm results\figs\subject1_svc_rbf_mirror_cm.png

# CNN (file: ultrasound_gesture_cnn_classification.py)
python ultrasound_gesture_cnn_classification.py --mode mirror --subject Subject_1 --epochs 5 --batch-size 64 --save-model results\cnn_mirror_subject1.keras --out results\subject1_cnn_mirror.json --cm results\figs\subject1_cnn_mirror_cm.png

# ViT (file: ultrasound_gesture_vit_classification.py)
python ultrasound_gesture_vit_classification.py --mode mirror --subject Subject_1 --epochs 5 --batch-size 64 --save-model results\vit_mirror_subject1.keras --out results\subject1_vit_mirror.json --cm results\figs\subject1_vit_mirror_cm.png
```
## Notes
- For faithful reproduction, match the data splits and hyperparameters reported in the paper. The runner scripts ship with sensible defaults aligned to the study.
- Training time varies by hardware. ViT typically needs multiple epochs; SVM (linear/RBF) may require many iterations. Expect anything from minutes to hours depending on CPU/GPU.
- ViT benefits from a GPU; SVM runs on CPU. Results are comparable across systems, but wall-clock time will differ.
- Full computation details (preprocessing, model settings, and evaluation protocol) are documented in the paper.
- The example commands were generated and validated on the machine described in “Tested system” below; performance may vary on other setups.

### Tested system

The commands and results in this repo were verified on the following machine. Performance may vary on other systems.

- Device: Lenovo Legion Slim 5 16APH9 (“Keshav”)
- OS: Windows 11, 64-bit (x64)
- CPU: AMD Ryzen 7 8845HS w/ Radeon 780M Graphics (3.80 GHz)
- GPU: NVIDIA GeForce RTX 4070 Laptop GPU (8 GB)
- RAM: 64.0 GB (63.3 GB usable)
- Storage: 954 GB (746 GB free at test time)
- Python: 3.10.11 (venv)
- Key libs: numpy==1.24.3, tensorflow==2.12.1 (CPU build: `tensorflow-intel`), scikit-learn==1.7.2, matplotlib==3.10.7

> Note: Runs above used the CPU build of TensorFlow on Windows. If you enable CUDA/cuDNN GPU acceleration (not required), training speeds will differ.
