# Acoustic Reflector Based Forearm Ultrasound for Hand Gesture Classification — Code

If you are using this repo, please cite it along with the dataset and the associated publication when used in academic or industrial research. Here is the plain text citation:
`K. Bimbraw, Y. Tang and H. K. Zhang, "Acoustic Reflector Based Forearm Ultrasound for Hand Gesture Classification," in IEEE Sensors Journal, doi: 10.1109/JSEN.2025.3621577.`

Here is the BibTeX:
```
@ARTICLE{11208539,
  author={Bimbraw, Keshav and Tang, Yichuan and Zhang, Haichong K.},
  journal={IEEE Sensors Journal}, 
  title={Acoustic Reflector Based Forearm Ultrasound for Hand Gesture Classification}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Ultrasonic imaging;Probes;Hands;Thumb;Acoustics;Indexes;Transducers;Wrist;Accuracy;Data acquisition;AI-driven signal processing;assistive technology;gesture recognition;machine learning for sensor data;wearable sensors},
  doi={10.1109/JSEN.2025.3621577}}
```
You may also cite 'Mirror-based ultrasound system for hand gesture classification through convolutional neural network and vision transformer' which is a precursor to this work. Here is the plain text citation: `Bimbraw, K., & Zhang, H. K. (2024, April). Mirror-based ultrasound system for hand gesture classification through convolutional neural network and vision transformer. In Medical Imaging 2024: Ultrasonic Imaging and Tomography (Vol. 12932, pp. 218-222). SPIE.`

And here is the bibtex:
```
@inproceedings{bimbraw2024mirror,
  title={Mirror-based ultrasound system for hand gesture classification through convolutional neural network and vision transformer},
  author={Bimbraw, Keshav and Zhang, Haichong K},
  booktitle={Medical Imaging 2024: Ultrasonic Imaging and Tomography},
  volume={12932},
  pages={218--222},
  year={2024},
  organization={SPIE}}
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
• If you previously installed packages that pulled in numpy>=2, you may see: “A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x …”. Fix by re-running the pinned install command above (step 4).
• Keep one-liner commands on a single line in PowerShell (avoid using ^ line continuations).

The script defaults to this root
```
C:\Users\bimbr\Documents\Mirror_Paper\Data_Upload\
├── mirror\Subject_1\{X_m_train.npy, X_m_test.npy, y_m_train.npy, y_m_test.npy}
└── perp\Subject_1\{X_m_train.npy, X_m_test.npy, y_m_train.npy, y_m_test.npy}
```
Override with `--root <path>` as needed.

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

