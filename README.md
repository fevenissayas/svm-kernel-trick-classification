# SVM & the Kernel Trick

Binary classification on `make_moons` comparing a linear SVM (baseline) with polynomial- and RBF-kernel
SVMs, plus a few small experiments that show how the kernel trick works.

## Files
| File | What it is |
|---|---|
| `svm_kernel_trick.ipynb` | Main notebook: data, models, tuning, comparison, kernel-trick demos |
| `svm_utils.py` | Reusable helpers (pipeline, grid search, metrics, plotting) |
| `SVM_Kernel_Trick_Report.docx` | 2-page report |
| `figures/` | Figures saved by the notebook |

## Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook svm_kernel_trick.ipynb   # Kernel -> Restart & Run All (~20 s)
```

## Results (test set, 150 points)
| Model | Best params (5-fold CV) | CV bal. acc. | Test acc. | F1 |
|---|---|---|---|---|
| Linear | C=0.1 | 0.862 | 0.860 | 0.863 |
| Polynomial | degree=5, C=100, coef0=1 | 0.936 | 0.947 | 0.947 |
| **RBF (selected)** | gamma=2, C=10 | **0.942** | 0.940 | 0.940 |

The scaler sits inside a `Pipeline`, so it is only fitted on training folds (no leakage).
I picked the model by CV score, not by test score.
