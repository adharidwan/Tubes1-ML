# Tugas Besar 1 IF3270 Pembelajaran Mesin
## Feedforward Neural Network (FFNN) from Scratch

Implementasi Feedforward Neural Network (FFNN) dari nol menggunakan Python dan NumPy, untuk memenuhi spesifikasi Tugas Besar 1 IF3270 Pembelajaran Mesin 2025/2026.

---

## Struktur Repository

```
├── README.md
├── data/
│   └── datasetml_2026.csv 
├── doc/
│   └── laporan.pdf                                
└── src/
```

---

## Setup & Cara Menjalankan

### 1. Clone repository

```bash
git clone https://github.com/adharidwan/Tubes1-ML.git
cd Tubes1-ML
```

### 2. Buat virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Dependensi

- Python 3.8+
- NumPy
- Matplotlib
- scikit-learn (untuk preprocessing dan perbandingan)
- pandas
- tqdm
- jupyter

---

## Pembagian Tugas

| Nama | NIM | Tugas |
|------|-----|-------|
| Muhammad Izzat Jundy | 13523092 | implementasi loss (Cross Entropy, MSE), backpropagation (chain rule), turunan fungsi aktivasi, optimizer gradient descent, dan batch processing. |
| Zulfaqqar Nayaka Athadiansyah | 13523094 | preprocessing data (missing values, one-hot encoding, feature scaling), K-Fold cross validation, metrik evaluasi (accuracy, precision, recall, F1-score), eksperimen pembanding dengan Scikit-Learn MLPClassifier, serta laporan dan visualisasi. |
| Muhammad Adha Ridwan | 13523098 | desain kelas FFNN dinamis, forward propagation, implementasi fungsi aktivasi forward (Linear, ReLU, Sigmoid, tanh, Softmax), inisialisasi bobot/bias (Xavier/He), Automatic Differentiation (Autograd). |

---

