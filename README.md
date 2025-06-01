# Geology Forecast Challenge — Solution by Nikita Shevyrev

This repository contains the complete solution for the [Kaggle Geology Forecast Challenge](https://www.kaggle.com/competitions/geology-forecast-challenge).

The solution combines:
- 📈 A powerful LSTM + Multi-head Attention architecture
- 🧠 A parallel **learnable attention projection** block
- 🧪 Synthetic training data generation
- 🔁 5-fold ensembling with 10 realization predictions per sample

---

## 📂 Repository Structure
GEOLOGY-FORECAST-CHALLENGE/
├── augmentations/
│   ├── __init__.py
│   └── mixup.py                  ← Mixup data augmentation
│
├── data/
│   ├── __init__.py
│   ├── data_generator.py         ← Data generation function
│   └── geology_dataset.py        ← Custom PyTorch Dataset
│
├── metrics/
│   ├── __init__.py
│   └── evaluation.py             ← Metric computation (e.g. NLL)
│
├── model/
│   ├── __init__.py
│   └── lstm_with_attention.py    ← Final model architecture
│
├── training/
│   ├── __init__.py
│   └── loop.py                   ← Training and validation functions
│
├── utils/
│   ├── __init__.py
│   └── logging.py                ← Weights & Biases setup
│
├── geologyforecast.ipynb         ← The final competition notebook
├── inference.py                  ← Script to generate submission.csv
├── LICENSE                       ← License file
├── README.md                     ← README file
├── requirements.txt              ← Python dependencies
└── train.py                      ← Training pipeline with data preparation

---

## ⚙️ Environment Setup

You can install required dependencies with:
```bash
pip install -r requirements.txt
```

Tested on: Python 3.10, PyTorch 1.13+

---

## 🏋️‍♂️ Training and Synthetic Data Generation

Steps:

1. Ensure your `input/` folder contains:
```
train.csv, sample_submission.csv, test.csv
```

2. Run the training script (set `RUN_DATA_GENERATION = False` if you’ve already generated data):
```bash
python train.py
```

3. Folded weights will be saved to:
```
weights/model_fold_0.pt ... model_fold_4.pt
```

Expected Folder Layout to run `train.py`:
GEOLOGY-FORECAST-CHALLENGE/
├── train.py
├── weights/
│   ├── model_fold_0.pt
│   ├── ...
├── input/
│   └── geology-forecast-challenge-open/
│       └── data/
│           ├── train.csv
│           ├── test.csv
│           ├── sample_submission.csv
│           └── train_raw/

---

## 📈 Inference

Steps:

1. Place the official test data in:
```
input/geology-forecast-challenge-open/data/test.csv
input/geology-forecast-challenge-open/data/sample_submission.csv
```

2. Place the 5 trained model weights in:
weights/
├── model_fold_0.pt
├── ...
└── model_fold_4.pt

3. Run Inference:
```bash
python inference.py
```

- This script performs ensemble inference across 5 folds × 10 realizations per sample.
- The resulting `submission.csv` is saved in the root folder and can be submitted to Kaggle.
- **Note:** This script does not require training. It loads only the provided weights and test files.

---

## 📄 Reproducibility & Competition Compliance
This repository contains:

✅ Full code for data preparation, model, training, and inference

✅ Scripts to train from scratch or reproduce predictions using saved weights

✅ A Kaggle notebook for competition submission

---

## 🔁 Note on Reproducibility & Model Weights
Due to small differences in numerical behavior across hardware (e.g., Kaggle’s P100 vs my local GPU) and library versions, I was **unable to reproduce identical weights across environments**. This can affect the final score even with the same code and seed.

For that reason:
- 🔒 I trained the final model **locally** using my own GPU (RTX 3080 Ti)
- 📦 The resulting weights are saved in a **private Kaggle dataset**, which is loaded in the final notebook to ensure accurate reproduction of `submission.csv`
- ✅ The notebook contains **full training code** (commented) in accordance with competition requirements

If anyone needs access to the weights for review or reproduction, feel free to:
- Open an issue here, or
- Contact me via Kaggle

---

## 🙏 Acknowledgements & Inspiration
Several Kaggle community members published excellent notebooks that inspired key aspects of this solution:

1. **Eva Koroleva**
- 🧠 Idea: Using **LSTM** as a base model
- 📓 Notebook: [LSTM KFold](https://www.kaggle.com/code/qmarva/lstm-kfold)
- 👤 Profile: [@qmarva](https://www.kaggle.com/qmarva)

2. **OlSch**
- 🧠 Idea: Using **skip connections (residual blocks)** inside neural network stacks
- 📓 Notebook: [Keras Geology Forecast Competition 58.4](https://www.kaggle.com/code/olasdsd/keras-geology-forecast-competition-58-4)
- 👤 Profile: [@olasdsd](https://www.kaggle.com/olasdsd)
- 💡 This structure (shown below) led me to introduce **residual LSTM stacks**, which significantly improved my model performance.
```python
# OlSch's solution (Keras) with a skip connection block
def initModel(num_features): 
    num_input = Input(shape=(num_features,), dtype="float32", name="num_input")
    x = Dense(UNITS, activation='linear')(num_input)
    for i in range(DEPT):
        skip = x
        x = Dense(UNITS, activation=ACTIVATION)(x)
        x = Dropout(DROPOUT)(x)
        x = x + skip
    y_pred = Dense(3000, activation='linear')(x)
        
    model = tf.keras.Model(
        inputs = [num_input],
        outputs = [y_pred]
    )
            
    opt = tf.keras.optimizers.Adam(learning_rate = 1e-4)
    model.compile(loss='mse', optimizer = opt)

    return model
```

Thanks to the Kaggle community for openly sharing ideas — it allowed me to iterate faster and build a stronger final solution.

---

## 📬 Contact
Feel free to open an issue or message me on Kaggle for questions or clarification.
