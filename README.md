# ML Mini Project: COVID-19 CT Triage (Reproduce and Extend)

This project reproduces and extends the Stanford CS229 project **"CT-based Patient Triage of COVID-19: Radiomics Prediction of ICU Admission, Mechanical Ventilation, and Death of Patients"** by Zhan and Li (2020).

The goal is to predict three outcomes for a COVID-19 patient when they arrive at hospital:

1. **ICU admission**
2. **Mechanical ventilation (MV)**
3. **Death**

using features from their chest CT scan (radiomics), clinical information and blood tests.

| Part | What it does | 
|---|---|
| Part 1: Reproduce | Rebuilds the paper's pipeline on a public dataset 
| Part 2: Extend | Adds XGBoost, a model the paper did not use, and compares it fairly 

## Dataset

The paper used private hospital data from China, and its code repository only contains example files. So we used the closest public dataset:

**Stony Brook University COVID-19 Positive Cases (COVID-19-NY-SBU)**, The Cancer Imaging Archive (TCIA).
Link: https://wiki.cancerimagingarchive.net/x/kAKtBQ

It has chest CT scans plus clinical records with all three outcomes.

| Step | Patients |
|---|---|
| Patients in the dataset | 1,384 |
| Patients with at least one CT scan | 458 |
| Patients with a thin lung CT scan (final cohort) | **286** |

The 286 patients were split once and the same split is used in both parts:

| Group | Patients | ICU | MV | Death |
|---|---|---|---|---|
| Train | 140 | 39 | 35 | 25 |
| Validation | 60 | 20 | 17 | 11 |
| Test | 86 | 25 | 21 | 15 |
| **Total** | **286** | **84** | **73** | **51** |

Raw scans and clinical files are not stored in this repository. Download them from TCIA using the link above.

## Repository structure

```
├── README.md
├── LICENSE
├── Data/
│   ├── features.csv            final table: 286 patients, features and outcomes
│   ├── split.csv               which patients are in train, validation and test
│   ├── boot_idx.npy            30 bootstrap resamples of the test set
│   └── test_predictions.csv    Part 1 model predictions on the test set
├── ML_Part1_Reproduce/
│   ├── ML_Part1_Reproduce.ipynb
│   └── Results/                Part 1 tables and figures
└── ML_Part2_Extend/
    ├── ML_min_part2.ipynb
    └── Results/                Part 2 tables and figures
```

## Part 1: Reproduce

**Pipeline**

1. Picked one thin lung CT scan per patient (2.5 mm slices or thinner), taking the earliest scan.
2. Found the lungs in each scan with **lungmask** and measured **107 radiomics features** (shape, intensity and texture) with **PyRadiomics**.
3. Joined these with clinical data (age, sex, illnesses, symptoms, vital signs) and blood tests. Information from later in the hospital stay was left out.
4. Built the paper's four feature groups:
   * **Radiom**: CT features only
   * **RadioClin**: CT + clinical
   * **RadioClinLab**: CT + clinical + blood tests
   * **ClinLab**: clinical + blood tests
5. Trained the paper's five models (**Logistic Regression, Random Forest, SVM, MLP, LightGBM**) with four data preparation options (none, **SMOTEENN**, SMOTEENN + **LASSO**, SMOTEENN + **PCA**), tuned with 5 fold cross validation.
6. Picked the best model for each outcome and feature group using the validation set, retrained it on train + validation, and tested it once.
7. Ran 30 bootstrap resamples of the test set with paired t tests, and found the top 10 features with permutation importance.

**Test results (AUROC)**

| Feature group | ICU | MV | Death |
|---|---|---|---|
| Radiom | 0.631 | 0.730 | 0.623 |
| RadioClin | 0.645 | 0.649 | 0.766 |
| RadioClinLab | 0.676 | **0.760** | **0.800** |
| ClinLab | **0.842** | 0.688 | 0.774 |
| *Paper (best)* | *0.91* | *0.92* | *0.83* |

**Key findings**

* Like the paper, combining all data (RadioClinLab) worked best for ventilation and death.
* Like the paper, adding clinical data to CT features clearly helped for ICU and death (p < 0.01).
* Unlike the paper, clinical + blood tests alone (ClinLab) was best for ICU.
* Top risk factors included low oxygen saturation, lactate, age, breathlessness, neutrophils and D dimer, which agree with the paper's findings.
* Scores are lower than the paper's, mainly because of the much smaller dataset (286 vs 2,362 patients).

## Part 2: Extend with XGBoost

**Why XGBoost:** it is a strong model for small tabular datasets, has built in regularisation to limit overfitting, and grows trees differently from LightGBM (level by level instead of leaf by leaf).

**Method:** same data, same split and same metrics as Part 1. Four XGBoost settings were tried for each outcome and feature group, the best was picked on the validation set, then retrained on train + validation and tested once. Both models were then compared on the same 30 bootstrap resamples.

**Comparison with the best Part 1 model for each outcome**

| Outcome | Feature group | Part 1 AUROC | XGBoost AUROC | Difference (95% interval) | Result |
|---|---|---|---|---|---|
| ICU | ClinLab | 0.842 | 0.741 | −0.109 (−0.173 to −0.062) | **Worse** |
| MV | RadioClinLab | 0.760 | 0.653 | −0.122 (−0.220 to −0.018) | **Worse** |
| Death | RadioClinLab | 0.800 | 0.786 | −0.007 (−0.110 to +0.128) | **Similar** |

AUROC values are on the test set. The difference column is the bootstrap average, and a result is called worse only when its interval does not include 0.

**Why XGBoost did not do better**

* The training set is very small (140 patients), and boosting usually needs more data.
* No class balancing was used, so the model mostly predicts "no" and misses many positive patients (recall 0.07 to 0.40). Its accuracy looks high only because most patients are negative.
* Part 1 chose from 20 model and preparation combinations for each case, while XGBoost had 4 settings.

## Main differences from the paper

* Public US dataset (286 patients) instead of private Chinese hospital data (2,362 patients)
* Random split instead of a split by admission date
* Only original radiomics features (107), without wavelet features, to keep extraction time manageable
* No radiologist score, since radiologist reads are not available
* Age only available in groups, and the blood test LDH is missing from this dataset
* Permutation importance instead of each model's built in importance

## How to run

Both notebooks run in **Google Colab** (CPU is enough, except the CT feature extraction in Part 1, which needs a GPU).

* **Part 1:** to skip the slow CT step, upload `Data/features.csv` and `Data/split.csv` and start from Step 12. Running the full extraction takes several hours.
* **Part 2:** upload the four files in `Data/` and `ML_Part1_Reproduce/Results/test_results.csv`, then run all cells.

All random seeds are set to 42, so results can be reproduced.

## References

* Zhan and Li (2020). *CT-based Patient Triage of COVID-19: Radiomics Prediction of ICU Admission, Mechanical Ventilation, and Death of Patients.* Stanford CS229.
* Original code: https://github.com/terryli710/COVID_19_Rapid_Triage_Risk_Predictor
* *Stony Brook University COVID-19 Positive Cases (COVID-19-NY-SBU)* [Data set]. The Cancer Imaging Archive. https://wiki.cancerimagingarchive.net/x/kAKtBQ

## Note

This is an academic project only. The models are not suitable for real clinical use.

The code is released under the MIT License. The dataset belongs to its original authors and is used under the terms set on its TCIA page.
