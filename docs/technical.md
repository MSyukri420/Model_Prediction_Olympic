← [Back to README](../README.md)

# Olympic Medal Prediction: technical details

## How it works

Each row is one athlete entered in one event at one Games. The label is whether that entry won any medal (gold, silver or bronze).

| Feature | Meaning |
|---|---|
| Age, Height, Weight | As recorded (years, cm, kg) |
| BMI | Weight / (height in m)² |
| PriorMedals | Medals the athlete won at earlier Games (strictly earlier years, so nothing from the Games being predicted) |
| Event_encoded | The event (765 in total), as an integer code; one-hot encoded for logistic regression |
| Sex_encoded | F = 0, M = 1 |

1. **Load and clean** ([`common.py`](../common.py)): drop entries missing age, height or weight; compute BMI and PriorMedals.
2. **Split by time**: train on 1896–2014 (192,722 entries), test on Rio 2016 (13,443 entries).
3. **Resample inside each model's pipeline**:
   - SMOTENC adds synthetic medal entries, taking event and sex from a real neighbour rather than averaging the codes.
   - Random undersampling drops non-medal entries until the classes are even.
   - Either way, resampling only touches the data a model is fitted on.
4. **Tune** ([`tune.py`](../tune.py)): randomized search, 10 candidates × 3 folds per model, scored by average precision, on training data only.
5. **Pick a cutoff**: each model picks the probability cutoff that maximises F1 on a held-out 20% of the training data, then refits on all of it.
6. **Models**:
   - logistic regression (scaled numbers, one-hot event and sex)
   - random forest
   - XGBoost
   - a stacked ensemble of the three, with a logistic-regression meta-model

   In the ensemble, each base model resamples inside its own pipeline, so the meta-model learns from out-of-fold predictions on real entries.
7. **Explain**: SHAP on the XGBoost (SMOTE) model.

## Evaluation

Rio 2016: 13,443 entries, of which 15.0% won a medal.

| Model | Cutoff | Precision | Recall | F1 | ROC AUC | PR AUC | Accuracy |
|---|---|---|---|---|---|---|---|
| Baseline (always no medal) | – | 0.000 | 0.000 | 0.000 | 0.500 | 0.150 | 0.850 |
| LogReg (SMOTE) | 0.622 | 0.285 | 0.575 | 0.381 | 0.729 | 0.336 | 0.720 |
| RF (SMOTE) | 0.534 | 0.275 | 0.554 | 0.368 | 0.705 | 0.313 | 0.715 |
| XGB (SMOTE) | 0.421 | 0.269 | 0.595 | 0.370 | 0.720 | 0.328 | 0.697 |
| Stacking (SMOTE) | 0.208 | 0.289 | 0.575 | 0.385 | 0.740 | 0.347 | 0.724 |
| LogReg (Undersampling) | 0.593 | 0.294 | 0.596 | 0.394 | 0.746 | 0.348 | 0.725 |
| RF (Undersampling) | 0.547 | 0.287 | 0.562 | 0.379 | 0.722 | 0.338 | 0.725 |
| XGB (Undersampling) | 0.592 | 0.307 | 0.560 | 0.396 | 0.738 | 0.343 | 0.744 |
| **Stacking (Undersampling)** | 0.195 | 0.291 | 0.627 | **0.398** | **0.753** | **0.359** | 0.716 |

The baseline predicts that nobody wins. Its ROC AUC of 0.5 is a coin flip, and its PR AUC equals the 15% base rate.

How to read this:

- **Ignore accuracy.** The baseline beats every model on accuracy because 85% of entries don't win a medal. F1, ROC AUC and PR AUC show whether a model actually finds medallists.
- **Precision of about 0.29 is roughly twice the 15% base rate.** A predicted medal is twice as likely to be real as a random guess, and the models find 55–63% of medallists.
- **Undersampling matches or beats SMOTE** on F1 and ROC AUC for every model, while training on about a sixth as many rows.
- **The model hardly matters.** Stacking with undersampling is best on F1, ROC AUC and PR AUC, but logistic regression with undersampling is within about 0.01 on each and is simpler.
- **The cutoff choice hardly matters either.** For random forest and XGBoost, the cutoff picked on held-out data changes F1 by at most 0.02 compared with 0.5. The stacking cutoffs sit near 0.2 because its meta-model is trained on real, imbalanced data rather than resampled data.

## Data

[120 years of Olympic history: athletes and results](https://www.kaggle.com/datasets/heesoo37/120-years-of-olympic-history-athletes-and-results) by rgriffin on Kaggle, scraped from sports-reference.com. It has 271,116 entries from 135,571 athletes across 51 Games (Athens 1896 to Rio 2016). Public domain ([CC0](https://creativecommons.org/publicdomain/zero/1.0/)).

- **Dropped entries**: 64,951 entries (24%) are missing age, height or weight and are dropped, leaving 206,165. 80% of the dropped entries are from before 1960.
- **PriorMedals** is computed before dropping, so medals from entries with missing measurements still count.
- **Team events**: every member of a medal-winning team is a medal entry. About 2 in 3 medal entries at Rio 2016 came from team events: events where one country won several medals of the same colour.
- **Training set**: every Games before 2016, Summer and Winter: 192,722 entries, 14.6% medals.
- **Test set**: Rio 2016, Summer only: 13,443 entries, 15.0% medals.
- **Returning athletes are included in the test set.** Their earlier results are known before the Games, so using PriorMedals isn't leakage.
- **Encoding**: event and sex codes are fitted on the whole dataset. These are only labels for categories, and carry no outcomes.

## Running and retraining

Python 3.12. The packages are pinned in [`requirements.txt`](../requirements.txt) to the tested versions. Times below are for a 12-core machine.

```bash
pip install -r requirements.txt

# train and evaluate all 8 models, print the results table, save figures/ (about 6 minutes)
python main.py

# retune hyperparameters and overwrite best_model_params.json (about 11 minutes)
python tune.py
```

`main.py` opens the figures in windows at the end. To run it without windows: `MPLBACKEND=Agg python main.py` (bash) or `$env:MPLBACKEND="Agg"; python main.py` (PowerShell).

## Project layout

```
common.py               data loading, PriorMedals, 2016 split, resampling and model pipelines
tune.py                 hyperparameter search on 1896–2014 data
main.py                 trains and evaluates every model, draws the figures and SHAP plot
athlete_events.csv      the Kaggle dataset
requirements.txt        Python packages

produced by tune.py:
best_model_params.json  tuned settings for each model and resampling method, read by main.py

produced by main.py:
figures/                Figure_1.png to Figure_6.png
```

## Figures

- [`Figure_1.png`](../figures/Figure_1.png): class balance before and after SMOTE and undersampling
- [`Figure_2.png`](../figures/Figure_2.png): ROC curves for all eight models
- [`Figure_3.png`](../figures/Figure_3.png): confusion matrices, SMOTE models
- [`Figure_4.png`](../figures/Figure_4.png): confusion matrices, undersampling models
- [`Figure_5.png`](../figures/Figure_5.png): F1 score by model
- [`Figure_6.png`](../figures/Figure_6.png): SHAP summary for XGBoost (SMOTE)

## Version history

- **v2 (current).** Rebuilt training and evaluation after a review of v1.
  - Tuning uses pre-2016 data only.
  - Resampling happens inside each fold and pipeline, and uses SMOTENC.
  - Events are one-hot encoded for logistic regression.
  - Each model's cutoff is picked on held-out data.
  - Results include a baseline row and PR AUC.
  - The test set includes every 2016 entry, and PriorMedals is a new feature.
  - Shared code is in `common.py`, and the code saves the figures.
  - Best result: F1 0.40, ROC AUC 0.75.
- **v1 (original)** (tag [`v1-original`](https://github.com/MSyukri420/Model_Prediction_Olympic/tree/v1-original)). Re-running and reviewing it in October 2026 found these problems:
  - It tuned on every year, including the 2016 test year, so the test results weren't fully unseen.
  - It resampled before cross-validation, and it tuned without the feature scaling that training used.
  - SMOTE averaged the event and sex codes, so 5.4% of synthetic entries were impossible (for example, women in men's events).
  - It tested only on first-time Olympians. That dropped 38% of 2016 entries, and those dropped entries won medals more often (19% vs 13%).
  - The SMOTE tree models found only 16–18% of medallists. Every model's accuracy was below the unreported 87% from always guessing "no medal".
  - Its best result was F1 0.31 and ROC AUC 0.70. That was on a different test set, so it isn't directly comparable with v2.
  - v1 also included a written report (`final_report.pdf`), removed in v2 because its numbers no longer match.
