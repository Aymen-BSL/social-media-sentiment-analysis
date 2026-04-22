# Social Media Sentiment Analysis

This project analyzes social media comments at two levels:

- Comment-level sentiment: `positive`, `negative`, `neutral`
- Post-level reaction: `Liked`, `Not Liked`, `Mixed Reaction`

It includes two approaches:

1. A traditional ML pipeline using TF-IDF with multiple classifiers
2. A pre-trained baseline using VADER

## Project Structure

- `data/raw/comments.csv`: labeled training data
- `data/raw/sample_posts.json`: demo posts with comment lists
- `data/processed/comments_clean.csv`: cleaned dataset generated from raw comments
- `models/`: saved ML models and reports
- `notebooks/experiments.ipynb`: starter notebook
- `src/preprocess.py`: text cleaning pipeline
- `src/train_ml.py`: train and compare ML models
- `src/predict_ml.py`: predict with the saved ML model
- `src/predict_pretrained.py`: predict with VADER
- `src/aggregate.py`: post-level reaction rules
- `src/evaluate.py`: evaluation helpers and confusion matrix export
- `src/demo.py`: run the final system on sample posts

## Simplest Way To Run

Create and use a project virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Then open the notebook with one command:

```powershell
.\run_notebook.ps1
```

Then in the notebook, run the cells from top to bottom.

## If You Prefer Scripts

These are optional now. The notebook does all of this for you.

```powershell
python src/preprocess.py
python src/train_ml.py
python src/predict_pretrained.py --comments "love this update" "pretty decent overall" "not good at all"
python src/predict_ml.py --comments "love this update" "pretty decent overall" "not good at all"
python src/demo.py
```

## Notes

- VADER is a real pre-existing sentiment tool for short social media text.
- The easiest tested path is the local `.venv` plus `notebooks/experiments.ipynb`.
- The included dataset is a realistic starter dataset suitable for a class project demo. You can replace it with a larger public dataset later without changing the code structure.
