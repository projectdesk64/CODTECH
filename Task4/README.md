## Email Spam Classifier

This project builds a simple yet strong email spam classifier using TF‑IDF features and a Multinomial Naive Bayes model. It is implemented in a Jupyter notebook: `Task4.ipynb`.

### Overview
- **Goal**: Classify messages as Ham (0) or Spam (1)
- **Features**: `TfidfVectorizer` with English stop words
- **Model**: `MultinomialNB`
- **Split**: 80% train / 20% test
- **Reported accuracy**: ~96.7% on the held-out test set

### Dataset
- Source (CSV): `https://raw.githubusercontent.com/Apaulgithub/oibsip_taskno4/main/spam.csv`
- Columns used: `Category` (ham/spam), `Message` (text)
- Preprocessing in notebook:
  - Keep first two columns
  - Rename to `Category` and `Message`
  - Map labels: `ham -> 0`, `spam -> 1` (stored as `label_num`)

### Requirements
- Python 3.8+
- Jupyter Notebook / JupyterLab
- Python packages:
  - pandas
  - scikit-learn

You can install dependencies with:

```bash
pip install pandas scikit-learn jupyter
```

### How to Run
1. Launch Jupyter:
   ```bash
   jupyter notebook
   ```
2. Open `Task4.ipynb`.
3. Execute all cells in order (Kernel → Restart & Run All) to:
   - Load and clean the dataset
   - Split into train/test
   - Vectorize with TF‑IDF
   - Train `MultinomialNB`
   - Evaluate accuracy, confusion matrix, and classification report
   - Try custom example messages

### Results (from the notebook)
- Accuracy: ~96.68%
- Confusion matrix and classification report are printed in the output cells.

### Predicting on New Messages
The notebook includes a cell where you can edit the `new_emails` list and re-run to see predictions using the already-fitted vectorizer and model.

### Project Structure
- `Task4.ipynb` — Main notebook with data loading, preprocessing, training, evaluation, and examples
- `README.md` — This file

### Notes
- The dataset is downloaded at runtime from a public URL; ensure an active internet connection.
- For reproducibility, a fixed `random_state=42` is used in the train/test split.



