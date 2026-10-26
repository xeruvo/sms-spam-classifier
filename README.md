# SMS Spam Classifier

A student machine learning project that classifies SMS messages as either
`ham` (legitimate) or `spam`.

## Project overview

The notebook demonstrates a complete text-classification workflow:

1. Load and inspect the CSV data.
2. Keep the label and message columns.
3. Normalize message text by lowercasing it, removing numbers and punctuation,
   and collapsing whitespace.
4. Encode `ham` as `0` and `spam` as `1`.
5. Split the data into stratified training and test sets.
6. Fit a TF-IDF vectorizer on the training messages and transform both splits.
7. Train and compare three supervised classifiers.
8. Evaluate the predictions with classification reports, confusion matrices,
   and accuracy.

## Technologies

- Python
- Jupyter Notebook or VS Code
- pandas and NumPy for data handling
- scikit-learn for preprocessing, models, splitting, and evaluation
- Matplotlib and Seaborn for plots

## Algorithms implemented

The notebook trains and evaluates:

- Logistic Regression
- Multinomial Naive Bayes
- Linear Support Vector Machine (`LinearSVC`)

Messages are represented with scikit-learn's `TfidfVectorizer`.

## Setup

From this project directory, create and activate a virtual environment if
desired, then install the project dependencies:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Open `spam_classifier.ipynb` in Jupyter or VS Code and select the environment
as the notebook kernel.

## Usage

1. Keep `spam.csv` in the same directory as `spam_classifier.ipynb`.
2. Open the notebook.
3. Run the cells from top to bottom.
4. Review the classification reports, confusion matrices, and accuracy
   comparison.

The notebook uses a fixed `random_state=42` and an 80/20 stratified split, so
the saved evaluation can be reproduced with the same package versions and
data.

## Dataset

This project uses the **SMS Spam Collection Dataset** from the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).
The dataset is not included in this repository. `spam.csv` is listed in
`.gitignore` and must remain on the local computer.

To obtain the dataset:

1. Visit the [official UCI dataset page](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).
2. Download the dataset archive from UCI.
3. Extract the SMS data file and place the project-compatible file in this
   directory with the filename `spam.csv`.
4. Confirm that `spam.csv` is in the same directory as
   `spam_classifier.ipynb`.

The notebook loads the file with `pd.read_csv("spam.csv", encoding="latin-1")`,
so it expects that relative path and filename. The local CSV has 5,572 data
rows, the columns `v1` and `v2`, and the labels `ham` and `spam`. The notebook
uses the first two columns and ignores the three additional columns in the
CSV; most are empty, but 50 rows contain a non-empty value in an additional
column.

The dataset contains message text that may include phone numbers, names, and
other personal or sensitive-looking content. Review UCI's current terms and
license before using or redistributing the data. This repository intentionally
does not publish the dataset.

## Evaluation results

These are the results stored in the notebook's executed outputs for the
fixed 80/20 split (1,115 test messages):

| Model | Accuracy | Spam precision | Spam recall | Spam F1 |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.97 | 0.99 | 0.79 | 0.88 |
| Multinomial Naive Bayes | 0.95 | 1.00 | 0.66 | 0.79 |
| Linear SVM | 0.99 | 0.99 | 0.90 | 0.94 |

The class distribution is imbalanced: 4,825 `ham` messages and 747 `spam`
messages. Accuracy should therefore be considered together with spam recall
and F1 when comparing the models. These figures are reported from the saved
notebook execution and are not a guarantee of results after changing the data,
dependencies, or random seed.

## Leakage review

The notebook performs deterministic, label-independent text cleaning before
the train/test split. This is not a material source of target leakage. TF-IDF
vectorization is handled correctly: the vectorizer is fitted only on
`X_train` and then used to transform `X_test`.

No leakage correction was applied, and the reported results were not silently
changed. For a stricter production workflow, the cleaning and vectorization
steps could be placed in a scikit-learn `Pipeline` and evaluated with
cross-validation, while keeping the test set untouched.
