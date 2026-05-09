# Alzheimer's Disease Predictor

A machine learning web application that predicts whether a patient 
is Nondemented or Demented based on clinical and brain imaging data 
from the OASIS Longitudinal dataset.
The model attempts to identify Converted patients but due to limited 
data this class has low reliability.

For a full non-technical explanation of this project including 
every decision made and why, please read the 
[Project Documentation](docs/documentation.md).

## Live Demo

Try the app here: https://alzheimers-predictor-madina.streamlit.app

## Project Structure

```
alzheimers_predictor/
├── app/
│   ├── app.py
│   ├── home.py
│   ├── form.py
│   └── results.py
├── data/
│   ├── oasis_longitudinal.csv
│   ├── oasis_cross-sectional.csv
│   ├── X_scaled.csv
│   └── y.csv
├── docs/
│   ├── documentation.md
│   ├── age_distribution.png
│   ├── brain_volume_distribution.png
│   ├── class_distribution.png
│   ├── confusion_matrix.png
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   ├── mmse_distribution.png
│   └── mmse_over_visits.png
├── model/
│   ├── alzheimers_model.pkl
│   └── scaler.pkl
├── notebooks/
│   ├── eda.ipynb
│   ├── preprocessing.ipynb
│   └── modeling.ipynb
├── .gitignore
├── requirements.txt
└── README.md
```

## Dataset
OASIS Longitudinal MRI Data -- 373 rows, 150 unique patients, 
aged 60 to 98. Source: oasis-brains.org

## Model
Random Forest Classifier trained with GroupKFold cross validation.
Mean accuracy: 71.29%

## How to Run

1. Clone the repository

```
git clone https://github.com/AlizadaMadina/alzheimers-predictor.git
```

2. Install dependencies

```
pip install -r requirements.txt
```

3. Run the web app

```
cd app
streamlit run app.py
```

## Tech Stack
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Streamlit

## Author
Madina Alizada