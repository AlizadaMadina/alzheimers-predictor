# Alzheimer's Disease Predictor

A machine learning web application that predicts whether a patient 
is Nondemented, Demented, or Converted based on clinical and brain 
imaging data from the OASIS Longitudinal dataset.

## Live Demo
Run locally using the instructions below.

## Project Structure
alzheimers_predictor/
├── app/
│   ├── app.py
│   ├── home.py
│   ├── form.py
│   └── results.py
├── data/
│   ├── oasis_longitudinal.csv
│   ├── X_scaled.csv
│   └── y.csv
├── docs/
│   └── documentation.md
├── model/
│   ├── alzheimers_model.pkl
│   └── scaler.pkl
├── notebooks/
│   ├── eda.ipynb
│   ├── preprocessing.ipynb
│   └── modeling.ipynb
└── README.md

## Dataset
OASIS Longitudinal MRI Data -- 373 rows, 150 unique patients, 
aged 60 to 98. Source: oasis-brains.org

## Model
Random Forest Classifier trained with GroupKFold cross validation.
Mean accuracy: 71.29%

## How to Run

1. Clone the repository
git clone https://github.com/AlizadaMadina/alzheimers-predictor.git

2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn streamlit

3. Run the web app
cd app
streamlit run app.py

## Tech Stack
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Streamlit

## Author
Madina Alizada