# Alzheimer's Disease Predictor - Full Project Documentation

Written by Madina Alizada

---

## Introduction

This document explains everything about my Alzheimer's Disease 
Predictor project from start to finish. I wrote this for anyone 
who wants to understand what I built, why I built it, and how 
every single decision was made. You do not need a technical 
background to read this. I will explain everything in plain 
English.

---

## What is This Project?

This project is a machine learning web application that predicts whether a patient 
is Nondemented or Demented based on clinical and brain imaging data 
from the OASIS Longitudinal dataset.
The model attempts to identify Converted patients but due to limited 
data this class has low reliability. Converted means 
the patient was cognitively healthy when they first came in but 
developed dementia over time.

The prediction is made by a machine learning model called a 
Random Forest classifier that I trained on real patient data 
from Washington University.

The goal was not just to build a model but to build a real 
product that anyone can open in a browser, fill in a form, 
and get a result with a confidence percentage.

---

## Why Alzheimer's?

Alzheimer's disease affects over 55 million people worldwide 
and that number is growing every year. One of the biggest 
challenges with Alzheimer's is that by the time symptoms 
are obvious, the disease has already been progressing in 
the brain for years, sometimes up to 20 years before any 
noticeable signs appear.

Early detection matters enormously because while there is 
currently no cure, early intervention can slow the progression 
and significantly improve the patient's quality of life.

Machine learning offers a way to identify patterns in clinical 
data that might not be obvious to the human eye, making it 
a promising tool for early detection support.

---

## The Dataset

I used the OASIS Longitudinal dataset which stands for Open 
Access Series of Imaging Studies. It was collected by 
Washington University Alzheimer's Disease Research Center 
and is freely available for research purposes.

The dataset contains data from 150 unique patients aged 60 
to 98 who were visited multiple times over several years. 
Each visit is a separate row in the data giving us 373 rows 
total and 15 columns.

Here is what each column means:

Subject ID - a unique identifier for each patient. The same 
patient appears multiple times with the same Subject ID but 
different visit numbers.

MRI ID - a unique identifier for each specific scan session.

Group - this is our target variable. It tells us whether 
the patient is Nondemented, Demented, or Converted.

Visit - which visit number this row represents. Some patients 
came back up to 5 times over several years.

MR Delay - the number of days between the first visit and 
this visit.

M/F - the gender of the patient.

Hand - the dominant hand. Almost all patients in this dataset 
are right handed.

Age - the age of the patient at the time of that specific 
scan.

EDUC - years of education completed. Higher education is 
believed to protect against dementia through what scientists 
call cognitive reserve.

SES - socioeconomic status on a scale of 1 to 5 where 1 
means highest status and 5 means lowest.

MMSE - Mini Mental State Examination score. This is a 
cognitive test scored from 0 to 30 where 30 means perfectly 
normal. It is one of the most important predictors of dementia.

CDR - Clinical Dementia Rating assigned by a doctor after 
examining the patient. Values are 0, 0.5, 1, and 2.

eTIV - Estimated Total Intracranial Volume, the total volume 
inside the skull in cubic millimeters.

nWBV - Normalized Whole Brain Volume. A ratio of actual brain 
tissue to total skull volume. As dementia progresses the brain 
physically shrinks and this number decreases.

ASF - Atlas Scaling Factor, a technical adjustment for head 
size differences between patients.

---

## Key Decisions I Made Before Writing Any Code

Before touching any code I spent time understanding the data 
and making important decisions about how to approach the problem.

The first decision was choosing the target variable. The dataset 
has two columns that relate to dementia severity - Group and CDR. 
I decided to use Group as my target and drop CDR entirely. The 
reason is that CDR directly measures dementia severity which is 
essentially the same thing as the target. Including it would be 
cheating because the model would just learn from CDR instead of 
the real brain and cognitive features.

The second decision was about the train/test split. Our dataset 
is longitudinal meaning the same patient appears multiple times. 
If I used a simple random split, the same patient could end up 
in both training and testing. The model would already know that 
patient from training so it would perform artificially well. 
This would give a fake accuracy score that does not reflect 
real world performance. To solve this I used GroupKFold cross 
validation which keeps all visits of the same patient together. 
Either all visits go into training or all go into testing, never 
both.

The third decision was dropping irrelevant columns. I removed 
Subject ID and MRI ID because they are just labels with no 
medical meaning. I removed Hand because almost everyone is 
right handed so it teaches the model nothing. I removed MR 
Delay because it is a scheduling detail not a clinical measurement.

---

## Phase 1 - Exploratory Data Analysis

Before building anything I spent time just looking at the data 
and asking questions. This phase is called Exploratory Data 
Analysis or EDA.

I discovered that the dataset has 373 rows and 15 columns with 
150 unique patients. The average patient age is 77 years. The 
average MMSE score is 27.3 out of 30.

I found 21 missing values in total - 19 in the SES column and 
2 in the MMSE column. Everything else was complete.

I created 5 visualizations to understand the data better.

The class distribution chart showed that we have 190 Nondemented 
patients, 146 Demented patients, and only 37 Converted patients. 
This unequal distribution is called class imbalance and it 
means the model needs special handling to not ignore the 
smaller Converted class.

The age distribution chart showed that age alone does not 
strongly separate the three groups. The boxes overlapped 
significantly which told me age would not be the most powerful 
predictor.

The MMSE score chart was the most dramatic. Nondemented patients 
clustered near perfect scores of 29 to 30 while Demented patients 
showed much lower scores with some as low as 5 or 6. This clear 
separation told me MMSE would be the most important feature.

The brain volume chart confirmed that demented patients have 
measurably smaller brain volumes than healthy patients which 
makes biological sense since Alzheimer's causes brain tissue 
to physically shrink.

The correlation heatmap revealed that MMSE and CDR have a 
strong negative correlation of -0.69 confirming they measure 
the same thing from different angles. It also showed that 
eTIV and ASF are almost perfectly correlated at -0.99 meaning 
they are mathematically related.

The MMSE over visits chart showed individual patient trajectories. 
Green lines representing healthy patients stayed flat and high. 
Red lines representing demented patients declined over time with 
one patient dropping from a score of 19 all the way down to 4 
over the course of the study.

---

## Phase 2 - Data Cleaning and Preprocessing

After understanding the data I cleaned and prepared it for 
the model.

I dropped 5 columns that would not help the model: Subject ID, 
MRI ID, Hand, MR Delay, and CDR. This reduced the dataset from 
15 columns to 10.

I filled the 21 missing values using the median value of each 
column. I used median instead of average because median is not 
affected by extreme outliers. This technique is called imputation 
and it is better than deleting the rows entirely because we 
would lose real patient data.

I converted text columns to numbers because machine learning 
models can only work with numbers. Female became 0 and Male 
became 1. Nondemented became 0, Demented became 1, and 
Converted became 2.

I scaled all numerical features using StandardScaler so they 
are all on the same range. Without scaling a column like eTIV 
with values in the thousands would dominate the model simply 
because its numbers are bigger even if it is not the most 
important feature.

---

## Phase 3 - Feature Engineering

Feature engineering is where I created three new columns from 
existing ones to give the model smarter information.

MMSE_decline measures how many points a patient's cognitive 
score has dropped since their very first visit. This captures 
the rate of cognitive decline which is more meaningful than 
just a single snapshot score.

nWBV_change measures how much a patient's brain volume has 
changed since their first visit. A positive number means the 
brain has shrunk which is a biological marker of Alzheimer's 
progression.

Age_first_visit records the patient's age when they first 
entered the study giving the model a stable baseline age.

These three features proved their value when the model later 
ranked Age_first_visit as the second most important feature 
out of all 12.

---

## Phase 4 - Training the Model

I trained a Random Forest classifier which works by building 
100 decision trees and having them all vote on the final 
prediction. The class with the most votes wins.

I used these settings for the model. 100 trees for stable 
and reliable predictions. Maximum depth of 10 to prevent 
the model from memorizing the training data too perfectly 
which is called overfitting. A fixed random seed of 42 so 
results are reproducible. And balanced class weights so the 
model pays equal attention to all three classes regardless 
of their size.

Using GroupKFold cross validation with 5 splits the model 
achieved a mean accuracy of 71.29 percent with a standard 
deviation of 6.09 percent.

The confusion matrix showed the model correctly identified 
25 out of 36 Nondemented patients and 25 out of 33 Demented 
patients. However it failed to correctly identify any of the 
5 Converted patients in the test fold due to the very small 
number of examples.

The feature importance chart confirmed our EDA findings. MMSE 
was the most important feature by a significant margin. 
Age_first_visit which we engineered ourselves came in second 
place proving that our feature engineering step genuinely 
added value. nWBV came in third confirming that brain volume 
is a strong predictor of dementia.

---

## Phase 5 - The Web Application

I built a three page web application using Streamlit.

Page 1 is the home page. It explains what the app is, why 
early detection matters, and has a button to go to the form.

Page 2 is the patient form. A user fills in the patient's 
age, gender, education, socioeconomic status, visit number, 
MMSE score, brain volume measurements, and atlas scaling 
factor. If it is not the patient's first visit, the form 
also asks for their measurements from Visit 1 to calculate 
decline automatically. The engineered features are calculated 
behind the scenes and never shown to the user.

Page 3 is the results dashboard. It shows the prediction 
with a color coded box, confidence percentages for all 
three classes, a feature importance chart showing what 
influenced the prediction most, and a comparison of the 
patient's key measurements against the dataset averages.

---

## Model Limitations

I want to be fully transparent about what this model can 
and cannot do.

This model was trained on patients aged 60 to 98. It is not 
designed for early onset dementia which affects younger 
patients and presents differently clinically. Predictions 
for patients outside this age range would not be reliable.

The dataset contains only 150 unique patients which is 
relatively small for a machine learning model. A larger 
dataset would improve reliability and generalizability.

The Converted class only had 37 patients total and the model 
struggles to predict it accurately. More data on patients in 
the conversion stage would significantly improve this weakness.

The model was trained on data from Washington University. It 
may not generalize perfectly to patients from different 
geographic regions or ethnic backgrounds due to potential 
differences in clinical presentation.

This tool is designed as a decision support aid only. It is 
not a replacement for clinical diagnosis by a qualified 
medical professional.

---

## What I Learned Building This Project

This project taught me so much more than just how to train a 
machine learning model.

Before this project I had never worked with real clinical data. 
Learning to look at a dataset and ask the right questions before 
writing any code was one of the most valuable skills I developed.

I learned that data cleaning and preprocessing takes much more 
time and thought than the actual modeling. Deciding which columns 
to drop, how to handle missing values, and why scaling matters 
are decisions that directly affect the quality of your model.

I learned the difference between a notebook and a product. Anyone 
can train a model. Building something that a real person can open 
in a browser, interact with, and get meaningful results from is 
a completely different challenge.

I learned to be honest about limitations. My model cannot 
reliably predict the Converted class due to limited data. 
Acknowledging this openly rather than hiding it is what 
separates a trustworthy project from a misleading one.

I learned that GroupKFold cross validation matters for 
longitudinal data. A simple train/test split would have 
given me fake accuracy because the same patient would 
appear in both training and testing. Understanding why 
this is a problem and how to fix it properly was one of 
my favorite moments in this project.

Most importantly I learned that the best way to understand 
something deeply is to build it yourself, make mistakes, 
catch them, fix them, and document everything along the way.

---

## Future Improvements

If I were to continue developing this project I would focus 
on these areas.

Collecting more data especially on Converted patients to 
improve prediction for that class.

Trying other machine learning models like XGBoost or a 
neural network to compare performance.

Adding proper MMSE decline tracking by storing previous 
visit data in a database so the app calculates decline 
automatically without the doctor needing to remember the 
first visit score.

Deploying the app to a public server so anyone can access 
it without running it locally.

Adding more languages to make the app accessible to 
non-English speaking medical professionals.

---

## Tools and Libraries Used

Python - the programming language used throughout.

Pandas - for loading, cleaning and manipulating the dataset.

NumPy - for numerical calculations behind the scenes.

Matplotlib and Seaborn - for creating all visualizations.

Scikit-learn - for the Random Forest model, cross validation, 
scaling, and evaluation metrics.

Streamlit - for building the web application.

Pickle - for saving and loading the trained model and scaler.

Jupyter Notebook - for writing the EDA, preprocessing, and 
modeling code in an interactive and well documented format.

Git and GitHub - for version control and sharing the project.

---

Author: Madina Alizada

---

Dataset: OASIS Longitudinal MRI Data, Washington University