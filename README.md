# Iris Classifier (Decision Tree)
##  Overview
This project demonstrates an end-to-end machine learning workflow using the classic Iris dataset.
A Decision Tree classifier is trained using scikit-learn, evaluated using accuracy and a confusion matrix.
The project includes:
- A Jupyter notebook for explanation and experimentation
- A reproducible Python training script
- Saved outputs (confusion matrix image and trained model)
## Quick start
```bash
git clone https://github.com/anamzulfiqar/Iris-classifier-project-.git
cd iris-classifier
## Activate virtual environment
python -m venv venv && source venv/bin/activate
## Install required packages
pip install -r requirements.txt
python src/train.py
## Train the model
python src/train.py --test-size 0.2 --random-state 42
```
## Project Structure
iris-classifier/
├── data/ # Empty (Iris loaded from scikit-learn)
├── notebooks/
│ └── iris_model.ipynb # Step-by-step notebook
├── src/
│ └── train.py # CLI training script
├── tests/
│ └── test_train.py # Basic pytest (optional)
├── outputs/
│ ├── confusion_matrix.png
│ └── iris_model.pkl
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt

