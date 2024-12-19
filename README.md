# Diabetes Prediction Model

This project is a diabetes prediction application built using an **Artificial Neural Network (ANN)**. The model predicts the probability of diabetes based on user input data such as age, hypertension, heart disease, BMI, HbA1c level, and blood glucose level. With a **97% validation accuracy**, this application provides a reliable tool for early diabetes risk assessment.

---

## Features

- **Real-Time Prediction**: Allows users to input data and instantly receive predictions.
- **High Accuracy**: The ANN model demonstrates a 97% validation accuracy.
- **User-Friendly Interface**: A clean and interactive web interface built using **Streamlit**.
- **Data Normalization**: Utilizes `StandardScaler` for efficient preprocessing of input data.

---

## Technology Stack

### **Machine Learning Frameworks**:

- **TensorFlow** and **Keras**: For building and training the ANN.

### **Frontend**:

- **Streamlit**: Provides a simple, interactive web app interface.

### **Preprocessing Tools**:

- **Pandas**: Used for handling and preprocessing the dataset.
- **scikit-learn (StandardScaler)**: Normalizes the input data for better model performance.

---

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/your-username/diabetes-prediction-model.git
   cd diabetes-prediction-model
   ```

2. **Install Dependencies**:
   Ensure you have Python installed, then run:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the App**:

   ```bash
   streamlit run main.py
   ```

4. **Model File**:
   Ensure the trained model file (`diabeties_model_advance.keras.keras`) is placed in the root directory of the project.

---

## Usage

1. Launch the app by running the command above.
2. Input the required data:
   - **Age**
   - **Hypertension (Yes/No)**
   - **Heart Disease (Yes/No)**
   - **BMI**
   - **HbA1c Level**
   - **Blood Glucose Level**
3. Click on the **Predict** button.
4. View the prediction result, displayed as the probability of having diabetes.

---

## Dataset

The dataset used for training is preprocessed by:

- Removing duplicates.
- Dropping unnecessary columns (`smoking_history`, `gender`).
- Splitting into features (`X`) and target (`y`).

The data is normalized using `StandardScaler` before being fed into the model.

---

## Results

- **Validation Accuracy**: 97%
- **Prediction Output**: Displays the probability of diabetes as a percentage.

---

## Project Structure

```
.
├── diabetes_prediction_dataset.csv   # Dataset used for training
├── diabeties_model_advance.keras.keras  # Trained ANN model
├── main.py                           # Streamlit application script
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

---

## Future Improvements

- Enhance the dataset by including more features.
- Provide detailed analysis for predictions.
- Deploy the app on a cloud platform for broader accessibility.

---

## License

This project is open-source and available under the [MIT License](LICENSE).

---

### Acknowledgements

Special thanks to all the resources and tutorials that guided me through this project. #MachineLearning #ANN #DiabetesPrediction #Streamlit #TensorFlow #Keras

Click the Colab IPYNB file link https\://colab.research.google.com/drive/1g7iHXQYL5RjGixhvibesX1c566rqgrEV?authuser=1#scrollTo=s5RG8Az7fOCi

If it is helpful for you please like this repo.

