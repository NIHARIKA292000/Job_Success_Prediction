```markdown
# 🎯 Job Success Prediction

This project predicts whether an employee will stay in their job or leave, based on various factors like satisfaction level, number of projects, average monthly hours, and more. The project uses a Random Forest classifier and provides a clean, interactive Gradio web interface for real-time predictions.

---

## 📁 Project Structure

```

Job\_Success\_Prediction/
│
├── artifacts/                     # Stores the trained model and preprocessor
│   ├── rf\_model.pkl
│   └── preprocessor.pkl
│
├── src/                           # Source code
│   ├── pipeline/
│   │   ├── predict\_pipeline.py    # Prediction and data transformation logic
│
├── app.py                         # Gradio UI application
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation (this file)

````

---

## 🚀 Features

- 🔍 **Predict** whether an employee will stay or leave
- ⚙️ **Modular code** using pipeline design for scalability
- 📊 **Interactive UI** using Gradio
- 💾 Model and preprocessor saved using Pickle in `artifacts/`

---

## 🧠 Model Overview

- **Algorithm**: Random Forest Classifier
- **Target Variable**: Employee Turnover (0 = Stay, 1 = Leave)
- **Input Features**:
  - satisfaction_level
  - last_evaluation
  - number_project
  - average_montly_hours
  - time_spend_company
  - Work_accident
  - promotion_last_5years
  - Department
  - salary

---

## 💡 How It Works

1. **User Inputs**:
   - Fill in employee-related details in the Gradio web app
2. **Pipeline**:
   - Inputs are converted into a DataFrame using `CustomData` class
   - The preprocessor scales/encodes the input
   - The trained model predicts the outcome
3. **Output**:
   - Displays whether the employee is likely to stay or leave

---

## ⚙️ Installation

1. **Clone the repo**
```bash
git clone https://github.com/your-username/job-success-prediction.git
cd job_success_prediction
````

2. **Create and activate a virtual environment**

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
python app.py
```

Then open the link provided in your terminal (usually [http://127.0.0.1:7860](http://127.0.0.1:7860)).

---

## 📦 Dependencies

All dependencies are listed in `requirements.txt`. Key ones include:

* `scikit-learn`
* `pandas`
* `numpy`
* `gradio`

---

## 🧩 Example Input (Used for Testing)

| Feature                 | Value  |
| ----------------------- | ------ |
| satisfaction\_level     | 0.7    |
| last\_evaluation        | 0.8    |
| number\_project         | 4      |
| average\_montly\_hours  | 160    |
| time\_spend\_company    | 3      |
| Work\_accident          | 0      |
| promotion\_last\_5years | 0      |
| Department              | sales  |
| salary                  | medium |

---

## ✨ Future Improvements

* Add probability scores (e.g., 87% chance of leaving)
* Include feature importance plots
* Deploy on cloud (e.g., HuggingFace Spaces, Heroku)
* Add unit tests and CI/CD pipeline

---

## 🙋‍♀️ Author

**Niharika H.**

---

