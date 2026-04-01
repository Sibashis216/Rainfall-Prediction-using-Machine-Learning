# 🌧️ Rainfall Prediction using Machine Learning

A Machine Learning model that predicts the **amount of rainfall (in mm)** based on meteorological parameters such as temperature, humidity, dew point, wind speed, and more. Built using **Random Forest Regressor** and documented step-by-step in a Jupyter Notebook.

---

## 🚀 Overview

The system analyzes **2,362 historical weather records**, where each record contains 7 meteorological features used to predict daily rainfall.

- **Objective :** Predict the amount of rainfall (in mm) based on weather parameters
- **Algorithm :** Random Forest Regressor
- **Dataset :** `fall.csv` — Historical daily weather records (2011 onwards)
- **Environment :** Jupyter Notebook / Google Colab

---

## 📂 Project Structure

```
rainfall-prediction/
│
├── Rainfall_Prediction.ipynb   # Main Jupyter Notebook
├── fall.csv                    # Dataset
└── README.md                   # Project documentation
```

---

## 📊 Dataset

The dataset `fall.csv` contains **2,362 rows** of historical weather observations.

| Column | Description |
|---|---|
| `year` | Year of observation |
| `month` | Month *(dropped during preprocessing)* |
| `day` | Day *(dropped during preprocessing)* |
| `tempavg` | Average Temperature (°C) |
| `DPavg` | Average Dew Point (°C) |
| `humidity avg` | Relative Humidity (%) |
| `SLPavg` | Sea Level Pressure (hPa) |
| `visibilityavg` | Visibility (km) |
| `windavg` | Wind Speed (km/h) |
| `Rainfall` | **Target** — Rainfall (mm) |

> `month` and `day` are dropped as they are not used for prediction.

---

## ✨ Notebook Workflow

The notebook follows a clean, step-by-step structure:

| Step | Description |
|---|---|
| Step 1 | Import Libraries |
| Step 2 | Load Dataset |
| Step 3 | Data Preprocessing (drop columns, check nulls) |
| Step 4 | Feature & Target Selection |
| Step 5 | Train-Test Split (80/20) |
| Step 6 | Feature Scaling (StandardScaler) |
| Step 7 | Model Building — Random Forest Regressor |
| Step 8 | Prediction & Evaluation (R² Score) |
| Step 9 | Feature Importance Plot |
| Step 10 | Scatter Plots — Each Feature vs Rainfall |
| Step 11 | Correlation Heatmap |
| Step 12 | Predict a New Data Record |

---

## 📈 Model Performance

| Metric | Value |
|---|---|
| Algorithm | Random Forest Regressor |
| Number of Trees | 500 |
| Train / Test Split | 80% / 20% |
| Training Size | 1,889 records |
| Testing Size | 473 records |
| **Training Accuracy (R²)** | **0.85** |
| **Testing Accuracy (R²)** | **0.29** |

> **Note :** The dataset is highly zero-inflated — **75% of records have 0 mm rainfall**. This naturally limits the R² score on unseen data, which is expected and realistic for rainfall regression tasks.

---

## 📊 Visualizations

The notebook generates the following plots inline:

| Plot | Description |
|---|---|
| Feature Importance Bar Chart | Shows which features influence rainfall the most |
| Temperature vs Rainfall | Scatter plot — training set |
| Dew Point vs Rainfall | Scatter plot — training set |
| Humidity vs Rainfall | Scatter plot — training set |
| SLP vs Rainfall | Scatter plot — training set |
| Visibility vs Rainfall | Scatter plot — training set |
| Wind vs Rainfall | Scatter plot — training set |
| Correlation Heatmap | Seaborn heatmap of all feature correlations |

### 🔑 Key Insight — Feature Importances

| Feature | Importance |
|---|---|
| 💧 Humidity | 30.7% |
| 🌡️ SLP (Pressure) | 17.4% |
| 💨 Wind Speed | 13.2% |
| 📅 Year | 12.0% |
| 🌡️ Dew Point | 10.6% |
| 🌡️ Temperature | 10.3% |
| 👁️ Visibility | 5.7% |

> **Humidity** is the strongest predictor of rainfall — consistent with meteorological science!

---

## 🔮 Sample Prediction

```python
# Input: [year, tempavg, DPavg, humidity, SLP, visibility, wind]
new_input = np.array([[2020, 18, 16, 65, 1013, 6, 8]])
new_input_scaled = scaler.transform(new_input)

ypred = regressor.predict(new_input_scaled)
print(f"Predicted Rainfall: {round(ypred[0], 2)} mm")
```

**Output:**
```
Predicted Rainfall: 0.13 mm
```

---

## ⚙️ How to Run

### Jupyter Notebook

```bash
git clone https://github.com/Sibashis216/Rainfall-Prediction-using-Machine-Learning.git
cd rainfall-prediction
pip install pandas numpy matplotlib seaborn scikit-learn notebook
jupyter notebook Rainfall_Prediction.ipynb
```
Run each cell using **Shift + Enter**.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.8+ | Core programming language |
| Jupyter Notebook | Interactive development environment |
| Pandas | Data loading and preprocessing |
| NumPy | Numerical computations |
| Matplotlib | Scatter plots and bar charts |
| Seaborn | Correlation heatmap |
| Scikit-learn | Model training, scaling, and evaluation |

---

<div align="center">
Made with ❤️ | ⭐ Star this repo if you found it helpful!
</div>
