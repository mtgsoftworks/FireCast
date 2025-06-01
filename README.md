**FireCast**
====================

**Table of Contents**

- [About](#about)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Descriptive Statistics](#descriptive-statistics)
- [Hypothesis Testing](#hypothesis-testing)
- [Statistical Modeling](#statistical-modeling)
- [Results and Outputs](#results-and-outputs)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## About

FireCast is a Python-based data science project developed as part of the ISE-216 course titled "Statistics for Data Science." The primary goal of this project is to analyze and model factors influencing forest fire occurrences in the Montesinho Natural Park region of northeast Portugal. Using the well-known "Forest Fires" dataset from the UCI Machine Learning Repository, FireCast applies a comprehensive statistical workflow, including exploratory data analysis (EDA), descriptive statistics, hypothesis testing, and regression modeling, to derive insights into wildfire risk factors and area burned.

Key objectives of FireCast:

1. **Explore the Forest Fires dataset** to understand variable distributions, detect outliers, and visualize relationships among features.
2. **Compute descriptive statistics** (mean, median, variance, etc.) for relevant variables to characterize typical fire conditions.
3. **Perform hypothesis tests** (e.g., t-tests, ANOVA) to evaluate whether specific factors—such as month, day of the week, or meteorological conditions—have a statistically significant impact on the burned area.
4. **Build and validate regression models** (e.g., linear regression, polynomial regression) to predict the burned area of a forest fire based on environmental factors (temperature, humidity, wind speed, etc.).
5. **Document and visualize results** in a clear, reproducible format with charts, tables, and a comprehensive report.

This project serves as an educational example for students, researchers, and data science enthusiasts who wish to learn how to structure and execute a full statistical analysis pipeline on a real-world dataset. The code is modular and well-documented, making it easy to adapt for similar environmental or risk-related analyses.

---

## Dataset

The dataset used in this project is the **Forest Fires** dataset ([Carvalho, F. & Tome, I., 2007](https://archive.ics.uci.edu/ml/datasets/Forest+Fires)). It contains meteorological and temporal information about forest fires that occurred in the Montesinho Natural Park from 2000 to 2003. The target variable is the **burned area** (measured in hectares). Key attributes include:

- `X`: x-axis spatial coordinate within the Montesinho park map (values 1 to 9)
- `Y`: y-axis spatial coordinate within the Montesinho park map (values 2 to 9)
- `month`: month of the year (categorical: jan to dec)
- `day`: day of the week (categorical: mon to sun)
- `FFMC`: Fine Fuel Moisture Code index (numeric)
- `DMC`: Duff Moisture Code index (numeric)
- `DC`: Drought Code index (numeric)
- `ISI`: Initial Spread Index (numeric)
- `temp`: temperature in Celsius degrees (numeric)
- `RH`: relative humidity in percentage (numeric)
- `wind`: wind speed in km/h (numeric)
- `rain`: outside rain in mm/m² (numeric)
- `area`: burned area of the forest fire in hectares (numeric; continuous)

For more details on the dataset attributes and source, refer to the UCI repository page.

---

## Project Structure

```plaintext
FireCast/
├── data/                  # Raw and cleaned data files (CSV format)
│   └── forestfires.csv    # Original dataset obtained from UCI
├── docs/                  # Documentation, reports, and supplementary materials
│   ├── ANALYSIS_REPORT.md  # Detailed analysis, methodology, and interpretation
│   └── figures/           # Saved plots and charts from EDA and modeling
├── output/                # Generated outputs (plots, model summaries, tables)
│   ├── eda_plots/         # EDA visualizations (histograms, scatter plots, box plots)
│   ├── stats_summary/     # Descriptive statistics tables
│   ├── hypothesis_tests/  # Hypothesis test results and associated tables
│   └── models/            # Trained model artifacts and evaluation metrics
├── main.py                # Main Python script orchestrating EDA, stats, and modeling
├── requirements.txt       # List of required Python packages and versions
└── README.md              # This README file (detailed project overview)
```

---

## Prerequisites

Before running any analysis or model scripts, ensure that the following are installed on your system:

- **Python 3.7+**
- **pip** (Python package installer)

Required Python libraries (listed in `requirements.txt`):

```plaintext
pandas          # Data manipulation and analysis
numpy           # Numerical operations
matplotlib      # Data visualization
seaborn         # Statistical data visualization
scipy           # Scientific computing (statistical tests)
statsmodels     # Advanced statistical modeling
scikit-learn    # Machine learning and regression modeling
jupyter         # Interactive notebooks (for exploring data)
``` 

Install dependencies via:

```bash
pip install -r requirements.txt
```

---

## Installation

1. **Clone the repository**

   ```bash
git clone https://github.com/mtgsoftworks/FireCast.git
cd FireCast
   ```

2. **Install required packages** (see Prerequisites section above):

   ```bash
pip install -r requirements.txt
   ```

3. **Verify dataset availability**: Ensure the `data/forestfires.csv` file exists. If missing, download it from the UCI Machine Learning Repository and place it under `data/`.

---

## Usage

The `main.py` script is structured into modular functions covering EDA, descriptive statistics, hypothesis testing, and modeling. You can run the entire pipeline or call specific functions interactively via a Jupyter notebook.

1. **Run full pipeline** (command-line):

   ```bash
python main.py
   ```
