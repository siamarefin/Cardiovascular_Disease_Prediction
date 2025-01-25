# Dataset Description

The dataset contains the following columns along with their data types and descriptions:

| Column Name             | Data Type | Description                                          |
| ----------------------- | --------- | ---------------------------------------------------- |
| **id**                  | `int64`   | Identifier for each entry.                           |
| **age**                 | `int64`   | Age in days.                                         |
| **gender**              | `int64`   | Gender of the individual.                            |
| **height**              | `int64`   | Height in centimeters.                               |
| **weight**              | `float64` | Weight in kilograms.                                 |
| **ap_hi**               | `int64`   | Systolic blood pressure.                             |
| **ap_lo**               | `int64`   | Diastolic blood pressure.                            |
| **cholesterol**         | `int64`   | Cholesterol level categories.                        |
| **gluc**                | `int64`   | Glucose level categories.                            |
| **smoke**               | `int64`   | Smoking status (binary).                             |
| **alco**                | `int64`   | Alcohol consumption status (binary).                 |
| **active**              | `int64`   | Physical activity status (binary).                   |
| **cardio**              | `int64`   | Presence of cardiovascular disease (binary).         |
| **age_years**           | `int64`   | Age in years (derived from age in days).             |
| **bmi**                 | `float64` | Body Mass Index (calculated from height and weight). |
| **bp_category**         | `object`  | Blood pressure category (e.g., normal, high, etc.).  |
| **bp_category_encoded** | `object`  | Encoded representation of blood pressure categories. |

## Notes:

- The `cardio` column indicates the presence (`1`) or absence (`0`) of cardiovascular disease.
- Derived columns such as `age_years` and `bmi` are calculated for ease of analysis.
- Categorical columns like `gender`, `cholesterol`, and `gluc` represent predefined categories.

This dataset can be used for a variety of analyses, including predictive modeling, statistical studies, and visualization tasks related to cardiovascular health.

## Data Source and Acknowledgment

Data was sourced from the UCI Machine Learning Repository and [Kaggle](https://www.kaggle.com/datasets/colewelkins/cardiovascular-disease/data).

All patient data has been anonymized to ensure privacy.
