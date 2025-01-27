import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.preprocessing import MinMaxScaler
import os 
from sklearn.feature_selection import RFE
import seaborn as sns
import matplotlib.pyplot as plt 
from threading import Thread
from matplotlib.figure import Figure
import matplotlib
matplotlib.use('Agg')
from sklearn.preprocessing import LabelEncoder

def data_analysis():
    # Path to the dataset
    file_path = "data/raw/cardio_data.csv"
    save_path = r"H:\Cardiovascular_Disease_Prediction\files"

    try:
        # Read the CSV
        data = pd.read_csv(file_path)

        # Create a string to store all results
        analysis_results = ""

        # Dataset Overview
        analysis_results += "<h2>Dataset Overview</h2>"
        analysis_results += f"<p>Shape of the dataset: {data.shape}</p>"

        # Columns in the dataset
        analysis_results += "<h3>Columns in the dataset:</h3>"
        analysis_results += f"<p>{', '.join(data.columns)}</p>"

        # Sample data
        analysis_results += "<h3>Sample Data:</h3>"
        analysis_results += data.head().to_html(classes="dataframe", index=False)

        # Statistical summary
        analysis_results += "<h3>Statistical Summary:</h3>"
        analysis_results += data.describe().to_html(classes="dataframe")


        # Missing value 
        analysis_results += "<h3>Missing Values:</h3>"
        missing_values = data.isnull().sum()
        missing_values_list = missing_values.to_dict()  # Convert to a dictionary for easier iteration

        # Format the missing values as an HTML unordered list
        analysis_results += "<ul>"
        for column, missing in missing_values_list.items():
            analysis_results += f"<li>{column}: {missing}</li>"
        analysis_results += "</ul>"
    
        # Return the path of the saved file along with analysis
        return analysis_results

    except Exception as e:
        # Handle errors
        return f"<h2>Error:</h2><p>{str(e)}</p>"

def data_visualization():
    try:
        # Load the dataset
        file_path = "data/raw/cardio_data.csv"
        if not os.path.exists(file_path):
            return "<h2>Error:</h2><p>File not found: Ensure the dataset is located at 'data/raw/cardio_data.csv'</p>"
        data = pd.read_csv(file_path)

        # Ensure the save directory exists
        save_dir = "files"
        os.makedirs(save_dir, exist_ok=True)

        # --------------------------------------------
        # First Visualization: Histograms by Categorical Columns
        # --------------------------------------------
        categorical_columns = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bp_category']
        final_table = pd.DataFrame(columns=['Column', 'Value', 'Cardio=0', 'Cardio=1'])

        for column in categorical_columns:
            grouped = data.groupby([column, 'cardio']).size().unstack(fill_value=0).reset_index()
            grouped.columns = ['Value', 'Cardio=0', 'Cardio=1']
            grouped['Column'] = column
            final_table = pd.concat([final_table, grouped], ignore_index=True)

        final_table = pd.concat([final_table, grouped], ignore_index=True)

        for column in categorical_columns:
            column_data = final_table[final_table['Column'] == column]
            plt.figure(figsize=(8, 6))
            plt.bar(column_data['Value'], column_data['Cardio=0'], alpha=0.7, label='Cardio=0')
            plt.bar(column_data['Value'], column_data['Cardio=1'], alpha=0.7, label='Cardio=1', bottom=column_data['Cardio=0'])
            plt.title(f"{column} vs Cardio")
            plt.xlabel(f"{column} Values")
            plt.ylabel("Count")
            plt.legend()
            plt.grid(True)
            plot_file = os.path.join(save_dir, f"{column}_vs_cardio_histogram.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()

        # --------------------------------------------
        # Second Visualization: General Histograms
        # --------------------------------------------
        all_columns = ['gender', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio', 'age_years', 'bmi', 'bp_category']

        for column in all_columns:
            grouped = data.groupby([column, 'cardio']).size().unstack(fill_value=0).reset_index()
            grouped.columns = ['Value', 'Cardio=0', 'Cardio=1']
            grouped['Column'] = column

        for column in all_columns:
            column_data = final_table[final_table['Column'] == column]
            plt.figure(figsize=(8, 6))
            plt.bar(column_data['Value'], column_data['Cardio=0'], alpha=0.7, label='Cardio=0')
            plt.bar(column_data['Value'], column_data['Cardio=1'], alpha=0.7, label='Cardio=1', bottom=column_data['Cardio=0'])
            plt.title(f"{column} vs Cardio")
            plt.xlabel(f"{column} Values")
            plt.ylabel("Count")
            plt.legend()
            plt.grid(True)
            plot_file = os.path.join(save_dir, f"{column}_vs_cardio_histogram_general.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()

        # --------------------------------------------
        # Third Visualization: Pie Charts
        # --------------------------------------------
        columns_to_plot = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio', 'bp_category']
        for column in columns_to_plot:
            value_counts = data[column].value_counts()
            plt.figure(figsize=(6, 6))
            plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
            plt.title(f"Distribution of {column}")
            plot_file = os.path.join(save_dir, f"{column}_distribution_pie.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()

        # Save the final DataFrame to a CSV file
        final_table_file = os.path.join(save_dir, "final_table.csv")
        final_table.to_csv(final_table_file, index=False)
        # Convert the final DataFrame to HTML
        final_table_html = final_table.to_html(index=False, classes="dataframe")

        return f"""
            <h2>Data Visualization Completed</h2>
            <p>All plots and the final table have been saved to the 'files' directory.</p>
            <p>Final table CSV saved at: {final_table_html}</p>
        """
    except Exception as e:
        return f"<h2>Error:</h2><p>{str(e)}</p>"
    



def feature_importance():
    try:
        # File path for the dataset
        file_path = "data/raw/cardio_data.csv"
        if not os.path.exists(file_path):
            return "<h2>Error:</h2><p>File not found: Ensure the dataset is located at 'data/raw/cardio_data.csv'</p>"

        # Load the dataset
        df = pd.read_csv(file_path)
         # Ensure required columns exist
        required_columns = ['id', 'cardio']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return f"<h2>Error:</h2><p>Missing required columns: {missing_columns}</p>"
        
        df = df[['gender', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio', 'age_years', 'bmi', 'bp_category_encoded']]
        # Create a copy of the dataset to preserve the original
        encoded_data = df.copy()

        # Initialize the LabelEncoder
        label_encoder = LabelEncoder()

        # Apply label encoding to all columns (if applicable)
        for column in encoded_data.columns:
            if encoded_data[column].dtype == 'object':  # Encode only object (categorical) columns
                encoded_data[column] = label_encoder.fit_transform(encoded_data[column])
        df = encoded_data
        
       

        # Separate features and target
        X = df.drop(columns=[ 'cardio'])
        y = df['cardio']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Model and Recursive Feature Elimination (RFE)
        model = RandomForestClassifier(random_state=42)
        rfe = RFE(estimator=model, n_features_to_select=5)
        rfe.fit(X_train, y_train)

        # Create ranking DataFrame
        ranking = pd.DataFrame({
            'Feature': X.columns,
            'Ranking': rfe.ranking_,
            'Support': rfe.support_
        }).sort_values(by='Ranking')

        # Convert ranking to HTML
        ranking_html = ranking.to_html(index=False, classes="dataframe")

        # Sort for visualization
        ranking_sorted = ranking.sort_values(by='Ranking')

        # Generate color palette
        colors = sns.color_palette('husl', len(ranking_sorted))

        # Plot feature importance
        plt.figure(figsize=(12, 8))
        plt.bar(ranking_sorted['Feature'], ranking_sorted['Ranking'], color=colors, edgecolor='black')
        plt.title('Feature Importance Rankings Based on RFE', fontsize=16)
        plt.xlabel('Features', fontsize=14)
        plt.ylabel('Importance Level (Lower is Better)', fontsize=14)
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Save the plot
        save_dir = "files"
        os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists
        plot_path = os.path.join(save_dir, "feature_importance_plot.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Return HTML table and confirm plot was saved
        return f"""
            <h2>Feature Importance</h2>
            {ranking_html}
            <p>Feature importance plot saved at: {plot_path}</p>
        """

    except Exception as e:
        # Log the error and return it
        return f"<h2>Error:</h2><p>{str(e)}</p>"



def data_preprocessing():
    # Example preprocessing logic
    return {"message": "Data preprocessing complete!"}


def model_apply():
    # Example model training logic
    return {"message": "Model trained successfully!"}

# Load the model and scaler
model_path = "model/model.pkl"
try:
    best_model = joblib.load(model_path)  # Load the trained model
except Exception as e:
    raise Exception(f"Error loading model or scaler: {str(e)}")

def predict(input_data: dict):
    """
    Predict the outcome based on input data using the pre-trained model.
    :param input_data: Dictionary containing input features.
    :return: Dictionary with prediction and probabilities.
    """
    # Convert JSON input to a DataFrame
    input_df = pd.DataFrame([input_data])

    # Ensure all required columns are present
    required_columns = ["ap_hi", "ap_lo", "cholesterol", "age_years", "bmi"]
    for col in required_columns:
        if col not in input_df.columns:
            return {"error": f"Missing required field: {col}"}

    try:
        # Normalize the input data using the same scaler as in training

        # Make predictions
        prediction = best_model.predict(input_df)

        # Return the result as a dictionary
        return {
            "input": input_data,
            "predicted_cardio": int(prediction[0]),  # 0: No, 1: Yes
        }
    except Exception as e:
        return {"error": f"Prediction error: {str(e)}"}
