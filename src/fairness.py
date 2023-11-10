import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

features_of_interest = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num',
       'marital-status', 'occupation', 'relationship', 'race', 'gender',
       'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']

def plot_FPDP(df, features_of_interest, save=False, save_path=None):
    """Plot Fairness Partial Dependence Plots

    Args:
        df (pandas DataFrame): dataframe of the date with the column 'prediction'
        features_of_interest (list of String): list of features for which FPDP is plotted
    """

    # Initialize empty dictionaries to store feature values and fairness test statistics
    feature_values = {}
    fairness_p_vals = {}

    # Iterate over each feature of interest
    for feature_name in features_of_interest:
        # Get unique values for the current feature
        unique_values = np.sort(df[feature_name].unique())
    
        # Initialize lists to store feature values and corresponding fairness test statistics
        values = []
        p_vals = []
    
        # Iterate over each unique feature value
        for value in unique_values:
            # Create a contingency table for the current feature value and the predicted outputs of the model
            contingency_table = pd.crosstab(df[feature_name] == value, df['prediction'])
        
            # Perform chi-squared test for statistical parity
            chi2, p, _, _ = chi2_contingency(contingency_table)
        
            # Append feature value and fairness test statistic to the lists
            values.append(value)
            #test_stats.append(chi2)
            p_vals.append(p)

    
        # Store feature values and test statistics in dictionaries
        feature_values[feature_name] = values
        fairness_p_vals[feature_name] = p_vals

    # Assuming features_of_interest, feature_values, and fairness_p_vals are defined

    # Calculate the number of rows needed
    n_features = len(features_of_interest)
    n_cols = 2
    n_rows = (n_features + n_cols - 1) // n_cols  # Ceiling division to ensure enough rows

    # Create subplots grid
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 6 * n_rows))
    axs = axs.flatten()  # Flatten the 2D array of axes to simplify indexing

    # Plot the FPDP for each feature
    for i, feature_name in enumerate(features_of_interest):
        ax = axs[i]
        ax.plot(feature_values[feature_name], fairness_p_vals[feature_name], marker='o', linestyle='-')
        plt.xticks(rotation=45) # For readability
        # Add a red line at p-value = 0.05 to the plot
        ax.axhline(y=0.05, color='red', linestyle='--', label='P-Value (0.05)')
        ax.set_xlabel(feature_name)
        ax.set_ylabel('p-value')
        ax.set_title(f'FPDP for {feature_name}')

    # Hide any unused subplots
    for i in range(n_features, n_rows * n_cols):
        axs[i].axis('off')

    # Save figures
    if save:
        plt.savefig(save_path)

    plt.tight_layout()
    plt.show()
