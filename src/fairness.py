import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency


def statistical_parity(df, interest_feature):
    """Statistical parity chi-squarred test

    Args:
        df (pandas DataFrame): data with predicted outcomes of the model
        interest_feature (String): feature on which the test is done (protected attribute)
    """
    # Create contingency table
    contingency_table = pd.crosstab(df[interest_feature], df['prediction'])

    # Chi-squarred test
    chi2, p, _, _ = chi2_contingency(contingency_table, correction=True)

    # Print results
    print(f"chi2 = {chi2}")
    print(f"p_value = {p}")
    if p < 0.05:
        print("There is no statistical parity (H0 rejected).")
    else:
        print("There is no proof that there is no statistical parity (H0 not rejected).")


def conditional_statistical_parity(df, dict_X_features, protected_attribute):
    """Conditional parity chi-squarred test

    Args:
        df (pandas DataFrame): data with predicted outcomes of the model
        dict_X_features (dict): keys are features on which there are subgroups, and values are list (bins)
        protected_attribute (String): ex. 'gender'

    #TODO:
    Manage categorical features of interest (no bins)
    """
    # Create a DataFrame combining true labels, predicted labels, and protected attributes (Gender), as well as the conditional features
    results_df = pd.DataFrame({'True_Labels': df["income"], 'Predicted_Labels': df["prediction"], protected_attribute: df[protected_attribute]})
    for key in dict_X_features.keys():
        results_df[key] = df[key]

    # Create bins for considering conditional features (creating subgroups)
    for key, value in dict_X_features.items():
        results_df[key + '_binned'] = pd.cut(results_df[key], value)

    # Calculate conditional counts for CSP considering conditional features
    list_groupby = [protected_attributes]
    for key in dict_X_features.keys():
        list_groupby.append(key + '_binned')
    list_groupby.append('Predicted_Labels')
    csp_counts = results_df.groupby(list_groupby).size().unstack(fill_value=0)

    print(csp_counts)

    # Perform the chi-squared test for each gender group
    protected_groups = results_df[protected_attribute].unique()
    chi2_values = {}

    for protected_group in protected_groups:
        sub_df = csp_counts.loc[protected_group]
        chi2, p, _, _ = chi2_contingency(sub_df)
        chi2_values[protected_group] = chi2

    # Calculate the critical value at a given significance level (e.g., 0.05)
    alpha = 0.05
    df = (len(csp_counts.columns) - 1) * (len(csp_counts.index) - 1)

    critical_value = chi2_contingency(sub_df, correction=False)[1]

    # Check if the chi-squared statistics for all gender groups are below the critical value
    csp_satisfied = all(chi2 <= critical_value for chi2 in chi2_values.values())

    print("chi2_values",chi2_values)
    print("critical_value",critical_value)

    if csp_satisfied:
        print("There is no statistical parity (H0 rejected).")
    else:
        print("There is no proof that there is no statistical parity (H0 not rejected).")


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

features_of_interest = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num',
       'marital-status', 'occupation', 'relationship', 'race', 'gender',
       'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
