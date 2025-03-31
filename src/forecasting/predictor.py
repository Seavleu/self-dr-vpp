import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def classify_demand(value, thresholds=None):
    """
    classifying a single consumption value into demand categories
    
    Parameters:
        value (float): The consumption value in kW
        thresholds (dict): A dictionary with keys 'low' and 'moderate'.
                           Defaults to {'low': 1.5, 'moderate': 3.5}.
    Returns:
        str: 'Low', 'Moderate', or 'High'
    """
    if thresholds is None:
        thresholds = {'low': 1.5, 'moderate': 3.5}
    
    if value < thresholds['low']:
        return 'Low'
    elif value < thresholds['moderate']:
        return 'Moderate'
    else:
        return 'High'

def apply_demand_classification(values, thresholds=None):
    """
    apply demand classification to an array of consumption values
    """
    vectorized_classifier = np.vectorize(lambda x: classify_demand(x, thresholds))
    return vectorized_classifier(values)

def evaluate_demand_classification(actual, predicted, thresholds=None, plot=True):
    """
    evaluate the classification of predicted vs actual demand values
    
    Parameters:
        actual (array-like): Actual consumption values (kW)
        predicted (array-like): Predicted consumption values (kW)
        thresholds (dict): Thresholds for classification
        plot (bool): Whether to plot the distribution of predicted categories
    
    Returns:
        tuple: (confusion matrix, classification report as str)
    """
    actual_labels = apply_demand_classification(np.array(actual).flatten(), thresholds)
    predicted_labels = apply_demand_classification(np.array(predicted).flatten(), thresholds)
    
    # compute confusion matrix and classification report
    labels_order = ['Low', 'Moderate', 'High']
    cm = confusion_matrix(actual_labels, predicted_labels, labels=labels_order)
    report = classification_report(actual_labels, predicted_labels, labels=labels_order)
    
    if plot:
        plt.figure(figsize=(8, 4))
        sns.countplot(x=predicted_labels, order=labels_order, palette='viridis')
        plt.title('Predicted Demand Categories')
        plt.xlabel('Category')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()
    
    return cm, report

def simulate_supply_condition(demand, supply, tolerance=0.1):
    """
    simulate supply condition based on demand (kW) and supply (kW) values
    
    Parameters:
        demand (float or array-like): Demand value(s) in kW
        supply (float or array-like): Supply value(s) in kW
        tolerance (float): Tolerance ratio for balanced condition
        
    Returns:
        str or np.array: Condition(s) as 'Over-supplied', 'Under-supplied', or 'Balanced'
    """
    def condition(d, s):
        if s > d * (1 + tolerance):
            return 'Over-supplied'
        elif s < d * (1 - tolerance):
            return 'Under-supplied'
        else:
            return 'Balanced'
    
    if np.isscalar(demand):
        return condition(demand, supply)
    else:
        return np.array([condition(d, s) for d, s in zip(demand, supply)])

def plot_demand_vs_actual_with_labels(actual, predicted, thresholds=None):
    # values are numpy arr
    actual = np.array(actual).flatten()
    predicted = np.array(predicted).flatten()
    
    predicted_labels = apply_demand_classification(predicted, thresholds)
    labels_order = ['Low', 'Moderate', 'High']
    colors = {'Low': 'green', 'Moderate': 'orange', 'High': 'red'}
    
    plt.figure(figsize=(14, 6))
    plt.plot(actual, label='Actual', color='blue', linewidth=1.5)
    plt.plot(predicted, label='Predicted', color='gray', alpha=0.5, linewidth=1.5)
    
    # scatter plot for predicted values 
    for label in labels_order:
        idx = np.where(predicted_labels == label)[0]
        plt.scatter(idx, predicted[idx], label=f'Predicted {label}', color=colors[label], s=10)
    
    plt.title('Predicted vs Actual Demand (Color-coded by Category)')
    plt.xlabel('Time Steps')
    plt.ylabel('Power (kW)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
