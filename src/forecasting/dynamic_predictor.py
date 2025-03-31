# File: src/forecasting/dynamic_scenarios.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def dynamic_thresholds(values, base_thresholds=None, dynamic_factor=0.1):
    """
    Compute dynamic thresholds based on the distribution of recent demand value
    For example, thresholds can be adjusted by a factor of the standard deviation

    Parameters:
        values (array-like): Array of demand values in kW
        base_thresholds (dict): Base thresholds with keys 'low' and 'moderate'
                                 Defaults to {'low': 1.5, 'moderate': 3.5}
        dynamic_factor (float): Factor to adjust thresholds based on std deviation

    Returns:
        dict: New thresholds.
    """
    if base_thresholds is None:
        base_thresholds = {'low': 1.5, 'moderate': 3.5}
    values = np.array(values)
    std_val = np.std(values)
    
    # adjust thresholds dynamically
    new_thresholds = {
        'low': base_thresholds['low'] + dynamic_factor * std_val,
        'moderate': base_thresholds['moderate'] + dynamic_factor * std_val
    }
    return new_thresholds

def classify_demand_dynamic(value, thresholds):
    """
    classify a single demand value based on dynamic thresholds
    """
    if value < thresholds['low']:
        return 'Low'
    elif value < thresholds['moderate']:
        return 'Moderate'
    else:
        return 'High'

def apply_dynamic_classification(values, base_thresholds=None, dynamic_factor=0.1):
    """
    Apply dynamic classification on an array of demand values.
    
    Parameters:
        values (array-like): Demand values (kW).
        base_thresholds (dict): Base thresholds.
        dynamic_factor (float): Factor to adjust thresholds.
        
    Returns:
        tuple: (Array of classification labels, used thresholds)
    """
    values = np.array(values).flatten()
    thresholds = dynamic_thresholds(values, base_thresholds, dynamic_factor)
    vectorized_classifier = np.vectorize(lambda x: classify_demand_dynamic(x, thresholds))
    labels = vectorized_classifier(values)
    return labels, thresholds

def evaluate_demand_classification_dynamic(actual, predicted, base_thresholds=None, dynamic_factor=0.1, plot=True):
    """
    Evaluate the dynamic classification of predicted vs. actual demand values """
    actual_labels, thresholds = apply_dynamic_classification(actual, base_thresholds, dynamic_factor)
    predicted_labels, _ = apply_dynamic_classification(predicted, base_thresholds, dynamic_factor)
    
    labels_order = ['Low', 'Moderate', 'High']
    cm = confusion_matrix(actual_labels, predicted_labels, labels=labels_order)
    report = classification_report(actual_labels, predicted_labels, labels=labels_order)
    
    if plot:
        plt.figure(figsize=(8, 4))
        sns.countplot(x=predicted_labels, order=labels_order, palette='viridis')
        plt.title('Predicted Demand Categories (Dynamic)')
        plt.xlabel('Category')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()
    
    return cm, report, thresholds

def dynamic_supply_simulation(demand_values, supply_function, tolerance=0.1):
    """
    Simulate supply conditions dynamically based on a provided supply function.
    
    Parameters:
        demand_values (array-like): Demand values in kW.
        supply_function (callable): A function that accepts an array of indices (or time values) and returns an array of supply values.
        tolerance (float): Tolerance ratio for balanced condition.
    
    Returns:
        tuple: (Array of supply condition labels, supply values)
    """
    demand_values = np.array(demand_values).flatten()
    time_indices = np.arange(len(demand_values))
    supply_values = supply_function(time_indices)
    
    def condition(d, s):
        if s > d * (1 + tolerance):
            return 'Over-supplied'
        elif s < d * (1 - tolerance):
            return 'Under-supplied'
        else:
            return 'Balanced'
    
    conditions = np.array([condition(d, s) for d, s in zip(demand_values, supply_values)])
    return conditions, supply_values

def plot_demand_vs_actual_with_dynamic_labels(actual, predicted, base_thresholds=None, dynamic_factor=0.1, supply_values=None):
    """
    Plot actual vs predicted demand with dynamic classification labels.
    
    Parameters:
        actual (array-like): Actual demand values (kW).
        predicted (array-like): Predicted demand values (kW).
        base_thresholds (dict): Base thresholds for classification.
        dynamic_factor (float): Factor for dynamic threshold adjustment.
        supply_values (array-like, optional): Supply values for overlay.
    """
    actual = np.array(actual).flatten()
    predicted = np.array(predicted).flatten()
    predicted_labels, thresholds = apply_dynamic_classification(predicted, base_thresholds, dynamic_factor)
    
    labels_order = ['Low', 'Moderate', 'High']
    colors = {'Low': 'green', 'Moderate': 'orange', 'High': 'red'}
    
    plt.figure(figsize=(14, 6))
    plt.plot(actual, label='Actual', color='blue', linewidth=1.5)
    plt.plot(predicted, label='Predicted', color='gray', alpha=0.6, linewidth=1.5)
    
    # Scatter predicted points color-coded by their label
    for label in labels_order:
        idx = np.where(predicted_labels == label)[0]
        plt.scatter(idx, predicted[idx], label=f'Predicted {label}', color=colors.get(label, 'black'), s=10)
    
    if supply_values is not None:
        plt.plot(supply_values, label='Supply', color='purple', linestyle='--', linewidth=1.5)
    
    plt.title('Predicted vs Actual Demand with Dynamic Classification')
    plt.xlabel('Time Steps')
    plt.ylabel('Power (kW)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
