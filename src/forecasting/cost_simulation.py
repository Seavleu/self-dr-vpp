import numpy as np
import matplotlib.pyplot as plt
import joblib

def recursive_forecast(model, initial_input, n_steps):
    """
    Generate a multi-step forecast using a recursive approach.
    
    Parameters:
        model: Trained one-step LSTM model
        initial_input (np.array): Input sequence of shape (look_back, 1)
        n_steps (int): Number of time steps (hours) to forecast
        
    Returns:
        np.array: Forecasted values (1D array) for n_steps
    """
    # initial_input is 2D
    if initial_input.ndim == 1:
        current_input = initial_input.reshape(-1, 1)
    else:
        current_input = initial_input.copy()
    
    forecast = []
    for i in range(n_steps):
        # reshape to (1, look_back, 1) as expected by the model
        input_for_model = current_input[np.newaxis, :, :]
        pred = model.predict(input_for_model)[0, 0]
        forecast.append(pred)
        # shift the input window and append the new prediction
        current_input = np.roll(current_input, -1, axis=0)
        current_input[-1, 0] = pred
    return np.array(forecast)


# --- Tariff Calculator Function for Residential Low Voltage ---
def calculate_residential_low_voltage_bill(monthly_kwh, is_summer=False):
    """
    calc monthly electricity bill for a residential low-voltage customer
    
    Parameters:
        monthly_kwh (float): Total monthly energy consumption (kWh)
        is_summer (bool): Whether the billing period is in summer
    
    Returns:
        float: Total estimated bill in KRW
    """
    # Tariff parameters (for both seasons, they are identical)
    if is_summer:
        demand_charge_1 = 910
        demand_charge_2 = 1600
        demand_charge_3 = 7300
        energy_charge_1 = 112.0
        energy_charge_2 = 206.6
        energy_charge_3 = 299.3
    else:
        demand_charge_1 = 910
        demand_charge_2 = 1600
        demand_charge_3 = 7300
        energy_charge_1 = 112.0
        energy_charge_2 = 206.6
        energy_charge_3 = 299.3

    # usage in tiers
    tier1_usage = min(300, monthly_kwh)
    tier2_usage = min(150, max(0, monthly_kwh - 300))  # 301-450 kWh => 150 kWh span
    tier3_usage = max(0, monthly_kwh - 450)

    # Demand charge (using highest applicable tier)
    if monthly_kwh <= 300:
        demand_charge = demand_charge_1
    elif monthly_kwh <= 450:
        demand_charge = demand_charge_2
    else:
        demand_charge = demand_charge_3

    # Energy charges per tier
    cost_energy_1 = tier1_usage * energy_charge_1
    cost_energy_2 = tier2_usage * energy_charge_2
    cost_energy_3 = tier3_usage * energy_charge_3
    total_energy_charge = cost_energy_1 + cost_energy_2 + cost_energy_3

    # Add fixed Climate Change & Fuel Cost pass-through (9 + 5 KRW/kWh)
    climate_fuel_cost = 14.0 * monthly_kwh

    total_bill = demand_charge + total_energy_charge + climate_fuel_cost
    return total_bill

# --- Integration with Forecasting Pipeline ---
# monthly forecast, forecast the next 30 days (30*24 hours)
n_hours_forecast = 30 * 24  # 720 hours
# Use the last sequence from test set as the initial input
# Define or load X_test before using it
# Assuming X_test is a NumPy array of shape (n_samples, look_back, 1)
# Replace the following line with the actual loading or definition of X_test
X_test = np.random.rand(100, 24, 1)  # Example: Random data for demonstration
initial_input = X_test[-1]  # Should be of shape (24, 1)
forecast_values = recursive_forecast(model, initial_input, n_hours_forecast)

# Inverse transform the forecasted values to kW
forecast_values_inv = scaler.inverse_transform(forecast_values.reshape(-1, 1)).flatten()

plt.figure(figsize=(8, 4))
plt.plot(forecast_values_inv[:100], marker='o', linestyle='-', label='Forecast')
plt.title('24-Hour (or longer) Forecast of Global Active Power Consumption')
plt.xlabel('Hour Ahead')
plt.ylabel('Predicted Demand (kW)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Aggregate the hourly forecast to a monthly total consumption (assuming constant usage for each hour)
monthly_kwh = forecast_values_inv.sum()  # kW * 1 hour = kWh
print(f"Predicted monthly consumption: {monthly_kwh:.2f} kWh")

# Assume if we know the season 
is_summer = True

# calc the estimated monthly bill
monthly_bill = calculate_residential_low_voltage_bill(monthly_kwh, is_summer)
print(f"Estimated monthly bill: {monthly_bill:,.0f} KRW")
