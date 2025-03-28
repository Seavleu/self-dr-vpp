'''
Dataset requried for this model is the same as the one used for ARIMA model.

Historical:
- Timestamps (datetime)
- Power usage (kWh) - ideally in 15 min/hrs/daily granularity intervals
- Consumer Id (optional)

Weather Data:
- Temperature (°C)
- Humidity (%)
- Solar radiation (W/m²)
- Wind speed (m/s)
- Rainfall (optional)
- Weather forecast (for long-term Transformer prediction)

Optional
- Day of week/holiday indicator
- Tariff rate per time block (dynamic pricing)
'''