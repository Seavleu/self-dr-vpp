def classify_demand(consumption, thresholds):
    if consumption < thresholds['low']:
        return "Low"
    elif consumption <= thresholds['moderate']:
        return "Moderate"
    else:
        return "High"