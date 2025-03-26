import joblib

# Load the feature names
feature_names = joblib.load('feature_names.pkl')

# Normalize feature names to lowercase
feature_names = [name.strip().lower() for name in feature_names]

# Re-save the feature names
joblib.dump(feature_names, 'feature_names.pkl')

print("Normalized feature names:", feature_names)