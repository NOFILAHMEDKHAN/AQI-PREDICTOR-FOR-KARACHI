import hopsworks

project = hopsworks.login(api_key_value="HOPSWORKS_API_KEY")
mr = project.get_model_registry()

model_names = ["aqi_gb_pro", "aqi_rf_pro", "aqi_ridge_pro"]

for name in model_names:
    try:
        remote = mr.get_model(name)
        print(f"\nModel: {name}")
        print("Stored Metrics:", remote.training_metrics)
    except Exception as e:
        print(f"Cannot load {name}: {e}")
