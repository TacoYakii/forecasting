import pickle 


def get_data(model_input_root, observed_root): 
    test_path = model_input_root / "test" / "sampled_forecasts.pkl" 
    train_validation_path = model_input_root / "validation" / "sampled_forecasts.pkl" 
    observed_path = observed_root / "observed.pkl" 
    is_valid = observed_root / "is_valid.pkl" 
    
    with open(test_path, "rb") as f: 
        test_data = pickle.load(f)
    with open(train_validation_path, "rb") as f: 
        train_validation_data = pickle.load(f)   
    with open(observed_path, "rb") as f: 
        observed_data = pickle.load(f)   
    with open(is_valid, "rb") as f: 
        is_valid_data = pickle.load(f)   
    
    return test_data, train_validation_data, observed_data, is_valid_data