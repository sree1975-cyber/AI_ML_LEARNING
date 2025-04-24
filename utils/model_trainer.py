def train_xgboost(X, y, params):
    # Convert params
    if params['scale_pos_weight'] == 'auto':
        params['scale_pos_weight'] = sum(y == 0) / sum(y == 1)
    
    model = XGBClassifier(**params)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    model.save_model("model.xgb")
    return model
