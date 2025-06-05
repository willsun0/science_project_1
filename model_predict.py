import pickle
import numpy as np



def predict_water_quality(features, selected_model):

    if selected_model=='xgb':
        # load the model from disk
        fileName_model = 'xgboost.pkl'
        loaded_model = pickle.load(open(fileName_model, 'rb'))
        print('Loaded ' +fileName_model+ ' successfully!!!')
    elif selected_model=='xgb_lag':
        # load the model from disk
        fileName_model = 'xgboost_lag.pkl'
        loaded_model = pickle.load(open(fileName_model, 'rb'))
        print('Loaded ' +fileName_model+ ' successfully!!!')       

    # features: list or np.array
    prediction = loaded_model.predict(features)
    # prediction = loaded_model.predict(np.array([features]))

    return prediction[0]