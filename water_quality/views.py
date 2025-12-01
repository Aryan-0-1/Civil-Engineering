from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os

# -------------------------
# RENDER MAIN PAGE
# -------------------------
def water_quality(request):
    return render(request, 'water_quality.html')


# -------------------------
# CUSTOM INVERSE SCALER FOR MULTI-TARGET
# -------------------------
def inverse_targets(final_scaler, y_scaled, target_indices):
    """
    Undo scaling only for selected target indices from a MinMaxScaler that expects
    the full feature shape.
    """
    n_features = final_scaler.scale_.shape[0]
    dummy = np.zeros((len(y_scaled), n_features))

    for j, idx in enumerate(target_indices):
        dummy[:, idx] = y_scaled[:, j]

    inv = final_scaler.inverse_transform(dummy)
    return inv[:, target_indices]


# -------------------------
# MODEL PREDICTION FUNCTION
# -------------------------
def water_quality_model(start_date, end_date):

    df = pd.read_csv(r'water_quality/static/AssiGhat_processed.csv')

    # Ensure proper dtype for date
    df['date'] = pd.to_datetime(df['date'])

    # Add cyclic date encodings
    df['doy'] = df['date'].dt.dayofyear
    df['sin_doy'] = np.sin(2 * np.pi * df['doy'] / 365.25)
    df['cos_doy'] = np.cos(2 * np.pi * df['doy'] / 365.25)

    # Load GRU model
    model = load_model(
        "water_quality/static/gru_model.h5",
        custom_objects={"mse": tf.keras.losses.MeanSquaredError()}
    )

    LOOKBACK = 7
    features = ['DO', 'pH', 'sin_doy', 'cos_doy']

    # Fit single scaler for all features
    final_scaler = MinMaxScaler()
    final_scaler.fit(df[features].values.astype(float))

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    prediction_dates = pd.date_range(start=start_date, end=end_date)

    forecasted_data = []

    df_original = df.sort_values('date').reset_index(drop=True)

    # Last LOOKBACK rows as initial input sequence
    current_sequence = df_original[features].tail(LOOKBACK).values.copy()

    for current_pred_date in prediction_dates:

        # Compute DOY encoding for predicted date
        current_doy = current_pred_date.dayofyear
        sin_doy = np.sin(2 * np.pi * current_doy / 365.25)
        cos_doy = np.cos(2 * np.pi * current_doy / 365.25)

        # Scale input sequence
        scaled_sequence = final_scaler.transform(current_sequence)
        model_input = scaled_sequence.reshape(1, LOOKBACK, len(features))

        # Predict
        predicted_scaled = model.predict(model_input, verbose=0)

        target_idx = [features.index('DO'), features.index('pH')]
        predicted_values = inverse_targets(final_scaler, predicted_scaled, target_idx)

        pred_do = predicted_values[0, 0]
        pred_ph = predicted_values[0, 1]

        forecasted_data.append([current_pred_date, pred_do, pred_ph])

        # Construct new row
        new_row = [pred_do, pred_ph, sin_doy, cos_doy]

        # Update sequence (remove oldest, append newest)
        current_sequence = np.vstack([current_sequence[1:], np.array(new_row)])

    # Build output DataFrame
    forecast_df = pd.DataFrame(forecasted_data, columns=['date', 'DO_predicted', 'pH_predicted'])

    return forecast_df


# -------------------------
# PREDICT API VIEW
# -------------------------
def water_predict(request):
    if request.method == 'POST':
        start_date = request.POST.get('start_date')
        end_date = request.POST.get('end_date')

        forecast_df = water_quality_model(start_date, end_date)

        # Convert DataFrame to HTML
        table_html = forecast_df.to_html(index=False, classes="forecast-table")

        return JsonResponse({
            "dates": forecast_df['date'].astype(str).tolist(),
            "do": forecast_df['DO_predicted'].tolist(),
            "ph": forecast_df['pH_predicted'].tolist(),
            "table_html": table_html,
        })

    return JsonResponse({"error": "Invalid request"}, status=400)
