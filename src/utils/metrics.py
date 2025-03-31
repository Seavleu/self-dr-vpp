import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import joblib
import json
import os
import datetime


def evaluate_model(model, X_test, y_test, scaler=None, plot=True, title='Prediction vs Actual', save_path=None):
    """
    Evaluate and visualize predictions from a trained model
    """
    y_pred = model.predict(X_test)

    if scaler is not None:
        y_test_inv = scaler.inverse_transform(y_test)
        y_pred_inv = scaler.inverse_transform(y_pred)
    else:
        y_test_inv = y_test
        y_pred_inv = y_pred

    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    r2 = r2_score(y_test_inv, y_pred_inv)

    if plot:
        plt.figure(figsize=(14, 6))
        plt.plot(y_test_inv[:500], label='Actual')
        plt.plot(y_pred_inv[:500], label='Predicted')
        plt.title(title)
        plt.xlabel('Time Steps')
        plt.ylabel('Power (kW)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")

    return mae, rmse, r2


def save_model(model, path='models/lstm_model.h5', scaler=None, scaler_path='models/scaler.pkl', history=None, history_path=None):
    """
    Save trained model, scaler, and training history
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    model.save(path)
    if scaler is not None:
        joblib.dump(scaler, scaler_path)
    if history is not None and history_path is not None:
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        with open(history_path, 'w') as f:
            json.dump(history.history, f)
    print(f"Model saved to {path}, scaler to {scaler_path}, and history to {history_path}")


def build_lstm_model(input_shape, units1=64, units2=32, dropout1=0.2, dropout2=0.2):
    """
    multi-layer LSTM model
    """
    model = Sequential([
        LSTM(units1, return_sequences=True, input_shape=input_shape),
        Dropout(dropout1),
        LSTM(units2),
        Dropout(dropout2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def train_model(model, X_train, y_train, config):
    """
    Train LSTM model with early stopping, TensorBoard logging, and model checkpointing.
    """
    save_cfg = config['save']
    early_stop = EarlyStopping(
        monitor=save_cfg['best_model_metric'],
        patience=save_cfg['best_model_metric_early_stopping'],
        restore_best_weights=True
    )

    checkpoint = ModelCheckpoint(
        filepath=save_cfg['best_model_path'],
        monitor=save_cfg['best_model_metric'],
        save_best_only=save_cfg['best_model_metric_save_best_only'],
        save_weights_only=save_cfg['best_model_metric_save_weights_only'],
        mode=save_cfg['best_model_metric_mode'],
        save_freq=save_cfg['best_model_metric_save_freq']
    )

    log_dir = os.path.join(save_cfg['tensorboard_log_dir'], datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_cb = TensorBoard(log_dir=log_dir)

    history = model.fit(
        X_train, y_train,
        epochs=config['train']['epochs'],
        batch_size=config['train']['batch_size'],
        validation_split=config['train']['validation_split'],
        callbacks=[early_stop, checkpoint, tensorboard_cb],
        verbose=1
    )

    return model, history


def walk_forward_cv(dataset, n_splits=3, look_back=24):
    split_size = len(dataset) // (n_splits + 1)
    results = []

    for i in range(n_splits):
        train = dataset[i * split_size:(i + 1) * split_size]
        test = dataset[(i + 1) * split_size:(i + 2) * split_size]

        def create_dataset(data):
            X, Y = [], []
            for j in range(len(data) - look_back):
                X.append(data[j:j + look_back])
                Y.append(data[j + look_back])
            return np.array(X), np.array(Y)

        X_train, y_train = create_dataset(train)
        X_test, y_test = create_dataset(test)

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        model = build_lstm_model((look_back, 1))
        model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=0)

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results.append({'fold': i + 1, 'MAE': mae, 'RMSE': rmse, 'R2': r2})
        print(f"Fold {i + 1} -- MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

    return results
