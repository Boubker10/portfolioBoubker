import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime

# Fonction pour afficher la série temporelle avec Plotly
def plot_time_series(data, title, ylabel, key):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data, mode='lines', name='Prix Ajusté', line=dict(color='blue')))
    fig.update_layout(title=title, xaxis_title='Date', yaxis_title=ylabel, hovermode="x unified")
    st.plotly_chart(fig, key=key)
    return fig

# Fonction pour afficher les log-returns
def plot_log_returns(data, key):
    log_returns = np.log(data / data.shift(1)).dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=log_returns.index, y=log_returns, mode='lines', name='Log Returns', line=dict(color='green')))
    fig.update_layout(title="Rendement (Log Returns)", xaxis_title='Date', yaxis_title='Log Returns', hovermode="x unified")
    st.plotly_chart(fig, key=key)
    return fig, log_returns

# Test de Dickey-Fuller pour stationnarité
def adf_test(series):
    result = adfuller(series.dropna())
    st.write(f"Test de Dickey-Fuller augmenté :")
    st.write(f"Statistique de test : {result[0]}")
    st.write(f"p-value : {result[1]}")
    st.write(f"Valeurs critiques : {result[4]}")

# Affichage des graphiques ACF et PACF
def plot_acf_pacf(data):
    fig_acf = plt.figure(figsize=(12, 6))
    ax1 = fig_acf.add_subplot(211)
    plot_acf(data, ax=ax1, lags=30)
    ax2 = fig_acf.add_subplot(212)
    plot_pacf(data, ax=ax2, lags=30)
    st.pyplot(fig_acf)  # Pas de 'key' ici

# Charger les données depuis yfinance
def load_yfinance_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data['Date'] = stock_data.index
    stock_data.set_index('Date', inplace=True)
    return stock_data

# Fonction pour exécuter Prophet
def run_prophet(stock_data_adj_close, forecast_steps):
    df_prophet = stock_data_adj_close.reset_index().rename(columns={"Date": "ds", "Adj Close": "y"})
    model_prophet = Prophet(daily_seasonality=True)
    model_prophet.fit(df_prophet)
    
    future_dates = model_prophet.make_future_dataframe(periods=forecast_steps)
    forecast = model_prophet.predict(future_dates)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], mode='lines', name='Données Historiques', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Prévisions', line=dict(color='red')))
    fig.update_layout(title=f'Prévisions avec Prophet pour {forecast_steps} jours', xaxis_title='Date', yaxis_title='Prix Ajusté (USD)', hovermode="x unified")
    st.plotly_chart(fig, key='prophet_forecast')

# Application principale Streamlit
def app():
    st.title("Prévision des Actions avec SARIMA, Prophet et Auto-ARIMA")

    # Sélectionner l'entreprise
    company = st.selectbox("Sélectionner une entreprise", ["AAPL - Apple", "NVDA - Nvidia", "FB - Facebook (Meta)", "TSLA - Tesla"])
    
    # Mapping des symboles
    tickers = {
        "AAPL - Apple": "AAPL",
        "NVDA - Nvidia": "NVDA",
        "FB - Facebook (Meta)": "META",
        "TSLA - Tesla": "TSLA"
    }

    # Sélectionner la plage de dates
    start_date = st.date_input("Sélectionner la date de début", value=datetime(2020, 1, 1))
    end_date = st.date_input("Sélectionner la date de fin", value=datetime.now())

    # Si les données n'existent pas dans session_state, les charger
    if 'stock_data' not in st.session_state:
        st.session_state['stock_data'] = None

    if st.button("Afficher les données timeseries"):
        # Télécharger les données via yfinance
        st.session_state['stock_data'] = load_yfinance_data(tickers[company], start_date, end_date)
        st.session_state['stock_data_adj_close'] = st.session_state['stock_data']['Adj Close'].astype(np.float32)

        st.subheader(f"Données Historiques - {company}")
        st.session_state['time_series_fig'] = plot_time_series(st.session_state['stock_data_adj_close'], f"Prix Ajusté des Actions ({start_date} - {end_date})", "Prix Ajusté (USD)", key="time_series")
        
        # Affichage des log returns
        st.session_state['log_returns_fig'], log_returns = plot_log_returns(st.session_state['stock_data_adj_close'], key="log_returns")

        # Test de Dickey-Fuller pour stationnarité
        st.subheader("Test de Stationnarité (ADF)")
        adf_test(st.session_state['stock_data_adj_close'])

        # Afficher les graphiques ACF et PACF
        st.subheader("ACF et PACF")
        st.session_state['acf_pacf_fig'] = plot_acf_pacf(log_returns)

    # Si les données sont déjà chargées
    if st.session_state['stock_data'] is not None:
        stock_data_adj_close = st.session_state['stock_data_adj_close']

        # Afficher les graphiques déjà générés
        if 'time_series_fig' in st.session_state:
            st.plotly_chart(st.session_state['time_series_fig'], key="time_series_reuse")
        if 'log_returns_fig' in st.session_state:
            st.plotly_chart(st.session_state['log_returns_fig'], key="log_returns_reuse")

        # Sélection du modèle de prévision
        model_choice = st.selectbox("Sélectionner le modèle de prévision", ["SARIMA", "Prophet"])

        # Sélection du nombre de jours pour la prévision
        st.subheader("Sélectionner le nombre de jours pour la prévision")
        forecast_steps = st.number_input("Combien de jours souhaitez-vous prévoir ?", min_value=1, max_value=365, value=30)

        if model_choice == "SARIMA":
            param_choice = st.radio("Choisir la méthode", ("Auto-ARIMA (optimisation automatique)", "Paramètres manuels"))

            if param_choice == "Paramètres manuels":
                # Champs de saisie pour les paramètres ARIMA/SARIMA
                p = st.number_input("Paramètre p (ordre AR)", min_value=0, max_value=5, value=1)
                d = st.number_input("Paramètre d (ordre de différenciation)", min_value=0, max_value=2, value=1)
                q = st.number_input("Paramètre q (ordre MA)", min_value=0, max_value=5, value=1)

                seasonal = st.checkbox("Inclure des composantes saisonnières (SARIMA)")
                if seasonal:
                    P = st.number_input("Paramètre P (ordre AR saisonnier)", min_value=0, max_value=5, value=1)
                    D = st.number_input("Paramètre D (ordre de différenciation saisonnier)", min_value=0, max_value=2, value=1)
                    Q = st.number_input("Paramètre Q (ordre MA saisonnier)", min_value=0, max_value=5, value=1)
                    s = st.number_input("Paramètre s (période saisonnière)", min_value=1, max_value=365, value=12)

            # Si l'utilisateur choisit Auto-ARIMA
            if param_choice == "Auto-ARIMA (optimisation automatique)":
                if st.button("Lancer la prévision avec SARIMA"):
                    st.subheader("Optimisation des paramètres avec auto-ARIMA")
                    stock_data_log = np.log(stock_data_adj_close)  # Transformation logarithmique pour stationnarité
                    auto_model = auto_arima(stock_data_log, start_p=1, start_q=1,
                                            max_p=5, max_q=5, m=12,
                                            start_P=0, seasonal=True,
                                            d=1, D=1, trace=True,
                                            error_action='ignore',
                                            suppress_warnings=True,
                                            stepwise=True)

                    st.write(auto_model.summary())
                    model = SARIMAX(stock_data_log, order=auto_model.order, seasonal_order=auto_model.seasonal_order)
                    results = model.fit()

                    forecast = results.get_forecast(steps=forecast_steps)
                    forecast_values = np.exp(forecast.predicted_mean)

                    # Affichage du graphique
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=stock_data_adj_close.index, y=stock_data_adj_close, mode='lines', name='Données Historiques'))
                    fig.add_trace(go.Scatter(x=forecast_values.index, y=forecast_values, mode='lines', name='Prévisions'))
                    fig.update_layout(title=f'Prévisions SARIMA (Auto-ARIMA) pour {forecast_steps} jours', xaxis_title='Date', yaxis_title='Prix Ajusté (USD)')
                    st.plotly_chart(fig, key='sarima_forecast')

            # Si l'utilisateur entre les paramètres manuels
            elif param_choice == "Paramètres manuels":
                if st.button("Lancer la prévision avec les paramètres manuels"):
                    stock_data_log = np.log(stock_data_adj_close)  # Transformation logarithmique pour stationnarité
                    if seasonal:
                        model = SARIMAX(stock_data_log, order=(p, d, q), seasonal_order=(P, D, Q, s))
                    else:
                        model = SARIMAX(stock_data_log, order=(p, d, q))

                    results = model.fit()

                    forecast = results.get_forecast(steps=forecast_steps)
                    forecast_values = np.exp(forecast.predicted_mean)

                    # Affichage du graphique
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=stock_data_adj_close.index, y=stock_data_adj_close, mode='lines', name='Données Historiques'))
                    fig.add_trace(go.Scatter(x=forecast_values.index, y=forecast_values, mode='lines', name='Prévisions'))
                    fig.update_layout(title=f'Prévisions SARIMA (paramètres manuels) pour {forecast_steps} jours', xaxis_title='Date', yaxis_title='Prix Ajusté (USD)')
                    st.plotly_chart(fig, key='sarima_manual_forecast')

        elif model_choice == "Prophet":
            if st.button("Lancer la prévision avec Prophet"):
                # Prévisions avec Prophet
                run_prophet(stock_data_adj_close, forecast_steps)

if __name__ == "__main__":
    app()
