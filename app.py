import sqlite3
from data import SQLRepository
from model import GarchModel
import streamlit as st
from config import settings
from data import AlphaVantageAPI
from glob import glob
import plotly.express as px
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import plotly.graph_objects as go
from plotly import figure_factory as ff



import joblib      # used for sumping and loading the model
from arch.univariate.base import ARCHModelResult
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf



av = AlphaVantageAPI()


#making a PredictIn class
class PredictIn:
    def __init__(self, ticker, n_days):
        self.ticker = ticker
        self.n_days = n_days

#making a PredictOut class
class PredictOut(PredictIn):
    def __init__(self, ticker, n_days, success, forecast, message):
        super().__init__(ticker, n_days)
        self.success = success
        self.forecast = forecast
        self.message = message

#making a FitIn class
class FitIn:
    def __init__(self, ticker, use_new_data, n_observations, p, q):
        self.ticker = ticker
        self.use_new_data = use_new_data
        self.n_observations = n_observations
        self.p = p
        self.q = q

#making a FitOut class
class FitOut(FitIn):
    def __init__(self, ticker, use_new_data, n_observations, p, q, success, message):
        super().__init__(ticker, use_new_data, n_observations, p, q)
        self.success = success
        self.message = message


def fetch_data(ticker):
    connection = sqlite3.connect(settings.db_name, check_same_thread=False)
    repo = SQLRepository(connection=connection)
    
    # Create the SQL query to fetch data for the given ticker
    sql = f"SELECT * FROM '{ticker}'"
    
    # Read the SQL query into a pandas DataFrame
    df = pd.read_sql(sql, con=connection, parse_dates=["date"], index_col="date")
    
    return df.dropna()


print(fetch_data("SUZLON.BSE"))



def build_model(ticker, use_new_data):
    # Create DB connection
    connection = sqlite3.connect(settings.db_name, check_same_thread=False)

    # Create SQLRepository
    repo = SQLRepository(connection=connection)
    
    # Create model
    model = GarchModel(ticker=ticker, repo=repo, use_new_data=use_new_data)

    # Return model
    return model





def fit_model(request):
    response = request.__dict__

    try:
        model = build_model(ticker=request.ticker, use_new_data=request.use_new_data)
        model.wrangle_data(n_observations=request.n_observations)
        model.fit(p=request.p, q=request.q)
        filename = model.dump()
        response["success"] = True
        response["message"] = f"Trained and saved '{filename}'."

    except Exception as e:
        response["success"] = False
        response["message"] = str(e)

    return response




def get_prediction(request):
    response = request.__dict__

    try:
        model = build_model(ticker=request.ticker, use_new_data=False)
        model.load()
        prediction = model.predict_volatility(horizon=request.n_days)
        response["success"] = True
        response["forecast"] = prediction
        response["message"] = ""

    except Exception as e:
        response["success"] = False
        response["forecast"] = {}
        response["message"] = str(e)

    return response




st.title('Stock Trend Prediction')
st.title('Made by Saif Chouaya')
st.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-Visit-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/saif-chouaya/)")
st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-Visit-blue?style=flat&logo=GitHub)](https://github.com/SaifChouaya)")


# User input for stock ticker
user_input = st.text_input('Enter Stock Ticker', 'SHOPERSTOP.BSE')

# User input for number of observations
n_observations = st.number_input('Number of Observations', min_value=1, value=2000)

# User input for p value
p_value = st.number_input('p Value', min_value=0, value=1)

# User input for q value
q_value = st.number_input('q Value', min_value=0, value=1)

# Button to trigger model fitting
if st.button('Fit Model'):
    # Call the fit_model function
    fit_request = FitIn(
        ticker=user_input,
        use_new_data=True,
        n_observations=n_observations,
        p=p_value,
        q=q_value
    )
    fit_response = fit_model(fit_request)
    st.write(fit_response)

# User input for number of days to predict
n_days = st.number_input('Number of Days to Predict', min_value=1, value=5)

# Button to trigger prediction
if st.button('Get Prediction'):
    # Call the get_prediction function
    predict_request = PredictIn(ticker=user_input, n_days=n_days)
    predict_response = get_prediction(predict_request)
    st.write(predict_response)
    

    # Describe Data
    st.subheader('Forecasting Volatility of the stock')
    # .write(df.describe())
    # Get the prediction data
    forecast = predict_response.get('forecast', {})
    print(forecast)
    fore = pd.DataFrame({'Date': list(forecast.keys()), 'Volatility': list(forecast.values())})

    if forecast:
        # Convert forecast to a pandas DataFrame for plotting
        fore = pd.DataFrame({'Date': list(forecast.keys()), 'Volatility': list(forecast.values())})
        Date = fore["Date"]
        

        # Get the number of periods
        num_periods = len(Date)

        # Calculate forecast start date
        start = pd.to_datetime(Date[0]) + pd.DateOffset(days=1)

        # Create date range
        Date = pd.bdate_range(start=start, periods=num_periods)

        # Create prediction index labels, ISO 8601 format
        Date = [d.isoformat().split('T')[0] for d in Date]
        Date = pd.to_datetime(fore['Date'])

        print(Date)
        print(fore['Volatility'])

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(Date, fore['Volatility'], label='Forecasted Volatility', color='blue')
        ax.set_title(f'Forecasted Volatility for {user_input} stock')
        ax.set_xlabel('Date')
        ax.set_ylabel('Volatility ($)')
        ax.grid(True)
        ax.legend()

# Display the plot in Streamlit
        st.pyplot(fig)

        # Set ticks to show every n-th date (adjust n as needed)
        n = 10  # Show every 5th date
        plt.xticks(Date[::n])

    else:
        st.write("No forecast data available.")



    # Display the DataFrame
    
        
    st.subheader('Information of the stock the stock')  
    data = fetch_data(ticker=user_input)
    st.write(data)

    st.subheader('The price of the stock')
    
    fig = px.line(data, x=data.index, y='close', title=f'Closing Price for {user_input} stock')
    st.plotly_chart(fig)


    # Return
    st.subheader('The Return of the stock')
    # Sort DataFrame ascending by date
    data.sort_index(ascending=True, inplace=True)
    # Create "return" column
    data['return'] = data["close"].pct_change() * 100
    
    fig = px.line(data, x=data.index, y='return', title=f'Return for {user_input} stock')
    st.plotly_chart(fig)



    # Distribution of Return
    # Create histogram of `data[return]`, 25 bins
    st.subheader('Distribution of the Return')
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.hist(data["return"].dropna(), bins=25, color='C0')
    ax.set_title("Distribution of Daily Returns")
    ax.set_xlabel("Daily Returns (%)")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)


    # Squared Return
    st.subheader('Squared Return of the stock')
    fig = px.line(data, x=data.index, y=data['return']**2, title=f'Squared Return for {user_input} stock')
    st.plotly_chart(fig)










    # Conditional Volatility Plot
    st.subheader('Conditional Volatility of the stock')

    # Calculate the rolling standard deviation of returns
    rolling_volatility = data['return'].rolling(window=20).std()

    # Plot data and conditional volatility using Plotly
    fig = go.Figure()

    # Add daily returns trace
    fig.add_trace(go.Scatter(x=data.index, y=data['return'], mode='lines', name='Daily Returns'))

    # Add rolling volatility trace
    fig.add_trace(go.Scatter(x=data.index, y=rolling_volatility, mode='lines', name='Rolling Volatility', line=dict(color='orange', width=3)))

    # Add -2 SD Conditional Volatility trace
    fig.add_trace(go.Scatter(x=data.index,
                            y=-2 * rolling_volatility,
                            mode='lines', name='-2 SD Conditional Volatility', line=dict(color='red', width=1), showlegend=False))

    # Add 2 SD Conditional Volatility trace
    fig.add_trace(go.Scatter(x=data.index,
                            y=2 * rolling_volatility,
                            mode='lines', name='2 SD Conditional Volatility', line=dict(color='red', width=1), showlegend=False))

    # Update layout
    fig.update_layout(title=f'{user_input} Daily Returns and Rolling Volatility',
                    xaxis_title='Date',
                    yaxis_title='Value',
                    legend_title='Legend')

    # Show figure
    st.plotly_chart(fig)






st.set_option('deprecation.showPyplotGlobalUse', False)