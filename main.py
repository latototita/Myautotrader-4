import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pandas_ta as ta
import asyncio,os
from metaapi_cloud_sdk import MetaApi


# Initialize MetaApi client
token = os.getenv('TOKEN') or 'eyJhbGciOiJSUzUxMiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiI2YjI0NTQ0ZWYzMWI0NzQ4NWMxNzQ1NmUzNzdmYTlhZiIsInBlcm1pc3Npb25zIjpbXSwidG9rZW5JZCI6IjIwMjEwMjEzIiwiaW1wZXJzb25hdGVkIjpmYWxzZSwicmVhbFVzZXJJZCI6IjZiMjQ1NDRlZjMxYjQ3NDg1YzE3NDU2ZTM3N2ZhOWFmIiwiaWF0IjoxNjgzOTcwNDk4fQ.XzOt-R6egTLGb0fpmbxzDrLHqqlSbqdskeX3OSbx585bi_jG9BhSp-PtEyZ4kqJBafXcGmGRa8IMYQ6BMtDRmoiUd6InEjioBhPlKa6wrylTruPK6_YYq3LsZGd-GctHqW5-_pv3UtKyYriHO-P61dE-zpH6AAAO-NeAru-GKvOQeNwhwSVW_Q8Ov6Q6dljt0q9psxZYOU2jZiR1N3d0d_pQpvKLCgXFk71TL93GyEj-7csQ5Z0py0ChVioeWY7Cf-MlzEJdnSFgcHeFaKfny680C-5srBJwCO4EBVSEEqJao71fhnnK7UsW_QVMUoamVEBvbxD2Wr0F2pHcdIkVUoMrJeNiWdCTvdEONsg9xMFREqGdvlx66khNhvOpVvK_obsSEwMUS7Qvk3-3yh5F7PaT0qsQW4WdZVRaTLbayA7ChbYqCGvp4EAA4mxYTSxWjihDFCWHy6QWmHVzDw5JzhUxus-bWtOTiVVGUjg5e5uPNSHYUzN2D0Pl4p6QxnGISQCmRTuNtbEEm_9yLF_5xuRAdQez1VS0rYP0x3YauLmLIdhpmNKjNNfi13uAiwJVmjIj__9VDALqiGje25WFWr9BLQCUZdemGHe4q9bc2IcAjSZIo6auI6aVqkGwm7UpkHat_FMTxynZnhDNkrGebjgvuW4_1nbmWVUGZ4y2mTc'
accountId = os.getenv('ACCOUNT_ID') or '44d6fa31-2cd5-4aaa-b5ed-8189b2d4a0b5'



# Define trading parameters
symbol = 'EURUSD'
trade_volume = 0.01
stop_loss_factor = 0.01
take_profit_factor = 0.02

# Define technical indicator parameters
macd_fast_period = 12
macd_slow_period = 26
macd_signal_period = 9

# Define RandomForestClassifier parameters
n_estimators = 2
max_depth = 6
random_state = 42

# Define feature column names
feature_columns = ['MACD', 'RSI', 'ADX', 'SMI']




def calculate_features(df):
    # Calculate technical indicators using pandas_ta
    df.ta.macd(fast=macd_fast_period, slow=macd_slow_period, signal=macd_signal_period)
    df.ta.rsi()
    df.ta.adx()
    df.ta.smi()
    return df

def preprocess_data(df):
    # Preprocess the data
    df.dropna(inplace=True)
    df['MACD'] = np.where(df['MACD'] > 0, 1, -1)
    df['RSI'] = np.where(df['RSI'] > 70, -1, np.where(df['RSI'] < 30, 1, 0))
    df['ADX'] = np.where(df['ADX'] > 25, 1, 0)
    df['SMI'] = np.where(df['SMI'] > 0, 1, -1)
    return df

def prepare_features_labels(df):
    # Prepare feature matrix and labels
    X = df[feature_columns].values
    y = df['label'].values
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    return X, y

def train_model(X_train, y_train):
    # Train a Random Forest classifier
    rf_clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    rf_clf.fit(X_train, y_train)
    return rf_clf



async def create_market_order(trade_type, df):
    # Calculate historical volatility
    volatility = calculate_volatility(df)
    last_volatility = volatility.iloc[-1]

    # Calculate stop-loss and take-profit levels based on volatility
    stop_loss = df['close'].iloc[-1] - (last_volatility * stop_loss_factor)
    take_profit = df['close'].iloc[-1] + (last_volatility * take_profit_factor)

    # Place a trade using MetaAPI-Cloud-SDK
    trade = {
        'symbol': symbol,
        'volume': trade_volume,
        'stopLoss': stop_loss,
        'takeProfit': take_profit
    }
    if trade_type == 'buy':
        print('bought',stop_loss,take_profit)
    else:
        print('Sold',stop_loss,take_profit)
    



async def main():
    # Run the trading strategy
    while True:
        api = MetaApi(token)
        account = await api.metatrader_account_api.get_account(accountId)
        initial_state = account.state
        deployed_states = ['DEPLOYING', 'DEPLOYED']
        if initial_state not in deployed_states:
                # wait until account is deployed and connected to broker
                print('Deploying account')
                await account.deploy()
        print('Waiting for API server to connect to broker (may take a few minutes)')
        await account.wait_connected()
        
        # connect to MetaApi API
        connection = account.get_rpc_connection()
        await connection.connect()
        
        # wait until terminal state synchronized to the local state
        print('Waiting for SDK to synchronize to terminal state (may take some time depending on your history size)')
        await connection.wait_synchronized()
        print('start')
        # Fetch candlestick data
        print('getting candles')
        data = await account.get_historical_candles(symbol=symbol, timeframe='1m', start_time=None, limit=100)
        print(data)
        df = pd.DataFrame(data['candles'])
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        print('1')
        # Calculate technical indicators
        df = calculate_features(df)

        # Preprocess the data
        df = preprocess_data(df)
        print('2')
        # Prepare feature matrix and labels
        X, y = prepare_features_labels(df)

        # Train the ensemble of models
        print('3')
        models = []
        for _ in range(n_estimators):
            bootstrap_indices = np.random.choice(range(len(X)), size=len(X), replace=True)
            X_train = X[bootstrap_indices]
            y_train = y[bootstrap_indices]
            model = train_model(X_train, y_train)
            models.append(model)
            print('3.4')
        # Place trades based on predictions
        print('4')
        # Make predictions using the ensemble of models
        predictions = []
        for model in models:
            prediction = model.predict([df[feature_columns].values])[0]
            predictions.append(prediction)
        
        # Perform majority voting
        majority_prediction = np.bincount(predictions).argmax()
        predicted_label = label_encoder.inverse_transform([majority_prediction])[0]

        # Place trades based on majority prediction
        if predicted_label == 'buy':
            await create_market_order('buy', df)
        elif predicted_label == 'sell':
            print('Sell')
        print(predicted_label ,'ORM NONE')
        print('start end')
        await asyncio.sleep(3)  # Run the strategy every hour

asyncio.run(main())
