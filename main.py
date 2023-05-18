import asyncio
import numpy as np
import os
from metaapi_cloud_sdk import MetaApi

# Initialize MetaApi client
# Initialize MetaApi client
token = os.getenv('TOKEN') or 'eyJhbGciOiJSUzUxMiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiI2YjI0NTQ0ZWYzMWI0NzQ4NWMxNzQ1NmUzNzdmYTlhZiIsInBlcm1pc3Npb25zIjpbXSwidG9rZW5JZCI6IjIwMjEwMjEzIiwiaW1wZXJzb25hdGVkIjpmYWxzZSwicmVhbFVzZXJJZCI6IjZiMjQ1NDRlZjMxYjQ3NDg1YzE3NDU2ZTM3N2ZhOWFmIiwiaWF0IjoxNjgzOTcwNDk4fQ.XzOt-R6egTLGb0fpmbxzDrLHqqlSbqdskeX3OSbx585bi_jG9BhSp-PtEyZ4kqJBafXcGmGRa8IMYQ6BMtDRmoiUd6InEjioBhPlKa6wrylTruPK6_YYq3LsZGd-GctHqW5-_pv3UtKyYriHO-P61dE-zpH6AAAO-NeAru-GKvOQeNwhwSVW_Q8Ov6Q6dljt0q9psxZYOU2jZiR1N3d0d_pQpvKLCgXFk71TL93GyEj-7csQ5Z0py0ChVioeWY7Cf-MlzEJdnSFgcHeFaKfny680C-5srBJwCO4EBVSEEqJao71fhnnK7UsW_QVMUoamVEBvbxD2Wr0F2pHcdIkVUoMrJeNiWdCTvdEONsg9xMFREqGdvlx66khNhvOpVvK_obsSEwMUS7Qvk3-3yh5F7PaT0qsQW4WdZVRaTLbayA7ChbYqCGvp4EAA4mxYTSxWjihDFCWHy6QWmHVzDw5JzhUxus-bWtOTiVVGUjg5e5uPNSHYUzN2D0Pl4p6QxnGISQCmRTuNtbEEm_9yLF_5xuRAdQez1VS0rYP0x3YauLmLIdhpmNKjNNfi13uAiwJVmjIj__9VDALqiGje25WFWr9BLQCUZdemGHe4q9bc2IcAjSZIo6auI6aVqkGwm7UpkHat_FMTxynZnhDNkrGebjgvuW4_1nbmWVUGZ4y2mTc'
accountId = os.getenv('ACCOUNT_ID') or '44d6fa31-2cd5-4aaa-b5ed-8189b2d4a0b5'

# Define parameters
symbol_list = ['XAUUSDm', 'GBPUSDm', 'XAGUSDm', 'AUDUSDm', 'EURUSDm', 'USDJPYm', 'GBPTRYm']
# Define indicator parameters
smi_period = 14
ema_period = 20
rsi_period = 14
bollinger_period = 20
bollinger_std = 2
adx_period = 14
ichimoku_conversion_line_period = 9
ichimoku_base_line_period = 26
ichimoku_lagging_span_period = 52
ichimoku_displacement = 26

async def main():
    # Connect to the MetaTrader account
    while True:
        api = MetaApi(token)
        try:
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
            k = True
        except Exception as e:
            print(f"Error connecting to MetaTrader account: {e}")
            k = False
            return

        while k:
            # Check for open trades
            trades = await connection.get_orders()
            if trades:
                print("There are open trades. Skipping analysis.")
            else:
                for symbol in symbol_list:
                    try:
                        # Fetch historical price data
                        candles = await account.get_historical_candles(symbol=symbol, timeframe='1m', start_time=None, limit=1000)
                        print('Fetched the latest candle data successfully')
                    except Exception as e:
                        print(f"Error retrieving candle data: {e}")

                    # Extract closing prices
                    close_prices = np.array([candle['close'] for candle in candles])
                    highest_high = np.max([candle['high'] for candle in candles[-smi_period:]])
                    lowest_low = np.min([candle['low'] for candle in candles[-smi_period:]])
                    smi_ema = np.convolve(close_prices[-smi_period:], np.ones(smi_period) / smi_period, mode='valid')

                    delta = np.diff(close_prices)
                    gain = np.zeros_like(delta)
                    loss = np.zeros_like(delta)
                    gain[delta > 0] = delta[delta > 0]
                    loss[delta < 0] = -delta[delta < 0]
                    avg_gain = np.convolve(gain, np.ones(rsi_period) / rsi_period, mode='valid')
                    avg_loss = np.convolve(loss, np.ones(rsi_period) / rsi_period, mode='valid')
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))

                    middle_band = np.mean(close_prices[-bollinger_period:])
                    std_dev = np.std(close_prices[-bollinger_period:])
                    upper_band = middle_band + bollinger_std * std_dev
                    lower_band = middle_band - bollinger_std * std_dev

                    up_move = np.diff(np.array([candle['high'] for candle in candles[-adx_period:]]))
                    down_move = -np.diff(np.array([candle['low'] for candle in candles[-adx_period:]]))
                    plus_dm = np.where(up_move > down_move, up_move, 0)
                    minus_dm = np.where(down_move > up_move, down_move, 0)
                    tr = np.maximum(np.maximum(candles[-adx_period:]['high'] - candles[-adx_period:]['low'],
                                            np.abs(candles[-adx_period:]['high'] - np.roll(candles[-adx_period:]['close'], 1))),
                                np.abs(candles[-adx_period:]['low'] - np.roll(candles[-adx_period:]['close'], 1)))
                    atr = np.convolve(tr, np.ones(adx_period) / adx_period, mode='valid')
                    plus_di = 100 * np.convolve(plus_dm, np.ones(adx_period) / atr, mode='valid')
                    minus_di = 100 * np.convolve(minus_dm, np.ones(adx_period) / atr, mode='valid')
                    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
                    adx = np.convolve(dx, np.ones(adx_period) / adx_period, mode='valid')

                    conversion_line = (np.max(np.array([candle['high'] for candle in candles[-ichimoku_conversion_line_period:]])) +
                                    np.min(np.array([candle['low'] for candle in candles[-ichimoku_conversion_line_period:]]))) / 2
                    base_line = (np.max(np.array([candle['high'] for candle in candles[-ichimoku_base_line_period:]])) +
                                np.min(np.array([candle['low'] for candle in candles[-ichimoku_base_line_period:]]))) / 2
                    leading_span_a = (conversion_line + base_line) / 2
                    leading_span_b = (np.max(np.array([candle['high'] for candle in candles[-ichimoku_lagging_span_period:]])) +
                                    np.min(np.array([candle['low'] for candle in candles[-ichimoku_lagging_span_period:]]))) / 2
                    leading_span_a_shifted = np.roll(leading_span_a, ichimoku_displacement)
                    leading_span_b_shifted = np.roll(leading_span_b, ichimoku_displacement)

                    # Identify buying and selling opportunities
                    buy_signal = (close_prices[-1] > smi_ema[-1] and rsi[-1] < 30 and
                                  close_prices[-1] < lower_band and adx[-1] > 25 and
                                  close_prices[-1] > leading_span_a_shifted[-1] and
                                  close_prices[-2] < leading_span_a_shifted[-2] and
                                  close_prices[-1] > leading_span_b_shifted[-1] and
                                  close_prices[-2] < leading_span_b_shifted[-2])

                    sell_signal = (close_prices[-1] < smi_ema[-1] and rsi[-1] > 70 and
                                   close_prices[-1] > upper_band and adx[-1] > 25 and
                                   close_prices[-1] < leading_span_a_shifted[-1] and
                                   close_prices[-2] > leading_span_a_shifted[-2] and
                                   close_prices[-1] < leading_span_b_shifted[-1] and
                                   close_prices[-2] > leading_span_b_shifted[-2])

                    # Execute trading orders
                    prices = await connection.get_symbol_price(symbol)
                    current_price = prices['ask']
                    if buy_signal:
                        # Calculate prices at pips above and below the current price
                        take_profit = current_price + (20 * 0.0001)  # Assuming 5 decimal places
                        stop_loss = current_price - (10 * 0.0001)
                        try:
                            # calculate margin required for trade
                            first_margin= await connection.calculate_margin({
                                'symbol': symbol,
                                'type': 'ORDER_TYPE_BUY',
                                'volume': 0.01,
                                'openPrice':  current_price
                            })
                            
                            if first_margin<((4/100)*10):
                                result = await connection.create_market_buy_order(
                                    symbol,
                                    0.01,
                                    stop_loss,
                                    take_profit,
                                    {'trailingStopLoss': {
                                            'distance': {
                                                'distance': 5,
                                                'units':'RELATIVE_BALANCE_PERCENTAGE'
                                            }
                                        }
                                    })
                            else:
                                pass
                            print('Trade successful, result code is ' + result['stringCode'])
                        except Exception as err:
                            print('Trade failed with error:')
                            print(api.format_error(err))
                    if sell_signal:
                        # Calculate prices at pips above and below the current price
                        stop_loss = current_price + (10 * 0.0001)  # Assuming 5 decimal places
                        take_profit = current_price - (20 * 0.0001)
                        try:
                            # calculate margin required for trade
                            first_margin= await connection.calculate_margin({
                                'symbol': symbol,
                                'type': 'ORDER_TYPE_SELL',
                                'volume': 0.01,
                                'openPrice':  current_price,
                            })
                            
                            if first_margin<((4/100)*10):
                                result = await connection.create_market_sell_order(
                                    symbol,
                                    0.01,
                                    stop_loss,
                                    take_profit,
                                    {'trailingStopLoss': {
                                            'distance': {
                                                'distance': 5,
                                                'units':'RELATIVE_BALANCE_PERCENTAGE'
                                            }
                                        }
                                    })
                            else:
                                pass
                        except Exception as err:
                            print('Trade failed with error:')
                            print(api.format_error(err))
                            
                    trades = await connection.get_orders()
                    if trades:
                        print("There are open trades. Skipping analysis.")
                        break
            await asyncio.sleep(60)  # Sleep for 1 minute before the next iteration
        await asyncio.sleep(60)  # Sleep for 1 minute before the next iteration
asyncio.run(main())
