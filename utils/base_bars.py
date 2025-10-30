import numpy as np 
import pandas as pd 

class BaseBars:
    MIN_TICKS = 20
    def __init__(self):
        pass
    
    def get_prob_buy(self, bar_b_t, current_prob, lambada_):
        recent_prob = float((bar_b_t == 1).mean())
        return (1 - lambada_**2) * recent_prob + lambada_**2 * current_prob
    
    def get_tick_directions(self, df: pd.DataFrame) -> pd.Series:
        """Calcula la dirección de cada tick (buy=1, sell=-1)"""
        price_diff = df['price'].diff()
        b_t = np.sign(price_diff)
        b_t = b_t.replace(0, np.nan).ffill().fillna(1)
        return b_t 

    def _form_bar(self, bar):
        """Forma una barra OHLC a partir de un conjunto de ticks"""
        return {
            'open_time': bar['timestamp'].iloc[0],
            'close_time': bar['timestamp'].iloc[-1],
            'open': bar['price'].iloc[0],
            'high': bar['price'].max(),
            'low': bar['price'].min(),
            'close': bar['price'].iloc[-1],
            'n_ticks': len(bar),
            'volume': bar['quantity'].sum(),
            'dollar_value': bar['dollar_value'].sum(),
        }