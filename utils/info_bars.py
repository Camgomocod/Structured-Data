import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import mplfinance as mpf

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class InfoBars:
    MIN_TICKS = 20
    
    def __init__(self, save_path):
        self.save_path = save_path
    
    def get_tick_directions(self, df: pd.DataFrame) -> pd.Series:
        """Calcula la dirección de cada tick usando la regla del tick"""
        price_diff = df['price'].diff()
        b_t = np.sign(price_diff)
        # Forward fill para ticks sin cambio de precio
        b_t = b_t.replace(0, np.nan).ffill().fillna(1)
        return b_t 
    
    def _form_bar(self, bar):
        """Forma una barra con la información del periodo"""
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
    
    def get_tibs(self, df: pd.DataFrame, exp_lambda=0.9, init_exp_T=1600) -> pd.DataFrame:
        """Tick Imbalance Bars (TIBs)"""
        df = df.copy()
        df['b_t'] = self.get_tick_directions(df)
        
        # Inicialización
        E_T = init_exp_T
        v_plus = 0.5  # P[b_t = 1]
        v_minus = 0.5  # P[b_t = -1]
        bars = []
        theta = 0  # Acumulador de imbalance
        last_index = 0
        
        for i in range(1, len(df)):
            theta += df['b_t'].iloc[i]
            
            # Calcular el threshold: E_0[θ_T] = E_0[T](v+ - v-)
            expected_theta = E_T * (v_plus - v_minus)
            
            # Condición de parada
            if abs(theta) >= abs(expected_theta) and (i - last_index + 1) >= InfoBars.MIN_TICKS:
                bar = df.iloc[last_index:i+1]
                bars.append(self._form_bar(bar))
                
                # Actualizar E_0[T] con EWMA
                T_actual = len(bar)
                E_T = exp_lambda * E_T + (1 - exp_lambda) * T_actual
                
                # Actualizar v+ y v- con EWMA
                buys = (bar['b_t'] == 1).sum()
                sells = (bar['b_t'] == -1).sum()
                total = buys + sells
                
                if total > 0:
                    v_plus_new = buys / total
                    v_minus_new = sells / total
                    v_plus = exp_lambda * v_plus + (1 - exp_lambda) * v_plus_new
                    v_minus = exp_lambda * v_minus + (1 - exp_lambda) * v_minus_new
                
                # Reset
                theta = 0
                last_index = i + 1
        
        return pd.DataFrame(bars)
    
    def get_vibs(self, df: pd.DataFrame, exp_lambda=0.9, init_exp_T=1600) -> pd.DataFrame:
        """Volume Imbalance Bars (VIBs)"""
        df = df.copy()
        df['b_t'] = self.get_tick_directions(df)
        
        # Inicialización
        E_T = init_exp_T
        E_v = df['quantity'].iloc[:init_exp_T].mean()  # E_0[v_t]
        v_plus = 0.5  # P[b_t = 1] * E_0[v_t|b_t = 1]
        bars = []
        theta = 0  # Acumulador de volume imbalance
        last_index = 0
        
        for i in range(1, len(df)):
            # θ_T = Σ(b_t * v_t)
            theta += df['b_t'].iloc[i] * df['quantity'].iloc[i]
            
            # Calcular el threshold: E_0[θ_T] = E_0[T](2v+ - E_0[v_t])
            expected_theta = E_T * (2 * v_plus - E_v)
            
            # Condición de parada
            if abs(theta) >= abs(expected_theta) and (i - last_index + 1) >= InfoBars.MIN_TICKS:
                bar = df.iloc[last_index:i+1]
                bars.append(self._form_bar(bar))
                
                # Actualizar E_0[T] con EWMA
                T_actual = len(bar)
                E_T = exp_lambda * E_T + (1 - exp_lambda) * T_actual
                
                # Actualizar E_0[v_t] con EWMA
                E_v_new = bar['quantity'].mean()
                E_v = exp_lambda * E_v + (1 - exp_lambda) * E_v_new
                
                # Actualizar v+ = P[b_t = 1] * E_0[v_t|b_t = 1]
                buy_volume = bar.loc[bar['b_t'] == 1, 'quantity'].sum()
                total_volume = bar['quantity'].sum()
                
                if total_volume > 0:
                    v_plus_new = buy_volume / total_volume
                    v_plus = exp_lambda * v_plus + (1 - exp_lambda) * v_plus_new
                
                # Reset
                theta = 0
                last_index = i + 1
        
        return pd.DataFrame(bars)
    
    def get_dibs(self, df: pd.DataFrame, exp_lambda=0.9, init_exp_T=1600) -> pd.DataFrame:
        """Dollar Imbalance Bars (DIBs)"""
        df = df.copy()
        df['b_t'] = self.get_tick_directions(df)
        
        # Asegurar que existe la columna dollar_value
        if 'dollar_value' not in df.columns:
            df['dollar_value'] = df['price'] * df['quantity']
        
        # Inicialización
        E_T = init_exp_T
        E_v = df['dollar_value'].iloc[:init_exp_T].mean()  # E_0[v_t] donde v_t es dollar value
        v_plus = 0.5  # P[b_t = 1] * E_0[v_t|b_t = 1]
        bars = []
        theta = 0  # Acumulador de dollar imbalance
        last_index = 0
        
        for i in range(1, len(df)):
            # θ_T = Σ(b_t * dollar_value_t)
            theta += df['b_t'].iloc[i] * df['dollar_value'].iloc[i]
            
            # Calcular el threshold: E_0[θ_T] = E_0[T](2v+ - E_0[v_t])
            expected_theta = E_T * (2 * v_plus - E_v)
            
            # Condición de parada
            if abs(theta) >= abs(expected_theta) and (i - last_index + 1) >= InfoBars.MIN_TICKS:
                bar = df.iloc[last_index:i+1]
                bars.append(self._form_bar(bar))
                
                # Actualizar E_0[T] con EWMA
                T_actual = len(bar)
                E_T = exp_lambda * E_T + (1 - exp_lambda) * T_actual
                
                # Actualizar E_0[v_t] con EWMA
                E_v_new = bar['dollar_value'].mean()
                E_v = exp_lambda * E_v + (1 - exp_lambda) * E_v_new
                
                # Actualizar v+ = P[b_t = 1] * E_0[v_t|b_t = 1]
                buy_dollars = bar.loc[bar['b_t'] == 1, 'dollar_value'].sum()
                total_dollars = bar['dollar_value'].sum()
                
                if total_dollars > 0:
                    v_plus_new = buy_dollars / total_dollars
                    v_plus = exp_lambda * v_plus + (1 - exp_lambda) * v_plus_new
                
                # Reset
                theta = 0
                last_index = i + 1
        
        return pd.DataFrame(bars)
    
    def plot_comparison(self, df: pd.DataFrame, tibs: pd.DataFrame, 
                       vibs: pd.DataFrame, dibs: pd.DataFrame):
        """Compara las tres tipos de barras"""
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        
        # TIBs
        axes[0].plot(df['timestamp'], df['price'], alpha=0.3, label='Trades', linewidth=0.5)
        axes[0].scatter(tibs['close_time'], tibs['close'], color='red', s=30, 
                       label=f'TIBs (n={len(tibs)})', alpha=0.7, marker='x')
        axes[0].set_ylabel('Price (USD)')
        axes[0].set_title('Tick Imbalance Bars')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # VIBs
        axes[1].plot(df['timestamp'], df['price'], alpha=0.3, label='Trades', linewidth=0.5)
        axes[1].scatter(vibs['close_time'], vibs['close'], color='green', s=30, 
                       label=f'VIBs (n={len(vibs)})', alpha=0.7, marker='o')
        axes[1].set_ylabel('Price (USD)')
        axes[1].set_title('Volume Imbalance Bars')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # DIBs
        axes[2].plot(df['timestamp'], df['price'], alpha=0.3, label='Trades', linewidth=0.5)
        axes[2].scatter(dibs['close_time'], dibs['close'], color='blue', s=30, 
                       label=f'DIBs (n={len(dibs)})', alpha=0.7, marker='s')
        axes[2].set_ylabel('Price (USD)')
        axes[2].set_xlabel('Time')
        axes[2].set_title('Dollar Imbalance Bars')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_bars(self, df: pd.DataFrame, info_bar: pd.DataFrame, bar_type='TIB'):
        """Plot para un tipo específico de barra"""
        plt.figure(figsize=(14, 6))
        plt.plot(df['timestamp'], df['price'], alpha=0.4, label='Trades', linewidth=0.8)
        plt.scatter(info_bar['close_time'], info_bar['close'], color='red', s=40, 
                   label=f'{bar_type} Close (n={len(info_bar)})', zorder=5)
        plt.legend()
        plt.title(f'{bar_type} - BTCUSDT')
        plt.xlabel('Time')
        plt.ylabel('Price (USD)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def candle_bars(self, info_bar: pd.DataFrame, bar_type='TIB'):
        """Grafica velas para las barras"""
        df_plot = info_bar.copy()
        df_plot.set_index('close_time', inplace=True)
        df_plot.index = pd.to_datetime(df_plot.index)
        
        mpf.plot(
            df_plot[['open', 'high', 'low', 'close', 'volume']],
            type='candle',
            style='charles',
            volume=True,
            title=f'{bar_type} - BTCUSDT',
            ylabel='Price (USD)',
            ylabel_lower='Volume',
        )