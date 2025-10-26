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
        
    def get_prob_buy(self, bar_b_t, current_prob, lambada_):
        recent_prob = float((bar_b_t == 1).mean())
        return (1 - lambada_**2) * recent_prob + lambada_**2 * current_prob
    
    def get_tick_directions(self, df: pd.DataFrame) -> pd.Series:
        price_diff = df['price'].diff()
        b_t = np.sign(price_diff)

        b_t = b_t.replace(0, np.nan).ffill().fillna(1)
        return b_t 

    def _form_bar(self, bar):
        return{
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

    def get_tibs(
        self, df: pd.DataFrame, exp_lambda = 0.9, init_exp_T = 1600
        )-> pd.DataFrame:

        df = df.copy()
        df['b_t'] = self.get_tick_directions(df)

        # initial
        E_T = init_exp_T
        prob_buy = 0.5
        bars = []
        tick_imbalance = 0 
        last_index = 0

        for i in range (1, len(df)):
            tick_imbalance += df['b_t'].iloc[i]

            # stop condition
            thereshold = E_T * abs(2 * prob_buy - 1)
            if abs(tick_imbalance) >= thereshold and (i - last_index + 1) >= InfoBars.MIN_TICKS:
                bar = df.iloc[last_index:i+1]
                bars.append(self._form_bar(bar))

                # update waited
                E_T = (1 - exp_lambda) * len(bar) + exp_lambda * E_T 
                prob_buy = float(self.get_prob_buy(bar['b_t'], prob_buy, exp_lambda))
                
                # Reset
                tick_imbalance = 0 
                last_index = i + 1

        return pd.DataFrame(bars)
    
    def get_dibs(self, df: pd.DataFrame, exp_lambda=0.9, init_exp_T=1600) -> pd.DataFrame:
        """
        Dollar Imbalance Bars (DIBs)
        Formula: E_0[θ_T] = E_0[T](2v+ - E_0[v_t])
        donde: v+ = P[b_t=1] * E_0[v_t|b_t=1]
               θ_T = Σ(b_t * dollar_value_t)
               v_t = dollar_value_t
        """
        df = df.copy()
        df['b_t'] = self.get_tick_directions(df)
        
        # Asegurar que existe la columna dollar_value
        if 'dollar_value' not in df.columns:
            df['dollar_value'] = df['price'] * df['quantity']
        
        # Inicialización usando los primeros ticks
        init_data = df.iloc[:min(init_exp_T, len(df))]
        E_T = init_exp_T
        E_v = init_data['dollar_value'].mean()  # E_0[v_t] donde v_t es dollar_value
        
        # Calcular v+ inicial
        buy_mask = init_data['b_t'] == 1
        if buy_mask.sum() > 0:
            E_v_buy = init_data.loc[buy_mask, 'dollar_value'].mean()  # E_0[v_t|b_t=1]
            P_buy = buy_mask.sum() / len(init_data)  # P[b_t=1]
            v_plus = P_buy * E_v_buy
        else:
            v_plus = E_v / 2
        
        bars = []
        theta = 0  # θ_T = Σ(b_t * dollar_value_t)
        last_index = 0
        
        for i in range(1, len(df)):
            # Acumular: θ_T = Σ(b_t * dollar_value_t)
            theta += df['b_t'].iloc[i] * df['dollar_value'].iloc[i]
            
            # E_0[θ_T] = E_0[T](2v+ - E_0[v_t])
            expected_theta = E_T * abs(2 * v_plus - E_v)
            
            # Condición de parada: |θ_T| >= E_0[|θ_T|]
            if abs(theta) >= expected_theta and (i - last_index + 1) >= InfoBars.MIN_TICKS:
                bar = df.iloc[last_index:i+1]
                bars.append(self._form_bar(bar))
                
                # Actualizar E_0[T] con EWMA
                T_actual = len(bar)
                E_T = exp_lambda * E_T + (1 - exp_lambda) * T_actual
                
                # Actualizar E_0[v_t] con EWMA
                E_v_new = bar['dollar_value'].mean()
                E_v = exp_lambda * E_v + (1 - exp_lambda) * E_v_new
                
                # Actualizar v+ = P[b_t=1] * E_0[v_t|b_t=1]
                buy_mask = bar['b_t'] == 1
                if buy_mask.sum() > 0:
                    E_v_buy_new = bar.loc[buy_mask, 'dollar_value'].mean()
                    P_buy_new = buy_mask.sum() / len(bar)
                    v_plus_new = P_buy_new * E_v_buy_new
                    v_plus = exp_lambda * v_plus + (1 - exp_lambda) * v_plus_new
                
                # Reset
                theta = 0
                last_index = i + 1
        
        return pd.DataFrame(bars)
    
    def get_vibs(self, df: pd.DataFrame, exp_lambda=0.9, init_exp_T=1600) -> pd.DataFrame:
        """
        Volume Imbalance Bars (VIBs)
        Formula: E_0[θ_T] = E_0[T](2v+ - E_0[v_t])
        donde: v+ = P[b_t=1] * E_0[v_t|b_t=1]
               θ_T = Σ(b_t * v_t)
        """
        df = df.copy()
        df['b_t'] = self.get_tick_directions(df)
        
        # Inicialización usando los primeros ticks
        init_data = df.iloc[:min(init_exp_T, len(df))]
        E_T = init_exp_T
        E_v = init_data['quantity'].mean()  # E_0[v_t]
        
        # Calcular v+ inicial
        buy_mask = init_data['b_t'] == 1
        if buy_mask.sum() > 0:
            E_v_buy = init_data.loc[buy_mask, 'quantity'].mean()  # E_0[v_t|b_t=1]
            P_buy = buy_mask.sum() / len(init_data)  # P[b_t=1]
            v_plus = P_buy * E_v_buy
        else:
            v_plus = E_v / 2
        
        bars = []
        theta = 0  # θ_T = Σ(b_t * v_t)
        last_index = 0
        
        for i in range(1, len(df)):
            # Acumular: θ_T = Σ(b_t * v_t)
            theta += df['b_t'].iloc[i] * df['quantity'].iloc[i]
            
            # E_0[θ_T] = E_0[T](2v+ - E_0[v_t])
            expected_theta = E_T * abs(2 * v_plus - E_v)
            
            # Condición de parada: |θ_T| >= E_0[|θ_T|]
            if abs(theta) >= expected_theta and (i - last_index + 1) >= InfoBars.MIN_TICKS:
                bar = df.iloc[last_index:i+1]
                bars.append(self._form_bar(bar))
                
                # Actualizar E_0[T] con EWMA
                T_actual = len(bar)
                E_T = exp_lambda * E_T + (1 - exp_lambda) * T_actual
                
                # Actualizar E_0[v_t] con EWMA
                E_v_new = bar['quantity'].mean()
                E_v = exp_lambda * E_v + (1 - exp_lambda) * E_v_new
                
                # Actualizar v+ = P[b_t=1] * E_0[v_t|b_t=1]
                buy_mask = bar['b_t'] == 1
                if buy_mask.sum() > 0:
                    E_v_buy_new = bar.loc[buy_mask, 'quantity'].mean()
                    P_buy_new = buy_mask.sum() / len(bar)
                    v_plus_new = P_buy_new * E_v_buy_new
                    v_plus = exp_lambda * v_plus + (1 - exp_lambda) * v_plus_new
                
                # Reset
                theta = 0
                last_index = i + 1
        
        return pd.DataFrame(bars)    
    

    def plot_comparison(self, df: pd.DataFrame, tibs: pd.DataFrame, 
                       vibs: pd.DataFrame, dibs: pd.DataFrame):
        """Compara los tres tipos de barras"""
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
            ylabel_lower='Volume')