import pandas as pd 
import numpy as np 

from utils.bars.base_bars import BaseBars as bb

class InfoBars:

    def __init__(self, save_path):
        self.save_path = save_path
        self.base_bars = bb()

    def get_tibs(
        self, df: pd.DataFrame, exp_lambda = 0.9, init_exp_T = 1600
        )-> pd.DataFrame:

        df = df.copy()
        df['b_t'] = self.base_bars.get_tick_directions(df)

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
            if abs(tick_imbalance) >= thereshold and (i - last_index + 1) >= bb.MIN_TICKS:
                bar = df.iloc[last_index:i+1]
                bars.append(self.base_bars._form_bar(bar))

                # update waited
                E_T = (1 - exp_lambda) * len(bar) + exp_lambda * E_T 
                prob_buy = float(self.base_bars.get_prob_buy(bar['b_t'], prob_buy, exp_lambda))
                
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
        df['b_t'] = self.base_bars.get_tick_directions(df)
        
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
            if abs(theta) >= expected_theta and (i - last_index + 1) >= bb.MIN_TICKS:
                bar = df.iloc[last_index:i+1]
                bars.append(self.base_bars._form_bar(bar))
                
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
        df['b_t'] = self.base_bars.get_tick_directions(df)
        
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
            if abs(theta) >= expected_theta and (i - last_index + 1) >= bb.MIN_TICKS:
                bar = df.iloc[last_index:i+1]
                bars.append(self.base_bars._form_bar(bar))
                
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