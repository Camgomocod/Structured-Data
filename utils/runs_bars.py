import pandas as pd 
import numpy as np 

from utils.base_bars import BaseBars as bb

class RunBars:

    def __init__(self, save_path):
        self.save_path = save_path
        self.base_bars = bb()

    def get_trbs(self, df: pd.DataFrame, exp_lambda=0.9, init_exp_T=1600) -> pd.DataFrame:
        """
        Tick Run Bars (TRBs)
        Formula: θ_T = max{Σ(b_t|b_t=1), -Σ(b_t|b_t=-1)}
                 E_0[θ_T] = E_0[T] * max{P[b_t=1], P[b_t=-1]}
        """
        df = df.copy()
        df['b_t'] = self.base_bars.get_tick_directions(df)
        
        # Inicialización
        init_data = df.iloc[:min(init_exp_T, len(df))]
        E_T = init_exp_T
        P_buy = (init_data['b_t'] == 1).sum() / len(init_data)
        
        bars = []
        buy_run = 0   # Σ(b_t|b_t=1)
        sell_run = 0  # Σ(b_t|b_t=-1)
        last_index = 0
        
        for i in range(1, len(df)):
            b_t = df['b_t'].iloc[i]
            
            # Acumular runs sin offsetting
            if b_t == 1:
                buy_run += 1
            else:  # b_t == -1
                sell_run += 1
            
            # θ_T = max{buy_run, sell_run}
            theta = max(buy_run, sell_run)
            
            # E_0[θ_T] = E_0[T] * max{P[b_t=1], P[b_t=-1]}
            expected_theta = E_T * max(P_buy, 1 - P_buy)
            
            # Condición de parada: θ_T >= E_0[θ_T]
            if theta >= expected_theta and (i - last_index + 1) >= bb.MIN_TICKS:
                bar = df.iloc[last_index:i+1]
                bars.append(self.base_bars._form_bar(bar))
                
                # Actualizar E_0[T] con EWMA
                T_actual = len(bar)
                E_T = exp_lambda * E_T + (1 - exp_lambda) * T_actual
                
                # Actualizar P[b_t=1] con EWMA
                P_buy_new = (bar['b_t'] == 1).sum() / len(bar)
                P_buy = exp_lambda * P_buy + (1 - exp_lambda) * P_buy_new
                
                # Reset
                buy_run = 0
                sell_run = 0
                last_index = i + 1
        
        return pd.DataFrame(bars)
    
    def get_vrbs(self, df: pd.DataFrame, exp_lambda=0.9, init_exp_T=1600) -> pd.DataFrame:
        """
        Volume Run Bars (VRBs)
        Formula: θ_T = max{Σ(v_t|b_t=1), Σ(v_t|b_t=-1)}
                 E_0[θ_T] = E_0[T] * max{P[b_t=1]*E_0[v_t|b_t=1], P[b_t=-1]*E_0[v_t|b_t=-1]}
        """
        df = df.copy()
        df['b_t'] = self.base_bars.get_tick_directions(df)
        
        # Inicialización
        init_data = df.iloc[:min(init_exp_T, len(df))]
        E_T = init_exp_T
        
        buy_mask = init_data['b_t'] == 1
        sell_mask = init_data['b_t'] == -1
        
        P_buy = buy_mask.sum() / len(init_data)
        E_v_buy = init_data.loc[buy_mask, 'quantity'].mean() if buy_mask.sum() > 0 else 0
        E_v_sell = init_data.loc[sell_mask, 'quantity'].mean() if sell_mask.sum() > 0 else 0
        
        bars = []
        buy_run = 0   # Σ(v_t|b_t=1)
        sell_run = 0  # Σ(v_t|b_t=-1)
        last_index = 0
        
        for i in range(1, len(df)):
            b_t = df['b_t'].iloc[i]
            v_t = df['quantity'].iloc[i]
            
            # Acumular volume runs
            if b_t == 1:
                buy_run += v_t
            else:  # b_t == -1
                sell_run += v_t
            
            # θ_T = max{buy_run, sell_run}
            theta = max(buy_run, sell_run)
            
            # E_0[θ_T] = E_0[T] * max{P[b_t=1]*E_0[v_t|b_t=1], P[b_t=-1]*E_0[v_t|b_t=-1]}
            expected_buy = P_buy * E_v_buy
            expected_sell = (1 - P_buy) * E_v_sell
            expected_theta = E_T * max(expected_buy, expected_sell)
            
            # Condición de parada
            if theta >= expected_theta and (i - last_index + 1) >= bb.MIN_TICKS:
                bar = df.iloc[last_index:i+1]
                bars.append(self.base_bars._form_bar(bar))
                
                # Actualizar E_0[T]
                T_actual = len(bar)
                E_T = exp_lambda * E_T + (1 - exp_lambda) * T_actual
                
                # Actualizar probabilidades y valores esperados
                buy_mask = bar['b_t'] == 1
                sell_mask = bar['b_t'] == -1
                
                P_buy_new = buy_mask.sum() / len(bar)
                P_buy = exp_lambda * P_buy + (1 - exp_lambda) * P_buy_new
                
                if buy_mask.sum() > 0:
                    E_v_buy_new = bar.loc[buy_mask, 'quantity'].mean()
                    E_v_buy = exp_lambda * E_v_buy + (1 - exp_lambda) * E_v_buy_new
                
                if sell_mask.sum() > 0:
                    E_v_sell_new = bar.loc[sell_mask, 'quantity'].mean()
                    E_v_sell = exp_lambda * E_v_sell + (1 - exp_lambda) * E_v_sell_new
                
                # Reset
                buy_run = 0
                sell_run = 0
                last_index = i + 1
        
        return pd.DataFrame(bars)
    
    def get_drbs(self, df: pd.DataFrame, exp_lambda=0.9, init_exp_T=1600) -> pd.DataFrame:
        """
        Dollar Run Bars (DRBs)
        Formula: θ_T = max{Σ(d_t|b_t=1), Σ(d_t|b_t=-1)}
                 E_0[θ_T] = E_0[T] * max{P[b_t=1]*E_0[d_t|b_t=1], P[b_t=-1]*E_0[d_t|b_t=-1]}
        donde d_t = dollar_value_t
        """
        df = df.copy()
        df['b_t'] = self.base_bars.get_tick_directions(df)
        
        # Asegurar que existe la columna dollar_value
        if 'dollar_value' not in df.columns:
            df['dollar_value'] = df['price'] * df['quantity']
        
        # Inicialización
        init_data = df.iloc[:min(init_exp_T, len(df))]
        E_T = init_exp_T
        
        buy_mask = init_data['b_t'] == 1
        sell_mask = init_data['b_t'] == -1
        
        P_buy = buy_mask.sum() / len(init_data)
        E_d_buy = init_data.loc[buy_mask, 'dollar_value'].mean() if buy_mask.sum() > 0 else 0
        E_d_sell = init_data.loc[sell_mask, 'dollar_value'].mean() if sell_mask.sum() > 0 else 0
        
        bars = []
        buy_run = 0   # Σ(d_t|b_t=1)
        sell_run = 0  # Σ(d_t|b_t=-1)
        last_index = 0
        
        for i in range(1, len(df)):
            b_t = df['b_t'].iloc[i]
            d_t = df['dollar_value'].iloc[i]
            
            # Acumular dollar runs
            if b_t == 1:
                buy_run += d_t
            else:  # b_t == -1
                sell_run += d_t
            
            # θ_T = max{buy_run, sell_run}
            theta = max(buy_run, sell_run)
            
            # E_0[θ_T] = E_0[T] * max{P[b_t=1]*E_0[d_t|b_t=1], P[b_t=-1]*E_0[d_t|b_t=-1]}
            expected_buy = P_buy * E_d_buy
            expected_sell = (1 - P_buy) * E_d_sell
            expected_theta = E_T * max(expected_buy, expected_sell)
            
            # Condición de parada
            if theta >= expected_theta and (i - last_index + 1) >= bb.MIN_TICKS:
                bar = df.iloc[last_index:i+1]
                bars.append(self.base_bars._form_bar(bar))
                
                # Actualizar E_0[T]
                T_actual = len(bar)
                E_T = exp_lambda * E_T + (1 - exp_lambda) * T_actual
                
                # Actualizar probabilidades y valores esperados
                buy_mask = bar['b_t'] == 1
                sell_mask = bar['b_t'] == -1
                
                P_buy_new = buy_mask.sum() / len(bar)
                P_buy = exp_lambda * P_buy + (1 - exp_lambda) * P_buy_new
                
                if buy_mask.sum() > 0:
                    E_d_buy_new = bar.loc[buy_mask, 'dollar_value'].mean()
                    E_d_buy = exp_lambda * E_d_buy + (1 - exp_lambda) * E_d_buy_new
                
                if sell_mask.sum() > 0:
                    E_d_sell_new = bar.loc[sell_mask, 'dollar_value'].mean()
                    E_d_sell = exp_lambda * E_d_sell + (1 - exp_lambda) * E_d_sell_new
                
                # Reset
                buy_run = 0
                sell_run = 0
                last_index = i + 1
        
        return pd.DataFrame(bars)
