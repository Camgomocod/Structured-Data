import pandas as pd
import numpy as np


class ETFTrick:
    """
    Implementación del ETF Trick de López de Prado.
    
    Convierte una cesta de futuros en una serie temporal que refleja el valor
    de $1 invertido en el spread, evitando problemas como:
    - Pesos que cambian en el tiempo
    - Valores negativos en spreads
    - Desalineación de tiempos de trading
    - Costos de ejecución
    
    Basado en: Advances in Financial Machine Learning, Section 2.3
    """
    
    def __init__(self, save_path=None):
        self.save_path = save_path
        self.K_series = None  # Serie de valores $1 investment
        self.holdings_history = None  # Historial de holdings
        self.costs_history = None  # Historial de costos1
    
    def create_etf_series(
        self,
        bars_df: pd.DataFrame,
        weights: pd.DataFrame,
        rebalance_bars: list,
        transaction_costs: dict = None,
        K0: float = 1.0
    ) -> pd.DataFrame:
        """
        Crea la serie ETF virtual a partir de barras de múltiples instrumentos.
        
        Parameters:
        -----------
        bars_df : pd.DataFrame
            DataFrame con columnas multi-index (instrument, field) donde field incluye:
            - 'open': precio de apertura (o_i,t)
            - 'close': precio de cierre (p_i,t)
            - 'phi': valor USD de un punto (φ_i,t)
            - 'volume': volumen (v_i,t)
            - 'carry': carry/dividendo/cupón (d_i,t)
            
        weights : pd.DataFrame
            DataFrame con pesos ω_i,t para cada instrumento en cada barra.
            Index: bars, Columns: instruments
            
        rebalance_bars : list
            Lista de índices de barras donde se rebalancea (B ⊆ {1,...,T})
            
        transaction_costs : dict, optional
            Diccionario {instrument: cost} donde cost es τ_i (e.g., 1e-4 = 1bp)
            Si None, asume 0 para todos
            
        K0 : float, default=1.0
            AUM inicial
            
        Returns:
        --------
        pd.DataFrame con columnas:
            - 'K_t': Valor del ETF virtual
            - 'rebalance_cost': Costo de rebalanceo (c_t)
            - 'bidask_cost': Costo bid-ask para trading (c̃_t)
            - 'tradeable_volume': Volumen tradeable (v_t)
        """
        if transaction_costs is None:
            transaction_costs = {inst: 0.0 for inst in weights.columns}
        
        T = len(bars_df)
        instruments = weights.columns.tolist()
        I = len(instruments)
        
        # Inicializar estructuras
        K = np.zeros(T + 1)
        K[0] = K0
        
        holdings = pd.DataFrame(0.0, index=range(T + 1), columns=instruments)
        rebalance_costs = np.zeros(T)
        bidask_costs = np.zeros(T)
        tradeable_volumes = np.zeros(T)
        
        # Convertir rebalance_bars a set para búsqueda rápida
        rebalance_set = set(rebalance_bars)
        
        for t in range(1, T + 1):
            bar_idx = t - 1  # Índice del DataFrame (0-based)
            
            # Calcular holdings h_i,t
            if t in rebalance_set:
                # Rebalanceo: calcular nuevos holdings
                total_weight = sum(abs(weights.iloc[bar_idx, i]) for i in range(I))
                
                for i, inst in enumerate(instruments):
                    omega_it = weights.iloc[bar_idx, i]
                    
                    # Usar open del siguiente período como proxy
                    # En práctica, sería o_i,t+1, aquí usamos close como aproximación
                    if bar_idx + 1 < len(bars_df):
                        price_next = bars_df.iloc[bar_idx + 1][(inst, 'open')]
                    else:
                        price_next = bars_df.iloc[bar_idx][(inst, 'close')]
                    
                    phi_it = bars_df.iloc[bar_idx][(inst, 'phi')]
                    
                    if total_weight > 0:
                        holdings.iloc[t, i] = (omega_it * K[t-1]) / (price_next * phi_it * total_weight)
                    else:
                        holdings.iloc[t, i] = 0
            else:
                # No rebalanceo: mantener holdings previos
                holdings.iloc[t] = holdings.iloc[t-1]
            
            # Calcular δ_i,t (cambio de valor de mercado)
            delta = np.zeros(I)
            for i, inst in enumerate(instruments):
                if (t - 1) in rebalance_set:
                    # Si el período anterior fue rebalanceo, usar p_t - o_t
                    delta[i] = (bars_df.iloc[bar_idx][(inst, 'close')] - 
                               bars_df.iloc[bar_idx][(inst, 'open')])
                else:
                    # De otro modo, usar Δp_t (cambio de close)
                    if bar_idx > 0:
                        delta[i] = (bars_df.iloc[bar_idx][(inst, 'close')] - 
                                   bars_df.iloc[bar_idx-1][(inst, 'close')])
                    else:
                        delta[i] = 0
            
            # Actualizar K_t
            pnl = 0
            for i, inst in enumerate(instruments):
                h_prev = holdings.iloc[t-1, i]
                phi_it = bars_df.iloc[bar_idx][(inst, 'phi')]
                d_it = bars_df.iloc[bar_idx][(inst, 'carry')]
                
                pnl += h_prev * phi_it * (delta[i] + d_it)
            
            K[t] = K[t-1] + pnl
            
            # Calcular costos de rebalanceo c_t
            if t in rebalance_set:
                cost = 0
                for i, inst in enumerate(instruments):
                    h_prev = holdings.iloc[t-1, i]
                    h_curr = holdings.iloc[t, i]
                    p_it = bars_df.iloc[bar_idx][(inst, 'close')]
                    
                    if bar_idx + 1 < len(bars_df):
                        o_next = bars_df.iloc[bar_idx + 1][(inst, 'open')]
                    else:
                        o_next = p_it
                    
                    phi_it = bars_df.iloc[bar_idx][(inst, 'phi')]
                    tau_i = transaction_costs[inst]
                    
                    cost += (abs(h_prev) * p_it + abs(h_curr) * o_next) * phi_it * tau_i
                
                rebalance_costs[bar_idx] = cost
            
            # Calcular costo bid-ask c̃_t (para trading una unidad del ETF)
            bidask_cost = 0
            for i, inst in enumerate(instruments):
                h_prev = holdings.iloc[t-1, i]
                p_it = bars_df.iloc[bar_idx][(inst, 'close')]
                phi_it = bars_df.iloc[bar_idx][(inst, 'phi')]
                tau_i = transaction_costs[inst]
                
                bidask_cost += abs(h_prev) * p_it * phi_it * tau_i
            
            bidask_costs[bar_idx] = bidask_cost
            
            # Calcular volumen tradeable v_t
            min_volume = float('inf')
            for i, inst in enumerate(instruments):
                h_prev = holdings.iloc[t-1, i]
                v_it = bars_df.iloc[bar_idx][(inst, 'volume')]
                
                if abs(h_prev) > 0:
                    tradeable = v_it / abs(h_prev)
                    min_volume = min(min_volume, tradeable)
            
            tradeable_volumes[bar_idx] = min_volume if min_volume != float('inf') else 0
        
        # Crear DataFrame de salida
        result = pd.DataFrame({
            'K_t': K[1:],  # Excluir K0
            'rebalance_cost': rebalance_costs,
            'bidask_cost': bidask_costs,
            'tradeable_volume': tradeable_volumes
        }, index=bars_df.index)
        
        # Guardar para referencia
        self.K_series = result
        self.holdings_history = holdings.iloc[1:]  # Excluir t=0
        
        return result
    
    def get_returns(self) -> pd.Series:
        """Calcula los retornos del ETF virtual."""
        if self.K_series is None:
            raise ValueError("Debe ejecutar create_etf_series primero")
        
        return self.K_series['K_t'].pct_change().fillna(0)
    
    def get_log_returns(self) -> pd.Series:
        """Calcula los log-retornos del ETF virtual."""
        if self.K_series is None:
            raise ValueError("Debe ejecutar create_etf_series primero")
        
        return np.log(self.K_series['K_t'] / self.K_series['K_t'].shift(1)).fillna(0)
    
    def get_sharpe_ratio(self, periods_per_year: int = 252) -> float:
        """
        Calcula el Sharpe Ratio del ETF virtual.
        
        Parameters:
        -----------
        periods_per_year : int
            Número de períodos por año (252 para días, 52 para semanas, etc.)
        """
        returns = self.get_returns()
        
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        return np.sqrt(periods_per_year) * returns.mean() / returns.std()
    
    def save_series(self, filename: str = None):
        """Guarda la serie ETF en disco."""
        if self.K_series is None:
            raise ValueError("No hay serie para guardar")
        
        path = filename if filename else self.save_path
        if path:
            self.K_series.to_csv(path)
            print(f"Serie ETF guardada en: {path}")
        else:
            raise ValueError("Debe especificar un path para guardar")
    
    @staticmethod
    def prepare_bars_multiindex(
        instruments_data: dict,
        required_fields: list = ['open', 'close', 'phi', 'volume', 'carry']
    ) -> pd.DataFrame:
        """
        Método auxiliar para preparar el DataFrame con multi-index requerido.
        
        Parameters:
        -----------
        instruments_data : dict
            Diccionario {instrument_name: DataFrame} donde cada DataFrame tiene
            columnas 'open', 'close', 'phi', 'volume', 'carry'
            
        required_fields : list
            Lista de campos requeridos
            
        Returns:
        --------
        pd.DataFrame con multi-index (instrument, field)
        """
        result = pd.DataFrame()
        
        for inst_name, inst_df in instruments_data.items():
            for field in required_fields:
                if field not in inst_df.columns:
                    raise ValueError(f"Campo '{field}' no encontrado en {inst_name}")
                
                result[(inst_name, field)] = inst_df[field].values
        
        return result
