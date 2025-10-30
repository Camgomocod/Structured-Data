import matplotlib.pyplot as plt 
import seaborn as sns 
import mplfinance as mpf
import pandas as pd 

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class BarsVisualizer:
    """Clase responsable de toda la visualización de barras"""
    
    def __init__(self, save_path=None):
        self.save_path = save_path
    
    def plot_comparison(self, df: pd.DataFrame, bars_dict: dict):
        """
        Compara múltiples tipos de barras
        
        Args:
            df: DataFrame con los trades originales
            bars_dict: Dict con formato {'nombre': (bars_df, color, marker)}
                      Ejemplo: {'TIBs': (tibs_df, 'red', 'x')}
        """
        n_bars = len(bars_dict)
        fig, axes = plt.subplots(n_bars, 1, figsize=(14, 3.5*n_bars), sharex=True)
        
        # Si solo hay un tipo de barra, axes no es lista
        if n_bars == 1:
            axes = [axes]
        
        for idx, (bar_name, (bars_df, color, marker)) in enumerate(bars_dict.items()):
            axes[idx].plot(df['timestamp'], df['price'], alpha=0.3, 
                          label='Trades', linewidth=0.5, color='gray')
            axes[idx].scatter(bars_df['close_time'], bars_df['close'], 
                            color=color, s=30, label=f'{bar_name} (n={len(bars_df)})', 
                            alpha=0.7, marker=marker)
            axes[idx].set_ylabel('Price (USD)')
            axes[idx].set_title(bar_name)
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time')
        plt.tight_layout()
        
        if self.save_path:
            plt.savefig(f'{self.save_path}/comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_single_bar(self, df: pd.DataFrame, bars_df: pd.DataFrame, 
                        bar_type: str, color='red'):
        """Plot para un tipo específico de barra"""
        plt.figure(figsize=(14, 6))
        plt.plot(df['timestamp'], df['price'], alpha=0.4, 
                label='Trades', linewidth=0.8, color='gray')
        plt.scatter(bars_df['close_time'], bars_df['close'], color=color, s=40, 
                   label=f'{bar_type} Close (n={len(bars_df)})', zorder=5)
        plt.legend()
        plt.title(f'{bar_type} - BTCUSDT')
        plt.xlabel('Time')
        plt.ylabel('Price (USD)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if self.save_path:
            plt.savefig(f'{self.save_path}/{bar_type}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def candle_chart(self, bars_df: pd.DataFrame, bar_type: str):
        """Grafica velas para las barras"""
        df_plot = bars_df.copy()
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
            savefig=f'{self.save_path}/{bar_type}_candles.png' if self.save_path else None
        )
    
    def plot_bar_statistics(self, bars_df: pd.DataFrame, bar_type: str):
        """Estadísticas de las barras generadas"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        
        # Distribución de ticks por barra
        axes[0, 0].hist(bars_df['n_ticks'], bins=50, color='steelblue', alpha=0.7)
        axes[0, 0].set_title('Distribution of Ticks per Bar')
        axes[0, 0].set_xlabel('Number of Ticks')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Distribución de volumen por barra
        axes[0, 1].hist(bars_df['volume'], bins=50, color='green', alpha=0.7)
        axes[0, 1].set_title('Distribution of Volume per Bar')
        axes[0, 1].set_xlabel('Volume')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Distribución de dollar value por barra
        axes[1, 0].hist(bars_df['dollar_value'], bins=50, color='orange', alpha=0.7)
        axes[1, 0].set_title('Distribution of Dollar Value per Bar')
        axes[1, 0].set_xlabel('Dollar Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Serie temporal del número de ticks
        axes[1, 1].plot(bars_df['close_time'], bars_df['n_ticks'], 
                       color='purple', linewidth=1, alpha=0.7)
        axes[1, 1].set_title('Ticks per Bar Over Time')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Number of Ticks')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'{bar_type} - Statistics', fontsize=16, y=1.00)
        plt.tight_layout()
        
        if self.save_path:
            plt.savefig(f'{self.save_path}/{bar_type}_stats.png', dpi=300, bbox_inches='tight')
        plt.show()