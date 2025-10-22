import requests
import pandas as pd
from datetime import datetime, timedelta
import time

class DowloadDataBinance: 
    def __init__(self, save_path):
        self.save_path = save_path
    
    def download_trades_binance(self, symbol='BTCUSDT', hours=12, max_trades=100000) -> pd.DataFrame:
        """
        Descarga trades individuales de Binance
        """
        print(f"🔄 Descargando trades de {symbol}...")
        
        url = "https://api.binance.com/api/v3/aggTrades"
        
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(hours=hours)).timestamp() * 1000)
        
        all_trades = []
        current_time = start_time
        iterations = 0
        max_iterations = 200
        
        while len(all_trades) < max_trades and iterations < max_iterations:
            params = {
                'symbol': symbol,
                'startTime': current_time,
                'endTime': end_time,
                'limit': 1000
            }
            
            try:
                response = requests.get(url, params=params)
                
                if response.status_code == 429:
                    print("⏸️  Rate limit, esperando 5s...")
                    time.sleep(5)
                    continue
                
                data = response.json()
                
                if isinstance(data, dict) and 'code' in data:
                    print(f"❌ Error API: {data.get('msg')}")
                    break
                
                if not data:
                    break
                
                all_trades.extend(data)
                current_time = data[-1]['T'] + 1
                
                print(f"📊 Trades: {len(all_trades):,} | Iter: {iterations+1}", end='\r')
                
                iterations += 1
                time.sleep(0.15)
                
            except Exception as e:
                print(f"\n❌ Error: {e}")
                break
        
        print(f"\n✓ Descargados: {len(all_trades):,} trades")
        
        if len(all_trades) < 100:
            print("⚠️  Muy pocos trades descargados")
            return None
        
        # Crear DataFrame
        df = pd.DataFrame(all_trades)
        df['timestamp'] = pd.to_datetime(df['T'], unit='ms')
        df['price'] = df['p'].astype(float)
        df['quantity'] = df['q'].astype(float)
        df['dollar_value'] = df['price'] * df['quantity']
        
        df = df.sort_values('timestamp').drop_duplicates(subset=['T']).reset_index(drop=True)
        df = df[['timestamp', 'price', 'quantity', 'dollar_value']].copy()
        
        # Guardar a CSV
        filename = f"{self.save_path}{symbol}_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filename, index=False)
        print(f"💾 Guardado en: {filename}")
        
        return df
    
    def show_info(self, trades: pd.DataFrame):
        print("\n" + "="*60)
        print("📊 INFORMACIÓN DE LOS DATOS")
        print("="*60)
        print(f"Total trades: {len(trades):,}")
        print(f"Período: {trades['timestamp'].min()} a {trades['timestamp'].max()}")
        print(f"Duración: {(trades['timestamp'].max() - trades['timestamp'].min()).total_seconds() / 3600:.1f} horas")
        print(f"\nPrimeros 5 trades:")
        print(trades.head())
        print(f"\nEstadísticas:")
        print(trades.describe())
         
