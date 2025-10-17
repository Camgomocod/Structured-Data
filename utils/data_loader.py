import pandas as pd 
import numpy as np 
# funtions of data loader 

# class for convert object date to datetime
class DateConverter:

    def robust_to_datetime(series):
        # si ya es datetime, devolver
        if np.issubdtype(series.dtype, np.datetime64):
            return series
        
        s = series.copy()
        # 4a) Si todos (o mayoría) son dígitos -> intentar como epoch ms o s
        as_str = s.dropna().astype(str)
        is_all_digits = as_str.str.match(r'^\d+$').mean()  # proporción de strings sólo dígitos
        
        if is_all_digits > 0.5:
            # la mayoría son timestamps numéricos; inferir ms vs s por magnitud
            # convertir a float para inspección del tamaño (usa sample para evitar overflow)
            sample_num = as_str.sample(min(100, len(as_str))).astype(float)
            median_sample = sample_num.median()
            print("mediana de muestra numérica:", median_sample)
            if median_sample > 1e12:  # típico de ms epoch ( > ~10^12)
                print("-> inferido como epoch en MILLISEGUNDOS")
                out = pd.to_datetime(s.astype(float), unit='ms', errors='coerce')
            elif median_sample > 1e9:  # típico de s epoch ( > ~10^9)
                print("-> inferido como epoch en SEGUNDOS")
                out = pd.to_datetime(s.astype(float), unit='s', errors='coerce')
            else:
                # números pequeños: tratar como strings normalmente
                out = pd.to_datetime(s, errors='coerce')
            return out
        else:
            # 4b) hay muchos strings legibles - intentar parse directo
            # algunos formatos con milisegundos tienen punto decimal: '2024-12-01 05:00:59.999'
            # pd.to_datetime normalmente lo maneja. Usamos errors='coerce' y luego mostraremos fallos.
            out = pd.to_datetime(s, errors='coerce', utc=False)
            # si demasiados NaT, intentar forzar formatos comunes
            nat_frac = out.isna().mean()
            print(f"frac NaT tras parse directo: {nat_frac:.3f}")
            if nat_frac > 0.2:
                # intentar parse con varias plantillas comunes
                fmts = [
                    "%Y-%m-%d %H:%M:%S.%f",
                    "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%dT%H:%M:%S.%fZ",
                    "%Y-%m-%dT%H:%M:%SZ",
                ]
                for fmt in fmts:
                    try:
                        test = pd.to_datetime(s, format=fmt, errors='coerce')
                        nat_frac2 = test.isna().mean()
                        print(f"  intento con format {fmt} => NaT frac: {nat_frac2:.3f}")
                        if nat_frac2 < nat_frac:
                            out = test
                            nat_frac = nat_frac2
                    except Exception:
                        pass
            return out