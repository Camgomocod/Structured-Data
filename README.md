# Structured Data Project 

Theory involve in the project: 

- Time Bars 
- Tick Bars 
- Volume Bars 
- Dollar Bars (fixed and dynamics) 

## Structure

'''txt 
bars_comparison_project/
│
├── data/
│   ├── raw/               # Datos crudos (descargados desde la API)
│   ├── processed/         # Datos ya limpiados / con bars generadas
│
├── notebooks/
│   ├── 01_data_acquisition.ipynb
│   ├── 02_data_cleaning_and_preprocessing.ipynb
│   ├── 03_time_bars.ipynb
│   ├── 04_tick_bars.ipynb
│   ├── 05_volume_bars.ipynb
│   ├── 06_dollar_bars.ipynb
│   ├── 07_dynamic_dollar_bars.ipynb
│   ├── 08_comparison_and_statistics.ipynb
│   ├── 09_conclusions_and_visuals.ipynb
│
├── utils/
│   ├── data_loader.py         # Funciones para bajar datos
│   ├── bar_constructors.py    # Funciones para construir cada tipo de bar
│   ├── stats_tools.py         # Tests estadísticos (normalidad, autocorrelación, etc.)
│
└── README.md
'''


