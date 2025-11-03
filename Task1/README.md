# Weather Analysis Dashboard

A comprehensive weather data analysis and visualization tool that fetches real-time weather data from OpenWeatherMap API and generates professional dashboards with statistical insights.

## Features

- **Multi-City Weather Comparison**: Track weather across multiple cities simultaneously (Delhi, Visakhapatnam, London, New York)
- **Real-Time Data Fetching**: Get current weather conditions and 5-day forecasts
- **Professional Visualizations**: Generate publication-quality dashboards
- **Statistical Analysis**: Correlation analysis, temperature distributions, and trends
- **Data Export**: Save all data as CSV files for further analysis
- **Interactive CLI Menu**: User-friendly command-line interface

## Visualizations

### 1. Main Summary Dashboard
- **Current Temperature Snapshot**: Bar chart comparing temperatures across cities
- **Correlation Heatmap**: Relationships between temperature, humidity, wind, and pressure
- **5-Day Average Temperature Forecast**: Multi-city temperature trends over time

### 2. Deep Analysis Dashboard (Per City)
- **Temperature Distribution**: Histogram with KDE overlay showing temperature patterns
- **Diurnal Temperature Cycle**: Daily temperature variations across the forecast period
- **Wind Speed Distribution**: Box plots showing wind patterns by day
- **Precipitation Forecast**: Bar chart of rain probability over 5 days

### 3. Correlation Pairplot
- Scatter plots and correlations between all current weather metrics

## Requirements

- Python 3.7+
- OpenWeatherMap API key (free tier available)

## Installation

1. Clone or download this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

Note: If you encounter issues with `dateutil`, install it separately:
```bash
pip install python-dateutil
```

3. Create a `.env` file in the project root:
```
OPENWEATHER_API_KEY=your_api_key_here
```

Get your free API key from [OpenWeatherMap](https://openweathermap.org/api)

## Usage

Run the script:
```bash
python fetch_weather.py
```

### Menu Options

**Option 1: Main Summary Dashboard**
- Generates the multi-city comparison dashboard
- Saves current weather data as CSV

**Option 2: Deep Analysis Dashboard**  
- Generates detailed analysis for a selected city or all cities
- Includes temperature distribution, diurnal patterns, wind, and precipitation

**Option 3: All Plots**
- Generates all dashboards and correlation plots
- Complete analysis suite

**Option 4: CSVs Only**
- Fetches and saves all data without generating plots
- Useful for data export only

## Output Structure

```
outputs/
├── data/
│   ├── {timestamp}_current_weather.csv        # All cities current weather
│   ├── {timestamp}_{city}_3hour.csv            # 3-hour forecast (detailed)
│   └── {timestamp}_{city}_daily.csv            # Daily summary
└── figures/
    ├── {timestamp}_00_MAIN_DASHBOARD.png       # Main summary dashboard
    ├── {timestamp}_{city}_ANALYSIS_DASHBOARD.png  # City-specific analysis
    └── {timestamp}_01_current_correlations.png    # Correlation pairplot
```

## Data Metrics

### Current Weather
- Temperature (°C)
- Feels Like (°C)
- Humidity (%)
- Atmospheric Pressure (hPa)
- Wind Speed (m/s)
- Cloud Cover (%)
- Weather Conditions
- Visibility (m)
- Timestamp

### Forecast Data (3-hour intervals)
- All current metrics
- Precipitation Probability (%)
- Daily summaries (min/max temp, averages)

## Technical Details

- **API**: OpenWeatherMap Free Tier (5-day/3-hour forecast)
- **Timezone Handling**: Automatic conversion using dateutil with fallback
- **Error Handling**: Retry logic with exponential backoff for API calls
- **Visualization**: Matplotlib with Seaborn styling
- **Data Processing**: Pandas for efficient data manipulation

## Configuration

Edit `CITIES` dictionary in `fetch_weather.py` to add or modify cities:
```python
CITIES = {
    "City Name": {"lat": latitude, "lon": longitude, "country": "Country"},
}
```

Adjust request delay (default 1.2 seconds):
```python
REQUEST_DELAY_SEC = 1.2  # Modify as needed
```

## Example Output

The script provides terminal output with summary statistics:

```
--- SUMMARY STATISTICS ---
CURRENT WEATHER SUMMARY:
          City  Temperature (°C)  Humidity (%) Weather
       Delhi             20.06          88   Mist
 Visakhapatnam           26.78          89   Mist
      London             14.12          88   Rain
     New York            12.42          45  Clear

PER-CITY FORECAST STATISTICS:
>> Delhi
  • Avg Temp: 20.25°C
  • Min Temp: 18.50°C
  • Max Temp: 23.10°C
  • Avg Humidity: 85.23%
  • Most common: Mist
```

## Dependencies

- `python-dotenv` - Environment variable management
- `requests` - HTTP requests with retry logic
- `pandas` - Data processing and analysis
- `numpy` - Numerical operations
- `matplotlib` - Plotting and visualization
- `seaborn` - Statistical visualizations
- `urllib3` - HTTP adapter with retry strategy
- `python-dateutil` - Robust timezone handling

## License

This project is open source and available for educational and personal use.

## Acknowledgments

- Weather data provided by [OpenWeatherMap](https://openweathermap.org/)
- Visualization styling inspired by FiveThirtyEight

