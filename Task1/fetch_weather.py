import os
import re
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta

import logging
import requests
import pandas as pd
import numpy as np

# --- Set matplotlib backend for headless environments ---
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
# ---

import seaborn as sns
from dotenv import load_dotenv
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# --- Import for robust timezone handling ---
from dateutil import tz


# -------------------------
# Configuration: Load Environment Variables
# -------------------------
load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")
if not API_KEY:
    raise EnvironmentError("OPENWEATHER_API_KEY not found. Add it to a .env file or environment variables.")

# -------------------------
# Configuration: API Endpoints and Settings
# -------------------------
CURRENT_URL = "https://api.openweathermap.org/data/2.5/weather"
FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"

# --- Configurable request delay ---
REQUEST_DELAY_SEC = float(os.getenv("REQUEST_DELAY_SEC", "1.2"))

# City coordinates for weather data fetching (latitude, longitude)
CITIES = {
    "Delhi": {"lat": 28.7041, "lon": 77.1025, "country": "India"},
    "Visakhapatnam": {"lat": 17.6869, "lon": 83.2185, "country": "India"},
    "London": {"lat": 51.5074, "lon": -0.1278, "country": "United Kingdom"},
    "New York": {"lat": 40.7128, "lon": -74.0060, "country": "United States"},
}

# Output directory structure
OUTPUT_DIR = Path("outputs")
FIGURES_DIR = OUTPUT_DIR / "figures"
DATA_DIR = OUTPUT_DIR / "data"
OUTPUT_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Logging Configuration
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("weather_final")

# -------------------------
# Visualization Style Configuration
# -------------------------
DEFAULT_STYLE = "fivethirtyeight"
try:
    plt.style.use(DEFAULT_STYLE)
except Exception:
    plt.style.use("seaborn-darkgrid")
sns.set_palette("husl")


# --- Filename slugify helper ---
def _slugify(text):
    """Converts a string to a safe, clean filename slug."""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9]+', '-', text).strip('-')
    return text


# -------------------------
# HTTP Fetcher Class with Retry Logic
# -------------------------
class WeatherDataFetcher:
    """
    Handles HTTP requests to OpenWeatherMap API with automatic retries.
    Includes exponential backoff for failed requests and proper error handling.
    """
    def __init__(self, api_key, timeout=10, retries=3, backoff=1):
        """
        Initialize fetcher with retry configuration.
        
        Args:
            api_key: OpenWeatherMap API key
            timeout: Request timeout in seconds (default: 10)
            retries: Maximum number of retries (default: 3)
            backoff: Exponential backoff factor (default: 1)
        """
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()
        retry_strategy = Retry(total=retries, status_forcelist=[429, 500, 502, 503, 504], backoff_factor=backoff, raise_on_status=False)
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def fetch_current_weather(self, lat, lon, units="metric"):
        """
        Fetch current weather data for a specific location.
        
        Args:
            lat: Latitude coordinate
            lon: Longitude coordinate
            units: Temperature units ('metric', 'imperial', 'kelvin')
            
        Returns:
            JSON response from API or None if request fails
        """
        params = {"lat": lat, "lon": lon, "units": units, "appid": self.api_key}
        try:
            resp = self.session.get(CURRENT_URL, params=params, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as he:
            # --- Safer status code access ---
            status = getattr(he.response, "status_code", None)
            if status == 401:
                logger.error("401 Unauthorized when fetching CURRENT weather. Check OPENWEATHER_API_KEY and plan.")
            else:
                logger.warning("HTTP error for CURRENT lat=%s lon=%s: %s", lat, lon, he)
            return None
        except requests.exceptions.RequestException as exc:
            logger.warning("Request error for CURRENT lat=%s lon=%s: %s", lat, lon, exc)
            return None

    def fetch_forecast_data(self, lat, lon, units="metric"):
        """
        Fetch 5-day weather forecast data (3-hour intervals) for a specific location.
        
        Args:
            lat: Latitude coordinate
            lon: Longitude coordinate
            units: Temperature units ('metric', 'imperial', 'kelvin')
            
        Returns:
            JSON response from API or None if request fails
        """
        params = {"lat": lat, "lon": lon, "units": units, "appid": self.api_key}
        try:
            resp = self.session.get(FORECAST_URL, params=params, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as he:
            # --- Safer status code access ---
            status = getattr(he.response, "status_code", None)
            if status == 401:
                logger.error("401 Unauthorized when fetching FORECAST. Check OPENWEATHER_API_KEY and plan.")
            else:
                logger.warning("HTTP error for FORECAST lat=%s lon=%s: %s", lat, lon, he)
            return None
        except requests.exceptions.RequestException as exc:
            logger.warning("Request error for FORECAST lat=%s lon=%s: %s", lat, lon, exc)
            return None

# -------------------------
# Data Processor Class: Timezone Handling & DataFrames
# -------------------------
class WeatherDataProcessor:
    """
    Processes raw weather API data into structured pandas DataFrames.
    Handles timezone conversions and data aggregation.
    """
    
    # --- Robust timezone conversion ---
    @staticmethod
    def _to_timestamp(dt_seconds, tz_name=None, tz_offset_seconds=None):
        """
        Convert UNIX seconds to tz-aware pandas.Timestamp
        using dateutil.tz.tzoffset for robust fixed offsets.
        """
        ts_utc = pd.to_datetime(dt_seconds, unit="s", utc=True)
        # 1. Try named timezone (e.g., 'Asia/Kolkata')
        if tz_name:
            try:
                return ts_utc.tz_convert(tz_name)
            except Exception:
                logger.debug("tz_name conversion failed (%s) — falling back to offset", tz_name)
        # 2. Fallback: numeric offset (seconds)
        if tz_offset_seconds is not None:
            try:
                # Use dateutil.tz.tzoffset for robust, fixed offset
                tzinfo_obj = tz.tzoffset(None, int(tz_offset_seconds))
                return ts_utc.tz_convert(tzinfo_obj)
            except Exception as e:
                logger.warning(f"tz_offset conversion failed ({e}); returning UTC timestamp")
        # 3. Last resort
        return ts_utc

    def extract_current_weather_dataframe(self, all_cities_current_data):
        """
        Convert current weather JSON data to a pandas DataFrame.
        
        Args:
            all_cities_current_data: Dictionary of {city_name: json_data}
            
        Returns:
            DataFrame with current weather metrics for all cities
        """
        records = []
        for city, resp in all_cities_current_data.items():
            if not resp:
                continue
            tz_offset = resp.get("timezone", 0)
            ts = self._to_timestamp(resp.get("dt", 0), tz_offset_seconds=tz_offset)
            weather_desc = (resp.get("weather") or [{}])[0].get("main", "Unknown")
            records.append({
                "City": city,
                "Temperature (°C)": round(resp.get("main", {}).get("temp", np.nan), 2),
                "Feels Like (°C)": round(resp.get("main", {}).get("feels_like", np.nan), 2),
                "Humidity (%)": resp.get("main", {}).get("humidity", np.nan),
                "Pressure (hPa)": resp.get("main", {}).get("pressure", np.nan),
                "Wind Speed (m/s)": round(resp.get("wind", {}).get("speed", np.nan), 2),
                "Cloud Cover (%)": resp.get("clouds", {}).get("all", np.nan),
                "Weather": weather_desc,
                "Visibility (m)": resp.get("visibility", np.nan),
                "Timestamp": ts,
            })
        return pd.DataFrame(records)

    def extract_3hour_forecast_dataframe(self, forecast_json, city_name):
        """
        Convert 5-day forecast JSON to a pandas DataFrame (3-hour intervals).
        
        Args:
            forecast_json: Raw forecast JSON from API
            city_name: Name of the city
            
        Returns:
            DataFrame with forecast data including timestamps, temperatures, and weather conditions
        """
        records = []
        if not forecast_json or "list" not in forecast_json:
            return pd.DataFrame(records)
        tz_offset = forecast_json.get("city", {}).get("timezone", 0)
        for item in forecast_json["list"]:
            weather_desc = (item.get("weather") or [{}])[0].get("main", "Unknown")
            ts = self._to_timestamp(item.get("dt", 0), tz_offset_seconds=tz_offset)
            records.append({
                "City": city_name,
                "Timestamp": pd.Timestamp(ts),
                "DateStr": pd.Timestamp(ts).strftime('%a %d'),
                "Hour": pd.Timestamp(ts).hour,
                "Temperature (°C)": round(item.get("main", {}).get("temp", np.nan), 2),
                "Feels Like (°C)": round(item.get("main", {}).get("feels_like", np.nan), 2),
                "Humidity (%)": item.get("main", {}).get("humidity", np.nan),
                "Pressure (hPa)": round(item.get("main", {}).get("pressure", np.nan), 2),
                "Wind Speed (m/s)": round(item.get("wind", {}).get("speed", np.nan), 2),
                "Clouds (%)": item.get("clouds", {}).get("all", np.nan),
                "Weather": weather_desc,
                "Precipitation Prob (%)": round(item.get("pop", 0) * 100, 2) if item.get("pop") is not None else np.nan,
            })
        df = pd.DataFrame(records)
        if not df.empty:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        return df

    def summarize_daily_from_hourly(self, df_hourly):
        """
        Aggregate 3-hourly forecast data into daily summaries.
        
        Args:
            df_hourly: DataFrame with hourly forecast data
            
        Returns:
            DataFrame with daily min/max temps, averages, and day names
        """
        if df_hourly is None or df_hourly.empty:
            return pd.DataFrame()
        df = df_hourly.set_index("Timestamp")
        agg = df.resample("D").agg(
            Temp_Min=("Temperature (°C)", "min"),
            Temp_Max=("Temperature (°C)", "max"),
            Avg_Humidity=("Humidity (%)", "mean"),
            Avg_Wind=("Wind Speed (m/s)", "mean"),
            Precip_Prob_Max=("Precipitation Prob (%)", "max"),
        )
        agg = agg.reset_index().rename(columns={"Timestamp": "Date"})
        agg["Date"] = pd.to_datetime(agg["Date"])
        agg["Day"] = agg["Date"].dt.strftime("%A")
        return agg

# -------------------------
# Visualization Class: Dashboard Generation
# -------------------------
class WeatherVisualizations:
    """
    Generates publication-quality weather analysis dashboards.
    Creates summary dashboards, correlation plots, and city-specific analyses.
    """
    
    @staticmethod
    def plot_summary_dashboard(df_current, all_daily, run_ts, save_path):
        """
        Generates a comprehensive 2x2 dashboard with multi-city weather comparison.
        
        Includes:
        - Current temperature comparison (bar chart)
        - Correlation heatmap (temperature, humidity, wind, pressure)
        - 5-day average temperature forecast (line chart across cities)
        
        Args:
            df_current: DataFrame with current weather for all cities
            all_daily: Dictionary of {city: daily_summary_df}
            run_ts: Timestamp string for file naming
            save_path: Path where dashboard will be saved
        """
        logger.info("Generating Main Summary Dashboard...")
        with plt.style.context(DEFAULT_STYLE):
            fig = plt.figure(figsize=(20, 16))
            gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.2)
            
            ax1 = fig.add_subplot(gs[0, 0]) # Top-left
            ax2 = fig.add_subplot(gs[0, 1]) # Top-right
            ax3 = fig.add_subplot(gs[1, :]) # Bottom row (full width)
            
            # --- Plot 1: Current Temperature Comparison (on ax1) ---
            if not df_current.empty:
                cities = df_current["City"]
                temps = df_current["Temperature (°C)"]
                norm = plt.Normalize(temps.min() - 5, temps.max() + 5)
                colors = plt.cm.coolwarm(norm(temps.values))
                bars = ax1.bar(cities, temps, color=colors, edgecolor="black", linewidth=0.8, zorder=3)
                ax1.set_title("Current Temperature Snapshot", fontsize=16, fontweight="bold", pad=10)
                ax1.set_ylabel("Temperature (°C)")
                ax1.grid(axis="y", linestyle="--", alpha=0.6)
                for spine in ["top", "right"]: ax1.spines[spine].set_visible(False)
                for bar in bars:
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height}°C", ha="center", va="bottom", fontsize=10, fontweight="bold")
            else:
                ax1.text(0.5, 0.5, "Current data not available", ha='center', va='center', fontsize=16, alpha=0.5)

            # --- Plot 2: Correlation Heatmap (on ax2) ---
            cols = ["Temperature (°C)", "Humidity (%)", "Wind Speed (m/s)", "Pressure (hPa)"]
            df_plot = df_current[cols].dropna()
            if not df_plot.empty:
                corr = df_plot.corr()
                sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax2, linewidths=.5, cbar=False)
                ax2.set_title("Current Metrics Correlation", fontsize=16, fontweight="bold", pad=10)
                ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
            else:
                ax2.text(0.5, 0.5, "Correlation data not available", ha='center', va='center', fontsize=16, alpha=0.5)

            # --- Plot 3: Multi-City Daily Comparison (on ax3) ---
            rows = []
            for city, df in all_daily.items():
                if df is None or df.empty: continue
                tmp = df.copy()
                if "Temp_Min" in tmp.columns and "Temp_Max" in tmp.columns:
                    tmp["Temp_Avg"] = (tmp["Temp_Min"] + tmp["Temp_Max"]) / 2.0
                    series = tmp.set_index("Date")["Temp_Avg"].rename(city)
                    rows.append(series)
            
            if rows:
                combined = pd.concat(rows, axis=1).sort_index() # Note: No dropna(how='all') here
                colors = plt.cm.Dark2(np.linspace(0, 1, len(combined.columns)))
                
                if not combined.empty: # Guard
                    for i, col in enumerate(combined.columns):
                        # --- THIS IS THE FIX ---
                        # Drop NaNs from each city's series *before* plotting
                        line = combined[col].dropna() 
                        # --- END FIX ---
                        
                        if not line.empty: # Only plot if there's data left
                            # Add a subtle shadow line underneath
                            ax3.plot(line.index, line, color='black', linewidth=5.5, alpha=0.1, zorder=i)
                            # Plot the main, thicker line on top (no marker)
                            ax3.plot(line.index, line, marker=None, label=col, color=colors[i], linewidth=3.5, zorder=i+1)
                            
                            ax3.text(line.index[-1] + pd.Timedelta(hours=2), line.iloc[-1], f" {col}", 
                                     color=colors[i], fontweight="bold", va="center", zorder=i+2)
                    
                    ax3.set_title("5-Day Average Temperature Forecast", fontsize=16, fontweight="bold", pad=10)
                    ax3.set_ylabel("Average Temperature (°C)")
                    ax3.grid(axis='y', linestyle="--", alpha=0.6)
                    for spine in ["top", "right"]: ax3.spines[spine].set_visible(False)
                    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%a %d"))
                    ax3.xaxis.set_major_locator(mdates.DayLocator(interval=1))
                    
                    # --- Guard xlim ---
                    ax3.set_xlim(right=combined.index[-1] + pd.Timedelta(days=1))
                else:
                    ax3.text(0.5, 0.5, "Daily forecast data empty", ha='center', va='center', fontsize=16, alpha=0.5)
            else:
                ax3.text(0.5, 0.5, "Daily forecast data not available", ha='center', va='center', fontsize=16, alpha=0.5)

            # --- Final Touches (with Provenance) ---
            fig.suptitle("Weather Dashboard", fontsize=24, fontweight='bold')
            fig.text(0.5, 0.95, f"Run: {run_ts} | Source: OpenWeatherMap (Free API)", 
                     fontsize=12, ha='center', va='bottom', style='italic', color='gray')
            
            fig.tight_layout(rect=[0, 0.03, 1, 0.94])
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info("Saved Main Dashboard: %s", save_path)
            plt.close()

    @staticmethod
    def plot_single_city_analysis_dashboard(df_hourly, city_name, run_ts, save_path):
        """
        Generates a detailed 2x2 analysis dashboard for a single city.
        
        Includes:
        - Temperature distribution (histogram + KDE with mean line)
        - Diurnal temperature cycle (hourly patterns across days)
        - Wind speed distribution (box plots by day)
        - Precipitation forecast (probability over 5 days)
        
        Args:
            df_hourly: DataFrame with hourly forecast data for the city
            city_name: Name of the city being analyzed
            run_ts: Timestamp string for file naming
            save_path: Path where dashboard will be saved
        """
        logger.info(f"Generating Deep Analysis Dashboard for {city_name}...")
        if df_hourly is None or df_hourly.empty:
            logger.warning(f"No hourly data for {city_name}, skipping analysis dashboard.")
            return

        with plt.style.context(DEFAULT_STYLE):
            fig = plt.figure(figsize=(18, 14))
            gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)
            
            ax1 = fig.add_subplot(gs[0, 0]) # Top-left
            ax2 = fig.add_subplot(gs[0, 1]) # Top-right
            ax3 = fig.add_subplot(gs[1, 0]) # Bottom-left
            ax4 = fig.add_subplot(gs[1, 1]) # Bottom-right
            
            # --- Plot 1: Temperature Histogram & KDE (ax1) ---
            temp_data = df_hourly["Temperature (°C)"].dropna()
            if not temp_data.empty:
                sns.histplot(temp_data, ax=ax1, kde=True, color="#d62728", bins=15, stat="density")
                ax1.set_title("5-Day Temp Distribution (Histogram + KDE)", fontsize=14, fontweight="bold")
                ax1.set_xlabel("Temperature (°C)")
                ax1.axvline(temp_data.mean(), color='black', linestyle='--', linewidth=2, label=f"Mean: {temp_data.mean():.1f}°C")
                ax1.legend()
            else:
                ax1.text(0.5, 0.5, "Temp data not available", ha='center', va='center', alpha=0.5)

            # --- Plot 2: Diurnal (Seasonal) Plot (ax2) ---
            if "Hour" in df_hourly.columns and "DateStr" in df_hourly.columns:
                sns.lineplot(ax=ax2, data=df_hourly, x="Hour", y="Temperature (°C)", hue="DateStr", marker="o", palette="viridis")
                ax2.set_title("Daily (Diurnal) Temperature Cycle", fontsize=14, fontweight="bold")
                ax2.set_xlabel("Hour of Day (Local Time)")
                ax2.set_ylabel("Temperature (°C)")
                ax2.set_xticks(range(0, 24, 3))
                ax2.legend(title="Day", bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                ax2.text(0.5, 0.5, "Diurnal data not available", ha='center', va='center', alpha=0.5)

            # --- Plot 3: Wind Speed Box Plot (ax3) ---
            if "Wind Speed (m/s)" in df_hourly.columns and "DateStr" in df_hourly.columns:
                day_order = df_hourly.drop_duplicates(subset=["DateStr"])["DateStr"].tolist()
                sns.boxplot(ax=ax3, x="DateStr", y="Wind Speed (m/s)", data=df_hourly, order=day_order, palette="coolwarm", medianprops=dict(color='red', linewidth=2))
                ax3.set_title("5-Day Wind Speed Distribution", fontsize=14, fontweight="bold")
                ax3.set_xlabel("Date")
                ax3.set_ylabel("Wind Speed (m/s)")
            else:
                ax3.text(0.5, 0.5, "Wind data not available", ha='center', va='center', alpha=0.5)
                
            # --- Plot 4: Precipitation Forecast (ax4) ---
            if "Precipitation Prob (%)" in df_hourly.columns:
                # --- Use Timedelta for bar width ---
                bar_width = pd.Timedelta(hours=2) / pd.Timedelta(days=1) 
                colors = plt.cm.Blues(df_hourly["Precipitation Prob (%)"] / 100.0)
                ax4.bar(df_hourly["Timestamp"], df_hourly["Precipitation Prob (%)"], width=bar_width, color=colors, edgecolor='black', linewidth=0.5)
                ax4.set_title("5-Day Precipitation Forecast", fontsize=14, fontweight="bold")
                ax4.set_ylabel("Probability (%)")
                ax4.xaxis.set_major_formatter(mdates.DateFormatter("%a %d"))
                ax4.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            else:
                ax4.text(0.5, 0.5, "Precip data not available", ha='center', va='center', alpha=0.5)

            # --- Final Touches (with Provenance) ---
            fig.suptitle(f"Deep Analysis Dashboard for {city_name}", fontsize=24, fontweight='bold')
            fig.text(0.5, 0.95, f"Run: {run_ts} | Source: OpenWeatherMap (Free API)", 
                     fontsize=12, ha='center', va='bottom', style='italic', color='gray')
            
            fig.tight_layout(rect=[0, 0.03, 1, 0.94])
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info("Saved Analysis Dashboard: %s", save_path)
            plt.close()

    @staticmethod
    def plot_current_weather_pairplot(df_current, save_path):
        """
        Generates correlation pairplot between current weather metrics.
        
        Shows scatter plots and regression lines for:
        - Temperature vs Humidity
        - Temperature vs Wind Speed
        - Temperature vs Pressure
        - And all other metric combinations
        
        Falls back to heatmap if pairplot generation fails.
        
        Args:
            df_current: DataFrame with current weather data
            save_path: Path where plot will be saved
        """
        if df_current.empty:
            logger.warning("No current data for pairplot.")
            return
        
        cols = ["Temperature (°C)", "Humidity (%)", "Wind Speed (m/s)", "Pressure (hPa)"]
        df_plot = df_current[cols].dropna()
        
        if df_plot.empty:
            logger.warning("No data for pairplot after dropping NaNs.")
            return

        logger.info("Generating Current Weather Pairplot...")
        with plt.style.context(DEFAULT_STYLE):
            try:
                # --- Use 'hist' to avoid scipy dependency ---
                g = sns.pairplot(df_plot, kind='reg', diag_kind='hist',
                                 plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.5, 'edgecolor': 'k', 'linewidth': 0.5}},
                                 diag_kws={'bins': 15, 'color': '#1f77b4'})
                
                g.fig.suptitle("Correlation Matrix for Current Weather Metrics", fontsize=18, fontweight='bold', y=1.03)
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                logger.info("Saved: %s", save_path)
            except Exception as e:
                logger.warning(f"Pairplot failed ({e}). Falling back to correlation heatmap.")
                # Fallback: Correlation Heatmap
                fig, ax = plt.subplots(figsize=(10, 8))
                corr = df_plot.corr()
                sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, linewidths=.5)
                ax.set_title("Correlation Heatmap (Fallback)", fontsize=16, fontweight="bold")
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                logger.info("Saved fallback heatmap: %s", save_path)
            
            plt.close()

# -------------------------
# CLI Menu Functions
# -------------------------
def get_user_choice():
    """
    Display interactive menu and get user's visualization choice.
    
    Returns:
        User's choice as string ('1', '2', '3', '4', or 'q')
    """
    print("=" * 50)
    print(" Weather Analysis Plot Generator")
    print("=" * 50)
    print("What would you like to generate?")
    print("\n[1] Main Summary Dashboard (Multi-City)")
    print("[2] Deep Analysis Dashboard (Single-City)")
    print("[3] All Plots (Both Dashboards + Pairplot)")
    print("[4] CSVs Only (No Plots)")
    print("\n[q] Quit")
    print("-" * 50)
    
    while True:
        choice = input("Enter your choice (1, 2, 3, 4, q): ").strip().lower()
        if choice in ['1', '2', '3', '4', 'q']:
            return choice
        else:
            print("Invalid choice. Please enter 1, 2, 3, 4, or q.")

def get_city_choice():
    """
    Display city selection menu and get user's choice.
    
    Returns:
        List of city names to analyze (or single-city list)
    """
    print("\nWhich city do you want to analyze?")
    city_list = list(CITIES.keys())
    for i, city in enumerate(city_list):
        print(f"[{i+1}] {city}")
    print("[all] All cities (generate a separate dashboard for each)")
    print("-" * 50)
    
    while True:
        choice = input(f"Enter choice (1-{len(city_list)}, all): ").strip().lower()
        if choice == 'all':
            return city_list
        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(city_list):
                return [city_list[choice_idx]] # Return as a list
        except ValueError:
            pass
        print("Invalid choice. Please enter a number from the list or 'all'.")

# -------------------------
# Main Execution Function
# -------------------------
def main():
    """
    Main function orchestrating the weather data fetching and visualization pipeline.
    
    Flow:
    1. Prompt user for visualization type
    2. Fetch current weather and forecasts for all cities
    3. Process data into DataFrames
    4. Save CSV exports
    5. Generate selected visualizations
    6. Display summary statistics
    """
    logger.info("Starting final multi-city run.")
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("Run timestamp: %s", run_ts)

    plot_choice = get_user_choice()
    if plot_choice == 'q':
        logger.info("User quit. Exiting.")
        return
    if plot_choice == '4':
        logger.info("Mode: CSVs Only. Plots will be skipped.")

    fetcher = WeatherDataFetcher(API_KEY)
    processor = WeatherDataProcessor()
    viz = WeatherVisualizations()

    all_current_json = {}
    all_hourly = {}
    all_daily = {}
    
    failed_current = []
    failed_forecast = []

    for city, coords in CITIES.items():
        logger.info("Fetching for %s", city)
        current_json = fetcher.fetch_current_weather(coords["lat"], coords["lon"])
        forecast_json = fetcher.fetch_forecast_data(coords["lat"], coords["lon"])
        
        if current_json:
            all_current_json[city] = current_json
        else:
            failed_current.append(city)
            
        if forecast_json:
            df_hourly = processor.extract_3hour_forecast_dataframe(forecast_json, city)
            df_daily = processor.summarize_daily_from_hourly(df_hourly)
        else:
            failed_forecast.append(city)
            df_hourly = pd.DataFrame()
            df_daily = pd.DataFrame()

        # Convert city name to safe filename slug (e.g., "New York" -> "new-york")
        city_key = _slugify(city)
        
        # Save hourly forecast data as CSV
        if not df_hourly.empty:
            hourly_path = DATA_DIR / f"{run_ts}_{city_key}_3hour.csv"
            df_hourly.to_csv(hourly_path, index=False)
            logger.info("Saved hourly CSV for %s: %s", city, hourly_path)
        else:
            logger.warning("Hourly DF empty for %s", city)

        # Save daily aggregated data as CSV
        if not df_daily.empty:
            daily_path = DATA_DIR / f"{run_ts}_{city_key}_daily.csv"
            df_daily.to_csv(daily_path, index=False)
            logger.info("Saved daily CSV for %s: %s", city, daily_path)
        else:
            logger.warning("Daily DF empty for %s", city)

        all_hourly[city] = df_hourly
        all_daily[city] = df_daily
        
        # Rate limiting: respect API call limits with configurable delay
        time.sleep(REQUEST_DELAY_SEC)

    # Log any failures for user visibility
    if failed_current:
        logger.warning(f"Failed to fetch CURRENT data for: {', '.join(failed_current)}")
    if failed_forecast:
        logger.warning(f"Failed to fetch FORECAST data for: {', '.join(failed_forecast)}")

    # Generate visualizations based on user choice
    if plot_choice == '4':  # CSVs Only
        logger.info("Skipping all plot generation as requested.")
    
    else:
        df_current = processor.extract_current_weather_dataframe(all_current_json)
        
        # Generate Main Summary Dashboard (options 1 or 3)
        if plot_choice in ('1', '3'):
            if not df_current.empty:
                current_csv = DATA_DIR / f"{run_ts}_current_weather.csv"
                df_current.to_csv(current_csv, index=False)
                logger.info("Saved combined current CSV: %s", current_csv)
            
            available_daily = {c: df for c, df in all_daily.items() if df is not None and not df.empty}
            
            viz.plot_summary_dashboard(
                df_current, 
                available_daily, 
                run_ts,
                FIGURES_DIR / f"{run_ts}_00_MAIN_DASHBOARD.png"
            )
            
            # Generate correlation pairplot for "All Plots" option
            if plot_choice == '3' and not df_current.empty:
                 viz.plot_current_weather_pairplot(df_current, FIGURES_DIR / f"{run_ts}_01_current_correlations.png")

        
        # Generate Per-City Deep Analysis Dashboard (options 2 or 3)
        if plot_choice in ('2', '3'):
            cities_to_analyze = get_city_choice()
            logger.info(f"Generating Deep Analysis Dashboard(s) for: {', '.join(cities_to_analyze)}")
            
            # Generate analysis dashboard for each selected city
            for city in cities_to_analyze:
                df_hour = all_hourly.get(city)
                city_key = _slugify(city)
                
                viz.plot_single_city_analysis_dashboard(
                    df_hour,
                    city,
                    run_ts,
                    FIGURES_DIR / f"{run_ts}_{city_key}_ANALYSIS_DASHBOARD.png"
                )

    # Display summary statistics (always runs regardless of visualization choice)
    logger.info("--- SUMMARY STATISTICS ---")
    logger.info("CURRENT WEATHER SUMMARY:")
    if 'df_current' not in locals():
        df_current = processor.extract_current_weather_dataframe(all_current_json)
        
    if not df_current.empty:
        print(df_current[["City", "Temperature (°C)", "Humidity (%)", "Weather"]].to_string(index=False))
    else:
        logger.info("  • No current data to display.")

    logger.info("PER-CITY FORECAST STATISTICS:")
    # Calculate and display forecast statistics for each city
    for city, df_h in all_hourly.items():
        logger.info(">> %s", city)
        if df_h is None or df_h.empty:
            logger.info("  • No hourly forecast data")
            continue
        if "Temperature (°C)" in df_h.columns and not df_h["Temperature (°C)"].dropna().empty:
            logger.info("  • Avg Temp: %.2f°C", df_h["Temperature (°C)"].mean())
            logger.info("  • Min Temp: %.2f°C", df_h["Temperature (°C)"].min())
            logger.info("  • Max Temp: %.2f°C", df_h["Temperature (°C)"].max())
        else:
            logger.info("  • Temperature data: N/A")
        if "Humidity (%)" in df_h.columns and not df_h["Humidity (%)"].dropna().empty:
            logger.info("  • Avg Humidity: %.2f%%", df_h["Humidity (%)"].mean())
        else:
            logger.info("  • Humidity: N/A")
        if "Weather" in df_h.columns and not df_h["Weather"].dropna().empty:
            logger.info("  • Most common: %s", df_h["Weather"].mode().values[0])
        else:
            logger.info("  • Weather: N/A")

    logger.info("Run complete. Outputs: %s", OUTPUT_DIR.resolve())

# -------------------------
# Entry Point
# -------------------------
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nRun interrupted by user. Exiting gracefully.")