"""
Generate realistic weather dataset for Dakshina Kannada region, Karnataka, India.
Based on IMD and NASA POWER historical data for coordinates: 12.87N, 74.88E (Mangalore)

Key climate characteristics:
- Tropical monsoon climate (Koppen: Am)
- Southwest monsoon: June-September (heavy rainfall, 80-90% of annual)
- Pre-monsoon: March-May (hot, humid, occasional thunderstorms)
- Post-monsoon: October-November (Northeast monsoon, moderate rainfall)
- Winter: December-February (mild, dry, pleasant)
- Annual rainfall: ~3500-4000mm
- Temperature range: 18-36C
- Humidity: 60-95%
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

def generate_dakshina_kannada_dataset(start_year=2000, end_year=2024):
    dates = pd.date_range(start=f"{start_year}-01-01", end=f"{end_year}-12-31", freq="D")
    n_days = len(dates)
    
    precipitation = np.zeros(n_days)
    temp_max = np.zeros(n_days)
    temp_min = np.zeros(n_days)
    humidity = np.zeros(n_days)
    wind_speed = np.zeros(n_days)
    solar_radiation = np.zeros(n_days)
    pressure = np.zeros(n_days)
    
    for i, date in enumerate(dates):
        month = date.month
        day_of_year = date.dayofyear
        
        # Day of year effects (seasonal cycles)
        sin_day = np.sin(2 * np.pi * day_of_year / 365.25)
        cos_day = np.cos(2 * np.pi * day_of_year / 365.25)
        
        # Year trend (slight warming trend per IPCC data for coastal Karnataka)
        year_offset = (date.year - 2000) * 0.02
        
        # ============================================================
        # PRECIPITATION (mm/day)
        # Based on IMD data for Mangalore/Dakshina Kannada
        # Annual: ~3500-4000mm
        # Southwest monsoon (Jun-Sep): ~2800-3200mm (80-85%)
        # Northeast monsoon (Oct-Nov): ~200-300mm
        # Winter (Dec-Feb): ~20-50mm
        # Pre-monsoon (Mar-May): ~200-300mm
        # ============================================================
        
        # Base monsoon pattern (realistic for coastal Karnataka)
        if month in [6, 7, 8, 9]:  # Southwest monsoon peak
            if month == 6:
                base_rain_prob = 0.70
                base_rain_amount = 15.0
            elif month == 7:
                base_rain_prob = 0.82
                base_rain_amount = 20.0
            elif month == 8:
                base_rain_prob = 0.78
                base_rain_amount = 18.0
            else:  # September
                base_rain_prob = 0.65
                base_rain_amount = 13.0
        elif month in [10, 11]:  # Post-monsoon / NE monsoon
            base_rain_prob = 0.35
            base_rain_amount = 5.0
        elif month in [3, 4, 5]:  # Pre-monsoon
            base_rain_prob = 0.20
            base_rain_amount = 3.0
        else:  # Winter (Dec, Jan, Feb)
            base_rain_prob = 0.10
            base_rain_amount = 1.0
        
        # Rain occurrence
        if np.random.random() < base_rain_prob:
            # Rain amount (lognormal distribution for realistic heavy rainfall)
            rain = np.random.lognormal(mean=np.log(base_rain_amount), sigma=1.2)
            # Cap at realistic maximum (extreme events can reach 200-300mm/day)
            rain = min(rain, 250.0)
            precipitation[i] = max(0.1, rain)
        
        # Add occasional extreme events during monsoon (consistent with DK history)
        if month in [7, 8] and np.random.random() < 0.03:  # 3% chance of extreme day
            precipitation[i] = max(precipitation[i], np.random.uniform(100, 200))
        
        # ============================================================
        # TEMPERATURE (Celsius)
        # Based on IMD data for Mangalore
        # Annual mean: ~27C
        # Max: 32-36C (Mar-May), 28-32C (monsoon), 27-31C (winter)
        # Min: 20-24C (winter), 23-25C (monsoon), 24-26C (summer)
        # ============================================================
        
        # Temperature seasonal cycle
        if month in [3, 4, 5]:  # Pre-monsoon / Summer
            base_tmax = 33.0 + year_offset
            base_tmin = 24.5
        elif month in [6, 7, 8, 9]:  # Monsoon (cooler due to clouds)
            base_tmax = 29.5
            base_tmin = 23.5
        elif month in [10, 11]:  # Post-monsoon
            base_tmax = 30.5
            base_tmin = 23.0
        else:  # Winter
            base_tmax = 31.0
            base_tmin = 20.5
        
        # Daily variation
        temp_max[i] = base_tmax + np.random.normal(0, 1.5)
        temp_min[i] = base_tmin + np.random.normal(0, 1.2)
        
        # Cool down on heavy rain days
        if precipitation[i] > 30:
            temp_max[i] -= np.random.uniform(1, 3)
            temp_min[i] -= np.random.uniform(0.5, 1.5)
        
        # Ensure temp_min < temp_max
        temp_min[i] = min(temp_min[i], temp_max[i] - 2.0)
        
        # ============================================================
        # HUMIDITY (%)
        # Coastal Karnataka: very humid year-round
        # Monsoon: 80-95%
        # Winter: 60-80%
        # Summer: 65-85%
        # ============================================================
        
        if month in [6, 7, 8, 9]:  # Monsoon
            base_humidity = 88.0
        elif month in [12, 1, 2]:  # Winter
            base_humidity = 72.0
        elif month in [3, 4, 5]:  # Summer
            base_humidity = 75.0
        else:  # Post-monsoon
            base_humidity = 80.0
        
        # Higher humidity on rainy days
        if precipitation[i] > 5:
            base_humidity += 5.0
        
        humidity[i] = np.clip(base_humidity + np.random.normal(0, 5), 50, 98)
        
        # ============================================================
        # WIND SPEED (m/s)
        # SW monsoon: stronger winds (2-6 m/s)
        # Winter: lighter winds (1-3 m/s)
        # Pre-monsoon thunderstorms: gusty
        # ============================================================
        
        if month in [6, 7, 8, 9]:  # Monsoon
            base_wind = 3.5
        elif month in [3, 4, 5]:  # Pre-monsoon
            base_wind = 2.5
        elif month in [12, 1, 2]:  # Winter
            base_wind = 1.8
        else:
            base_wind = 2.2
        
        # Higher wind during rain
        if precipitation[i] > 20:
            base_wind += 1.5
        
        wind_speed[i] = max(0.2, base_wind + np.random.normal(0, 0.8))
        
        # ============================================================
        # SOLAR RADIATION (kWh/m2/day)
        # Based on NASA POWER data for Mangalore
        # Clear sky: 5-7 kWh/m2/day
        # Monsoon (cloudy): 2-4 kWh/m2/day
        # ============================================================
        
        if month in [6, 7, 8, 9]:  # Monsoon (cloudy)
            base_solar = 3.5
        elif month in [3, 4, 5]:  # Summer (clear)
            base_solar = 6.0
        elif month in [12, 1, 2]:  # Winter (mostly clear)
            base_solar = 5.5
        else:
            base_solar = 5.0
        
        # Reduce solar radiation on rainy days
        if precipitation[i] > 10:
            base_solar -= 2.0
        elif precipitation[i] > 2:
            base_solar -= 1.0
        
        solar_radiation[i] = max(1.0, base_solar + np.random.normal(0, 0.8))
        
        # ============================================================
        # ATMOSPHERIC PRESSURE (kPa)
        # Based on NASA POWER data for coastal Karnataka
        # Range: ~100.0 - 101.5 kPa
        # Lower during monsoon
        # ============================================================
        
        if month in [6, 7, 8, 9]:  # Monsoon
            base_pressure = 100.5
        elif month in [12, 1, 2]:  # Winter
            base_pressure = 101.2
        else:
            base_pressure = 100.9
        
        pressure[i] = base_pressure + np.random.normal(0, 0.3)
    
    # Create DataFrame
    df = pd.DataFrame({
        "Date": dates.strftime("%Y-%m-%d"),
        "precipitation_mm": np.round(precipitation, 2),
        "temp_max": np.round(temp_max, 2),
        "temp_min": np.round(temp_min, 2),
        "humidity": np.round(humidity, 2),
        "wind_speed": np.round(wind_speed, 2),
        "solar_radiation": np.round(solar_radiation, 2),
        "pressure": np.round(pressure, 2),
    })
    
    return df


if __name__ == "__main__":
    print("Generating realistic Dakshina Kannada weather dataset (2000-2024)...")
    df = generate_dakshina_kannada_dataset(2000, 2024)
    
    output_path = r".\Dakshina_Kannada_Weather_2000_2024.csv"
    df.to_csv(output_path, index=False)
    
    print(f"\nDataset saved to: {output_path}")
    print(f"\nDataset shape: {df.shape}")
    print(f"\nDate range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"\nSummary Statistics:")
    print(f"  Precipitation (mm):")
    print(f"    Mean: {df['precipitation_mm'].mean():.2f}")
    print(f"    Annual total: {df.groupby(df['Date'].str[:4])['precipitation_mm'].sum().mean():.0f} mm/year")
    print(f"    Max daily: {df['precipitation_mm'].max():.2f} mm")
    print(f"    Rainy days (>1mm): {(df['precipitation_mm'] > 1).sum()} ({(df['precipitation_mm'] > 1).sum() / len(df) * 100:.1f}%)")
    print(f"  Temperature (C):")
    print(f"    Max range: {df['temp_max'].min():.1f} - {df['temp_max'].max():.1f}")
    print(f"    Min range: {df['temp_min'].min():.1f} - {df['temp_min'].max():.1f}")
    print(f"  Humidity: {df['humidity'].min():.1f}% - {df['humidity'].max():.1f}%")
    print(f"  Wind Speed: {df['wind_speed'].min():.2f} - {df['wind_speed'].max():.2f} m/s")
    print(f"  Solar Radiation: {df['solar_radiation'].min():.2f} - {df['solar_radiation'].max():.2f} kWh/m2/day")
    print(f"  Pressure: {df['pressure'].min():.2f} - {df['pressure'].max():.2f} kPa")
    
    # Monthly statistics (averaged across years)
    print(f"\nMonthly Averages (per month, averaged over all years):")
    df['datetime'] = pd.to_datetime(df['Date'])
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    
    # Compute yearly totals first, then average
    yearly_monthly = df.groupby(['year', 'month'])['precipitation_mm'].sum().reset_index()
    monthly_avg = yearly_monthly.groupby('month')['precipitation_mm'].agg(['mean', 'std']).round(1)
    monthly_avg.columns = ['Avg_Monthly_Total', 'Std']
    
    # Temperature and humidity averages
    temp_monthly = df.groupby('month').agg({
        'temp_max': 'mean',
        'temp_min': 'mean',
        'humidity': 'mean',
    }).round(1)
    
    monthly_stats = monthly_avg.join(temp_monthly)
    monthly_stats.columns = ['Avg_Monthly_Rainfall(mm)', 'Std', 'Avg_Max_Temp(C)', 'Avg_Min_Temp(C)', 'Avg_Humidity(%)']
    print(monthly_stats.to_string())
    
    print(f"\nAnnual Rainfall by Year:")
    annual = df.groupby('year')['precipitation_mm'].sum().round(0)
    print(f"  Mean: {annual.mean():.0f} mm/year")
    print(f"  Min: {annual.min():.0f} mm ({annual.idxmin()})")
    print(f"  Max: {annual.max():.0f} mm ({annual.idxmax()})")
