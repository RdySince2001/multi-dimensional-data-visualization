import os
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def ensure_output_dir(output_dir: str) -> None:
    """Ensure that the output directory exists."""
    os.makedirs(output_dir, exist_ok=True)

def plot_weather_heatmap(df: pd.DataFrame, output_dir: str) -> str:
    """Generates a heatmap of monthly average temperature by city."""
    # Bản sao để tránh SettingWithCopyWarning
    df_copy = df.copy()
    
    # Tính toán ma trận pivot: Thành phố x Tháng, giá trị là trung bình của avg_temp
    # Đảm bảo tháng sắp xếp từ 1 đến 12
    pivot_df = df_copy.pivot_table(
        index='city', 
        columns='month', 
        values='avg_temp', 
        aggfunc='mean'
    )
    
    plt.figure(figsize=(10, 5))
    sns.heatmap(
        pivot_df, 
        annot=True, 
        fmt=".1f", 
        cmap="coolwarm", 
        cbar_kws={'label': 'Average temperature (°F)'}
    )
    
    plt.title('Monthly Average Temperature by City', fontsize=12, fontweight='bold', pad=10)
    plt.xlabel('Month')
    plt.ylabel('City')
    
    output_path = os.path.join(output_dir, "weather_heatmap.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path

def plot_weather_scatter(df: pd.DataFrame, output_dir: str) -> str:
    """Generates a scatter plot of Daily Temperature vs Humidity integrated with Precipitation."""
    df_copy = df.copy()
    
    # Xử lý lượng mưa: 'T' (Trace) -> 0, ép kiểu về float
    df_copy['precip_numeric'] = df_copy['precip'].replace('T', '0')
    df_copy['precip_numeric'] = pd.to_numeric(df_copy['precip_numeric'], errors='coerce').fillna(0)
    
    plt.figure(figsize=(10, 6))
    
    # Vẽ scatter plot: x=avg_humidity, y=avg_temp, màu=city, kích thước=lượng mưa
    scatter = sns.scatterplot(
        data=df_copy,
        x='avg_humidity',
        y='avg_temp',
        hue='city',
        size='precip_numeric',
        sizes=(20, 300),
        alpha=0.6,
        palette='Set2'
    )
    
    plt.title('Daily Temperature vs. Humidity Integrated with Precipitation', fontsize=12, fontweight='bold', pad=10)
    plt.xlabel('Average Humidity (%)')
    plt.ylabel('Average Temperature (°F)')
    
    # Đặt Legend ra ngoài biểu đồ để không che khuất dữ liệu
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title='Indicators')
    
    output_path = os.path.join(output_dir, "weather_scatter.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path

def plot_global_temp_heatmap(df: pd.DataFrame, output_dir: str) -> str:
    """Generates a heatmap of global land-ocean temperature anomalies (1880-2025)."""
    # Lấy danh sách 12 tháng từ cột dữ liệu có sẵn
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Đặt Year làm index và lọc chỉ lấy 12 tháng để vẽ ma trận Năm x Tháng
    matrix_df = df.set_index('Year')[months]
    
    # Chuyển đổi dữ liệu sang dạng số (phòng trường hợp kiểu dữ liệu gốc bị nhận diện sai)
    matrix_df = matrix_df.apply(pd.to_numeric, errors='coerce')
    
    # Thiết lập kích thước lớn theo chiều dọc để thể hiện rõ các năm từ 1880 - 2025
    plt.figure(figsize=(9, 14))
    
    sns.heatmap(
        matrix_df, 
        cmap='coolwarm', 
        vmin=-1.5, 
        vmax=1.5,
        cbar_kws={'label': 'Temperature Anomaly (°C)', 'shrink': 0.8}
    )
    
    plt.title('Global Land-Ocean Temperature Anomalies (1880-2025)', fontsize=12, fontweight='bold', pad=10)
    plt.xlabel('Month')
    plt.ylabel('Year')
    
    output_path = os.path.join(output_dir, "global_temp_heatmap.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path

def plot_minnesota_precip_line(df: pd.DataFrame, output_dir: str) -> str:
    """Generates a line chart of monthly precipitation by site in Minnesota (1927-1936)."""
    df_copy = df.copy()
    
    # Tạo cột trục thời gian liên tục 'date' từ năm (year) và tháng (mo)
    df_copy['date'] = pd.to_datetime(df_copy[['year', 'mo']].rename(columns={'mo': 'month'}).assign(day=1))
    
    # Sắp xếp thứ tự thời gian để các đường vẽ không bị rối loạn
    df_copy = df_copy.sort_values(by=['site', 'date'])
    
    plt.figure(figsize=(12, 5))
    
    # Biểu diễn đường lượng mưa theo thời gian, phân loại màu theo địa điểm (site)
    sns.lineplot(
        data=df_copy, 
        x='date', 
        y='precip', 
        hue='site', 
        marker='o', 
        linewidth=1, 
        markersize=3
    )
    
    plt.title('Monthly Precipitation by Site in Minnesota (1927-1936)', fontsize=12, fontweight='bold', pad=10)
    plt.xlabel('Year')
    plt.ylabel('Precipitation (inches)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title='Site')
    
    output_path = os.path.join(output_dir, "minnesota_precip_line.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path

def main() -> List[str]:
    """Run all visualizations and return a list of generated file paths."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    out_dir = os.path.join(base_dir, "output")
    ensure_output_dir(out_dir)
    figures: List[str] = []

    # 1 & 2. Load and plot weather data
    weather_path = os.path.join(data_dir, "weather_data.csv")
    weather_df = pd.read_csv(weather_path)
    figures.append(plot_weather_heatmap(weather_df, out_dir))
    figures.append(plot_weather_scatter(weather_df, out_dir))

    # 3. Load and plot global temperature anomalies
    global_path = os.path.join(data_dir, "global_temp.csv")
    # skiprows=1 vì dòng đầu tiên của file chứa text tiêu đề "Land-Ocean: Global Means"
    global_df = pd.read_csv(global_path, skiprows=1)
    
    # Thay thế dấu kí hiệu khuyết '***' bằng NA thực sự và ép kiểu số
    global_df = global_df.replace("***", pd.NA)
    for col in global_df.columns[1:]:
        global_df[col] = pd.to_numeric(global_df[col], errors="coerce")
    figures.append(plot_global_temp_heatmap(global_df, out_dir))

    # 4. Load and plot Minnesota barley weather summary
    minnesota_path = os.path.join(data_dir, "minnesota_weather.csv")
    minnesota_df = pd.read_csv(minnesota_path)
    figures.append(plot_minnesota_precip_line(minnesota_df, out_dir))

    print("🎉 All charts have been generated successfully in the 'output/' directory!")
    return figures

if __name__ == "__main__":
    main()