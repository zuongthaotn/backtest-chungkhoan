import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Đọc dữ liệu
CSV_PATH = "VN30F1M_5minutes.csv"
df = pd.read_csv(CSV_PATH, index_col='Date', parse_dates=True).tail(48)

# --- Xác định range giá ---
low_price = df["Low"].min()
high_price = df["High"].max()
n_bins = 24

# --- Tạo bins ---
bins = np.linspace(low_price, high_price, n_bins + 1)
bin_centers = (bins[:-1] + bins[1:]) / 2
volume_profile = pd.Series(0.0, index=bin_centers)

# --- Phân bổ volume vào bins ---
for _, row in df.iterrows():
    # các giá nằm trong khoảng nến
    mask = (bin_centers >= row["Low"]) & (bin_centers <= row["High"])
    count = mask.sum()
    if count > 0:
        vol_per_bin = row["Volume"] / count
        volume_profile.loc[mask] += vol_per_bin

# --- Tính POC, VAH, VAL ---
poc_price = volume_profile.idxmax()

sorted_vol = volume_profile.sort_values(ascending=False)
cum_vol = sorted_vol.cumsum() / sorted_vol.sum()
value_area = sorted_vol[cum_vol <= 0.7]
vah = value_area.index.max()
val = value_area.index.min()

print(f"POC: {poc_price:.2f}")
print(f"VAH: {vah:.2f}")
print(f"VAL: {val:.2f}")

# --- Vẽ biểu đồ Volume Profile ---
plt.figure(figsize=(8,6))
plt.barh(volume_profile.index, volume_profile.values, height=(bins[1]-bins[0])*0.9, color='skyblue')
plt.axhline(poc_price, color='red', linestyle='--', label=f'POC {poc_price:.2f}')
# plt.axhline(vah, color='green', linestyle='--', label=f'VAH {vah:.2f}')
# plt.axhline(val, color='orange', linestyle='--', label=f'VAL {val:.2f}')
plt.xlabel("Volume")
plt.ylabel("Price")
plt.title("Volume Profile (24 bins)")
plt.legend()
plt.savefig("volume_profile.png", dpi=150, bbox_inches='tight')
print("✅ Saved chart to volume_profile.png")