import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('/Bi-Annual_Pedestrian_Counts.csv')

def parse_date(col):
    prefix = col.split('_')[0]
    first_digit_idx = next(i for i, c in enumerate(prefix) if c.isdigit())
    month_str = prefix[:first_digit_idx]
    year_suffix = prefix[first_digit_idx:]
    month_lookup = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
        'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
        'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }
    month_abbr = month_str[:3].title()
    if month_abbr not in month_lookup:
        raise ValueError(f"Unknown month: {month_abbr}")
    month = month_lookup[month_abbr]
    year = int('20' + year_suffix)
    return pd.Timestamp(year=year, month=month, day=1)

#totals per date
ped_cols = [col for col in df.columns if '_' in col and col.split('_')[0][:3] in ['May', 'Oct', 'Jun']]
date_prefixes = sorted(set(col.split('_')[0] for col in ped_cols))

for date in date_prefixes:
    relevant_cols = [f"{date}_AM", f"{date}_PM", f"{date}_MD"]
    existing_cols = [col for col in relevant_cols if col in df.columns]
    if len(existing_cols) == 3:
        df[f'{date}_Total'] = df[existing_cols].sum(axis=1, min_count=1)

total_cols = [col for col in df.columns if col.endswith('_Total')]

#Group by Borough and sum totals
borough_totals = pd.DataFrame()

for borough, group in df.groupby('Borough'):
    sums = group[total_cols].sum(axis=0)
    dates = [parse_date(col) for col in sums.index]
    series = pd.Series(sums.values, index=dates).sort_index()
    borough_totals[borough] = series

overall_total = borough_totals.sum(axis=1)

plt.figure(figsize=(16, 8))

#Plot boroughs w/ polyfit 3
degree_borough = 3
for borough in borough_totals.columns:
    x_dates = (borough_totals.index - borough_totals.index.min()).days.to_numpy()
    y_values = borough_totals[borough].to_numpy()

    coeffs = np.polyfit(x_dates, y_values, degree_borough)
    poly = np.poly1d(coeffs)
    y_fit = poly(x_dates)

    ss_res = np.sum((y_values - y_fit) ** 2)
    ss_tot = np.sum((y_values - np.mean(y_values)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    plt.plot(borough_totals.index.to_numpy(), y_values, 'o', label=f"{borough} data")
    x_fit = np.linspace(x_dates.min(), x_dates.max(), 200)
    y_fit_smooth = poly(x_fit)
    dates_fit = borough_totals.index.min() + pd.to_timedelta(x_fit, unit='D')
    plt.plot(dates_fit.to_numpy(), y_fit_smooth, label=f"{borough} fit (deg {degree_borough}) $R^2$={r2:.3f}")

#Plot total w/ polyfit 4
degree_overall = 4
x_dates = (overall_total.index - overall_total.index.min()).days.to_numpy()
y_values = overall_total.to_numpy()

coeffs = np.polyfit(x_dates, y_values, degree_overall)
poly = np.poly1d(coeffs)
print("Overall polynomial equation:", poly) #give equation
y_fit = poly(x_dates)

ss_res = np.sum((y_values - y_fit) ** 2)
ss_tot = np.sum((y_values - np.mean(y_values)) ** 2)
r2 = 1 - (ss_res / ss_tot)

plt.plot(overall_total.index.to_numpy(), y_values, 'kD', label="Overall total data")
x_fit = np.linspace(x_dates.min(), x_dates.max(), 300)
y_fit_smooth = poly(x_fit)
dates_fit = overall_total.index.min() + pd.to_timedelta(x_fit, unit='D')
plt.plot(dates_fit.to_numpy(), y_fit_smooth, 'k-', linewidth=3, label=f"Overall fit (deg {degree_overall}) $R^2$={r2:.3f}")

plt.title("Pedestrian Foot Traffic by Borough + Overall Total with Polynomial Fits")
plt.xlabel("Date")
plt.ylabel("Total Pedestrian Count")
plt.legend(loc='upper left', fontsize='small')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
