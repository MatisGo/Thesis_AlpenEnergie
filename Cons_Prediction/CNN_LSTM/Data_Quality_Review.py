"""
Data Quality Review Script for Consumption Forecasting
=======================================================
This script analyzes the input data to detect:
- Missing values (NaN, NULL, empty)
- Outliers using multiple methods (IQR, Z-score, isolation forest)
- Data inconsistencies and anomalies
- Temporal gaps in the time series

The results are visualized and summarized for data preprocessing decisions.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_FILE = '../Data_January.csv'
OUTPUT_DIR = './Data_Quality_Reports/'

# Outlier detection thresholds
IQR_MULTIPLIER = 1.5          # Standard IQR multiplier for outlier detection
ZSCORE_THRESHOLD = 3.0        # Z-score threshold for outliers
ISOLATION_CONTAMINATION = 0.05  # Expected proportion of outliers

# Columns to analyze
NUMERIC_COLUMNS = ['Consumption', 'Production', 'Temperature', 'Temperature_Predicted',
                   'Irradiance', 'Rain', 'Level_Bidmi', 'Level_Haselholz']

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def detect_outliers_iqr(data, column, multiplier=1.5):
    """
    Detect outliers using Interquartile Range (IQR) method.

    Outliers are defined as values below Q1 - multiplier*IQR or
    above Q3 + multiplier*IQR.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    outliers = (data[column] < lower_bound) | (data[column] > upper_bound)

    return outliers, lower_bound, upper_bound


def detect_outliers_zscore(data, column, threshold=3.0):
    """
    Detect outliers using Z-score method.

    Outliers are defined as values with |z-score| > threshold.
    """
    z_scores = np.abs(stats.zscore(data[column].dropna()))

    # Create mask for full dataset (including NaN positions)
    outliers = pd.Series(False, index=data.index)
    valid_idx = data[column].dropna().index
    outliers.loc[valid_idx] = z_scores > threshold

    return outliers


def detect_outliers_isolation_forest(data, columns, contamination=0.05):
    """
    Detect outliers using Isolation Forest algorithm.

    This is useful for multivariate outlier detection.
    """
    # Prepare data (drop rows with NaN)
    clean_data = data[columns].dropna()

    if len(clean_data) == 0:
        return pd.Series(False, index=data.index)

    # Fit Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    predictions = iso_forest.fit_predict(clean_data)

    # Create mask for full dataset
    outliers = pd.Series(False, index=data.index)
    outliers.loc[clean_data.index] = predictions == -1

    return outliers


def detect_temporal_gaps(data, datetime_column, expected_freq='5min'):
    """
    Detect gaps in the time series data.
    """
    time_diffs = data[datetime_column].diff()
    expected_delta = pd.Timedelta(expected_freq)

    gaps = time_diffs > expected_delta
    gap_info = data.loc[gaps, [datetime_column]].copy()
    gap_info['Previous_Time'] = data[datetime_column].shift(1).loc[gaps]
    gap_info['Gap_Duration'] = time_diffs.loc[gaps]

    return gaps, gap_info


def plot_data_quality_summary(data, numeric_cols, outlier_results, output_dir):
    """
    Create comprehensive data quality visualization.
    """
    n_cols = len(numeric_cols)
    fig, axes = plt.subplots(n_cols, 3, figsize=(18, 4*n_cols))

    if n_cols == 1:
        axes = axes.reshape(1, -1)

    for i, col in enumerate(numeric_cols):
        if col not in data.columns:
            continue

        # Time series plot with outliers highlighted
        ax1 = axes[i, 0]
        ax1.plot(data.index, data[col], 'b-', alpha=0.6, linewidth=0.5, label='Data')

        if col in outlier_results:
            outlier_mask = outlier_results[col]['iqr']
            ax1.scatter(data.index[outlier_mask], data[col][outlier_mask],
                       c='red', s=10, label=f'Outliers ({outlier_mask.sum()})')

        ax1.set_title(f'{col} - Time Series', fontsize=10, fontweight='bold')
        ax1.set_xlabel('Sample Index')
        ax1.set_ylabel(col)
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Histogram
        ax2 = axes[i, 1]
        valid_data = data[col].dropna()
        ax2.hist(valid_data, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax2.axvline(valid_data.mean(), color='red', linestyle='--', label=f'Mean: {valid_data.mean():.2f}')
        ax2.axvline(valid_data.median(), color='green', linestyle='--', label=f'Median: {valid_data.median():.2f}')
        ax2.set_title(f'{col} - Distribution', fontsize=10, fontweight='bold')
        ax2.set_xlabel(col)
        ax2.set_ylabel('Frequency')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # Box plot
        ax3 = axes[i, 2]
        bp = ax3.boxplot(valid_data, vert=True, patch_artist=True)
        bp['boxes'][0].set_facecolor('steelblue')
        bp['boxes'][0].set_alpha(0.7)
        ax3.set_title(f'{col} - Box Plot', fontsize=10, fontweight='bold')
        ax3.set_ylabel(col)
        ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'data_quality_overview.png'), dpi=300, bbox_inches='tight')
    plt.show()


def plot_missing_values_heatmap(data, output_dir):
    """
    Create a heatmap showing missing values pattern.
    """
    # Sample data for visualization (every 100th point for readability)
    sample_step = max(1, len(data) // 500)
    sample_data = data.iloc[::sample_step]

    missing_matrix = sample_data.isnull().astype(int)

    fig, ax = plt.subplots(figsize=(14, 8))

    im = ax.imshow(missing_matrix.T, aspect='auto', cmap='RdYlGn_r', interpolation='nearest')

    ax.set_yticks(range(len(missing_matrix.columns)))
    ax.set_yticklabels(missing_matrix.columns)
    ax.set_xlabel('Sample Index (sampled)')
    ax.set_title('Missing Values Pattern (Red = Missing)', fontsize=14, fontweight='bold')

    plt.colorbar(im, ax=ax, label='Missing (1) / Present (0)')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'missing_values_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.show()


def generate_summary_report(data, numeric_cols, outlier_results, gap_info, output_dir):
    """
    Generate a comprehensive text report of data quality findings.
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("DATA QUALITY REVIEW REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"\nDataset Shape: {data.shape[0]} rows x {data.shape[1]} columns")
    report_lines.append(f"Date Range: {data['DateTime'].min()} to {data['DateTime'].max()}")

    # Missing Values Summary
    report_lines.append("\n" + "-" * 80)
    report_lines.append("1. MISSING VALUES ANALYSIS")
    report_lines.append("-" * 80)

    missing_summary = data.isnull().sum()
    missing_pct = (missing_summary / len(data) * 100).round(2)

    report_lines.append(f"\n{'Column':<30} {'Missing Count':>15} {'Missing %':>12}")
    report_lines.append("-" * 60)
    for col in data.columns:
        if missing_summary[col] > 0:
            report_lines.append(f"{col:<30} {missing_summary[col]:>15} {missing_pct[col]:>11.2f}%")

    total_missing = missing_summary.sum()
    total_cells = data.shape[0] * data.shape[1]
    report_lines.append("-" * 60)
    report_lines.append(f"{'TOTAL':<30} {total_missing:>15} {total_missing/total_cells*100:>11.2f}%")

    # Outlier Summary
    report_lines.append("\n" + "-" * 80)
    report_lines.append("2. OUTLIER ANALYSIS")
    report_lines.append("-" * 80)

    report_lines.append(f"\n{'Column':<25} {'IQR Outliers':>15} {'Z-Score Outliers':>18} {'IF Outliers':>15}")
    report_lines.append("-" * 75)

    for col in numeric_cols:
        if col in outlier_results:
            iqr_count = outlier_results[col]['iqr'].sum()
            zscore_count = outlier_results[col]['zscore'].sum()
            if_count = outlier_results[col].get('isolation_forest', pd.Series([False])).sum()
            report_lines.append(f"{col:<25} {iqr_count:>15} {zscore_count:>18} {if_count:>15}")

    # Statistical Summary
    report_lines.append("\n" + "-" * 80)
    report_lines.append("3. STATISTICAL SUMMARY")
    report_lines.append("-" * 80)

    for col in numeric_cols:
        if col in data.columns:
            valid_data = data[col].dropna()
            if len(valid_data) > 0:
                report_lines.append(f"\n{col}:")
                report_lines.append(f"  Count:     {len(valid_data):>12}")
                report_lines.append(f"  Mean:      {valid_data.mean():>12.4f}")
                report_lines.append(f"  Std:       {valid_data.std():>12.4f}")
                report_lines.append(f"  Min:       {valid_data.min():>12.4f}")
                report_lines.append(f"  25%:       {valid_data.quantile(0.25):>12.4f}")
                report_lines.append(f"  50%:       {valid_data.median():>12.4f}")
                report_lines.append(f"  75%:       {valid_data.quantile(0.75):>12.4f}")
                report_lines.append(f"  Max:       {valid_data.max():>12.4f}")

    # Temporal Gaps
    report_lines.append("\n" + "-" * 80)
    report_lines.append("4. TEMPORAL GAPS ANALYSIS")
    report_lines.append("-" * 80)

    if len(gap_info) > 0:
        report_lines.append(f"\nFound {len(gap_info)} temporal gaps in the data:")
        for idx, row in gap_info.head(20).iterrows():
            report_lines.append(f"  Gap at {row['DateTime']}: Duration = {row['Gap_Duration']}")
        if len(gap_info) > 20:
            report_lines.append(f"  ... and {len(gap_info) - 20} more gaps")
    else:
        report_lines.append("\nNo temporal gaps detected (data is continuous at 5-min intervals)")

    # Recommendations
    report_lines.append("\n" + "-" * 80)
    report_lines.append("5. RECOMMENDATIONS")
    report_lines.append("-" * 80)

    recommendations = []

    if total_missing > 0:
        recommendations.append("- Handle missing values through interpolation or forward-fill")

    for col in numeric_cols:
        if col in outlier_results and outlier_results[col]['iqr'].sum() > len(data) * 0.01:
            recommendations.append(f"- Review outliers in {col} ({outlier_results[col]['iqr'].sum()} detected)")

    if len(gap_info) > 0:
        recommendations.append("- Fill temporal gaps or handle them during sequence creation")

    if not recommendations:
        recommendations.append("- Data quality appears good for model training")

    for rec in recommendations:
        report_lines.append(rec)

    report_lines.append("\n" + "=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)

    # Save report
    report_text = "\n".join(report_lines)
    report_path = os.path.join(output_dir, 'data_quality_report.txt')
    with open(report_path, 'w') as f:
        f.write(report_text)

    print(report_text)

    return report_text


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("DATA QUALITY REVIEW FOR CONSUMPTION FORECASTING")
    print("=" * 80)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    print("\n1. Loading data...")
    data = pd.read_csv(DATA_FILE, skiprows=3, header=None, encoding='latin-1')

    # Assign column names
    data.columns = ['DateTime_str', 'Date', 'DayTime', 'Forecast_Prod', 'Forecast_Load',
                    'Consumption', 'Production', 'Level_Bidmi', 'Level_Haselholz',
                    'Temperature', 'Irradiance', 'Rain', 'SDR_Mode', 'Forecast_Mode',
                    'Transfer_Mode', 'Waterlevel_Mode', 'Temp_Forecast']

    # Parse DateTime
    data['DateTime'] = pd.to_datetime(data['DateTime_str'], format='%d.%m.%Y %H:%M:%S', errors='coerce')

    # Handle Temperature_Predicted
    data['Temp_Forecast'] = pd.to_numeric(data['Temp_Forecast'], errors='coerce')
    data.rename(columns={'Temp_Forecast': 'Temperature_Predicted'}, inplace=True)

    print(f"   Loaded {len(data)} rows x {len(data.columns)} columns")
    print(f"   Date range: {data['DateTime'].min()} to {data['DateTime'].max()}")

    # Update numeric columns to only include those present
    available_numeric_cols = [col for col in NUMERIC_COLUMNS if col in data.columns]

    # 2. Missing Values Analysis
    print("\n2. Analyzing missing values...")
    missing_counts = data.isnull().sum()
    print(f"   Total missing values: {missing_counts.sum()}")

    for col in available_numeric_cols:
        if missing_counts[col] > 0:
            print(f"   - {col}: {missing_counts[col]} missing ({missing_counts[col]/len(data)*100:.2f}%)")

    # 3. Outlier Detection
    print("\n3. Detecting outliers...")
    outlier_results = {}

    for col in available_numeric_cols:
        if col not in data.columns:
            continue

        print(f"   Analyzing {col}...")
        outlier_results[col] = {}

        # IQR method
        iqr_outliers, lower, upper = detect_outliers_iqr(data, col, IQR_MULTIPLIER)
        outlier_results[col]['iqr'] = iqr_outliers
        outlier_results[col]['iqr_bounds'] = (lower, upper)

        # Z-score method
        zscore_outliers = detect_outliers_zscore(data, col, ZSCORE_THRESHOLD)
        outlier_results[col]['zscore'] = zscore_outliers

        print(f"      IQR outliers: {iqr_outliers.sum()} (bounds: [{lower:.2f}, {upper:.2f}])")
        print(f"      Z-score outliers: {zscore_outliers.sum()}")

    # Multivariate outlier detection using Isolation Forest
    print("\n   Running Isolation Forest for multivariate outlier detection...")
    available_for_if = [col for col in available_numeric_cols if col in data.columns and data[col].notna().sum() > 100]

    if len(available_for_if) >= 2:
        if_outliers = detect_outliers_isolation_forest(data, available_for_if, ISOLATION_CONTAMINATION)
        print(f"   Isolation Forest outliers: {if_outliers.sum()}")

        for col in available_for_if:
            outlier_results[col]['isolation_forest'] = if_outliers

    # 4. Temporal Gap Detection
    print("\n4. Detecting temporal gaps...")
    gaps, gap_info = detect_temporal_gaps(data, 'DateTime', '5min')
    print(f"   Found {gaps.sum()} gaps in the time series")

    # 5. Generate Visualizations
    print("\n5. Generating visualizations...")
    plot_data_quality_summary(data, available_numeric_cols, outlier_results, OUTPUT_DIR)
    plot_missing_values_heatmap(data, OUTPUT_DIR)

    # 6. Generate Summary Report
    print("\n6. Generating summary report...")
    report = generate_summary_report(data, available_numeric_cols, outlier_results, gap_info, OUTPUT_DIR)

    # 7. Save outlier indices for later use
    print("\n7. Saving outlier data...")
    outlier_df = pd.DataFrame(index=data.index)
    outlier_df['DateTime'] = data['DateTime']

    for col in available_numeric_cols:
        if col in outlier_results:
            outlier_df[f'{col}_IQR_Outlier'] = outlier_results[col]['iqr']
            outlier_df[f'{col}_ZScore_Outlier'] = outlier_results[col]['zscore']

    outlier_df.to_csv(os.path.join(OUTPUT_DIR, 'outlier_flags.csv'), index=False)
    print(f"   Outlier flags saved to {OUTPUT_DIR}outlier_flags.csv")

    # 8. Summary Statistics
    print("\n" + "=" * 80)
    print("DATA QUALITY REVIEW COMPLETE")
    print("=" * 80)
    print(f"\nOutput files saved to: {OUTPUT_DIR}")
    print("  - data_quality_overview.png")
    print("  - missing_values_heatmap.png")
    print("  - data_quality_report.txt")
    print("  - outlier_flags.csv")
