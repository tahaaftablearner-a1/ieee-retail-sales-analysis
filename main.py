from pathlib import Path

import numpy as np
import pandas as pd


def run_retail_analysis():
  print('--- 1. Loading Dataset ---')
  csv_path = Path(__file__).resolve().parent / 'data' / 'sales.csv'
  df = pd.read_csv(csv_path)
  print(f'Original rows: {len(df)}')

  print('\n--- 2. Data Cleaning ---')
  # Remove duplicate rows[cite: 4]
  df.drop_duplicates(inplace=True)

  # Handle missing or non-numeric values in units_sold and unit_price[cite: 4]
  df['units_sold'] = pd.to_numeric(df['units_sold'], errors='coerce')
  df['unit_price'] = pd.to_numeric(df['unit_price'], errors='coerce')

  # Impute missing values
  df['units_sold'] = df['units_sold'].fillna(df['units_sold'].median())
  df['unit_price'] = df['unit_price'].fillna(df['unit_price'].mean())

  print(f'Cleaned rows: {len(df)}')

  print('\n--- 3. NumPy Revenue Calculations ---')
  # Calculate row-level revenue vector calculation
  df['total_revenue'] = df['units_sold'] * df['unit_price']

  # NumPy statistical aggregations[cite: 4]
  revenues = df['total_revenue'].to_numpy()
  units = df['units_sold'].to_numpy()

  total_revenue = np.sum(revenues)
  total_units = np.sum(units)
  aov = np.mean(revenues)

  print(f'Total Business Revenue: ${total_revenue:.2f}')
  print(f'Average Order Value (AOV): ${aov:.2f}')
  print(f'Total Units Sold: {int(total_units)}')

  print('\n--- 4. Groupby & Business Insights ---')
  category_group = (
      df.groupby('category')['total_revenue'].sum().reset_index()
  )
  category_group = category_group.sort_values(
      by='total_revenue', ascending=False
  )
  print('Revenue by Category:')
  print(category_group.to_string(index=False))

  top_product = (
      df.groupby('product_name')['total_revenue'].sum().idxmax()
  )
  print(f'\nHighest-Performing Product: {top_product}')


if __name__ == '__main__':
  run_retail_analysis()
