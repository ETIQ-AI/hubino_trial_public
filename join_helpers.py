
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_and_clean_data():
    """Load the generated datasets"""
    oracle_df = pd.read_csv('hubino_trial/data/oracle_demo.csv')
    pos_df = pd.read_csv('hubino_trial/data/pos_demo.csv')
    return oracle_df, pos_df

def analyze_join_keys(oracle_df, pos_df):
    """Analyze the join keys to understand overlap"""
    # Create combined keys for analysis
    oracle_df['join_key'] = oracle_df['MRN_NO'].astype(str) + '|' + oracle_df['VISIT_NUMBER'].astype(str)
    pos_df['join_key'] = pos_df['MRN'].astype(str) + '|' + pos_df['VISIT_NO'].astype(str)
    
    # Remove null combinations
    oracle_valid = oracle_df[~oracle_df['join_key'].str.contains('None|nan', na=False)]
    pos_valid = pos_df[~pos_df['join_key'].str.contains('None|nan', na=False)]
    
    oracle_keys = set(oracle_valid['join_key'].unique())
    pos_keys = set(pos_valid['join_key'].unique())
    
    # Calculate overlap statistics
    overlap_count = len(oracle_keys.intersection(pos_keys))
    oracle_only = len(oracle_keys - pos_keys)
    pos_only = len(pos_keys - oracle_keys)
    
    return oracle_df, pos_df

def create_join_visualization(join_results_dict, filename='join_comparison.png'):
    """Create visualizations comparing different join results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Record counts by join type
    join_counts = {name: len(df) if df is not None else 0 
                   for name, (df, _) in join_results_dict.items()}
    
    colors = ['skyblue', 'lightcoral', 'gold', 'lightgreen', 'orange', 'purple']
    axes[0,0].bar(join_counts.keys(), join_counts.values(), color=colors[:len(join_counts)])
    axes[0,0].set_title('Record Counts by Join Type')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # 2. Missing data analysis
    missing_data = {}
    for join_name, (df, _) in join_results_dict.items():
        if df is not None:
            if 'MRN' in df.columns:
                missing_data[join_name] = df['MRN'].isna().sum()
            elif 'MRN_NO' in df.columns:
                missing_data[join_name] = df['MRN_NO'].isna().sum()
    
    if missing_data:
        axes[0,1].bar(missing_data.keys(), missing_data.values(), color=colors[:len(missing_data)])
        axes[0,1].set_title('Missing Data Records by Join Type')
        axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. Amount comparisons
    oracle_df, pos_df = load_and_clean_data()
    oracle_amounts = oracle_df['LINE_AMOUNT'].dropna()
    pos_amounts = pos_df['ITEM_LINE_AMOUNT'].dropna()
    
    axes[1,0].hist(oracle_amounts, alpha=0.7, label='Oracle', bins=20, color='skyblue')
    axes[1,0].hist(pos_amounts, alpha=0.7, label='POS', bins=20, color='lightcoral')
    axes[1,0].set_title('Original Amount Distribution Comparison')
    axes[1,0].set_xlabel('Amount')
    axes[1,0].legend()
    
    # 4. Data completeness summary
    completeness = {}
    for join_name, (df, _) in join_results_dict.items():
        if df is not None:
            if '_merge' in df.columns:
                both_count = (df['_merge'] == 'both').sum()
                completeness[join_name] = (both_count / len(df)) * 100 if len(df) > 0 else 0
            else:
                if 'MRN' in df.columns and 'MRN_NO' in df.columns:
                    both_not_null = (~df['MRN'].isna() & ~df['MRN_NO'].isna()).sum()
                    completeness[join_name] = (both_not_null / len(df)) * 100 if len(df) > 0 else 0
                else:
                    completeness[join_name] = 100
    
    if completeness:
        axes[1,1].bar(completeness.keys(), completeness.values(), color=colors[:len(completeness)])
        axes[1,1].set_title('Data Completeness % (Both Systems)')
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def print_join_summary(join_name, result_df, description):
    """Print a minimal summary for join results"""
    print(f"\n{join_name}: {len(result_df)} records - {description}")
    return result_df

def save_join_results(result_df, filename, description):
    """Save join results to CSV"""
    result_df.to_csv(filename, index=False)
