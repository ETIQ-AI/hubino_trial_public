"""
Right Join Analysis - Demonstrates Distribution Change Problems
==============================================================

This script shows how right joins can systematically change data distributions
and introduce bias by keeping all records from the right table while potentially
losing records from the left table.
"""

import pandas as pd
import numpy as np
from join_helpers import load_and_clean_data, analyze_join_keys, save_join_results

def perform_right_join_analysis():
    """Demonstrate how right joins can change data distributions"""
    # === RIGHT JOIN ANALYSIS: DISTRIBUTION CHANGE DEMONSTRATION ===
    # Right joins keep ALL POS records but can lose Oracle data, potentially biasing the results.
    
    # Load data
    oracle_df, pos_df = load_and_clean_data()
    oracle_df2, pos_df2 = analyze_join_keys(oracle_df, pos_df)
    
    # Perform right join - keeps all POS records
    right_join = oracle_df2.merge(
        pos_df2,
        left_on=['MRN_NO', 'VISIT_NUMBER'],
        right_on=['MRN', 'VISIT_NO'],
        how='right',
        suffixes=('_oracle', '_pos')
    )
    
    # Analyze the extent of missing Oracle data in the joined result.
    missing_oracle_data = right_join['MRN_NO'].isna().sum()
    total_pos_records = len(pos_df)
    missing_percentage = (missing_oracle_data / len(right_join)) * 100
    
    description = f"Right join preserving all POS records. {missing_oracle_data}/{len(right_join)} records missing Oracle data"
    
    # === MISSING ORACLE DATA EXAMPLES ===
    # Isolate examples of POS records that did not have a matching Oracle record.
    missing_examples = right_join[right_join['MRN_NO'].isna()][
        ['MRN', 'VISIT_NO', 'ITEM_LINE_AMOUNT', 'TRANS_TYPE_NAME', 'FACILITY_CODE']
    ].head(10)

    return right_join, description

def analyze_facility_distribution_changes(right_join):
    """Analyze how facility distributions change after right join"""
    # === FACILITY DISTRIBUTION ANALYSIS ===
    
    # Load original data for comparison
    oracle_df, pos_df = load_and_clean_data()
    
    # Calculate original POS facility distribution.
    pos_facility_dist = pos_df['FACILITY_CODE'].value_counts(normalize=True) * 100
    pos_facility_counts = pos_df['FACILITY_CODE'].value_counts()

    # The distribution after the join should be identical since all POS records are kept.
    # This analysis confirms the integrity of the right table's data.
    joined_facility_dist = right_join['FACILITY_CODE'].value_counts(normalize=True) * 100
    joined_facility_counts = right_join['FACILITY_CODE'].value_counts()

def analyze_missing_data_bias(right_join):
    """Analyze if missing Oracle data creates systematic bias"""
    # === SYSTEMATIC BIAS ANALYSIS ===
    
    oracle_df, pos_df = load_and_clean_data()
    
    # Check if the rate of missing Oracle data varies by facility, which would indicate bias.
    missing_by_facility = right_join.groupby('FACILITY_CODE')['MRN_NO'].apply(
        lambda x: (x.isna().sum(), len(x), x.isna().sum() / len(x) * 100)
    )
    
    # Check if certain POS transaction types are more or less likely to have a matching Oracle record.
    missing_by_trans_type = right_join.groupby('TRANS_TYPE_NAME')['MRN_NO'].apply(
        lambda x: (x.isna().sum(), len(x), x.isna().sum() / len(x) * 100)
    )
    
    # === AMOUNT DISTRIBUTION BIAS ===
    # Compare the financial amounts of POS records with and without Oracle matches to detect bias.
    pos_with_oracle = right_join[~right_join['MRN_NO'].isna()]['ITEM_LINE_AMOUNT'].dropna()
    pos_without_oracle = right_join[right_join['MRN_NO'].isna()]['ITEM_LINE_AMOUNT'].dropna()
    
    if len(pos_with_oracle) > 0 and len(pos_without_oracle) > 0:
        # A significant difference in mean or median amounts suggests the missing data is not random.
        mean_diff = abs(pos_with_oracle.mean() - pos_without_oracle.mean())
        relative_diff = mean_diff / pos_with_oracle.mean() * 100

def analyze_completeness_by_time_period(right_join):
    """Check if missing data varies by time period"""
    # === TEMPORAL BIAS ANALYSIS ===
    
    # Convert date strings to datetime objects to enable time-based analysis.
    if 'TRANS_DATE' in right_join.columns:
        try:
            right_join['trans_date_parsed'] = pd.to_datetime(right_join['TRANS_DATE'], format='%d/%m/%y', errors='coerce')
            right_join['year_month'] = right_join['trans_date_parsed'].dt.to_period('M')
            
            # Group by month to see if data completeness changes over time.
            temporal_missing = right_join.groupby('year_month')['MRN_NO'].apply(
                lambda x: (x.isna().sum(), len(x), x.isna().sum() / len(x) * 100)
            )
        except Exception as e:
            # Handle potential date parsing errors.
            pass

def demonstrate_analysis_distortion(right_join):
    """Show how right join distorts analysis results"""
    # === ANALYSIS DISTORTION DEMONSTRATION ===
    
    oracle_df, pos_df = load_and_clean_data()
    
    # Compare total revenue from original sources vs. the joined data.
    oracle_revenue = oracle_df['LINE_AMOUNT'].sum()
    pos_revenue = pos_df['ITEM_LINE_AMOUNT'].sum()
    
    # The Oracle revenue in the joined table will be incomplete due to lost records.
    joined_oracle_revenue = right_join['LINE_AMOUNT'].dropna().sum()
    joined_pos_revenue = right_join['ITEM_LINE_AMOUNT'].sum()
    
    # Calculate the amount and percentage of Oracle revenue lost in the join.
    oracle_loss = oracle_revenue - joined_oracle_revenue
    oracle_loss_pct = (oracle_loss / oracle_revenue) * 100
    
    # Compare revenue by business unit to see if the data loss is concentrated in specific areas.
    original_bu_revenue = oracle_df.groupby('BUSINESS_UNIT_NAME')['LINE_AMOUNT'].sum()
    joined_bu_revenue = right_join.groupby('BUSINESS_UNIT_NAME')['LINE_AMOUNT'].sum()

if __name__ == "__main__":
    # Run the main analysis
    result_df, description = perform_right_join_analysis()
    
    # Run detailed bias analyses
    analyze_facility_distribution_changes(result_df)
    analyze_missing_data_bias(result_df)
    analyze_completeness_by_time_period(result_df)
    demonstrate_analysis_distortion(result_df)
    
    # Save results
    save_join_results(result_df, 'results/right_join_results.csv', description)
    
    # === KEY PROBLEMS WITH RIGHT JOINS ===
    # 1. MISSING LEFT DATA: Oracle information is lost for any POS records that don't have a match.
    # 2. SYSTEMATIC BIAS: If the missing data is not random, it can skew the dataset (e.g., certain facilities having higher data loss).
    # 3. DISTORTED ANALYSIS: Metrics calculated from the left-side data (Oracle) will be incomplete and inaccurate.
    # 4. HIDDEN BIAS: The analysis might appear correct because all right-side data is present, but it's skewed by the non-random absence of left-side data.
    
    # === WHEN RIGHT JOINS GO WRONG ===
    # When losing records from the left table creates a systematic bias in the dataset.
    # When you need a complete financial picture that requires data from both tables.
    # When the likelihood of a match correlates with important variables like facility, time, or transaction type.
    # Only use when you specifically want to analyze from the right table's perspective and accept the loss of non-matching left-side data.