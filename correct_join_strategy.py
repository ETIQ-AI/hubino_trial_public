"""
Correct Join Strategy - Demonstrates Proper Data Integration
Shows the correct approach with cleaning, outer join, and proper aggregation.
"""

import pandas as pd
import numpy as np
from join_helpers import load_and_clean_data, analyze_join_keys, print_join_summary, save_join_results

def clean_and_standardize_keys(oracle_df, pos_df):
    """Step 1: Clean and standardize join keys"""
    oracle_clean = oracle_df.copy()
    pos_clean = pos_df.copy()
    
    # Standardize formatting (uppercase, replace underscores, strip whitespace)
    oracle_clean['MRN_CLEAN'] = oracle_clean['MRN_NO'].astype(str).str.upper().str.replace('_', '-').str.strip()
    oracle_clean['VISIT_CLEAN'] = oracle_clean['VISIT_NUMBER'].astype(str).str.upper().str.replace('_', '-').str.strip()
    
    pos_clean['MRN_CLEAN'] = pos_clean['MRN'].astype(str).str.upper().str.replace('_', '-').str.strip()
    pos_clean['VISIT_CLEAN'] = pos_clean['VISIT_NO'].astype(str).str.upper().str.replace('_', '-').str.strip()
    
    # Remove clearly invalid keys
    oracle_valid = oracle_clean[
        ~oracle_clean['MRN_CLEAN'].isin(['None', 'nan', 'NAN', '']) & 
        ~oracle_clean['VISIT_CLEAN'].isin(['None', 'nan', 'NAN', ''])
    ].copy()
    
    pos_valid = pos_clean[
        ~pos_clean['MRN_CLEAN'].isin(['None', 'nan', 'NAN', '']) & 
        ~pos_clean['VISIT_CLEAN'].isin(['None', 'nan', 'NAN', ''])
    ].copy()
    
    print(f"After standardization: Oracle {len(oracle_df)} → {len(oracle_valid)}, POS {len(pos_df)} → {len(pos_valid)}")
    
    return oracle_valid, pos_valid

def analyze_key_overlap(oracle_valid, pos_valid):
    """Step 2: Analyze key overlap after cleaning"""
    # Create composite join keys
    oracle_valid['join_key'] = oracle_valid['MRN_CLEAN'] + '|' + oracle_valid['VISIT_CLEAN']
    pos_valid['join_key'] = pos_valid['MRN_CLEAN'] + '|' + pos_valid['VISIT_CLEAN']
    
    oracle_keys = set(oracle_valid['join_key'].unique())
    pos_keys = set(pos_valid['join_key'].unique())
    
    overlap = oracle_keys.intersection(pos_keys)
    oracle_only = oracle_keys - pos_keys
    pos_only = pos_keys - oracle_keys
    
    print(f"Key overlap: {len(overlap)} matching, {len(oracle_only)} Oracle-only, {len(pos_only)} POS-only")
    
    return oracle_valid, pos_valid

def perform_correct_join(oracle_valid, pos_valid):
    """Step 3: Perform outer join to preserve all data"""
    # Outer join preserves all data from both systems
    correct_join = oracle_valid.merge(
        pos_valid,
        left_on=['MRN_CLEAN', 'VISIT_CLEAN'],
        right_on=['MRN_CLEAN', 'VISIT_CLEAN'],
        how='outer',
        suffixes=('_oracle', '_pos'),
        indicator=True
    )
    
    # Analyze merge results
    merge_stats = correct_join['_merge'].value_counts()
    total = len(correct_join)
    
    print(f"\nOuter join results: {total} unique patient visits")
    for category, count in merge_stats.items():
        print(f"  {category}: {count} ({count/total*100:.1f}%)")
    
    return correct_join

def handle_duplicates_through_aggregation(correct_join):
    """Step 4: Handle duplicates through proper aggregation"""
    # Check for duplicates at patient visit level
    duplicate_counts = correct_join.groupby(['MRN_CLEAN', 'VISIT_CLEAN']).size()
    duplicates = duplicate_counts[duplicate_counts > 1]
    
    print(f"\nPatient visits with multiple transactions: {len(duplicates)}")
    
    # Aggregate financial data by patient visit
    financial_summary = correct_join.groupby(['MRN_CLEAN', 'VISIT_CLEAN']).agg({
        'LINE_AMOUNT': ['sum', 'count', 'mean'],
        'ITEM_LINE_AMOUNT': ['sum', 'count', 'mean'],
        'BUSINESS_UNIT_NAME': 'first',
        'FACILITY_CODE': 'first',
        '_merge': 'first',
        'TRANSACTION_TYPE_NAME': lambda x: ', '.join(x.dropna().unique()),
        'TRANS_TYPE_NAME': lambda x: ', '.join(x.dropna().unique())
    }).reset_index()
    
    # Flatten column names
    financial_summary.columns = [
        'MRN', 'VISIT', 
        'Oracle_Total_Amount', 'Oracle_Transaction_Count', 'Oracle_Avg_Amount',
        'POS_Total_Amount', 'POS_Transaction_Count', 'POS_Avg_Amount',
        'Business_Unit', 'Facility', 'Data_Source',
        'Oracle_Transaction_Types', 'POS_Transaction_Types'
    ]
    
    print(f"After aggregation: {len(financial_summary)} unique patient visits")
    
    return financial_summary

def validate_financial_reconciliation(financial_summary, oracle_valid, pos_valid):
    """Step 5: Validate the financial reconciliation"""
    # Calculate totals
    original_oracle_total = oracle_valid['LINE_AMOUNT'].sum()
    original_pos_total = pos_valid['ITEM_LINE_AMOUNT'].sum()
    
    aggregated_oracle_total = financial_summary['Oracle_Total_Amount'].sum()
    aggregated_pos_total = financial_summary['POS_Total_Amount'].sum()
    
    # Check reconciliation accuracy
    oracle_diff = abs(original_oracle_total - aggregated_oracle_total)
    pos_diff = abs(original_pos_total - aggregated_pos_total)
    
    oracle_reconciled = oracle_diff < 0.01
    pos_reconciled = pos_diff < 0.01
    
    print(f"\nFinancial reconciliation:")
    print(f"  Oracle: ${original_oracle_total:,.2f} → ${aggregated_oracle_total:,.2f} ({'✓' if oracle_reconciled else '✗'})")
    print(f"  POS: ${original_pos_total:,.2f} → ${aggregated_pos_total:,.2f} ({'✓' if pos_reconciled else '✗'})")
    
    # Analyze data completeness
    both_systems = financial_summary[financial_summary['Data_Source'] == 'both']
    oracle_only = financial_summary[financial_summary['Data_Source'] == 'left_only']
    pos_only = financial_summary[financial_summary['Data_Source'] == 'right_only']
    
    print(f"\nData completeness:")
    print(f"  Both systems: {len(both_systems)} ({len(both_systems)/len(financial_summary)*100:.1f}%)")
    print(f"  Oracle only: {len(oracle_only)} ({len(oracle_only)/len(financial_summary)*100:.1f}%)")
    print(f"  POS only: {len(pos_only)} ({len(pos_only)/len(financial_summary)*100:.1f}%)")
    
    return financial_summary

def demonstrate_correct_analysis(financial_summary):
    """Step 6: Demonstrate proper analysis using correctly joined data"""
    both_systems = financial_summary[financial_summary['Data_Source'] == 'both']
    
    if len(both_systems) > 0:
        # Compare Oracle vs POS amounts for matched visits
        oracle_amounts = both_systems['Oracle_Total_Amount'].dropna()
        pos_amounts = both_systems['POS_Total_Amount'].dropna()
        
        if len(oracle_amounts) > 0 and len(pos_amounts) > 0:
            amount_correlation = oracle_amounts.corr(pos_amounts)
            mean_diff = abs(oracle_amounts.mean() - pos_amounts.mean())
            
            print(f"\nMatched visits analysis:")
            print(f"  Oracle vs POS correlation: {amount_correlation:.3f}")
            print(f"  Mean amount difference: ${mean_diff:.2f}")
    
    # Business unit analysis
    bu_analysis = financial_summary.groupby('Business_Unit').agg({
        'Oracle_Total_Amount': ['count', 'sum'],
        'POS_Total_Amount': ['sum'],
        'Data_Source': lambda x: (x == 'both').sum()
    }).round(2)
    
    # Data quality metrics
    total_visits = len(financial_summary)
    complete_data = len(financial_summary[financial_summary['Data_Source'] == 'both'])
    data_completeness = (complete_data / total_visits) * 100
    
    print(f"  Overall data completeness: {data_completeness:.1f}%")

def create_final_summary():
    """Print summary of the correct approach"""
    print("\n" + "="*60)
    print("CORRECT DATA INTEGRATION STRATEGY")
    print("="*60)
    print("✓ Step 1: Standardize and clean join keys")
    print("✓ Step 2: Analyze key overlap and relationships")
    print("✓ Step 3: Use outer join to preserve all data")
    print("✓ Step 4: Aggregate duplicates appropriately")
    print("✓ Step 5: Validate financial reconciliation")
    print("✓ Step 6: Perform proper analysis at correct granularity")
    print("="*60)

if __name__ == "__main__":
    print("=== Correct Data Integration Strategy ===")
    
    # Load original data
    oracle_df, pos_df = load_and_clean_data()
    
    # Execute the correct strategy step by step
    oracle_valid, pos_valid = clean_and_standardize_keys(oracle_df, pos_df)
    oracle_valid, pos_valid = analyze_key_overlap(oracle_valid, pos_valid)
    correct_join = perform_correct_join(oracle_valid, pos_valid)
    financial_summary = handle_duplicates_through_aggregation(correct_join)
    financial_summary = validate_financial_reconciliation(financial_summary, oracle_valid, pos_valid)
    demonstrate_correct_analysis(financial_summary)
    
    # Save results
    save_join_results(correct_join, 'results/correct_join_detailed.csv', 
                     "Complete outer join with all transaction details")
    save_join_results(financial_summary, 'results/financial_summary.csv',
                     "Aggregated financial summary by patient visit")
    
    # Final summary
    create_final_summary()
    
    print("\nAnalysis complete. Files saved:")
    print("  - results/correct_join_detailed.csv")
    print("  - results/financial_summary.csv")