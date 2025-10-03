"""
Left Join Analysis - Demonstrates Missing Data Problems
=====================================================

This script shows how left joins can cause missing data when the right table
doesn't have matching records for all left table keys.
"""

import pandas as pd
import numpy as np
from join_helpers import load_and_clean_data, analyze_join_keys, save_join_results

def perform_left_join_analysis():
    """Demonstrate a left join that causes missing data"""
    # === LEFT JOIN ANALYSIS: MISSING DATA DEMONSTRATION ===
    # Left joins keep ALL Oracle records but lose POS data for unmatched records.
    
    # Load data
    oracle_df, pos_df = load_and_clean_data()
    oracle_df, pos_df = analyze_join_keys(oracle_df, pos_df)
    
    # Perform left join - keeps all Oracle records
    left_join = oracle_df.merge(
        pos_df, 
        left_on=['MRN_NO', 'VISIT_NUMBER'], 
        right_on=['MRN', 'VISIT_NO'], 
        how='left',
        suffixes=('_oracle', '_pos')
    )
    
    # Analyze the impact
    missing_pos_data = left_join['MRN'].isna().sum()
    total_records = len(left_join)
    missing_percentage = (missing_pos_data / total_records) * 100
    
    description = f"Left join preserving all Oracle records. {missing_pos_data}/{total_records} records missing POS data"
    
    # === MISSING DATA EXAMPLES ===
    # Isolate and view Oracle records that did not have a corresponding match in the POS data.
    missing_examples = left_join[left_join['MRN'].isna()][
        ['MRN_NO', 'VISIT_NUMBER', 'LINE_AMOUNT', 'TRANSACTION_TYPE_NAME', 'BUSINESS_UNIT_NAME']
    ].head(10)
    
    # === FINANCIAL IMPACT ANALYSIS ===
    # Calculate the total financial value of Oracle records that do not have a match in the POS data.
    oracle_total_amount = oracle_df['LINE_AMOUNT'].sum()
    matched_oracle_amount = left_join[~left_join['MRN'].isna()]['LINE_AMOUNT'].sum()
    unmatched_amount = oracle_total_amount - matched_oracle_amount
    
    # === BUSINESS UNIT IMPACT ===
    # Analyze which business units are most affected by the missing POS data.
    missing_by_bu = left_join[left_join['MRN'].isna()]['BUSINESS_UNIT_NAME'].value_counts()
    total_by_bu = left_join['BUSINESS_UNIT_NAME'].value_counts()
    
    if not missing_by_bu.empty:
        # Calculate the percentage of missing data for each business unit.
        for bu in total_by_bu.index:
            missing_count = missing_by_bu.get(bu, 0)
            total_count = total_by_bu[bu]
            percentage = (missing_count / total_count) * 100
    
    # === SUCCESSFUL MATCHES EXAMPLES ===
    # Isolate and view examples of records that were successfully matched between Oracle and POS data.
    matched_examples = left_join[~left_join['MRN'].isna()][
        ['MRN_NO', 'VISIT_NUMBER', 'LINE_AMOUNT', 'ITEM_LINE_AMOUNT', 'BUSINESS_UNIT_NAME']
    ].head(5)
    
    # Save results
    save_join_results(left_join, 'results/left_join_results.csv', description)
    
    # === KEY PROBLEMS WITH LEFT JOINS ===
    # 1. MISSING DATA: POS information is lost for unmatched Oracle records.
    # 2. INCOMPLETE ANALYSIS: Financial analysis is missing the POS perspective for unmatched records.
    # 3. BUSINESS IMPACT: Some business units may be more affected by data loss than others.
    # 4. SILENT FAILURE: The join operation completes without errors, but the resulting dataset is incomplete.
    
    # === WHEN TO USE LEFT JOINS ===
    # When you need ALL records from the primary (left) table, regardless of matches in the right table.
    # When missing data from the secondary (right) table is acceptable or expected for the analysis.
    # When you're performing an analysis centered on the left table's universe of data.
    # Don't use when a complete, matched dataset from both systems is required for accurate analysis.
    
    return left_join, description

def compare_with_other_joins():
    """Show quick comparison with other join types"""
    # === QUICK COMPARISON WITH OTHER JOINS ===
    
    oracle_df, pos_df = load_and_clean_data()
    
    # Perform different join types to compare record counts.
    left_join = oracle_df.merge(pos_df, left_on=['MRN_NO', 'VISIT_NUMBER'], 
                               right_on=['MRN', 'VISIT_NO'], how='left', suffixes=('_oracle', '_pos'))
    
    inner_join = oracle_df.merge(pos_df, left_on=['MRN_NO', 'VISIT_NUMBER'], 
                                right_on=['MRN', 'VISIT_NO'], how='inner', suffixes=('_oracle', '_pos'))
    
    outer_join = oracle_df.merge(pos_df, left_on=['MRN_NO', 'VISIT_NUMBER'], 
                                right_on=['MRN', 'VISIT_NO'], how='outer', 
                                suffixes=('_oracle', '_pos'), indicator=True)
    
    # Original and resulting record counts are calculated but not displayed.
    # len(oracle_df)
    # len(pos_df)
    # len(left_join)
    # len(inner_join)
    # len(outer_join)

if __name__ == "__main__":
    # Run the analysis
    result_df, description = perform_left_join_analysis()
    
    # Optional comparison
    compare_with_other_joins()
