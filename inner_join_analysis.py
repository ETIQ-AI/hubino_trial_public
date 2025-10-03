"""
Inner Join Analysis - Demonstrates Data Duplication Problems
===========================================================

This script shows how inner joins can cause data duplication when both tables
have multiple records for the same join keys (many-to-many relationships).
"""

import pandas as pd
import numpy as np
from join_helpers import load_and_clean_data, analyze_join_keys, save_join_results

def perform_inner_join_analysis():
    """Demonstrate how inner joins can cause duplication"""
    # === INNER JOIN ANALYSIS: DATA DUPLICATION DEMONSTRATION ===
    # Inner joins can inadvertently multiply records when join keys are not unique in one or both tables.
    
    # Load data
    oracle_df, pos_df = load_and_clean_data()
    oracle_df, pos_df = analyze_join_keys(oracle_df, pos_df)
    
    # Count original valid records for comparison.
    oracle_valid = oracle_df[~oracle_df['MRN_NO'].isna() & ~oracle_df['VISIT_NUMBER'].isna()]
    pos_valid = pos_df[~pos_df['MRN'].isna() & ~pos_df['VISIT_NO'].isna()]
    
    # Perform inner join
    inner_join = oracle_df.merge(
        pos_df,
        left_on=['MRN_NO', 'VISIT_NUMBER'],
        right_on=['MRN', 'VISIT_NO'],
        how='inner',
        suffixes=('_oracle', '_pos')
    )
    
    # Calculate the duplication factor to quantify the record inflation.
    joined_records = len(inner_join)
    expected_max = min(len(oracle_valid), len(pos_valid))
    duplication_factor = joined_records / expected_max if expected_max > 0 else 0
    
    description = f"Inner join with {joined_records} records, {duplication_factor:.2f}x duplication factor"
    
    # === DUPLICATION PATTERN ANALYSIS ===
    
    # Find keys that appear multiple times in the joined result to identify many-to-many matches.
    key_counts = inner_join.groupby(['MRN_NO', 'VISIT_NUMBER']).size()
    duplicated_keys = key_counts[key_counts > 1].sort_values(ascending=False)
    
    if len(duplicated_keys) > 0:
        # === DETAILED DUPLICATION EXAMPLE ===
        # Select the most duplicated key combination to inspect the source of the multiplication.
        example_key = duplicated_keys.index[0]
        example_mrn, example_visit = example_key
        
        # Show the original records from each source table for the example key.
        oracle_example = oracle_df[
            (oracle_df['MRN_NO'] == example_mrn) & 
            (oracle_df['VISIT_NUMBER'] == example_visit)
        ][['TRANSACTION_NUMBER', 'LINE_AMOUNT', 'TRANSACTION_TYPE_NAME', 'ISSUE_DEPARTMENT']]
        
        pos_example = pos_df[
            (pos_df['MRN'] == example_mrn) & 
            (pos_df['VISIT_NO'] == example_visit)
        ][['DOC_NUMBER', 'ITEM_LINE_AMOUNT', 'TRANS_TYPE_NAME', 'ISSUE_DEPARTMENT']]
        
        # The number of resulting records is the product of the record counts from each table for that key.
        expected_combinations = len(oracle_example) * len(pos_example)
        
        # Isolate the resulting joined records for the example key.
        joined_example = inner_join[
            (inner_join['MRN_NO'] == example_mrn) & 
            (inner_join['VISIT_NUMBER'] == example_visit)
        ][['TRANSACTION_NUMBER', 'LINE_AMOUNT', 'DOC_NUMBER', 'ITEM_LINE_AMOUNT']].head(10)
    
    # === FINANCIAL IMPACT OF DUPLICATION ===
    # Calculate original totals for comparison against the inflated joined totals.
    oracle_total = oracle_df['LINE_AMOUNT'].sum()
    pos_total = pos_df['ITEM_LINE_AMOUNT'].sum()
    
    # Summing amounts on the joined table will lead to incorrect, inflated results.
    joined_oracle_total = inner_join['LINE_AMOUNT'].sum()
    joined_pos_total = inner_join['ITEM_LINE_AMOUNT'].sum()
    
    # === DUPLICATION BY BUSINESS UNIT ===
    # Check if duplication is more prevalent in certain business units.
    duplication_by_bu = inner_join.groupby('BUSINESS_UNIT_NAME').apply(
        lambda x: len(x) / len(x.groupby(['MRN_NO', 'VISIT_NUMBER']))
    ).sort_values(ascending=False)
    
    # Save results
    save_join_results(inner_join, 'results/inner_join_results.csv', description)
    
    # === KEY PROBLEMS WITH INNER JOINS ===
    # 1. DATA MULTIPLICATION: Records multiply when join keys aren't unique, leading to a Cartesian product for matched keys.
    # 2. INFLATED TOTALS: Financial amounts and other metrics become artificially inflated due to duplicated records.
    # 3. MANY-TO-MANY: An inner join on a many-to-many relationship pairs every record from the left with every record from the right for a given key.
    # 4. ANALYSIS DISTORTION: Metrics like averages, sums, and counts become unreliable and misleading.
    
    # === WHEN INNER JOINS GO WRONG ===
    # - When you assume a 1:1 or 1:many relationship, but the data actually has a many-to-many relationship.
    # - When both tables have multiple valid records per join key (e.g., multiple transactions for the same patient visit).
    # - When you need to sum amounts post-join without correcting for the duplication.
    # - Use only when you are certain about the relationship cardinality and that it will not cause unintended duplication.
    
    return inner_join, description

def analyze_relationship_cardinality():
    """Analyze the actual relationships between Oracle and POS data"""
    # === RELATIONSHIP CARDINALITY ANALYSIS ===
    
    oracle_df, pos_df = load_and_clean_data()
    
    # Analyze Oracle key frequency to check for duplicates.
    oracle_key_counts = oracle_df.groupby(['MRN_NO', 'VISIT_NUMBER']).size()
    oracle_duplicates = oracle_key_counts[oracle_key_counts > 1]
    
    # Analyze POS key frequency to check for duplicates.  
    pos_key_counts = pos_df.groupby(['MRN', 'VISIT_NO']).size()
    pos_duplicates = pos_key_counts[pos_key_counts > 1]
    
    # Determine the relationship type based on the presence of duplicate keys in each table.
    oracle_has_dupes = len(oracle_duplicates) > 0
    pos_has_dupes = len(pos_duplicates) > 0
    
    if not oracle_has_dupes and not pos_has_dupes:
        relationship = "1:1 (Ideal for inner join)"
    elif oracle_has_dupes and not pos_has_dupes:
        relationship = "Many:1 (Oracle to POS)"
    elif not oracle_has_dupes and pos_has_dupes:
        relationship = "1:Many (Oracle to POS)"
    else:
        relationship = "Many:Many (Problematic for inner join)"

if __name__ == "__main__":
    # Run the cardinality analysis first to understand the data relationship.
    analyze_relationship_cardinality()
    
    # Run the main analysis to demonstrate the effect of an inner join on this data.
    result_df, description = perform_inner_join_analysis()