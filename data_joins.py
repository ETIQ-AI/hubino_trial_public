import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_and_clean_data():
    """Load the generated datasets"""
    print("Loading datasets...")
    oracle_df = pd.read_csv('data/oracle_demo.csv')
    pos_df = pd.read_csv('data/pos_demo.csv')
    
    print(f"Oracle data: {len(oracle_df)} records")
    print(f"POS data: {len(pos_df)} records")
    
    return oracle_df, pos_df

def analyze_join_keys(oracle_df, pos_df):
    """Analyze the join keys to understand overlap"""
    print("\n=== JOIN KEY ANALYSIS ===")
    
    # Create combined keys for analysis
    oracle_df['join_key'] = oracle_df['MRN_NO'].astype(str) + '|' + oracle_df['VISIT_NUMBER'].astype(str)
    pos_df['join_key'] = pos_df['MRN'].astype(str) + '|' + pos_df['VISIT_NO'].astype(str)
    
    # Remove null combinations
    oracle_valid = oracle_df[~oracle_df['join_key'].str.contains('None|nan', na=False)]
    pos_valid = pos_df[~pos_df['join_key'].str.contains('None|nan', na=False)]
    
    oracle_keys = set(oracle_valid['join_key'].unique())
    pos_keys = set(pos_valid['join_key'].unique())
    
    print(f"Unique Oracle join keys: {len(oracle_keys)}")
    print(f"Unique POS join keys: {len(pos_keys)}")
    print(f"Overlapping keys: {len(oracle_keys.intersection(pos_keys))}")
    print(f"Oracle-only keys: {len(oracle_keys - pos_keys)}")
    print(f"POS-only keys: {len(pos_keys - oracle_keys)}")
    
    return oracle_df, pos_df

def join_causing_missingness(oracle_df, pos_df):
    """Demonstrate a join that causes missing data"""
    print("\n=== 1. JOIN CAUSING MISSING-NESS (LEFT JOIN) ===")
    
    # Left join - keeps all Oracle records, but many won't have POS matches
    left_join = oracle_df.merge(
        pos_df, 
        left_on=['MRN_NO', 'VISIT_NUMBER'], 
        right_on=['MRN', 'VISIT_NO'], 
        how='left',
        suffixes=('_oracle', '_pos')
    )
    
    # Analyze missingness
    missing_pos_data = left_join['MRN'].isna().sum()
    total_records = len(left_join)
    missing_percentage = (missing_pos_data / total_records) * 100
    
    print(f"Total records after left join: {total_records}")
    print(f"Records missing POS data: {missing_pos_data} ({missing_percentage:.1f}%)")
    
    # Show examples of missing data
    print("\nExamples of Oracle records without POS matches:")
    missing_examples = left_join[left_join['MRN'].isna()][
        ['MRN_NO', 'VISIT_NUMBER', 'LINE_AMOUNT', 'TRANSACTION_TYPE_NAME']
    ].head()
    print(missing_examples.to_string(index=False))
    
    # Analyze impact on financial data
    oracle_total_amount = oracle_df['LINE_AMOUNT'].sum()
    matched_oracle_amount = left_join[~left_join['MRN'].isna()]['LINE_AMOUNT'].sum()
    unmatched_amount = oracle_total_amount - matched_oracle_amount
    
    print(f"\nFinancial Impact:")
    print(f"Total Oracle amount: ${oracle_total_amount:,.2f}")
    print(f"Matched Oracle amount: ${matched_oracle_amount:,.2f}")
    print(f"Unmatched amount: ${unmatched_amount:,.2f}")
    
    return left_join

def join_causing_duplication(oracle_df, pos_df):
    """Demonstrate a join that causes duplication"""
    print("\n=== 2. JOIN CAUSING DUPLICATION (INNER JOIN) ===")
    
    # Inner join where both tables have multiple records for same keys
    inner_join = oracle_df.merge(
        pos_df,
        left_on=['MRN_NO', 'VISIT_NUMBER'],
        right_on=['MRN', 'VISIT_NO'],
        how='inner',
        suffixes=('_oracle', '_pos')
    )
    
    oracle_records = len(oracle_df[~oracle_df['MRN_NO'].isna() & ~oracle_df['VISIT_NUMBER'].isna()])
    pos_records = len(pos_df[~pos_df['MRN'].isna() & ~pos_df['VISIT_NO'].isna()])
    joined_records = len(inner_join)
    
    print(f"Oracle records (with valid keys): {oracle_records}")
    print(f"POS records (with valid keys): {pos_records}")
    print(f"Records after inner join: {joined_records}")
    print(f"Duplication factor: {joined_records / min(oracle_records, pos_records):.2f}x")
    
    # Find specific examples of duplication
    duplicated_keys = inner_join.groupby(['MRN_NO', 'VISIT_NUMBER']).size()
    duplicates = duplicated_keys[duplicated_keys > 1]
    
    print(f"\nKey combinations appearing multiple times: {len(duplicates)}")
    print("Examples of duplicated keys:")
    print(duplicates.head().to_string())
    
    # Show example of how one Oracle record becomes multiple
    if len(duplicates) > 0:
        example_key = duplicates.index[0]
        example_mrn, example_visit = example_key
        
        print(f"\nExample: MRN {example_mrn}, Visit {example_visit}")
        oracle_example = oracle_df[
            (oracle_df['MRN_NO'] == example_mrn) & 
            (oracle_df['VISIT_NUMBER'] == example_visit)
        ][['TRANSACTION_NUMBER', 'LINE_AMOUNT', 'TRANSACTION_TYPE_NAME']]
        
        pos_example = pos_df[
            (pos_df['MRN'] == example_mrn) & 
            (pos_df['VISIT_NO'] == example_visit)
        ][['DOC_NUMBER', 'ITEM_LINE_AMOUNT', 'TRANS_TYPE_NAME']]
        
        print("Oracle records:")
        print(oracle_example.to_string(index=False))
        print("POS records:")
        print(pos_example.to_string(index=False))
        print(f"This creates {len(oracle_example) * len(pos_example)} joined records!")
    
    return inner_join

def join_changing_distribution(oracle_df, pos_df):
    """Demonstrate how joins can change data distributions"""
    print("\n=== 3. JOIN CHANGING DISTRIBUTION ===")
    
    # Right join - keeps all POS records, loses Oracle records
    right_join = oracle_df.merge(
        pos_df,
        left_on=['MRN_NO', 'VISIT_NUMBER'],
        right_on=['MRN', 'VISIT_NO'],
        how='right',
        suffixes=('_oracle', '_pos')
    )
    
    print("Original vs Right Join Distribution Comparison:")
    
    # Analyze facility distribution changes
    print("\nFacility Distribution Change:")
    
    # Original POS facility distribution
    pos_facility_dist = pos_df['FACILITY_CODE'].value_counts(normalize=True) * 100
    
    # After right join (some facilities might be underrepresented if Oracle data is missing)
    joined_facility_dist = right_join['FACILITY_CODE'].value_counts(normalize=True) * 100
    
    facility_comparison = pd.DataFrame({
        'Original_POS_%': pos_facility_dist,
        'After_RightJoin_%': joined_facility_dist
    }).fillna(0)
    
    print(facility_comparison.round(1))
    
    # Analyze amount distribution changes
    print("\nAmount Distribution Changes:")
    
    # Original amounts
    oracle_amounts = oracle_df['LINE_AMOUNT'].dropna()
    pos_amounts = pos_df['ITEM_LINE_AMOUNT'].dropna()
    joined_oracle_amounts = right_join['LINE_AMOUNT'].dropna()
    
    print(f"Original Oracle amount stats:")
    print(f"  Mean: ${oracle_amounts.mean():.2f}, Std: ${oracle_amounts.std():.2f}")
    print(f"Original POS amount stats:")
    print(f"  Mean: ${pos_amounts.mean():.2f}, Std: ${pos_amounts.std():.2f}")
    print(f"Joined Oracle amount stats (right join):")
    print(f"  Mean: ${joined_oracle_amounts.mean():.2f}, Std: ${joined_oracle_amounts.std():.2f}")
    
    # Check for systematic bias
    missing_oracle = right_join['LINE_AMOUNT'].isna().sum()
    total_pos = len(right_join)
    print(f"\nPOS records without Oracle matches: {missing_oracle}/{total_pos} ({missing_oracle/total_pos*100:.1f}%)")
    
    # Analyze which facilities are more affected
    missing_by_facility = right_join.groupby('FACILITY_CODE')['LINE_AMOUNT'].apply(
        lambda x: x.isna().sum() / len(x) * 100
    ).sort_values(ascending=False)
    
    print("Missing Oracle data by facility:")
    print(missing_by_facility.round(1).to_string())
    
    return right_join

def correct_join_strategy(oracle_df, pos_df):
    """Demonstrate the correct join strategy for this dataset"""
    print("\n=== 4. CORRECT JOIN STRATEGY ===")
    
    # Step 1: Clean and standardize join keys
    oracle_clean = oracle_df.copy()
    pos_clean = pos_df.copy()
    
    # Standardize MRN format (remove inconsistencies)
    oracle_clean['MRN_CLEAN'] = oracle_clean['MRN_NO'].astype(str).str.upper().str.replace('_', '-')
    oracle_clean['VISIT_CLEAN'] = oracle_clean['VISIT_NUMBER'].astype(str).str.upper().str.replace('_', '-')
    
    pos_clean['MRN_CLEAN'] = pos_clean['MRN'].astype(str).str.upper().str.replace('_', '-')
    pos_clean['VISIT_CLEAN'] = pos_clean['VISIT_NO'].astype(str).str.upper().str.replace('_', '-')
    
    # Remove null keys
    oracle_valid = oracle_clean[
        ~oracle_clean['MRN_CLEAN'].isin(['None', 'nan']) & 
        ~oracle_clean['VISIT_CLEAN'].isin(['None', 'nan'])
    ]
    
    pos_valid = pos_clean[
        ~pos_clean['MRN_CLEAN'].isin(['None', 'nan']) & 
        ~pos_clean['VISIT_CLEAN'].isin(['None', 'nan'])
    ]
    
    print(f"After cleaning:")
    print(f"Oracle valid records: {len(oracle_valid)}/{len(oracle_df)}")
    print(f"POS valid records: {len(pos_valid)}/{len(pos_df)}")
    
    # Step 2: Perform outer join to capture all data
    correct_join = oracle_valid.merge(
        pos_valid,
        left_on=['MRN_CLEAN', 'VISIT_CLEAN'],
        right_on=['MRN_CLEAN', 'VISIT_CLEAN'],
        how='outer',
        suffixes=('_oracle', '_pos'),
        indicator=True
    )
    
    # Analyze the results
    merge_stats = correct_join['_merge'].value_counts()
    total = len(correct_join)
    
    print(f"\nCorrect Join Results:")
    print(f"Both datasets: {merge_stats.get('both', 0)} ({merge_stats.get('both', 0)/total*100:.1f}%)")
    print(f"Oracle only: {merge_stats.get('left_only', 0)} ({merge_stats.get('left_only', 0)/total*100:.1f}%)")
    print(f"POS only: {merge_stats.get('right_only', 0)} ({merge_stats.get('right_only', 0)/total*100:.1f}%)")
    print(f"Total unique patient visits: {total}")
    
    # Step 3: Handle duplicates by aggregating
    print(f"\nHandling Duplicates:")
    
    # Count duplicates before aggregation
    dup_count = correct_join.groupby(['MRN_CLEAN', 'VISIT_CLEAN']).size()
    duplicates = (dup_count > 1).sum()
    print(f"Patient visits with multiple records: {duplicates}")
    
    # Create aggregated view for analysis
    financial_summary = correct_join.groupby(['MRN_CLEAN', 'VISIT_CLEAN']).agg({
        'LINE_AMOUNT': ['sum', 'count'],
        'ITEM_LINE_AMOUNT': ['sum', 'count'],
        'BUSINESS_UNIT_NAME': 'first',
        'FACILITY_CODE': 'first',
        '_merge': 'first'
    }).reset_index()
    
    # Flatten column names
    financial_summary.columns = ['MRN', 'VISIT', 'Oracle_Total_Amount', 'Oracle_Transaction_Count', 
                                'POS_Total_Amount', 'POS_Transaction_Count', 'Business_Unit', 'Facility', 'Data_Source']
    
    print(f"After aggregation: {len(financial_summary)} unique patient visits")
    
    # Show summary statistics
    print(f"\nFinancial Summary:")
    oracle_total = financial_summary['Oracle_Total_Amount'].sum()
    pos_total = financial_summary['POS_Total_Amount'].sum()
    print(f"Total Oracle amount: ${oracle_total:,.2f}")
    print(f"Total POS amount: ${pos_total:,.2f}")
    print(f"Difference: ${abs(oracle_total - pos_total):,.2f}")
    
    # Show examples of well-matched records
    both_systems = financial_summary[financial_summary['Data_Source'] == 'both']
    if len(both_systems) > 0:
        print(f"\nExamples of records in both systems:")
        examples = both_systems[['MRN', 'VISIT', 'Oracle_Total_Amount', 'POS_Total_Amount', 'Business_Unit']].head()
        print(examples.to_string(index=False))
    
    return correct_join, financial_summary

def create_visualizations(oracle_df, pos_df, left_join, inner_join, right_join, correct_join):
    """Create visualizations to show the impact of different joins"""
    print("\n=== CREATING VISUALIZATIONS ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Record counts by join type
    join_counts = {
        'Original Oracle': len(oracle_df),
        'Original POS': len(pos_df),
        'Left Join': len(left_join),
        'Inner Join': len(inner_join),
        'Right Join': len(right_join),
        'Correct Join': len(correct_join)
    }
    
    axes[0,0].bar(join_counts.keys(), join_counts.values(), color=['skyblue', 'lightcoral', 'gold', 'lightgreen', 'orange', 'purple'])
    axes[0,0].set_title('Record Counts by Join Type')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # 2. Missing data by join type
    missing_data = {
        'Left Join': left_join['MRN'].isna().sum(),
        'Right Join': right_join['MRN_NO'].isna().sum(),
        'Inner Join': inner_join['MRN'].isna().sum() + inner_join['MRN'].isna().sum(),
        'Correct Join': correct_join['MRN_CLEAN'].isna().sum()
    }
    
    axes[0,1].bar(missing_data.keys(), missing_data.values(), color=['gold', 'orange', 'lightgreen', 'purple'])
    axes[0,1].set_title('Missing Data Records by Join Type')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. Amount distribution comparison
    oracle_amounts = oracle_df['LINE_AMOUNT'].dropna()
    pos_amounts = pos_df['ITEM_LINE_AMOUNT'].dropna()
    
    axes[1,0].hist(oracle_amounts, alpha=0.7, label='Oracle', bins=20, color='skyblue')
    axes[1,0].hist(pos_amounts, alpha=0.7, label='POS', bins=20, color='lightcoral')
    axes[1,0].set_title('Amount Distribution Comparison')
    axes[1,0].set_xlabel('Amount')
    axes[1,0].legend()
    
    # 4. Data completeness by join
    completeness = {
        'Left Join': (len(left_join) - left_join['MRN'].isna().sum()) / len(left_join) * 100,
        'Right Join': (len(right_join) - right_join['MRN_NO'].isna().sum()) / len(right_join) * 100,
        'Inner Join': 100,  # Complete by definition
        'Correct Join': correct_join['_merge'].value_counts()['both'] / len(correct_join) * 100
    }
    
    axes[1,1].bar(completeness.keys(), completeness.values(), color=['gold', 'orange', 'lightgreen', 'purple'])
    axes[1,1].set_title('Data Completeness % (Both Systems)')
    axes[1,1].tick_params(axis='x', rotation=45)
    axes[1,1].set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig('join_analysis_results.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'join_analysis_results.png'")
    plt.show()

def main():
    """Main execution function"""
    # Load data
    oracle_df, pos_df = load_and_clean_data()
    
    # Analyze join keys
    oracle_df, pos_df = analyze_join_keys(oracle_df, pos_df)
    
    # Demonstrate different join problems
    left_join = join_causing_missingness(oracle_df, pos_df)
    inner_join = join_causing_duplication(oracle_df, pos_df)
    right_join = join_changing_distribution(oracle_df, pos_df)
    correct_join, financial_summary = correct_join_strategy(oracle_df, pos_df)

    # Create visualizations
    create_visualizations(oracle_df, pos_df, left_join, inner_join, right_join, correct_join)
    
    print("\n=== SUMMARY ===")
    print("This analysis demonstrates four key data integration challenges:")
    print("1. MISSING DATA: Left joins lose POS data for unmatched Oracle records")
    print("2. DUPLICATION: Inner joins multiply records when keys aren't unique")
    print("3. DISTRIBUTION CHANGE: Right joins can bias results toward certain facilities")
    print("4. CORRECT APPROACH: Outer join with key standardization and aggregation")
    print("\nThese are exactly the types of issues your testing software should detect!")

if __name__ == "__main__":
    main()