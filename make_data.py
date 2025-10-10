import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import string

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_random_string(length):
    """Generate random string of given length"""
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

def generate_mrn():
    """Generate Medical Record Number"""
    facility = random.choice(['BJAL', 'SHAH', 'CHER', 'PUCH'])
    number = f"{random.randint(1, 999999):07d}"
    return f"{facility}-{number}"

def generate_visit_number():
    """Generate Visit Number"""
    facility = random.choice(['BJAL', 'SHAH', 'CHER', 'PUCH'])
    number = f"A{random.randint(1, 999999):09d}"
    return f"{number}-{facility}"

def generate_date(base_date, days_range):
    """Generate random date within range"""
    random_days = random.randint(-days_range, days_range)
    return base_date + timedelta(days=random_days)

# Generate Oracle demo data
def create_oracle_demo():
    oracle_data = []
    
    # Create some base MRN/Visit pairs that will be used in both datasets
    base_pairs = []
    for i in range(50):  # 50 matching pairs
        mrn = generate_mrn()
        visit = generate_visit_number()
        base_pairs.append((mrn, visit))
    
    # Create additional MRN/Visit pairs that won't match
    additional_pairs = []
    for i in range(30):  # 30 non-matching pairs
        mrn = generate_mrn()
        visit = generate_visit_number()
        additional_pairs.append((mrn, visit))
    
    all_pairs = base_pairs + additional_pairs
    
    business_units = ['CAH BUKIT JALIL', 'CAH SHAH ALAM', 'CAH CHERAS', 'CAH PUCHONG', 'CAH PETALING JAYA']
    transaction_sources = ['Care21 Imported', 'Manual Entry', 'System Generated', 'External Import']
    account_codes = ['412101', '412102', '412103', '413001', '413002', '414001']
    sub_account_codes = ['4001', '4002', '4003', '5001', '5002']
    transaction_types = ['IP Corporate INV', 'OP Corporate INV', 'Emergency INV', 'Pharmacy INV']
    doctors = ['YOGESVARAN.KANAPATY', 'AHMAD.IBRAHIM', 'SARAH.TAN', 'MUTHU.KRISHNAN', 'DAVID.LIM']
    specialities = ['EAR, NOSE AND THROAT', 'EMERGENCY MEDICINE', 'INTERNAL MEDICINE', 'SURGERY', 'PEDIATRICS']
    departments = ['ER', 'Ward A', 'Ward B', 'Pharmacy', 'Laboratory']
    
    base_date = datetime(2025, 5, 12)
    
    for i in range(100):  # 100 oracle records
        # Deliberately create some duplicates and missing data issues
        if i < 50:
            mrn, visit = base_pairs[i]
        else:
            mrn, visit = random.choice(all_pairs)
        
        # Create some duplicate entries (same MRN/Visit but different transactions)
        if i % 15 == 0 and i > 0:
            mrn, visit = random.choice(base_pairs[:20])  # Reuse early pairs
        
        # Sometimes make visit numbers slightly different to test fuzzy matching
        if random.random() < 0.1:
            visit = visit.replace('-', '_') if '-' in visit else visit + '_MOD'
        
        # Sometimes introduce missing MRN or Visit
        if random.random() < 0.05:
            mrn = None
        if random.random() < 0.03:
            visit = None
        
        transaction_date = generate_date(base_date, 30)
        creation_date = transaction_date + timedelta(hours=random.randint(0, 48))
        posted_date = creation_date + timedelta(days=random.randint(1, 5))
        
        record = {
            'BUSINESS_UNIT_NAME': random.choice(business_units),
            'TRANSACTION_SOURCE': random.choice(transaction_sources),
            'ACCOUNTING_CLASS_CODE': 'REVENUE',
            'ACCOUNT_CODE': random.choice(account_codes),
            'SUB_ACCOUNT_CODE': random.choice(sub_account_codes),
            'TRANSACTION_TYPE_NAME': random.choice(transaction_types),
            'TRANSACTION_DATE': transaction_date.strftime('%d/%m/%y'),
            'CREATION_DATE': creation_date.strftime('%d/%m/%y %H:%M'),
            'POSTED_DATE': posted_date.strftime('%d/%m/%y'),
            'TRANSACTION_NUMBER': f"BJ-IPC{random.randint(1000, 9999)}",
            'TRANSACTION_LINE_DESCRIPTION': f"Medical Item {random.randint(1, 100)}",
            'LINE_AMOUNT': round(random.uniform(10, 500), 2),
            'CONCATENATED_SEGMENT': f"20701-40001-100601-{random.choice(account_codes)}-{random.choice(sub_account_codes)}-99999-100970-999999-9999-9999",
            'MRN_NO': mrn,
            'VISIT_NUMBER': visit,
            'PRIMARY_ADMITTING_DOCTOR': random.choice(doctors),
            'PRIMARY_DOCTOR_SPECIALITY': random.choice(specialities),
            'ORDER_DOCTOR': random.choice(doctors),
            'ORDER_DOCTOR_SPECIALITY': random.choice(specialities),
            'ISSUE_DEPARTMENT': random.choice(departments),
            'LEGAL_ENTITY_NAME': "ASIA ONEHEALTHCARE SDN BHD (FORMERLY KNOWN AS COLUMBIA ASIA HEATHCARE SDN BHD), CAH GLOBAL BUSINESS CENTER SDN BHD",
            'BU_NAME_P': "ASIA ONEHEALTHCARE SDN BHD, CAH BUKIT JALIL, CAH CHERAS, CAH PUCHONG",
            'FROM_DATE_P': transaction_date.strftime('%d/%m/%y'),
            'TO_DATE_P': (transaction_date + timedelta(days=7)).strftime('%d/%m/%y')
        }
        oracle_data.append(record)
    
    return pd.DataFrame(oracle_data), base_pairs

# Generate POS demo data
def create_pos_demo(base_pairs):
    pos_data = []
    
    # Use different proportions of base pairs to create join challenges
    facilities = ['BJAL', 'SHAH', 'CHER', 'PUCH', 'KLNG']
    trans_types = ['IPCBill', 'OPCBill', 'EmergencyBill', 'PharmacyBill']
    payment_terms = ['30', '60', '90', 'COD']
    doc_types = ['Bill', 'Credit Note', 'Adjustment', 'Refund']
    currencies = ['RM', 'USD', 'SGD']
    cashiers = ['NURAFIFAH.ZAINI', 'AHMAD.HASSAN', 'SITI.AMINAH', 'RAJESH.KUMAR']
    doctors = ['YOGESVARAN.KANAPATY', 'AHMAD.IBRAHIM', 'SARAH.TAN', 'MUTHU.KRISHNAN', 'DAVID.LIM']
    departments = ['FA', 'ER', 'Ward A', 'Ward B', 'Pharmacy']
    
    base_date = datetime(2025, 5, 12)
    
    for i in range(120):  # 120 POS records (more than Oracle to test 1:many relationships)
        # Use base pairs for first 60 records, then mix with random data
        if i < 40:
            mrn, visit = base_pairs[i % len(base_pairs)]
        elif i < 80:
            # Create some records that partially match (same MRN, different visit)
            mrn, _ = base_pairs[i % len(base_pairs)]
            visit = generate_visit_number()
        else:
            # Completely new pairs
            mrn = generate_mrn()
            visit = generate_visit_number()
        
        # Create duplicates with different amounts (billing adjustments)
        if i % 12 == 0 and i > 0:
            mrn, visit = random.choice(base_pairs[:25])
        
        # Introduce data quality issues
        if random.random() < 0.08:
            mrn = mrn.replace('-', '') if mrn and '-' in mrn else mrn  # Remove hyphens
        if random.random() < 0.06:
            visit = visit.upper() if visit else visit  # Change case
        if random.random() < 0.04:
            mrn = None
        if random.random() < 0.03:
            visit = None
        
        trans_date = generate_date(base_date, 25)
        creation_date = trans_date + timedelta(hours=random.randint(0, 72))
        
        # Create distribution changes - some facilities have higher amounts
        if 'BJAL' in str(mrn):
            base_amount = random.uniform(100, 1000)  # Higher amounts for BJAL
        else:
            base_amount = random.uniform(20, 300)   # Lower amounts for others
        
        record = {
            'HEADER_ID': random.randint(700000, 800000),
            'FLOW_ID': generate_random_string(22),
            'LINE_NO': random.randint(1, 5),
            'LINE_STATUS': random.choice(['OPEN', 'CLOSED', 'PENDING', 'CANCELLED']),
            'ERROR': '' if random.random() > 0.1 else 'DATA_VALIDATION_ERROR',
            'TRANS_PRIMARY_ID': random.randint(90000, 99999),
            'FACILITY_CODE': random.choice(facilities),
            'TRANS_TYPE_NAME': random.choice(trans_types),
            'PAYMENT_TERMS': random.choice(payment_terms),
            'TRANS_DATE': trans_date.strftime('%d/%m/%y'),
            'DOC_NUMBER': f"BJ-{random.choice(['IPC', 'OPC', 'EMG'])}{random.randint(1000, 9999)}",
            'DEBTOR_CODE': random.randint(2000, 3000),
            'DOC_TYPE': random.choice(doc_types),
            'DOC_DETAIL_ID': random.randint(750000, 760000),
            'PACKAGE_LINE_ID': random.randint(0, 5),
            'ITEM_DESC': f"Medical Service {random.randint(1, 50)}",
            'ITEM_CODE': f"AC{random.randint(100000, 999999):06d}",
            'TAX_AMOUNT': round(base_amount * 0.06, 2) if random.random() > 0.3 else 0,
            'TAX_CODE': '' if random.random() > 0.7 else 'GST6',
            'DISCOUNT_AMOUNT': round(base_amount * random.uniform(0, 0.2), 2),
            'CURRENCY_CODE': random.choice(currencies),
            'ITEM_LINE_AMOUNT': round(base_amount, 2),
            'PAYABLE_AMOUNT': round(base_amount * random.uniform(0.8, 1.0), 2),
            'ITEM_LINE_QUANTITY': random.randint(1, 3),
            'UNIT_PRICE': round(base_amount, 2),
            'PATIENT_IDENT_NO': f"{random.randint(800000, 999999)}-{random.randint(10, 99)}-{random.randint(1000, 9999)}",
            'MRN': mrn,
            'VISIT_NO': visit,
            'GL_NUMBER': f"A{random.randint(100000000, 999999999)}",
            'CASHIER_CODE': random.choice(cashiers),
            'DOC_DATE': trans_date.strftime('%d/%m/%y %H:%M'),
            'PRIMARY_DOCTOR': random.choice(doctors),
            'ADMIT_DATE_TIME': (trans_date - timedelta(hours=random.randint(1, 48))).strftime('%d/%m/%y %H:%M'),
            'ADMIT_VISIT_DATE_TIME': '',
            'DISCHARGE_DATE_TIME': (trans_date + timedelta(hours=random.randint(24, 168))).strftime('%d/%m/%y %H:%M') if random.random() > 0.3 else '',
            'WARD': f"MD{random.randint(1, 10)}" if random.random() > 0.4 else '',
            'ROOM': random.randint(100, 600) if random.random() > 0.3 else '',
            'BED': f"{random.randint(100, 600)}{random.choice(['A', 'B'])}" if random.random() > 0.3 else '',
            'DISCOUNT_REASON': random.choice(['CORPORATE PRICING DISCOUNT', 'INSURANCE DISCOUNT', 'STAFF DISCOUNT', '']),
            'ORDER_DOCTOR': random.choice(doctors),
            'ISSUE_DEPARTMENT': random.choice(departments),
            'REV_ACCOUNT_CODE': random.randint(4000000, 5000000),
            'REC_ACC_CODE': random.randint(1500000, 1600000),
            'DISCOUNT_ACC_CODE': random.randint(5000000, 6000000),
            'TAX_ACC_CODE': '',
            'ROUNDING_ACC_CODE': random.randint(5300000, 5400000),
            'BILL_ALLOCATE_TO': '',
            'DOCUMENT_AMOUNT': round(base_amount * random.uniform(1.0, 1.5), 1),
            'ALLOCATE_CHARGE_IDENTIFIER': random.choice(['Y', 'N', '']),
            'CREATED_BY': 'integrationUser',
            'CREATION_DATE': f"{creation_date.hour}:{creation_date.minute}.0",
            'IS_INVENTORY': random.choice(['Yes', 'No']),
            'ERP_STATUS': random.choice(['SUCCESS', 'PENDING', 'FAILED']),
            'CARE21_CALLBACK_STATUS': random.choice(['S', 'F', 'P'])
        }
        pos_data.append(record)
    
    return pd.DataFrame(pos_data)

# Generate the datasets
print("Generating Oracle demo data...")
oracle_df, base_pairs = create_oracle_demo()

print("Generating POS demo data...")
pos_df = create_pos_demo(base_pairs)

# Save to CSV files
oracle_df.to_csv('data/oracle_demo.csv', index=False)
pos_df.to_csv('data/pos_demo.csv', index=False)

print(f"Generated oracle_demo.csv with {len(oracle_df)} records")
print(f"Generated pos_demo.csv with {len(pos_df)} records")

# Print some statistics about the generated data
print("\n=== DATA QUALITY CHALLENGES INTRODUCED ===")
print(f"Oracle records with missing MRN: {oracle_df['MRN_NO'].isna().sum()}")
print(f"Oracle records with missing VISIT_NUMBER: {oracle_df['VISIT_NUMBER'].isna().sum()}")
print(f"POS records with missing MRN: {pos_df['MRN'].isna().sum()}")
print(f"POS records with missing VISIT_NO: {pos_df['VISIT_NO'].isna().sum()}")

# Check for potential duplicates
oracle_duplicates = oracle_df.groupby(['MRN_NO', 'VISIT_NUMBER']).size()
pos_duplicates = pos_df.groupby(['MRN', 'VISIT_NO']).size()

print(f"\nOracle MRN/Visit combinations appearing multiple times: {(oracle_duplicates > 1).sum()}")
print(f"POS MRN/Visit combinations appearing multiple times: {(pos_duplicates > 1).sum()}")

print("\n=== SAMPLE ORACLE DATA ===")
print(oracle_df[['MRN_NO', 'VISIT_NUMBER', 'LINE_AMOUNT', 'BUSINESS_UNIT_NAME']].head())

print("\n=== SAMPLE POS DATA ===")
print(pos_df[['MRN', 'VISIT_NO', 'ITEM_LINE_AMOUNT', 'FACILITY_CODE']].head())

print("\nFiles generated successfully! The data includes:")
print("1. Tricky joins - only ~40% of records will have perfect matches")
print("2. Missing data - some MRN/Visit values are None")
print("3. Data quality issues - different formats, cases, separators")
print("4. Duplicates - some MRN/Visit pairs appear multiple times")
print("5. Distribution changes - BJAL facility has higher amounts than others")
