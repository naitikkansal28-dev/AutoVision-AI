"""
AutoVision AI - Database Seed & Synthetic Data Generator
Populates the database with realistic automotive dealership records across 36 months.
Generates users, vehicles, customers, employees, enquiries, bookings, sales, inventory, and expenses.
Also exports data/sample_sales.csv for machine learning benchmarks and CSV reporting.
"""

import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

import config
from database.connection import init_database, execute_query, fetch_dataframe, get_backend
from utils.authentication import hash_password

# Set random seed for reproducible research data
random.seed(42)
np.random.seed(42)


def seed_users():
    """Seeds default Admin and Manager accounts with hashed passwords."""
    print("[Seed] Seeding users...")
    execute_query("DELETE FROM users")
    
    for username, details in config.DEMO_CREDENTIALS.items():
        pwd_hash, salt = hash_password(details["password"])
        sql = """
            INSERT INTO users (username, password_hash, salt, full_name, email, role, is_active)
            VALUES (%s, %s, %s, %s, %s, %s, 1)
        """
        execute_query(sql, (
            username,
            pwd_hash,
            salt,
            details["full_name"],
            details["email"],
            details["role"]
        ))
    print("[Seed] Users seeded successfully.")


def seed_employees():
    """Seeds showroom sales representatives and managers."""
    print("[Seed] Seeding employees...")
    execute_query("DELETE FROM employees")
    
    reps = [
        ("Aarav Sharma", "Senior Sales Consultant", "aarav.sharma@autovision.ai", "+91 98201 11223", 8, 1.75, "2022-01-15"),
        ("Priya Verma", "Executive Sales Consultant", "priya.verma@autovision.ai", "+91 98202 22334", 7, 1.50, "2022-03-10"),
        ("Rohan Kulkarni", "EV & Luxury Specialist", "rohan.kulkarni@autovision.ai", "+91 98203 33445", 6, 2.00, "2022-06-01"),
        ("Ananya Iyer", "Customer Relationship Manager", "ananya.iyer@autovision.ai", "+91 98204 44556", 5, 1.25, "2023-01-12"),
        ("Vikram Singh", "Fleet & Commercial Sales", "vikram.singh@autovision.ai", "+91 98205 55667", 6, 1.50, "2023-04-20"),
        ("Neha Gupta", "Junior Sales Associate", "neha.gupta@autovision.ai", "+91 98206 66778", 4, 1.00, "2023-08-15")
    ]
    
    for name, role, email, phone, target, comm, hire_date in reps:
        sql = """
            INSERT INTO employees (name, role, email, phone, monthly_sales_target, commission_rate, hire_date, is_active)
            VALUES (%s, %s, %s, %s, %s, %s, %s, 1)
        """
        execute_query(sql, (name, role, email, phone, target, comm, hire_date))
    print("[Seed] Employees seeded successfully.")


def seed_vehicles():
    """Seeds comprehensive vehicle catalog covering hatchbacks, sedans, SUVs, EVs, and luxury."""
    print("[Seed] Seeding vehicles...")
    execute_query("DELETE FROM vehicles")
    
    vehicle_catalog = [
        # Brand, Model, Variant, Body, Fuel, Transmission, Year, Color, Cost, Ex-Showroom
        ("Tata", "Nexon", "Creative Plus", "Compact SUV", "Petrol", "Manual", 2024, "Daytona Grey", 920000, 1150000),
        ("Tata", "Nexon.ev", "Empowered Plus LR", "Compact SUV", "Electric", "Automatic", 2024, "Empowered White", 1550000, 1920000),
        ("Tata", "Harrier", "Fearless Dark Edition", "Mid-Size SUV", "Diesel", "Automatic", 2024, "Oberon Black", 1850000, 2350000),
        ("Tata", "Safari", "Accomplished Plus", "Full-Size SUV", "Diesel", "Automatic", 2024, "Cosmic Gold", 2050000, 2600000),
        ("Tata", "Punch", "Accomplished Dazzle", "Micro SUV", "Petrol", "Manual", 2024, "Tropical Mist", 620000, 780000),
        
        ("Mahindra", "XUV700", "AX7 Luxury Diesel", "Full-Size SUV", "Diesel", "Automatic", 2024, "Everest White", 1950000, 2480000),
        ("Mahindra", "Scorpio-N", "Z8 Select", "Full-Size SUV", "Diesel", "Manual", 2024, "Deep Forest", 1450000, 1850000),
        ("Mahindra", "Thar", "LX 4x4 Hard Top", "Lifestyle SUV", "Diesel", "Automatic", 2024, "Napoli Black", 1350000, 1720000),
        ("Mahindra", "XUV400", "EL Pro 39.4kWh", "Compact SUV", "Electric", "Automatic", 2024, "Infinity Blue", 1380000, 1740000),
        
        ("Hyundai", "Creta", "SX (O) Turbo DCT", "Mid-Size SUV", "Petrol", "Automatic", 2024, "Robust Emerald", 1580000, 2000000),
        ("Hyundai", "Venue", "SX 1.2 Kappa", "Compact SUV", "Petrol", "Manual", 2024, "Titan Grey", 840000, 1060000),
        ("Hyundai", "Verna", "SX(O) 1.5 Turbo DCT", "Sedan", "Petrol", "Automatic", 2024, "Abyss Black", 1420000, 1780000),
        ("Hyundai", "Ioniq 5", "Long Range RWD", "Luxury EV", "Electric", "Automatic", 2024, "Gravity Gold Matte", 3800000, 4650000),
        
        ("Maruti Suzuki", "Grand Vitara", "Alpha Plus Strong Hybrid", "Mid-Size SUV", "Hybrid", "Automatic", 2024, "Opulent Red", 1620000, 2020000),
        ("Maruti Suzuki", "Brezza", "ZXi Plus AT", "Compact SUV", "Petrol", "Automatic", 2024, "Magma Grey", 1020000, 1310000),
        ("Maruti Suzuki", "Swift", "ZXi Plus Dual Tone", "Hatchback", "Petrol", "Manual", 2024, "Luster Blue", 680000, 850000),
        ("Maruti Suzuki", "Fronx", "Alpha 1.0L Turbo 6AT", "Crossover", "Petrol", "Automatic", 2024, "Splendid Silver", 980000, 1250000),
        
        ("Toyota", "Innova Hycross", "ZX (O) Hybrid", "MPV", "Hybrid", "Automatic", 2024, "Blackish Ageha", 2550000, 3180000),
        ("Toyota", "Fortuner", "4x4 Legender AT", "Full-Size SUV", "Diesel", "Automatic", 2024, "Pearl White", 3650000, 4450000),
        ("Toyota", "Urban Cruiser Hyryder", "V e-Drive 2WD", "Mid-Size SUV", "Hybrid", "Automatic", 2024, "Cafe White", 1610000, 2010000),
        
        ("Kia", "Seltos", "GTX Plus 1.5 Turbo DCT", "Mid-Size SUV", "Petrol", "Automatic", 2024, "Pewter Olive", 1620000, 2050000),
        ("Kia", "Sonet", "HTX 1.5 Diesel AT", "Compact SUV", "Diesel", "Automatic", 2024, "Glacier White", 1050000, 1330000),
        ("Kia", "EV6", "GT-Line AWD", "Luxury EV", "Electric", "Automatic", 2024, "Moonscape Matte", 5200000, 6500000),
        
        ("Honda", "City", "ZX e:HEV Hybrid", "Sedan", "Hybrid", "Automatic", 2024, "Radiant Red", 1680000, 2080000),
        ("Honda", "Elevate", "ZX CVT", "Mid-Size SUV", "Petrol", "Automatic", 2024, "Phoenix Orange", 1250000, 1590000),
        
        ("BMW", "3 Series Gran Limousine", "330Li M Sport", "Luxury Sedan", "Petrol", "Automatic", 2024, "Portimao Blue", 4900000, 6150000),
        ("Mercedes-Benz", "C-Class", "C 200", "Luxury Sedan", "Petrol", "Automatic", 2024, "Cavansite Blue", 5100000, 6380000)
    ]
    
    for v in vehicle_catalog:
        sql = """
            INSERT INTO vehicles (brand, model, variant, body_type, fuel_type, transmission, model_year, color, cost_price, ex_showroom_price)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        execute_query(sql, v)
    print(f"[Seed] {len(vehicle_catalog)} vehicles seeded successfully.")


def seed_customers(n=150):
    """Seeds realistic customer demographics across cities and income brackets."""
    print(f"[Seed] Seeding {n} customers...")
    execute_query("DELETE FROM customers")
    
    first_names = [
        "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh", "Ayaan", "Krishna", "Ishaan",
        "Shaurya", "Atharv", "Dhruv", "Kabir", "Rohan", "Siddharth", "Aanya", "Diya", "Gauri", "Ananya",
        "Pari", "Isha", "Navya", "Aditi", "Meera", "Rhea", "Avani", "Tanvi", "Kavya", "Sneha", "Karan",
        "Manish", "Rajesh", "Sunil", "Amit", "Vikram", "Suresh", "Ramesh", "Deepak", "Nitin"
    ]
    last_names = [
        "Patel", "Sharma", "Verma", "Mehta", "Deshmukh", "Kulkarni", "Shah", "Reddy", "Nair", "Iyer",
        "Chopra", "Malhotra", "Kapoor", "Joshi", "Bhatia", "Singhania", "Aggarwal", "Gupta", "Bansal", "Rao"
    ]
    cities = [
        ("Mumbai", "Maharashtra"), ("Pune", "Maharashtra"), ("Delhi", "Delhi NCR"),
        ("Bengaluru", "Karnataka"), ("Hyderabad", "Telangana"), ("Ahmedabad", "Gujarat"),
        ("Chennai", "Tamil Nadu"), ("Jaipur", "Rajasthan"), ("Chandigarh", "Punjab/Haryana"),
        ("Kolkata", "West Bengal"), ("Nagpur", "Maharashtra"), ("Indore", "Madhya Pradesh")
    ]
    brands = ["Tata", "Mahindra", "Hyundai", "Maruti Suzuki", "Toyota", "Kia", "Honda", "BMW", "Mercedes-Benz"]
    fuels = ["Petrol", "Diesel", "Electric", "Hybrid"]

    for i in range(n):
        fn = random.choice(first_names)
        ln = random.choice(last_names)
        full_name = f"{fn} {ln}"
        email = f"{fn.lower()}.{ln.lower()}{random.randint(10, 99)}@example.com"
        phone = f"+91 {random.choice([98, 97, 99, 91, 93])}{random.randint(10000000, 99999999)}"
        city, state = random.choice(cities)
        
        # Annual income between 6 LPA and 55 LPA
        annual_income = round(random.uniform(600000, 5500000), -4)
        budget_max = round(annual_income * random.uniform(0.5, 1.2), -4)
        pref_brand = random.choice(brands)
        pref_fuel = random.choice(fuels)
        
        sql = """
            INSERT INTO customers (full_name, email, phone, city, state, annual_income, budget_max, preferred_brand, preferred_fuel)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        execute_query(sql, (full_name, email, phone, city, state, annual_income, budget_max, pref_brand, pref_fuel))
    print(f"[Seed] {n} customers seeded successfully.")


def seed_sales_and_history():
    """
    Generates 36 months of realistic historical sales records (from Jan 2023 to Dec 2025/early 2026),
    incorporating realistic automotive market dynamics:
    - Monthly volume ~ 20-45 units/month
    - Festive surge in Sep-Nov (Diwali/Dussehra)
    - Year-end closing push in March & December
    - Monsoon moderation in July-August
    - Segment-appropriate discounts, insurance, and financing
    - Realistic gross profit margins
    """
    print("[Seed] Seeding 36 months of realistic sales transactions...")
    execute_query("DELETE FROM sales")
    
    vehicles_df = fetch_dataframe("SELECT * FROM vehicles")
    customers_df = fetch_dataframe("SELECT * FROM customers")
    employees_df = fetch_dataframe("SELECT * FROM employees")
    
    finance_partners = ["HDFC Bank", "State Bank of India", "ICICI Bank", "Kotak Mahindra Bank", "Axis Bank", "Tata Capital"]
    payment_methods = ["Bank Loan", "Bank Loan", "Bank Loan", "Full Cash / Wire", "Corporate Lease"]
    
    sales_records = []
    invoice_seq = 1001

    # Date range: 36 months, Jan 1, 2023 to Dec 31, 2025
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2025, 12, 31)
    
    current_month_date = start_date
    while current_month_date <= end_date:
        year = current_month_date.year
        month = current_month_date.month
        
        # Base monthly volume
        base_volume = random.randint(22, 30)
        
        # Seasonality factors
        season_multiplier = 1.0
        if month in [10, 11]:  # Festival season (Diwali, Dussehra)
            season_multiplier = 1.45
        elif month == 3:       # FY close tax depreciation benefits
            season_multiplier = 1.30
        elif month == 12:      # Year-end clearance discounts
            season_multiplier = 1.25
        elif month in [7, 8]:  # Monsoon slowdown
            season_multiplier = 0.85
            
        # Year-over-year growth (dealership expanding)
        yoy_growth = 1.0 + (year - 2023) * 0.12
        monthly_sales_count = int(base_volume * season_multiplier * yoy_growth)
        
        # Days in month
        if month in [1, 3, 5, 7, 8, 10, 12]:
            days_in_month = 31
        elif month in [4, 6, 9, 11]:
            days_in_month = 30
        else:
            days_in_month = 28
            
        for _ in range(monthly_sales_count):
            day = random.randint(1, days_in_month)
            sale_date = datetime(year, month, day).strftime("%Y-%m-%d")
            
            # Select vehicle with weighted popularity
            # SUVs & Hybrids are very popular in current market
            weights = []
            for _, v in vehicles_df.iterrows():
                w = 1.0
                if v["body_type"] in ["Compact SUV", "Mid-Size SUV"]:
                    w *= 2.5
                if v["fuel_type"] in ["Hybrid", "Electric"]:
                    w *= 1.8
                if v["brand"] in ["Tata", "Mahindra", "Hyundai", "Toyota"]:
                    w *= 2.0
                weights.append(w)
            
            weights = np.array(weights) / sum(weights)
            v_idx = np.random.choice(len(vehicles_df), p=weights)
            vehicle = vehicles_df.iloc[v_idx]
            
            customer = customers_df.sample(1).iloc[0]
            sales_rep = employees_df.sample(1).iloc[0]
            
            ex_price = float(vehicle["ex_showroom_price"])
            cost_price = float(vehicle["cost_price"])
            
            # Discount depending on month (higher in Dec)
            base_discount_pct = random.uniform(0.01, 0.04)
            if month == 12:
                base_discount_pct += 0.02
            discount_amount = round(ex_price * base_discount_pct, -2)
            
            # Add-ons: Accessories & Insurance
            accessories_amount = round(random.uniform(15000, 55000), -2)
            insurance_amount = round(ex_price * random.uniform(0.03, 0.045), -2)
            
            # Selling price = Ex-Showroom - Discount
            selling_price = ex_price - discount_amount
            
            # Dealer Gross Profit = (selling_price - cost_price) + (dealer margin on insurance ~ 15%) + (margin on accessories ~ 35%)
            dealer_insurance_margin = insurance_amount * 0.15
            dealer_acc_margin = accessories_amount * 0.35
            gross_profit = round((selling_price - cost_price) + dealer_insurance_margin + dealer_acc_margin, 2)
            
            invoice_num = f"INV-{year}-{invoice_seq}"
            invoice_seq += 1
            
            payment = random.choice(payment_methods)
            financer = random.choice(finance_partners) if "Loan" in payment else "N/A"
            
            sales_records.append({
                "invoice_number": invoice_num,
                "booking_id": None,
                "customer_id": int(customer["id"]),
                "vehicle_id": int(vehicle["id"]),
                "sales_rep_id": int(sales_rep["id"]),
                "sale_date": sale_date,
                "selling_price": selling_price,
                "cost_price": cost_price,
                "discount_amount": discount_amount,
                "accessories_amount": accessories_amount,
                "insurance_amount": insurance_amount,
                "gross_profit": gross_profit,
                "payment_method": payment,
                "finance_partner": financer
            })
            
        # Advance to next month
        if month == 12:
            current_month_date = datetime(year + 1, 1, 1)
        else:
            current_month_date = datetime(year, month + 1, 1)
            
    # Insert batch into database
    for rec in sales_records:
        sql = """
            INSERT INTO sales (invoice_number, booking_id, customer_id, vehicle_id, sales_rep_id,
                               sale_date, selling_price, cost_price, discount_amount,
                               accessories_amount, insurance_amount, gross_profit,
                               payment_method, finance_partner)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        execute_query(sql, (
            rec["invoice_number"], rec["booking_id"], rec["customer_id"], rec["vehicle_id"],
            rec["sales_rep_id"], rec["sale_date"], rec["selling_price"], rec["cost_price"],
            rec["discount_amount"], rec["accessories_amount"], rec["insurance_amount"],
            rec["gross_profit"], rec["payment_method"], rec["finance_partner"]
        ))
        
    print(f"[Seed] {len(sales_records)} sales records generated across 2023-2025.")
    
    # Save a clean benchmark CSV in data/sample_sales.csv
    sales_export_df = pd.DataFrame(sales_records)
    # Join vehicle details for clean standalone CSV analytics
    merged_export = sales_export_df.merge(
        vehicles_df[["id", "brand", "model", "variant", "body_type", "fuel_type", "transmission"]],
        left_on="vehicle_id", right_on="id"
    )
    merged_export.drop(columns=["id"], inplace=True)
    csv_path = config.DATA_DIR / "sample_sales.csv"
    merged_export.to_csv(csv_path, index=False)
    print(f"[Seed] Exported standalone research benchmark to {csv_path}")


def seed_inventory_and_pipeline():
    """
    Seeds current showroom physical inventory (Available, Booked, Sold units),
    enquiries, test drives, bookings, and recurring monthly expenses.
    """
    print("[Seed] Seeding showroom inventory, bookings, and enquiries pipeline...")
    execute_query("DELETE FROM inventory")
    execute_query("DELETE FROM enquiries")
    execute_query("DELETE FROM bookings")
    execute_query("DELETE FROM expenses")
    
    vehicles_df = fetch_dataframe("SELECT * FROM vehicles")
    customers_df = fetch_dataframe("SELECT * FROM customers")
    employees_df = fetch_dataframe("SELECT * FROM employees")
    
    # 1. Inventory units
    vin_counter = 50001
    today = datetime(2026, 1, 15)
    
    # Create ~65 units in active inventory
    for _ in range(65):
        v = vehicles_df.sample(1).iloc[0]
        # Days in stock ranging from 5 days up to 130 days (creating realistic aging stock!)
        days_in_stock = random.choice([
            random.randint(5, 25),   # fast turnaround (fresh)
            random.randint(26, 55),  # normal turnaround
            random.randint(65, 85),  # aging (>60 days warning)
            random.randint(95, 135)  # critically aging (>90 days alert!)
        ])
        arrival_date = (today - timedelta(days=days_in_stock)).strftime("%Y-%m-%d")
        
        status = random.choices(["Available", "Booked", "Sold"], weights=[0.60, 0.25, 0.15])[0]
        vin = f"AV{v['brand'][:3].upper()}{vin_counter}"
        vin_counter += 1
        
        bay = random.choice(["Main Showroom Floor", "Bay A (Ground)", "Bay B (First Floor)", "Yard Stock Area"])
        
        sql = """
            INSERT INTO inventory (vin, vehicle_id, stock_status, arrival_date, showroom_location, days_in_stock)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        execute_query(sql, (vin, int(v["id"]), status, arrival_date, bay, days_in_stock))
        
    # 2. Enquiries & Test Drives (~80 enquiries)
    enquiry_statuses = ["New", "Contacted", "Test Drive Scheduled", "Test Drive Completed", "Negotiating", "Converted", "Lost"]
    for i in range(80):
        c = customers_df.sample(1).iloc[0]
        v = vehicles_df.sample(1).iloc[0]
        rep = employees_df.sample(1).iloc[0]
        
        enquiry_date = (today - timedelta(days=random.randint(1, 90))).strftime("%Y-%m-%d")
        status = random.choices(enquiry_statuses, weights=[0.20, 0.15, 0.15, 0.20, 0.15, 0.10, 0.05])[0]
        test_drive = 1 if status in ["Test Drive Scheduled", "Test Drive Completed", "Negotiating", "Converted"] else 0
        notes = f"Customer interested in {v['brand']} {v['model']} {v['variant']}. Budget max ₹{c['budget_max']:,.0f}."
        
        sql = """
            INSERT INTO enquiries (customer_id, vehicle_id, sales_rep_id, enquiry_date, status, test_drive_conducted, notes)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        execute_query(sql, (int(c["id"]), int(v["id"]), int(rep["id"]), enquiry_date, status, test_drive, notes))
        
    # 3. Active Bookings (~35 active bookings)
    booking_statuses = ["Confirmed", "Confirmed", "Confirmed", "Completed", "Cancelled"]
    for i in range(35):
        c = customers_df.sample(1).iloc[0]
        v = vehicles_df.sample(1).iloc[0]
        rep = employees_df.sample(1).iloc[0]
        
        b_date = (today - timedelta(days=random.randint(2, 45))).strftime("%Y-%m-%d")
        deliv_date = (today + timedelta(days=random.randint(5, 30))).strftime("%Y-%m-%d")
        b_ref = f"BK-2026-{100 + i}"
        status = random.choice(booking_statuses)
        b_amt = random.choice([25000, 50000, 100000, 150000])
        
        sql = """
            INSERT INTO bookings (booking_reference, customer_id, vehicle_id, sales_rep_id, booking_date, booking_amount, status, delivery_promised_date, notes)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        execute_query(sql, (b_ref, int(c["id"]), int(v["id"]), int(rep["id"]), b_date, b_amt, status, deliv_date, "Deposit received, loan clearance in progress."))
        
    # 4. Showroom Operational Expenses across 36 months
    expense_categories = [
        ("Showroom Facility Lease / Rent", 350000),
        ("Showroom Staff Fixed Salaries", 420000),
        ("Electricity & Commercial Utilities", 65000),
        ("Digital Marketing & Local Ads", 95000),
        ("Showroom Maintenance & Display Cleaning", 30000),
        ("Customer Hospitality & Test Drive Fuel", 25000)
    ]
    
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2025, 12, 31)
    current_date = start_date
    
    while current_date <= end_date:
        for cat, base_amt in expense_categories:
            fluctuation = random.uniform(0.92, 1.08)
            amt = round(base_amt * fluctuation, -2)
            exp_date = current_date.strftime("%Y-%m-05")
            sql = """
                INSERT INTO expenses (expense_date, category, amount, description)
                VALUES (%s, %s, %s, %s)
            """
            execute_query(sql, (exp_date, cat, amt, f"Monthly routine operational expense: {cat}"))
            
        # Next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)
            
    print("[Seed] Showroom pipeline and operational expenses seeded successfully.")


def run_full_seed():
    """Executes the complete end-to-end database initialization and data population."""
    print("=" * 70)
    print("AutoVision AI - Database Initialization & Seed Data Generation")
    print("=" * 70)
    init_database()
    seed_users()
    seed_employees()
    seed_vehicles()
    seed_customers(n=150)
    seed_sales_and_history()
    seed_inventory_and_pipeline()
    print("=" * 70)
    print("Database seeding completed successfully! All tables ready for production analytics.")
    print("=" * 70)


if __name__ == "__main__":
    run_full_seed()
