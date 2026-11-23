-- ==============================================================================
-- AutoVision AI - Car Showroom Database Schema
-- Compatible with MySQL 8.0+ and SQLite Fallback
-- ==============================================================================

-- 1. Users Table (Authentication & Access Control)
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(50) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    salt VARCHAR(64) NOT NULL,
    full_name VARCHAR(100) NOT NULL,
    email VARCHAR(100) NOT NULL UNIQUE,
    role VARCHAR(20) NOT NULL DEFAULT 'Manager',
    is_active TINYINT DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Employees Table (Sales Executives & Staff)
CREATE TABLE IF NOT EXISTS employees (
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(100) NOT NULL,
    role VARCHAR(50) NOT NULL,
    email VARCHAR(100) UNIQUE,
    phone VARCHAR(20),
    monthly_sales_target INT DEFAULT 5,
    commission_rate DECIMAL(4, 2) DEFAULT 1.50,
    hire_date DATE NOT NULL,
    is_active TINYINT DEFAULT 1
);

-- 3. Vehicles Master Table (Catalog of Models & Variants)
CREATE TABLE IF NOT EXISTS vehicles (
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    brand VARCHAR(50) NOT NULL,
    model VARCHAR(50) NOT NULL,
    variant VARCHAR(50) NOT NULL,
    body_type VARCHAR(30) NOT NULL,
    fuel_type VARCHAR(20) NOT NULL,
    transmission VARCHAR(20) NOT NULL,
    model_year INT NOT NULL,
    color VARCHAR(30) NOT NULL,
    cost_price DECIMAL(12, 2) NOT NULL,
    ex_showroom_price DECIMAL(12, 2) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 4. Inventory Table (Individual Physical Stock Units)
CREATE TABLE IF NOT EXISTS inventory (
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    vin VARCHAR(30) NOT NULL UNIQUE,
    vehicle_id INT NOT NULL,
    stock_status VARCHAR(20) NOT NULL DEFAULT 'Available',
    arrival_date DATE NOT NULL,
    showroom_location VARCHAR(50) DEFAULT 'Main Dealership Bay',
    days_in_stock INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (vehicle_id) REFERENCES vehicles(id) ON DELETE CASCADE
);

-- 5. Customers Table
CREATE TABLE IF NOT EXISTS customers (
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    full_name VARCHAR(100) NOT NULL,
    email VARCHAR(100),
    phone VARCHAR(20),
    city VARCHAR(50) NOT NULL,
    state VARCHAR(50) NOT NULL,
    annual_income DECIMAL(12, 2),
    budget_max DECIMAL(12, 2),
    preferred_brand VARCHAR(50),
    preferred_fuel VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 6. Enquiries Table
CREATE TABLE IF NOT EXISTS enquiries (
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    customer_id INT NOT NULL,
    vehicle_id INT NOT NULL,
    sales_rep_id INT NOT NULL,
    enquiry_date DATE NOT NULL,
    status VARCHAR(30) NOT NULL DEFAULT 'New',
    test_drive_conducted TINYINT DEFAULT 0,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(id) ON DELETE CASCADE,
    FOREIGN KEY (vehicle_id) REFERENCES vehicles(id) ON DELETE CASCADE,
    FOREIGN KEY (sales_rep_id) REFERENCES employees(id) ON DELETE CASCADE
);

-- 7. Bookings Table
CREATE TABLE IF NOT EXISTS bookings (
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    booking_reference VARCHAR(50) NOT NULL UNIQUE,
    customer_id INT NOT NULL,
    vehicle_id INT NOT NULL,
    sales_rep_id INT NOT NULL,
    booking_date DATE NOT NULL,
    booking_amount DECIMAL(10, 2) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'Confirmed',
    delivery_promised_date DATE,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(id) ON DELETE CASCADE,
    FOREIGN KEY (vehicle_id) REFERENCES vehicles(id) ON DELETE CASCADE,
    FOREIGN KEY (sales_rep_id) REFERENCES employees(id) ON DELETE CASCADE
);

-- 8. Sales Table (Completed Vehicle Transactions)
CREATE TABLE IF NOT EXISTS sales (
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    invoice_number VARCHAR(50) NOT NULL UNIQUE,
    booking_id INT,
    customer_id INT NOT NULL,
    vehicle_id INT NOT NULL,
    sales_rep_id INT NOT NULL,
    sale_date DATE NOT NULL,
    selling_price DECIMAL(12, 2) NOT NULL,
    cost_price DECIMAL(12, 2) NOT NULL,
    discount_amount DECIMAL(10, 2) DEFAULT 0.00,
    accessories_amount DECIMAL(10, 2) DEFAULT 0.00,
    insurance_amount DECIMAL(10, 2) DEFAULT 0.00,
    gross_profit DECIMAL(12, 2) NOT NULL,
    payment_method VARCHAR(30) DEFAULT 'Bank Loan',
    finance_partner VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(id) ON DELETE CASCADE,
    FOREIGN KEY (vehicle_id) REFERENCES vehicles(id) ON DELETE CASCADE,
    FOREIGN KEY (sales_rep_id) REFERENCES employees(id) ON DELETE CASCADE
);

-- 9. Expenses Table (Operational Dealership Costs for Net Profit)
CREATE TABLE IF NOT EXISTS expenses (
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    expense_date DATE NOT NULL,
    category VARCHAR(50) NOT NULL,
    amount DECIMAL(10, 2) NOT NULL,
    description VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
