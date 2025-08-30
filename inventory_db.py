import sqlite3
import json
from typing import List, Dict, Optional
from datetime import datetime
import os

class InventoryDatabase:
    def __init__(self, db_path: str = "inventory.db"):
        """Initialize the inventory database"""
        
        self.db_path = db_path
        self.conn = None
        self._initialize_database()
        self._load_sample_data()
    
    def _initialize_database(self):
        """Create database tables if they don't exist"""
        
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        cursor = self.conn.cursor()
        
        # Create products table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                part_number TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                subcategory TEXT,
                expected_quantity INTEGER DEFAULT 0,
                current_quantity INTEGER DEFAULT 0,
                location TEXT,
                description TEXT,
                manufacturer TEXT DEFAULT 'Medtronic',
                unit_price REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create inventory_checks table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS inventory_checks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                check_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                before_image_path TEXT,
                after_image_path TEXT,
                differences_json TEXT,
                notes TEXT,
                checked_by TEXT
            )
        ''')
        
        # Create inventory_differences table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS inventory_differences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                check_id INTEGER,
                product_id INTEGER,
                difference_type TEXT,
                quantity_difference INTEGER,
                confidence REAL,
                FOREIGN KEY (check_id) REFERENCES inventory_checks (id),
                FOREIGN KEY (product_id) REFERENCES products (id)
            )
        ''')
        
        self.conn.commit()
    
    def _load_sample_data(self):
        """Load sample inventory data if database is empty"""
        
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM products")
        
        if cursor.fetchone()[0] == 0:
            sample_products = [
                # Screws
                {
                    'part_number': 'MED-SCR-001',
                    'name': 'Cortical Screw 3.5mm x 20mm',
                    'category': 'Screws',
                    'subcategory': 'Cortical',
                    'expected_quantity': 20,
                    'current_quantity': 20,
                    'location': 'Tray A - Row 1',
                    'description': 'Self-tapping cortical bone screw',
                    'unit_price': 125.00
                },
                {
                    'part_number': 'MED-SCR-002',
                    'name': 'Cancellous Screw 6.5mm x 30mm',
                    'category': 'Screws',
                    'subcategory': 'Cancellous',
                    'expected_quantity': 15,
                    'current_quantity': 15,
                    'location': 'Tray A - Row 2',
                    'description': 'Partially threaded cancellous bone screw',
                    'unit_price': 145.00
                },
                {
                    'part_number': 'MED-SCR-003',
                    'name': 'Locking Screw 5.0mm x 25mm',
                    'category': 'Screws',
                    'subcategory': 'Locking',
                    'expected_quantity': 12,
                    'current_quantity': 12,
                    'location': 'Tray A - Row 3',
                    'description': 'Locking head screw for plates',
                    'unit_price': 165.00
                },
                
                # Instruments
                {
                    'part_number': 'MED-INS-001',
                    'name': 'Depth Gauge 0-110mm',
                    'category': 'Instruments',
                    'subcategory': 'Measuring',
                    'expected_quantity': 2,
                    'current_quantity': 2,
                    'location': 'Tray B - Slot 1',
                    'description': 'Precision depth measurement gauge',
                    'unit_price': 450.00
                },
                {
                    'part_number': 'MED-INS-002',
                    'name': 'Drill Guide 3.5mm',
                    'category': 'Instruments',
                    'subcategory': 'Guides',
                    'expected_quantity': 3,
                    'current_quantity': 3,
                    'location': 'Tray B - Slot 2',
                    'description': 'Drill guide for precise screw placement',
                    'unit_price': 325.00
                },
                {
                    'part_number': 'MED-INS-003',
                    'name': 'Hex Screwdriver 2.5mm',
                    'category': 'Instruments',
                    'subcategory': 'Drivers',
                    'expected_quantity': 2,
                    'current_quantity': 2,
                    'location': 'Tray B - Slot 3',
                    'description': 'Hexagonal screwdriver with quick-connect',
                    'unit_price': 275.00
                },
                
                # Drill Bits
                {
                    'part_number': 'MED-DRL-001',
                    'name': 'Drill Bit 2.5mm x 150mm',
                    'category': 'Drill Bits',
                    'subcategory': 'Standard',
                    'expected_quantity': 5,
                    'current_quantity': 5,
                    'location': 'Tray C - Row 1',
                    'description': 'Quick-connect drill bit',
                    'unit_price': 185.00
                },
                {
                    'part_number': 'MED-DRL-002',
                    'name': 'Drill Bit 3.5mm x 150mm',
                    'category': 'Drill Bits',
                    'subcategory': 'Standard',
                    'expected_quantity': 5,
                    'current_quantity': 5,
                    'location': 'Tray C - Row 2',
                    'description': 'Quick-connect drill bit',
                    'unit_price': 185.00
                },
                
                # Plates
                {
                    'part_number': 'MED-PLT-001',
                    'name': 'LC-DCP Plate 4.5mm 8-hole',
                    'category': 'Plates',
                    'subcategory': 'Compression',
                    'expected_quantity': 2,
                    'current_quantity': 2,
                    'location': 'Tray D - Top',
                    'description': 'Limited contact dynamic compression plate',
                    'unit_price': 850.00
                },
                {
                    'part_number': 'MED-PLT-002',
                    'name': 'Reconstruction Plate 3.5mm 12-hole',
                    'category': 'Plates',
                    'subcategory': 'Reconstruction',
                    'expected_quantity': 2,
                    'current_quantity': 2,
                    'location': 'Tray D - Bottom',
                    'description': 'Malleable reconstruction plate',
                    'unit_price': 925.00
                }
            ]
            
            for product in sample_products:
                self.add_product(product)
    
    def add_product(self, product: Dict) -> int:
        """Add a new product to the database"""
        
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO products (
                part_number, name, category, subcategory,
                expected_quantity, current_quantity, location,
                description, manufacturer, unit_price
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            product['part_number'],
            product['name'],
            product['category'],
            product.get('subcategory'),
            product.get('expected_quantity', 0),
            product.get('current_quantity', 0),
            product.get('location'),
            product.get('description'),
            product.get('manufacturer', 'Medtronic'),
            product.get('unit_price')
        ))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def get_product(self, part_number: str) -> Optional[Dict]:
        """Get a product by part number"""
        
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM products WHERE part_number = ?", (part_number,))
        row = cursor.fetchone()
        
        if row:
            return dict(row)
        return None
    
    def get_all_items(self) -> List[Dict]:
        """Get all inventory items"""
        
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM products ORDER BY category, name")
        result = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return result
    
    def update_quantity(self, part_number: str, new_quantity: int):
        """Update the current quantity of a product"""
        
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE products 
            SET current_quantity = ?, updated_at = CURRENT_TIMESTAMP
            WHERE part_number = ?
        ''', (new_quantity, part_number))
        
        self.conn.commit()
    
    def record_inventory_check(self, check_data: Dict) -> int:
        """Record an inventory check"""
        
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO inventory_checks (
                before_image_path, after_image_path,
                differences_json, notes, checked_by
            ) VALUES (?, ?, ?, ?, ?)
        ''', (
            check_data.get('before_image_path'),
            check_data.get('after_image_path'),
            json.dumps(check_data.get('differences', [])),
            check_data.get('notes'),
            check_data.get('checked_by', 'AI System')
        ))
        
        check_id = cursor.lastrowid
        
        # Record individual differences
        for diff in check_data.get('differences', []):
            cursor.execute('''
                INSERT INTO inventory_differences (
                    check_id, product_id, difference_type,
                    quantity_difference, confidence
                ) VALUES (?, ?, ?, ?, ?)
            ''', (
                check_id,
                diff.get('product_id'),
                diff.get('type'),
                diff.get('quantity_difference'),
                diff.get('confidence')
            ))
        
        self.conn.commit()
        return check_id
    
    def get_inventory_summary(self) -> Dict:
        """Get summary statistics of inventory"""
        
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Total items
        cursor.execute("SELECT COUNT(*) FROM products")
        total_items = cursor.fetchone()[0]
        
        # Items by category
        cursor.execute('''
            SELECT category, COUNT(*) as count 
            FROM products 
            GROUP BY category
        ''')
        categories = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Low stock items (current < expected)
        cursor.execute('''
            SELECT COUNT(*) 
            FROM products 
            WHERE current_quantity < expected_quantity
        ''')
        low_stock_count = cursor.fetchone()[0]
        
        # Total value
        cursor.execute('''
            SELECT SUM(current_quantity * unit_price) 
            FROM products 
            WHERE unit_price IS NOT NULL
        ''')
        total_value = cursor.fetchone()[0] or 0
        
        result = {
            'total_items': total_items,
            'categories': categories,
            'low_stock_count': low_stock_count,
            'total_value': total_value
        }
        conn.close()
        return result
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()