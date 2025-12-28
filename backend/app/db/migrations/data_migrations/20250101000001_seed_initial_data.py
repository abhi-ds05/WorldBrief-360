"""
20250101000001_seed_initial_data.py
Seed initial data for the application.

This is a data migration that runs after the initial schema is created.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import table, column
from sqlalchemy.dialects import postgresql
from datetime import datetime, date
import uuid


# revision identifiers, used by Alembic
revision = '20250101000001'
down_revision = '20250101000000'  # Points to the initial schema migration
branch_labels = None
depends_on = None


def upgrade():
    """Insert initial data into database tables."""
    
    # Get connection and metadata
    connection = op.get_bind()
    
    # ========== Insert Initial Users ==========
    users_table = table(
        'users',
        column('id', sa.Integer),
        column('email', sa.String),
        column('username', sa.String),
        column('full_name', sa.String),
        column('is_active', sa.Boolean),
        column('role', sa.String),
        column('created_at', sa.DateTime),
    )
    
    op.bulk_insert(users_table, [
        {
            'id': 1,
            'email': 'admin@example.com',
            'username': 'admin',
            'full_name': 'System Administrator',
            'is_active': True,
            'role': 'admin',
            'created_at': datetime.utcnow(),
        },
        {
            'id': 2,
            'email': 'user@example.com',
            'username': 'johndoe',
            'full_name': 'John Doe',
            'is_active': True,
            'role': 'user',
            'created_at': datetime.utcnow(),
        },
        {
            'id': 3,
            'email': 'manager@example.com',
            'username': 'janedoe',
            'full_name': 'Jane Doe',
            'is_active': True,
            'role': 'manager',
            'created_at': datetime.utcnow(),
        }
    ])
    
    # ========== Insert Product Categories ==========
    categories_table = table(
        'categories',
        column('id', sa.Integer),
        column('name', sa.String),
        column('slug', sa.String),
        column('description', sa.Text),
    )
    
    op.bulk_insert(categories_table, [
        {
            'id': 1,
            'name': 'Electronics',
            'slug': 'electronics',
            'description': 'Electronic devices and gadgets',
        },
        {
            'id': 2,
            'name': 'Books',
            'slug': 'books',
            'description': 'Books and reading materials',
        },
        {
            'id': 3,
            'name': 'Clothing',
            'slug': 'clothing',
            'description': 'Apparel and accessories',
        }
    ])
    
    # ========== Insert Products ==========
    products_table = table(
        'products',
        column('id', sa.Integer),
        column('name', sa.String),
        column('description', sa.Text),
        column('price', sa.Numeric(10, 2)),
        column('category_id', sa.Integer),
        column('in_stock', sa.Boolean),
        column('stock_quantity', sa.Integer),
        column('created_at', sa.DateTime),
    )
    
    op.bulk_insert(products_table, [
        {
            'id': 1,
            'name': 'Smartphone X',
            'description': 'Latest smartphone with advanced features',
            'price': 999.99,
            'category_id': 1,
            'in_stock': True,
            'stock_quantity': 100,
            'created_at': datetime.utcnow(),
        },
        {
            'id': 2,
            'name': 'Laptop Pro',
            'description': 'High-performance laptop for professionals',
            'price': 1499.99,
            'category_id': 1,
            'in_stock': True,
            'stock_quantity': 50,
            'created_at': datetime.utcnow(),
        },
        {
            'id': 3,
            'name': 'Python Programming Book',
            'description': 'Comprehensive guide to Python programming',
            'price': 49.99,
            'category_id': 2,
            'in_stock': True,
            'stock_quantity': 200,
            'created_at': datetime.utcnow(),
        }
    ])
    
    # ========== Insert Configuration Settings ==========
    settings_table = table(
        'settings',
        column('id', sa.String),
        column('value', sa.String),
        column('description', sa.Text),
        column('updated_at', sa.DateTime),
    )
    
    op.bulk_insert(settings_table, [
        {
            'id': 'site_name',
            'value': 'My Application',
            'description': 'Name of the application/site',
            'updated_at': datetime.utcnow(),
        },
        {
            'id': 'maintenance_mode',
            'value': 'false',
            'description': 'Whether the site is in maintenance mode',
            'updated_at': datetime.utcnow(),
        },
        {
            'id': 'default_currency',
            'value': 'USD',
            'description': 'Default currency for the application',
            'updated_at': datetime.utcnow(),
        }
    ])
    
    # ========== Insert Countries (Example of larger dataset) ==========
    # This shows how to handle larger datasets efficiently
    countries_table = table(
        'countries',
        column('id', sa.Integer),
        column('name', sa.String),
        column('code', sa.String(2)),
        column('continent', sa.String),
    )
    
    # Sample countries data
    countries_data = [
        {'id': 1, 'name': 'United States', 'code': 'US', 'continent': 'North America'},
        {'id': 2, 'name': 'Canada', 'code': 'CA', 'continent': 'North America'},
        {'id': 3, 'name': 'United Kingdom', 'code': 'GB', 'continent': 'Europe'},
        {'id': 4, 'name': 'Germany', 'code': 'DE', 'continent': 'Europe'},
        {'id': 5, 'name': 'Japan', 'code': 'JP', 'continent': 'Asia'},
    ]
    
    # Insert in batches if you have a large dataset
    op.bulk_insert(countries_table, countries_data)
    
    # ========== Using raw SQL for complex inserts ==========
    # Sometimes you need raw SQL for complex operations
    connection.execute(sa.text("""
        INSERT INTO audit_log (action, user_id, details, created_at)
        VALUES ('initial_seed', 1, 'Initial data migration completed', :now)
    """), {'now': datetime.utcnow()})
    
    # ========== Update sequences after manual ID inserts ==========
    # Important for PostgreSQL after manually setting IDs
    if connection.dialect.name == 'postgresql':
        connection.execute(sa.text("SELECT setval('users_id_seq', (SELECT MAX(id) FROM users))"))
        connection.execute(sa.text("SELECT setval('products_id_seq', (SELECT MAX(id) FROM products))"))
        connection.execute(sa.text("SELECT setval('categories_id_seq', (SELECT MAX(id) FROM categories))"))
        connection.execute(sa.text("SELECT setval('countries_id_seq', (SELECT MAX(id) FROM countries))"))
    
    print("✓ Initial data seeded successfully")


def downgrade():
    """Remove all seeded data (be careful with this in production!)."""
    
    # Get connection
    connection = op.get_bind()
    
    # IMPORTANT: Consider if you want to delete all data or just seeded data
    # This deletes ALL data from these tables - use with caution!
    
    # Delete in reverse order (respecting foreign key constraints)
    connection.execute(sa.text("DELETE FROM audit_log WHERE action = 'initial_seed'"))
    connection.execute(sa.text("DELETE FROM countries"))
    connection.execute(sa.text("DELETE FROM settings"))
    connection.execute(sa.text("DELETE FROM products"))
    connection.execute(sa.text("DELETE FROM categories"))
    connection.execute(sa.text("DELETE FROM users"))
    
    # Reset sequences for PostgreSQL
    if connection.dialect.name == 'postgresql':
        connection.execute(sa.text("ALTER SEQUENCE users_id_seq RESTART WITH 1"))
        connection.execute(sa.text("ALTER SEQUENCE products_id_seq RESTART WITH 1"))
        connection.execute(sa.text("ALTER SEQUENCE categories_id_seq RESTART WITH 1"))
        connection.execute(sa.text("ALTER SEQUENCE countries_id_seq RESTART WITH 1"))
    
    print("✓ Seeded data removed")


# Alternative: Data-safe downgrade (preserves user-added data)
def downgrade_safe():
    """
    Alternative downgrade that only removes the specific seeded data.
    This is safer but more complex to implement.
    """
    # You would need to track which records were inserted in the upgrade
    # and only delete those specific records
    pass