"""
20250101000000_init.py
Initial database schema creation.

This creates all the tables and constraints for the application.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.sql import expression


# revision identifiers, used by Alembic
revision = '20250101000000'
down_revision = None  # This is the first migration
branch_labels = None
depends_on = None


def upgrade():
    """Create initial database schema."""
    
    # ========== Users Table ==========
    op.create_table(
        'users',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('uuid', postgresql.UUID(as_uuid=True), 
                  server_default=sa.text('gen_random_uuid()'), 
                  nullable=False, unique=True),
        sa.Column('email', sa.String(255), nullable=False, unique=True, index=True),
        sa.Column('username', sa.String(50), nullable=False, unique=True, index=True),
        sa.Column('full_name', sa.String(100), nullable=True),
        sa.Column('hashed_password', sa.String(255), nullable=False),
        sa.Column('is_active', sa.Boolean, nullable=False, server_default=expression.true()),
        sa.Column('is_superuser', sa.Boolean, nullable=False, server_default=expression.false()),
        sa.Column('role', sa.String(20), nullable=False, server_default='user'),
        sa.Column('email_verified', sa.Boolean, nullable=False, server_default=expression.false()),
        sa.Column('last_login', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), 
                  nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(timezone=True), 
                  nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        
        # Composite index for common queries
        sa.Index('idx_users_active_email', 'is_active', 'email'),
        sa.Index('idx_users_role_active', 'role', 'is_active'),
    )
    
    # Add comment to table
    op.execute("COMMENT ON TABLE users IS 'System users and authentication information'")
    
    # ========== Profiles Table (One-to-one with Users) ==========
    op.create_table(
        'profiles',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('user_id', sa.Integer, 
                  sa.ForeignKey('users.id', ondelete='CASCADE'), 
                  nullable=False, unique=True),
        sa.Column('bio', sa.Text, nullable=True),
        sa.Column('avatar_url', sa.String(500), nullable=True),
        sa.Column('phone_number', sa.String(20), nullable=True),
        sa.Column('date_of_birth', sa.Date, nullable=True),
        sa.Column('country', sa.String(100), nullable=True),
        sa.Column('timezone', sa.String(50), nullable=True, server_default='UTC'),
        sa.Column('preferences', postgresql.JSONB, nullable=True, server_default='{}'),
        sa.Column('created_at', sa.DateTime(timezone=True), 
                  nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(timezone=True), 
                  nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
    )
    
    op.create_index('idx_profiles_user_id', 'profiles', ['user_id'], unique=True)
    
    # ========== Categories Table ==========
    op.create_table(
        'categories',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('slug', sa.String(100), nullable=False, unique=True),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('parent_id', sa.Integer, 
                  sa.ForeignKey('categories.id', ondelete='SET NULL'), 
                  nullable=True),
        sa.Column('sort_order', sa.Integer, nullable=False, server_default='0'),
        sa.Column('is_active', sa.Boolean, nullable=False, server_default=expression.true()),
        sa.Column('created_at', sa.DateTime(timezone=True), 
                  nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(timezone=True), 
                  nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
    )
    
    op.create_index('idx_categories_slug', 'categories', ['slug'], unique=True)
    op.create_index('idx_categories_parent', 'categories', ['parent_id'])
    op.create_index('idx_categories_active', 'categories', ['is_active'])
    
    # ========== Products Table ==========
    op.create_table(
        'products',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('sku', sa.String(50), nullable=False, unique=True),
        sa.Column('name', sa.String(200), nullable=False),
        sa.Column('slug', sa.String(200), nullable=False, unique=True),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('short_description', sa.String(500), nullable=True),
        sa.Column('price', sa.Numeric(10, 2), nullable=False),
        sa.Column('compare_at_price', sa.Numeric(10, 2), nullable=True),
        sa.Column('cost_price', sa.Numeric(10, 2), nullable=True),
        sa.Column('category_id', sa.Integer, 
                  sa.ForeignKey('categories.id', ondelete='SET NULL'), 
                  nullable=True),
        sa.Column('brand', sa.String(100), nullable=True),
        sa.Column('weight', sa.Numeric(8, 3), nullable=True),  # in kg
        sa.Column('dimensions', sa.String(100), nullable=True),  # "10x20x30"
        sa.Column('in_stock', sa.Boolean, nullable=False, server_default=expression.true()),
        sa.Column('stock_quantity', sa.Integer, nullable=False, server_default='0'),
        sa.Column('low_stock_threshold', sa.Integer, nullable=False, server_default='10'),
        sa.Column('is_featured', sa.Boolean, nullable=False, server_default=expression.false()),
        sa.Column('is_published', sa.Boolean, nullable=False, server_default=expression.false()),
        sa.Column('metadata', postgresql.JSONB, nullable=True, server_default='{}'),
        sa.Column('created_at', sa.DateTime(timezone=True), 
                  nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(timezone=True), 
                  nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('published_at', sa.DateTime(timezone=True), nullable=True),
    )
    
    op.create_index('idx_products_sku', 'products', ['sku'], unique=True)
    op.create_index('idx_products_slug', 'products', ['slug'], unique=True)
    op.create_index('idx_products_category', 'products', ['category_id'])
    op.create_index('idx_products_price', 'products', ['price'])
    op.create_index('idx_products_stock', 'products', ['in_stock', 'stock_quantity'])
    op.create_index('idx_products_published', 'products', ['is_published', 'published_at'])
    op.create_index('idx_products_featured', 'products', ['is_featured', 'is_published'])
    
    # ========== Orders Table ==========
    op.create_table(
        'orders',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('order_number', sa.String(50), nullable=False, unique=True),
        sa.Column('user_id', sa.Integer, 
                  sa.ForeignKey('users.id', ondelete='SET NULL'), 
                  nullable=True),
        sa.Column('status', sa.String(20), nullable=False, server_default='pending'),
        sa.Column('total_amount', sa.Numeric(10, 2), nullable=False),
        sa.Column('subtotal', sa.Numeric(10, 2), nullable=False),
        sa.Column('tax_amount', sa.Numeric(10, 2), nullable=False, server_default='0'),
        sa.Column('shipping_amount', sa.Numeric(10, 2), nullable=False, server_default='0'),
        sa.Column('discount_amount', sa.Numeric(10, 2), nullable=False, server_default='0'),
        sa.Column('currency', sa.String(3), nullable=False, server_default='USD'),
        sa.Column('customer_email', sa.String(255), nullable=False),
        sa.Column('customer_name', sa.String(100), nullable=False),
        sa.Column('shipping_address', postgresql.JSONB, nullable=True),
        sa.Column('billing_address', postgresql.JSONB, nullable=True),
        sa.Column('notes', sa.Text, nullable=True),
        sa.Column('metadata', postgresql.JSONB, nullable=True, server_default='{}'),
        sa.Column('created_at', sa.DateTime(timezone=True), 
                  nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(timezone=True), 
                  nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('paid_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('shipped_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('delivered_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('cancelled_at', sa.DateTime(timezone=True), nullable=True),
    )
    
    op.create_index('idx_orders_order_number', 'orders', ['order_number'], unique=True)
    op.create_index('idx_orders_user_id', 'orders', ['user_id'])
    op.create_index('idx_orders_status', 'orders', ['status'])
    op.create_index('idx_orders_created', 'orders', ['created_at'])
    op.create_index('idx_orders_customer_email', 'orders', ['customer_email'])
    
    # ========== Order Items Table ==========
    op.create_table(
        'order_items',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('order_id', sa.Integer, 
                  sa.ForeignKey('orders.id', ondelete='CASCADE'), 
                  nullable=False),
        sa.Column('product_id', sa.Integer, 
                  sa.ForeignKey('products.id', ondelete='RESTRICT'), 
                  nullable=True),
        sa.Column('sku', sa.String(50), nullable=False),
        sa.Column('name', sa.String(200), nullable=False),
        sa.Column('quantity', sa.Integer, nullable=False),
        sa.Column('unit_price', sa.Numeric(10, 2), nullable=False),
        sa.Column('total_price', sa.Numeric(10, 2), nullable=False),
        sa.Column('tax_amount', sa.Numeric(10, 2), nullable=False, server_default='0'),
        sa.Column('metadata', postgresql.JSONB, nullable=True, server_default='{}'),
    )
    
    op.create_index('idx_order_items_order', 'order_items', ['order_id'])
    op.create_index('idx_order_items_product', 'order_items', ['product_id'])
    op.create_index('idx_order_items_order_product', 'order_items', ['order_id', 'product_id'])
    
    # ========== Countries Table (Reference Data) ==========
    op.create_table(
        'countries',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('name', sa.String(100), nullable=False, unique=True),
        sa.Column('official_name', sa.String(200), nullable=True),
        sa.Column('code', sa.String(2), nullable=False, unique=True),
        sa.Column('code3', sa.String(3), nullable=False, unique=True),
        sa.Column('continent', sa.String(50), nullable=False),
        sa.Column('region', sa.String(50), nullable=True),
        sa.Column('phone_code', sa.String(10), nullable=True),
        sa.Column('currency_code', sa.String(3), nullable=True),
        sa.Column('currency_name', sa.String(50), nullable=True),
        sa.Column('is_active', sa.Boolean, nullable=False, server_default=expression.true()),
    )
    
    op.create_index('idx_countries_code', 'countries', ['code'], unique=True)
    op.create_index('idx_countries_continent', 'countries', ['continent'])
    
    # ========== Settings Table (Key-Value Store) ==========
    op.create_table(
        'settings',
        sa.Column('id', sa.String(100), primary_key=True),
        sa.Column('value', sa.Text, nullable=False),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('data_type', sa.String(20), nullable=False, server_default='string'),
        sa.Column('is_public', sa.Boolean, nullable=False, server_default=expression.false()),
        sa.Column('category', sa.String(50), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), 
                  nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_by', sa.Integer, 
                  sa.ForeignKey('users.id', ondelete='SET NULL'), 
                  nullable=True),
    )
    
    op.create_index('idx_settings_category', 'settings', ['category'])
    op.create_index('idx_settings_public', 'settings', ['is_public'])
    
    # ========== Audit Log Table ==========
    op.create_table(
        'audit_log',
        sa.Column('id', sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column('action', sa.String(100), nullable=False),
        sa.Column('resource_type', sa.String(50), nullable=True),
        sa.Column('resource_id', sa.String(100), nullable=True),
        sa.Column('user_id', sa.Integer, 
                  sa.ForeignKey('users.id', ondelete='SET NULL'), 
                  nullable=True),
        sa.Column('user_ip', sa.String(45), nullable=True),
        sa.Column('user_agent', sa.String(500), nullable=True),
        sa.Column('details', postgresql.JSONB, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), 
                  nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
    )
    
    op.create_index('idx_audit_action', 'audit_log', ['action'])
    op.create_index('idx_audit_user', 'audit_log', ['user_id'])
    op.create_index('idx_audit_resource', 'audit_log', ['resource_type', 'resource_id'])
    op.create_index('idx_audit_created', 'audit_log', ['created_at'])
    
    # ========== Refresh Tokens Table ==========
    op.create_table(
        'refresh_tokens',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('token', sa.String(500), nullable=False, unique=True),
        sa.Column('user_id', sa.Integer, 
                  sa.ForeignKey('users.id', ondelete='CASCADE'), 
                  nullable=False),
        sa.Column('device_id', sa.String(200), nullable=True),
        sa.Column('user_agent', sa.String(500), nullable=True),
        sa.Column('ip_address', sa.String(45), nullable=True),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('revoked_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), 
                  nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
    )
    
    op.create_index('idx_refresh_tokens_token', 'refresh_tokens', ['token'], unique=True)
    op.create_index('idx_refresh_tokens_user', 'refresh_tokens', ['user_id'])
    op.create_index('idx_refresh_tokens_expires', 'refresh_tokens', ['expires_at'])
    
    # ========== Create Functions and Triggers (PostgreSQL specific) ==========
    if op.get_context().dialect.name == 'postgresql':
        
        # Function to update updated_at timestamp
        op.execute("""
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $$ language 'plpgsql';
        """)
        
        # Create triggers for tables with updated_at columns
        tables_with_updated_at = ['users', 'profiles', 'categories', 'products']
        
        for table_name in tables_with_updated_at:
            op.execute(f"""
            CREATE TRIGGER update_{table_name}_updated_at
            BEFORE UPDATE ON {table_name}
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
            """)
        
        # Function to generate order number
        op.execute("""
        CREATE OR REPLACE FUNCTION generate_order_number()
        RETURNS TRIGGER AS $$
        DECLARE
            year_part TEXT;
            sequence_num INTEGER;
        BEGIN
            year_part := TO_CHAR(NEW.created_at, 'YYMM');
            
            -- Get next sequence value for this month
            SELECT COALESCE(MAX(SUBSTRING(order_number FROM 7)::INTEGER), 0) + 1
            INTO sequence_num
            FROM orders
            WHERE order_number LIKE 'ORD-' || year_part || '%';
            
            NEW.order_number := 'ORD-' || year_part || LPAD(sequence_num::TEXT, 6, '0');
            RETURN NEW;
        END;
        $$ language 'plpgsql';
        """)
        
        op.execute("""
        CREATE TRIGGER set_order_number
        BEFORE INSERT ON orders
        FOR EACH ROW
        WHEN (NEW.order_number IS NULL)
        EXECUTE FUNCTION generate_order_number();
        """)
        
        # Create full-text search index for products
        op.execute("""
        ALTER TABLE products 
        ADD COLUMN IF NOT EXISTS search_vector tsvector 
        GENERATED ALWAYS AS (
            setweight(to_tsvector('english', COALESCE(name, '')), 'A') ||
            setweight(to_tsvector('english', COALESCE(description, '')), 'B') ||
            setweight(to_tsvector('english', COALESCE(sku, '')), 'C')
        ) STORED;
        """)
        
        op.execute("""
        CREATE INDEX idx_products_search ON products USING GIN(search_vector);
        """)
    
    print("✓ Initial schema created successfully")


def downgrade():
    """Drop all tables and clean up."""
    
    # Drop triggers and functions first (PostgreSQL)
    if op.get_context().dialect.name == 'postgresql':
        tables_with_updated_at = ['users', 'profiles', 'categories', 'products']
        
        for table_name in tables_with_updated_at:
            op.execute(f"DROP TRIGGER IF EXISTS update_{table_name}_updated_at ON {table_name}")
        
        op.execute("DROP TRIGGER IF EXISTS set_order_number ON orders")
        op.execute("DROP FUNCTION IF EXISTS update_updated_at_column()")
        op.execute("DROP FUNCTION IF EXISTS generate_order_number()")
    
    # Drop tables in reverse order (respecting foreign key constraints)
    op.drop_table('refresh_tokens')
    op.drop_table('audit_log')
    op.drop_table('settings')
    op.drop_table('order_items')
    op.drop_table('orders')
    op.drop_table('countries')
    op.drop_table('products')
    op.drop_table('categories')
    op.drop_table('profiles')
    op.drop_table('users')
    
    print("✓ All tables dropped successfully")