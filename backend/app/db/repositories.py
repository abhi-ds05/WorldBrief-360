"""
Repository pattern implementation for data access.
"""

from typing import Type, TypeVar, Generic, Optional, List, Dict, Any
from uuid import UUID
from sqlalchemy import select, update, delete, func, and_, or_
from sqlalchemy.orm import Session, joinedload, selectinload
from sqlalchemy.ext.asyncio import AsyncSession

from .base import Base, TimestampMixin, SoftDeleteMixin

T = TypeVar('T', bound=Base)


class BaseRepository(Generic[T]):
    """
    Base repository class for CRUD operations.
    """
    
    def __init__(self, model_class: Type[T], session: Session):
        """
        Initialize repository.
        
        Args:
            model_class: SQLAlchemy model class
            session: Database session
        """
        self.model_class = model_class
        self.session = session
    
    def get(self, id: UUID, **kwargs) -> Optional[T]:
        """
        Get a record by ID.
        
        Args:
            id: Record ID
            **kwargs: Additional filter criteria
            
        Returns:
            Model instance or None
        """
        query = select(self.model_class).where(self.model_class.id == id)
        
        # Apply additional filters
        for key, value in kwargs.items():
            if hasattr(self.model_class, key):
                query = query.where(getattr(self.model_class, key) == value)
        
        return self.session.execute(query).scalar_one_or_none()
    
    def get_by(self, **kwargs) -> Optional[T]:
        """
        Get a record by multiple criteria.
        
        Args:
            **kwargs: Filter criteria
            
        Returns:
            Model instance or None
        """
        query = select(self.model_class)
        
        for key, value in kwargs.items():
            if hasattr(self.model_class, key):
                if isinstance(value, (list, tuple)):
                    query = query.where(getattr(self.model_class, key).in_(value))
                else:
                    query = query.where(getattr(self.model_class, key) == value)
        
        return self.session.execute(query).scalar_one_or_none()
    
    def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
        order_by: Optional[List] = None,
        **filters
    ) -> List[T]:
        """
        Get all records with optional filtering and pagination.
        
        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            order_by: List of columns to order by
            **filters: Filter criteria
            
        Returns:
            List of model instances
        """
        query = select(self.model_class)
        
        # Apply filters
        for key, value in filters.items():
            if hasattr(self.model_class, key) and value is not None:
                column = getattr(self.model_class, key)
                
                if isinstance(value, (list, tuple)):
                    query = query.where(column.in_(value))
                elif isinstance(value, dict):
                    # Handle comparison operators
                    for op, op_value in value.items():
                        if op == 'gt':
                            query = query.where(column > op_value)
                        elif op == 'gte':
                            query = query.where(column >= op_value)
                        elif op == 'lt':
                            query = query.where(column < op_value)
                        elif op == 'lte':
                            query = query.where(column <= op_value)
                        elif op == 'like':
                            query = query.where(column.like(f'%{op_value}%'))
                        elif op == 'ilike':
                            query = query.where(column.ilike(f'%{op_value}%'))
                        elif op == 'neq':
                            query = query.where(column != op_value)
                elif hasattr(value, '__iter__') and not isinstance(value, str):
                    # Handle arrays
                    query = query.where(column.any(value))
                else:
                    query = query.where(column == value)
        
        # Apply ordering
        if order_by:
            order_clauses = []
            for order in order_by:
                if order.startswith('-'):
                    order_clauses.append(getattr(self.model_class, order[1:]).desc())
                else:
                    order_clauses.append(getattr(self.model_class, order).asc())
            query = query.order_by(*order_clauses)
        
        # Apply pagination
        query = query.offset(skip).limit(limit)
        
        result = self.session.execute(query)
        return result.scalars().all()
    
    def create(self, **kwargs) -> T:
        """
        Create a new record.
        
        Args:
            **kwargs: Model attributes
            
        Returns:
            Created model instance
        """
        instance = self.model_class(**kwargs)
        self.session.add(instance)
        self.session.flush()
        return instance
    
    def create_many(self, items: List[Dict[str, Any]]) -> List[T]:
        """
        Create multiple records.
        
        Args:
            items: List of dictionaries with model attributes
            
        Returns:
            List of created model instances
        """
        instances = [self.model_class(**item) for item in items]
        self.session.add_all(instances)
        self.session.flush()
        return instances
    
    def update(self, id: UUID, **kwargs) -> Optional[T]:
        """
        Update a record.
        
        Args:
            id: Record ID
            **kwargs: Attributes to update
            
        Returns:
            Updated model instance or None if not found
        """
        instance = self.get(id)
        if instance:
            for key, value in kwargs.items():
                if hasattr(instance, key):
                    setattr(instance, key, value)
            self.session.flush()
        return instance
    
    def update_by(self, filters: Dict[str, Any], values: Dict[str, Any]) -> int:
        """
        Update multiple records matching filters.
        
        Args:
            filters: Filter criteria
            values: Values to update
            
        Returns:
            Number of records updated
        """
        stmt = update(self.model_class).where(
            self._build_filter_conditions(filters)
        ).values(**values).execution_options(synchronize_session="fetch")
        
        result = self.session.execute(stmt)
        self.session.flush()
        return result.rowcount
    
    def delete(self, id: UUID, soft: bool = True) -> bool:
        """
        Delete a record.
        
        Args:
            id: Record ID
            soft: If True, use soft delete if available
            
        Returns:
            True if deleted, False if not found
        """
        instance = self.get(id)
        if not instance:
            return False
        
        if soft and hasattr(instance, 'is_deleted'):
            # Soft delete
            instance.is_deleted = True
            if hasattr(instance, 'deleted_at'):
                from datetime import datetime
                instance.deleted_at = datetime.utcnow()
        else:
            # Hard delete
            self.session.delete(instance)
        
        self.session.flush()
        return True
    
    def delete_by(self, **kwargs) -> int:
        """
        Delete multiple records matching criteria.
        
        Args:
            **kwargs: Filter criteria
            
        Returns:
            Number of records deleted
        """
        stmt = delete(self.model_class).where(
            self._build_filter_conditions(kwargs)
        )
        
        result = self.session.execute(stmt)
        self.session.flush()
        return result.rowcount
    
    def count(self, **filters) -> int:
        """
        Count records matching criteria.
        
        Args:
            **filters: Filter criteria
            
        Returns:
            Count of records
        """
        query = select(func.count()).select_from(self.model_class)
        
        if filters:
            conditions = self._build_filter_conditions(filters)
            query = query.where(conditions)
        
        result = self.session.execute(query)
        return result.scalar()
    
    def exists(self, **kwargs) -> bool:
        """
        Check if a record exists matching criteria.
        
        Args:
            **kwargs: Filter criteria
            
        Returns:
            True if exists, False otherwise
        """
        return self.count(**kwargs) > 0
    
    def _build_filter_conditions(self, filters: Dict[str, Any]):
        """
        Build SQLAlchemy filter conditions from dictionary.
        
        Args:
            filters: Filter dictionary
            
        Returns:
            SQLAlchemy filter conditions
        """
        conditions = []
        
        for key, value in filters.items():
            if hasattr(self.model_class, key) and value is not None:
                column = getattr(self.model_class, key)
                
                if isinstance(value, (list, tuple)):
                    conditions.append(column.in_(value))
                elif isinstance(value, dict):
                    # Handle comparison operators
                    for op, op_value in value.items():
                        if op == 'eq':
                            conditions.append(column == op_value)
                        elif op == 'neq':
                            conditions.append(column != op_value)
                        elif op == 'gt':
                            conditions.append(column > op_value)
                        elif op == 'gte':
                            conditions.append(column >= op_value)
                        elif op == 'lt':
                            conditions.append(column < op_value)
                        elif op == 'lte':
                            conditions.append(column <= op_value)
                        elif op == 'like':
                            conditions.append(column.like(f'%{op_value}%'))
                        elif op == 'ilike':
                            conditions.append(column.ilike(f'%{op_value}%'))
                        elif op == 'is_null':
                            conditions.append(column.is_(None))
                        elif op == 'is_not_null':
                            conditions.append(column.is_not(None))
                        elif op == 'between':
                            if isinstance(op_value, (list, tuple)) and len(op_value) == 2:
                                conditions.append(column.between(op_value[0], op_value[1]))
                else:
                    conditions.append(column == value)
        
        return and_(*conditions) if conditions else True


class AsyncBaseRepository(Generic[T]):
    """
    Async base repository class for CRUD operations.
    """
    
    def __init__(self, model_class: Type[T], session: AsyncSession):
        """
        Initialize async repository.
        
        Args:
            model_class: SQLAlchemy model class
            session: Async database session
        """
        self.model_class = model_class
        self.session = session
    
    async def get(self, id: UUID, **kwargs) -> Optional[T]:
        """Async get a record by ID."""
        query = select(self.model_class).where(self.model_class.id == id)
        
        for key, value in kwargs.items():
            if hasattr(self.model_class, key):
                query = query.where(getattr(self.model_class, key) == value)
        
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    async def get_by(self, **kwargs) -> Optional[T]:
        """Async get a record by multiple criteria."""
        query = select(self.model_class)
        
        for key, value in kwargs.items():
            if hasattr(self.model_class, key):
                if isinstance(value, (list, tuple)):
                    query = query.where(getattr(self.model_class, key).in_(value))
                else:
                    query = query.where(getattr(self.model_class, key) == value)
        
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    async def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
        order_by: Optional[List] = None,
        **filters
    ) -> List[T]:
        """Async get all records."""
        query = select(self.model_class)
        
        # Apply filters
        for key, value in filters.items():
            if hasattr(self.model_class, key) and value is not None:
                column = getattr(self.model_class, key)
                
                if isinstance(value, (list, tuple)):
                    query = query.where(column.in_(value))
                else:
                    query = query.where(column == value)
        
        # Apply ordering
        if order_by:
            order_clauses = []
            for order in order_by:
                if order.startswith('-'):
                    order_clauses.append(getattr(self.model_class, order[1:]).desc())
                else:
                    order_clauses.append(getattr(self.model_class, order).asc())
            query = query.order_by(*order_clauses)
        
        # Apply pagination
        query = query.offset(skip).limit(limit)
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def create(self, **kwargs) -> T:
        """Async create a new record."""
        instance = self.model_class(**kwargs)
        self.session.add(instance)
        await self.session.flush()
        return instance
    
    async def update(self, id: UUID, **kwargs) -> Optional[T]:
        """Async update a record."""
        instance = await self.get(id)
        if instance:
            for key, value in kwargs.items():
                if hasattr(instance, key):
                    setattr(instance, key, value)
            await self.session.flush()
        return instance
    
    async def delete(self, id: UUID, soft: bool = True) -> bool:
        """Async delete a record."""
        instance = await self.get(id)
        if not instance:
            return False
        
        if soft and hasattr(instance, 'is_deleted'):
            instance.is_deleted = True
            if hasattr(instance, 'deleted_at'):
                from datetime import datetime
                instance.deleted_at = datetime.utcnow()
        else:
            await self.session.delete(instance)
        
        await self.session.flush()
        return True
    
    async def count(self, **filters) -> int:
        """Async count records."""
        query = select(func.count()).select_from(self.model_class)
        
        if filters:
            conditions = []
            for key, value in filters.items():
                if hasattr(self.model_class, key):
                    column = getattr(self.model_class, key)
                    if isinstance(value, (list, tuple)):
                        conditions.append(column.in_(value))
                    else:
                        conditions.append(column == value)
            if conditions:
                query = query.where(and_(*conditions))
        
        result = await self.session.execute(query)
        return result.scalar()
    
    async def exists(self, **kwargs) -> bool:
        """Async check if record exists."""
        return await self.count(**kwargs) > 0


# Repository factory functions
def get_repository(model_class: Type[T], session: Session) -> BaseRepository[T]:
    """
    Get a repository for a model class.
    
    Args:
        model_class: SQLAlchemy model class
        session: Database session
        
    Returns:
        Repository instance
    """
    return BaseRepository(model_class, session)


async def get_async_repository(model_class: Type[T], session: AsyncSession) -> AsyncBaseRepository[T]:
    """
    Get an async repository for a model class.
    
    Args:
        model_class: SQLAlchemy model class
        session: Async database session
        
    Returns:
        Async repository instance
    """
    return AsyncBaseRepository(model_class, session)