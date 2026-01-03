/ *--Copyright 2025 β ORI Inc.Canada All Rights Reserved.
 * Author: Awase Khirni Syed
* Hexagonal Flask Architecture -Artefacts generation from database. 
*/

import os
import re
import subprocess
from typing import Dict, List, Any, Optional, Union
from sqlalchemy.ext.automap import automap_base
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime

# Database configuration
DATABASE_URL = "postgresql+psycopg2://awasekhirnisyed:YOUR_PASSWORD_HERE@localhost:5432/profxdb"
PROJECT_DIR = "project"

# Folder structure with files
FOLDER_STRUCTURE = {
    "api/dtos": ["__init__.py"],
    "api/controllers": ["__init__.py"],
    "api/validations": ["__init__.py"],
    "api/mappers": ["__init__.py"],
    "domain": ["__init__.py"],
    "infrastructure/repositories": ["__init__.py"],
    "infrastructure/database": ["__init__.py"],
    "application/services": ["__init__.py"],
    "shared/utils": ["__init__.py"],
    "tests": ["__init__.py"],
}

def ensure_init_files(folder_path: str):
    """Ensure __init__.py exists in each directory level"""
    parts = folder_path.split(os.sep)
    current_path = PROJECT_DIR
    for part in parts:
        current_path = os.path.join(current_path, part)
        os.makedirs(current_path, exist_ok=True)
        init_file = os.path.join(current_path, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, "w") as f:
                f.write(f"# {part} package\n")

def create_folders_and_files():
    """Create the project folder structure and files"""
    for folder, files in FOLDER_STRUCTURE.items():
        folder_path = os.path.join(PROJECT_DIR, folder)
        os.makedirs(folder_path, exist_ok=True)
        ensure_init_files(folder)

        for file in files:
            file_path = os.path.join(folder_path, file)
            if not os.path.exists(file_path):
                with open(file_path, "w") as f:
                    if file.endswith(".py"):
                        f.write(f"# {file}\n")

def generate_sqlalchemy_models():
    """Generate SQLAlchemy models using sqlacodegen"""
    try:
        result = subprocess.run(
            ["sqlacodegen", DATABASE_URL, "--outfile", "models.py"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"Error generating models: {result.stderr}")
        else:
            print("Models generated successfully")
    except FileNotFoundError:
        print("sqlacodegen not found. Install with: pip install sqlacodegen")

def analyze_table(table_class) -> Dict[str, Any]:
    """Analyze table structure to determine special features"""
    analysis = {
        'has_timestamps': False,
        'has_status': False,
        'has_soft_delete': False,
        'has_user_relation': False,
        'has_enum_columns': [],
        'has_many_relations': [],
        'has_one_relations': []
    }

    for column in table_class.__table__.columns:
        col_name = column.name.lower()
        col_type = str(column.type).lower()

        # Check for timestamp columns
        if col_name in ('created_at', 'updated_at', 'modified_at'):
            analysis['has_timestamps'] = True

        # Check for status columns
        if col_name.endswith('_status'):
            analysis['has_status'] = True
            if col_type.startswith('varchar'):
                analysis['has_enum_columns'].append(column.name)

        # Check for soft delete
        if col_name in ('is_deleted', 'deleted_at', 'deleted'):
            analysis['has_soft_delete'] = True

        # Check for user relations
        if col_name.endswith('_user_id') or col_name == 'user_id':
            analysis['has_user_relation'] = True

        # Check for enum-like columns
        if col_name.endswith(('_type', '_role', '_kind')) and col_type.startswith('varchar'):
            analysis['has_enum_columns'].append(column.name)

    # Check relationships
    for rel in table_class.__mapper__.relationships:
        if rel.direction.name == 'MANYTOONE':
            analysis['has_one_relations'].append(rel.key)
        elif rel.direction.name == 'ONETOMANY':
            analysis['has_many_relations'].append(rel.key)

    return analysis

def generate_entity_file(table_name: str, class_name: str, table_class, analysis: Dict[str, Any]) -> str:
    """Generate entity file for a table"""
    imports = [
        "from enum import Enum, auto",
        "from uuid import uuid4",
        "from datetime import datetime",
        "from typing import Dict, Any, Optional, List",
        "from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, Enum as SqlEnum",
        "from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY",
        "from sqlalchemy.orm import relationship",
        "from infrastructure.database.db import Base",
        ""
    ]

    # Generate enum classes
    enum_classes = []
    for col_name in analysis['has_enum_columns']:
        enum_name = f"{col_name.split('_')[-1].capitalize()}Enum"
        enum_classes.append(f"""class {enum_name}(str, Enum):
    \"\"\"Enum for {col_name} column\"\"\"
    VALUE1 = "value1"
    VALUE2 = "value2"
    VALUE3 = "value3"\n""")

    # Class definition
    class_def = f"class {class_name}(Base):\n"
    class_def += f"    __tablename__ = '{table_name}'\n\n"

    # Columns
    for column in table_class.__table__.columns:
        col_type = str(column.type)
        col_def = f"    {column.name} = Column("

        # Map PostgreSQL types to SQLAlchemy types
        if 'UUID' in col_type:
            col_def += "UUID(as_uuid=True)"
        elif 'INTEGER' in col_type:
            col_def += "Integer"
        elif 'VARCHAR' in col_type or 'CHAR' in col_type:
            col_def += "String"
        elif 'BOOLEAN' in col_type:
            col_def += "Boolean"
        elif 'DATE' in col_type:
            col_def += "Date"
        elif 'TIMESTAMP' in col_type:
            col_def += "DateTime"
        elif 'FLOAT' in col_type or 'DOUBLE' in col_type or 'DECIMAL' in col_type:
            col_def += "Float"
        elif 'ARRAY' in col_type:
            col_def += "ARRAY(String)"
        elif 'JSON' in col_type or 'JSONB' in col_type:
            col_def += "JSONB"
        elif 'TEXT' in col_type:
            col_def += "Text"
        else:
            col_def += "String"

        # Column attributes
        if column.primary_key:
            col_def += ", primary_key=True"
        if not column.nullable and not column.primary_key:
            col_def += ", nullable=False"
        if column.server_default:
            if 'uuid_generate_v4()' in str(column.server_default):
                col_def += ", server_default=expression.text('uuid_generate_v4()')"
            else:
                col_def += f", server_default={column.server_default}"
        if column.name in analysis['has_enum_columns']:
            enum_name = f"{column.name.split('_')[-1].capitalize()}Enum"
            col_def += f", type_=SqlEnum({enum_name})"

        col_def += ")\n"
        class_def += col_def

    # Relationships
    for rel in table_class.__mapper__.relationships:
        if rel.direction.name == 'MANYTOONE':
            class_def += f"    {rel.key} = relationship('{rel.mapper.class_.__name__}')\n"
        elif rel.direction.name == 'ONETOMANY':
            class_def += f"    {rel.key} = relationship('{rel.mapper.class_.__name__}', back_populates='{rel.back_populates}')\n"

    # Methods
    methods = f"""
    def __repr__(self):
        return f"<{class_name}({', '.join([f'{col.name}={{self.{col.name}}}' for col in table_class.__table__.columns[:3]])})>"

    def to_dict(self, include_relationships: bool = False) -> Dict[str, Any]:
        \"\"\"Convert entity to dictionary\"\"\"
        data = {{
            column.name: getattr(self, column.name)
            for column in self.__table__.columns
        }}

        if include_relationships:
            for rel in self.__mapper__.relationships:
                if rel.direction.name == 'MANYTOONE':
                    data[rel.key] = getattr(self, rel.key).to_dict() if getattr(self, rel.key) else None
                elif rel.direction.name == 'ONETOMANY':
                    data[rel.key] = [item.to_dict() for item in getattr(self, rel.key)]

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        \"\"\"Create entity from dictionary\"\"\"
        return cls(**{{k: v for k, v in data.items() if k in cls.__table__.columns}})
"""

    if analysis['has_timestamps']:
        methods += """
    def update_timestamps(self):
        \"\"\"Update timestamp fields\"\"\"
        if hasattr(self, 'updated_at'):
            self.updated_at = datetime.utcnow()
        elif hasattr(self, 'created_at') and not self.created_at:
            self.created_at = datetime.utcnow()
"""

    # Combine all parts
    file_content = "\n".join(imports) + "\n"
    if enum_classes:
        file_content += "\n".join(enum_classes) + "\n"
    file_content += class_def + methods

    return file_content

def generate_repository_interface(table_name: str, class_name: str, analysis: Dict[str, Any]) -> str:
    """Generate repository interface for a table"""
    imports = [
        "from abc import ABC, abstractmethod",
        "from typing import List, Optional, Dict, Any, Union",
        "from datetime import datetime",
        f"from domain.{table_name}.{table_name}_entity import {class_name}",
        "from shared.async_base_repository import IAsyncBaseRepository",
        ""
    ]

    class_def = f"class IAsync{class_name}Repository(IAsyncBaseRepository[{class_name}], ABC):\n"
    class_def += f"    \"\"\"Async repository interface for {class_name}\"\"\"\n\n"

    # Basic CRUD methods
    methods = [
        "@abstractmethod\nasync def get_by_id(self, id: Union[int, str, UUID]) -> Optional[{class_name}]:\n    \"\"\"Get entity by ID\"\"\"\n    pass\n",
        "@abstractmethod\nasync def get_all(self, skip: int = 0, limit: int = 100) -> List[{class_name}]:\n    \"\"\"Get all entities\"\"\"\n    pass\n",
        "@abstractmethod\nasync def create(self, data: Dict[str, Any]) -> {class_name}:\n    \"\"\"Create new entity\"\"\"\n    pass\n",
        "@abstractmethod\nasync def update(self, id: Union[int, str, UUID], data: Dict[str, Any]) -> Optional[{class_name}]:\n    \"\"\"Update entity\"\"\"\n    pass\n",
        "@abstractmethod\nasync def delete(self, id: Union[int, str, UUID]) -> bool:\n    \"\"\"Delete entity\"\"\"\n    pass\n"
    ]

    # Context-specific methods
    if analysis['has_status']:
        methods.append("@abstractmethod\nasync def get_by_status(self, status: str, skip: int = 0, limit: int = 100) -> List[{class_name}]:\n    \"\"\"Get entities by status\"\"\"\n    pass\n")
        methods.append("@abstractmethod\nasync def change_status(self, id: Union[int, str, UUID], new_status: str) -> Optional[{class_name}]:\n    \"\"\"Change entity status\"\"\"\n    pass\n")

    if analysis['has_timestamps']:
        methods.append("@abstractmethod\nasync def get_created_after(self, timestamp: datetime, skip: int = 0, limit: int = 100) -> List[{class_name}]:\n    \"\"\"Get entities created after timestamp\"\"\"\n    pass\n")
        methods.append("@abstractmethod\nasync def get_updated_after(self, timestamp: datetime, skip: int = 0, limit: int = 100) -> List[{class_name}]:\n    \"\"\"Get entities updated after timestamp\"\"\"\n    pass\n")

    if analysis['has_soft_delete']:
        methods.append("@abstractmethod\nasync def soft_delete(self, id: Union[int, str, UUID]) -> bool:\n    \"\"\"Soft delete entity\"\"\"\n    pass\n")
        methods.append("@abstractmethod\nasync def restore(self, id: Union[int, str, UUID]) -> bool:\n    \"\"\"Restore soft-deleted entity\"\"\"\n    pass\n")

    if analysis['has_user_relation']:
        methods.append("@abstractmethod\nasync def get_by_user_id(self, user_id: Union[int, str, UUID], skip: int = 0, limit: int = 100) -> List[{class_name}]:\n    \"\"\"Get entities by user ID\"\"\"\n    pass\n")

    if analysis['has_many_relations'] or analysis['has_one_relations']:
        methods.append("@abstractmethod\nasync def get_with_relations(self, id: Union[int, str, UUID], relation_names: List[str]) -> Optional[{class_name}]:\n    \"\"\"Get entity with relations\"\"\"\n    pass\n")

    # Advanced methods
    methods.extend([
        "@abstractmethod\nasync def count(self) -> int:\n    \"\"\"Count entities\"\"\"\n    pass\n",
        "@abstractmethod\nasync def exists(self, **filters: Any) -> bool:\n    \"\"\"Check if entity exists\"\"\"\n    pass\n",
        "@abstractmethod\nasync def bulk_create(self, items: List[Dict[str, Any]]) -> List[{class_name}]:\n    \"\"\"Bulk create entities\"\"\"\n    pass\n"
    ])

    # Combine all parts
    file_content = "\n".join(imports) + "\n" + class_def + "\n".join(methods)
    return file_content

def generate_service_interface(table_name: str, class_name: str, analysis: Dict[str, Any]) -> str:
    """Generate service interface for a table"""
    imports = [
        "from abc import ABC, abstractmethod",
        "from typing import List, Optional, Dict, Any, Union",
        "from datetime import datetime",
        f"from domain.{table_name}.{table_name}_entity import {class_name}",
        "from shared.async_base_service import IAsyncBaseService",
        ""
    ]

    class_def = f"class IAsync{class_name}Service(IAsyncBaseService[{class_name}], ABC):\n"
    class_def += f"    \"\"\"Async service interface for {class_name}\"\"\"\n\n"

    # Basic CRUD methods
    methods = [
        "@abstractmethod\nasync def get_by_id(self, id: Union[int, str, UUID]) -> Optional[{class_name}]:\n    \"\"\"Get entity by ID\"\"\"\n    pass\n",
        "@abstractmethod\nasync def get_all(self, skip: int = 0, limit: int = 100) -> List[{class_name}]:\n    \"\"\"Get all entities\"\"\"\n    pass\n",
        "@abstractmethod\nasync def create(self, data: Dict[str, Any]) -> {class_name}:\n    \"\"\"Create new entity\"\"\"\n    pass\n",
        "@abstractmethod\nasync def update(self, id: Union[int, str, UUID], data: Dict[str, Any]) -> Optional[{class_name}]:\n    \"\"\"Update entity\"\"\"\n    pass\n",
        "@abstractmethod\nasync def delete(self, id: Union[int, str, UUID]) -> bool:\n    \"\"\"Delete entity\"\"\"\n    pass\n"
    ]

    # Context-specific methods
    if analysis['has_status']:
        methods.append("@abstractmethod\nasync def get_by_status(self, status: str, skip: int = 0, limit: int = 100) -> List[{class_name}]:\n    \"\"\"Get entities by status\"\"\"\n    pass\n")
        methods.append("@abstractmethod\nasync def change_status(self, id: Union[int, str, UUID], new_status: str, reason: Optional[str] = None) -> Optional[{class_name}]:\n    \"\"\"Change entity status\"\"\"\n    pass\n")

    if analysis['has_timestamps']:
        methods.append("@abstractmethod\nasync def get_recently_created(self, days: int = 7, skip: int = 0, limit: int = 100) -> List[{class_name}]:\n    \"\"\"Get recently created entities\"\"\"\n    pass\n")
        methods.append("@abstractmethod\nasync def get_recently_updated(self, days: int = 7, skip: int = 0, limit: int = 100) -> List[{class_name}]:\n    \"\"\"Get recently updated entities\"\"\"\n    pass\n")

    if analysis['has_soft_delete']:
        methods.append("@abstractmethod\nasync def soft_delete(self, id: Union[int, str, UUID], reason: Optional[str] = None) -> bool:\n    \"\"\"Soft delete entity\"\"\"\n    pass\n")
        methods.append("@abstractmethod\nasync def restore(self, id: Union[int, str, UUID]) -> bool:\n    \"\"\"Restore soft-deleted entity\"\"\"\n    pass\n")

    if analysis['has_user_relation']:
        methods.append("@abstractmethod\nasync def get_for_user(self, user_id: Union[int, str, UUID], skip: int = 0, limit: int = 100) -> List[{class_name}]:\n    \"\"\"Get entities for user\"\"\"\n    pass\n")

    # Business logic methods
    methods.extend([
        "@abstractmethod\nasync def search(self, query: str, fields: List[str] = None, skip: int = 0, limit: int = 100) -> List[{class_name}]:\n    \"\"\"Search entities\"\"\"\n    pass\n",
        "@abstractmethod\nasync def get_stats(self) -> Dict[str, Any]:\n    \"\"\"Get statistics\"\"\"\n    pass\n",
        "@abstractmethod\nasync def export_to_csv(self, filters: Optional[Dict[str, Any]] = None) -> str:\n    \"\"\"Export to CSV\"\"\"\n    pass\n",
        "@abstractmethod\nasync def import_from_csv(self, csv_data: str) -> int:\n    \"\"\"Import from CSV\"\"\"\n    pass\n"
    ])

    # Combine all parts
    file_content = "\n".join(imports) + "\n" + class_def + "\n".join(methods)
    return file_content

def generate_repository_implementation(table_name: str, class_name: str, analysis: Dict[str, Any]) -> str:
    """Generate repository implementation for a table"""
    imports = [
        "import logging",
        "from typing import List, Optional, Dict, Any, Union",
        "from datetime import datetime",
        "from uuid import UUID",
        "from sqlalchemy import select, update, delete, and_",
        "from sqlalchemy.ext.asyncio import AsyncSession",
        "from sqlalchemy.sql.expression import Select",
        f"from domain.{table_name}.{table_name}_entity import {class_name}",
        f"from domain.{table_name}.{table_name}_repository_interface import IAsync{class_name}Repository",
        "from infrastructure.database.db import async_db",
        "from shared.utils.base_repository import BaseRepository",
        ""
    ]

    class_def = f"class {class_name}Repository(BaseRepository, IAsync{class_name}Repository):\n"
    class_def += f"    \"\"\"Async repository implementation for {class_name}\"\"\"\n\n"
    class_def += "    def __init__(self):\n"
    class_def += "        super().__init__()\n"
    class_def += f"        self.model = {class_name}\n\n"

    # Basic CRUD methods
    methods = [
        "async def get_by_id(self, id: Union[int, str, UUID]) -> Optional[{class_name}]:\n"
        "    \"\"\"Get entity by ID\"\"\"\n"
        "    async with async_db.get_session() as session:\n"
        "        result = await session.get(self.model, id)\n"
        "        return result\n",

        "async def get_all(self, skip: int = 0, limit: int = 100) -> List[{class_name}]:\n"
        "    \"\"\"Get all entities\"\"\"\n"
        "    async with async_db.get_session() as session:\n"
        "        query = select(self.model).offset(skip).limit(limit)\n"
        "        result = await session.execute(query)\n"
        "        return result.scalars().all()\n",

        "async def create(self, data: Dict[str, Any]) -> {class_name}:\n"
        "    \"\"\"Create new entity\"\"\"\n"
        "    async with async_db.get_session() as session:\n"
        "        instance = self.model(**data)\n"
        "        session.add(instance)\n"
        "        await session.commit()\n"
        "        await session.refresh(instance)\n"
        "        return instance\n",

        "async def update(self, id: Union[int, str, UUID], data: Dict[str, Any]) -> Optional[{class_name}]:\n"
        "    \"\"\"Update entity\"\"\"\n"
        "    async with async_db.get_session() as session:\n"
        "        instance = await session.get(self.model, id)\n"
        "        if not instance:\n"
        "            return None\n"
        "        for key, value in data.items():\n"
        "            setattr(instance, key, value)\n"
        "        await session.commit()\n"
        "        await session.refresh(instance)\n"
        "        return instance\n",

        "async def delete(self, id: Union[int, str, UUID]) -> bool:\n"
        "    \"\"\"Delete entity\"\"\"\n"
        "    async with async_db.get_session() as session:\n"
        "        instance = await session.get(self.model, id)\n"
        "        if not instance:\n"
        "            return False\n"
        "        await session.delete(instance)\n"
        "        await session.commit()\n"
        "        return True\n"
    ]

    # Context-specific methods
    if analysis['has_status']:
        methods.append(
            "async def get_by_status(self, status: str, skip: int = 0, limit: int = 100) -> List[{class_name}]:\n"
            "    \"\"\"Get entities by status\"\"\"\n"
            "    async with async_db.get_session() as session:\n"
            "        query = select(self.model).where(self.model.status == status).offset(skip).limit(limit)\n"
            "        result = await session.execute(query)\n"
            "        return result.scalars().all()\n"
        )
        methods.append(
            "async def change_status(self, id: Union[int, str, UUID], new_status: str) -> Optional[{class_name}]:\n"
            "    \"\"\"Change entity status\"\"\"\n"
            "    async with async_db.get_session() as session:\n"
            "        stmt = update(self.model).where(self.model.id == id).values(status=new_status).returning(self.model)\n"
            "        result = await session.execute(stmt)\n"
            "        await session.commit()\n"
            "        return result.scalar_one_or_none()\n"
        )

    if analysis['has_timestamps']:
        methods.append(
            "async def get_created_after(self, timestamp: datetime, skip: int = 0, limit: int = 100) -> List[{class_name}]:\n"
            "    \"\"\"Get entities created after timestamp\"\"\"\n"
            "    async with async_db.get_session() as session:\n"
            "        query = select(self.model).where(self.model.created_at >= timestamp).offset(skip).limit(limit)\n"
            "        result = await session.execute(query)\n"
            "        return result.scalars().all()\n"
        )
        methods.append(
            "async def get_updated_after(self, timestamp: datetime, skip: int = 0, limit: int = 100) -> List[{class_name}]:\n"
            "    \"\"\"Get entities updated after timestamp\"\"\"\n"
            "    async with async_db.get_session() as session:\n"
            "        query = select(self.model).where(self.model.updated_at >= timestamp).offset(skip).limit(limit)\n"
            "        result = await session.execute(query)\n"
            "        return result.scalars().all()\n"
        )

    if analysis['has_soft_delete']:
        methods.append(
            "async def soft_delete(self, id: Union[int, str, UUID]) -> bool:\n"
            "    \"\"\"Soft delete entity\"\"\"\n"
            "    async with async_db.get_session() as session:\n"
            "        stmt = update(self.model).where(self.model.id == id).values(is_deleted=True, deleted_at=datetime.utcnow())\n"
            "        await session.execute(stmt)\n"
            "        await session.commit()\n"
            "        return True\n"
        )
        methods.append(
            "async def restore(self, id: Union[int, str, UUID]) -> bool:\n"
            "    \"\"\"Restore soft-deleted entity\"\"\"\n"
            "    async with async_db.get_session() as session:\n"
            "        stmt = update(self.model).where(self.model.id == id).values(is_deleted=False, deleted_at=None)\n"
            "        await session.execute(stmt)\n"
            "        await session.commit()\n"
            "        return True\n"
        )

    if analysis['has_user_relation']:
        methods.append(
            "async def get_by_user_id(self, user_id: Union[int, str, UUID], skip: int = 0, limit: int = 100) -> List[{class_name}]:\n"
            "    \"\"\"Get entities by user ID\"\"\"\n"
            "    async with async_db.get_session() as session:\n"
            "        query = select(self.model).where(self.model.user_id == user_id).offset(skip).limit(limit)\n"
            "        result = await session.execute(query)\n"
            "        return result.scalars().all()\n"
        )

    if analysis['has_many_relations'] or analysis['has_one_relations']:
        methods.append(
            "async def get_with_relations(self, id: Union[int, str, UUID], relation_names: List[str]) -> Optional[{class_name}]:\n"
            "    \"\"\"Get entity with relations\"\"\"\n"
            "    async with async_db.get_session() as session:\n"
            "        query = select(self.model).where(self.model.id == id)\n"
            "        for relation in relation_names:\n"
            "            query = query.options(selectinload(getattr(self.model, relation)))\n"
            "        result = await session.execute(query)\n"
            "        return result.scalar_one_or_none()\n"
        )

    # Advanced methods
    methods.extend([
        "async def count(self) -> int:\n"
        "    \"\"\"Count entities\"\"\"\n"
        "    async with async_db.get_session() as session:\n"
        "        query = select(func.count()).select_from(self.model)\n"
        "        result = await session.execute(query)\n"
        "        return result.scalar_one()\n",

        "async def exists(self, **filters: Any) -> bool:\n"
        "    \"\"\"Check if entity exists\"\"\"\n"
        "    async with async_db.get_session() as session:\n"
        "        query = select(exists().where(self._apply_filters(filters)))\n"
        "        result = await session.execute(query)\n"
        "        return result.scalar_one()\n",

        "async def bulk_create(self, items: List[Dict[str, Any]]) -> List[{class_name}]:\n"
        "    \"\"\"Bulk create entities\"\"\"\n"
        "    async with async_db.get_session() as session:\n"
        "        instances = [self.model(**item) for item in items]\n"
        "        session.add_all(instances)\n"
        "        await session.commit()\n"
        "        for instance in instances:\n"
        "            await session.refresh(instance)\n"
        "        return instances\n"
    ])

    # Combine all parts
    file_content = "\n".join(imports) + "\n" + class_def + "\n".join(methods)
    return file_content

def generate_service_implementation(table_name: str, class_name: str, analysis: Dict[str, Any]) -> str:
    """Generate service implementation for a table"""
    imports = [
        "import logging",
        "from typing import List, Optional, Dict, Any, Union",
        "from datetime import datetime, timedelta",
        "from uuid import UUID",
        f"from domain.{table_name}.{table_name}_entity import {class_name}",
        f"from domain.{table_name}.{table_name}_service_interface import IAsync{class_name}Service",
        f"from infrastructure.repositories.{table_name}.{table_name}_repository import {class_name}Repository",
        "from shared.utils.base_service import BaseService",
        ""
    ]

    class_def = f"class {class_name}Service(BaseService, IAsync{class_name}Service):\n"
    class_def += f"    \"\"\"Async service implementation for {class_name}\"\"\"\n\n"
    class_def += "    def __init__(self):\n"
    class_def += "        super().__init__()\n"
    class_def += f"        self.repository = {class_name}Repository()\n\n"

    # Basic CRUD methods
    methods = [
        "async def get_by_id(self, id: Union[int, str, UUID]) -> Optional[{class_name}]:\n"
        "    \"\"\"Get entity by ID\"\"\"\n"
        "    return await self.repository.get_by_id(id)\n",

        "async def get_all(self, skip: int = 0, limit: int = 100) -> List[{class_name}]:\n"
        "    \"\"\"Get all entities\"\"\"\n"
        "    return await self.repository.get_all(skip=skip, limit=limit)\n",

        "async def create(self, data: Dict[str, Any]) -> {class_name}:\n"
        "    \"\"\"Create new entity\"\"\"\n"
        "    # Add any business validation here\n"
        "    return await self.repository.create(data)\n",

        "async def update(self, id: Union[int, str, UUID], data: Dict[str, Any]) -> Optional[{class_name}]:\n"
        "    \"\"\"Update entity\"\"\"\n"
        "    # Add any business validation here\n"
        "    return await self.repository.update(id, data)\n",

        "async def delete(self, id: Union[int, str, UUID]) -> bool:\n"
        "    \"\"\"Delete entity\"\"\"\n"
        "    # Add any pre-delete checks here\n"
        "    return await self.repository.delete(id)\n"
    ]

    # Context-specific methods
    if analysis['has_status']:
        methods.append(
            "async def get_by_status(self, status: str, skip: int = 0, limit: int = 100) -> List[{class_name}]:\n"
            "    \"\"\"Get entities by status\"\"\"\n"
            "    return await self.repository.get_by_status(status, skip, limit)\n"
        )
        methods.append(
            "async def change_status(self, id: Union[int, str, UUID], new_status: str, reason: Optional[str] = None) -> Optional[{class_name}]:\n"
            "    \"\"\"Change entity status\"\"\"\n"
            "    if reason:\n"
            "        logging.info(f\"Changing status of {id} to {new_status}. Reason: {reason}\")\n"
            "    return await self.repository.change_status(id, new_status)\n"
        )

    if analysis['has_timestamps']:
        methods.append(
            "async def get_recently_created(self, days: int = 7, skip: int = 0, limit: int = 100) -> List[{class_name}]:\n"
            "    \"\"\"Get recently created entities\"\"\"\n"
            "    cutoff = datetime.utcnow() - timedelta(days=days)\n"
            "    return await self.repository.get_created_after(cutoff, skip, limit)\n"
        )
        methods.append(
            "async def get_recently_updated(self, days: int = 7, skip: int = 0, limit: int = 100) -> List[{class_name}]:\n"
            "    \"\"\"Get recently updated entities\"\"\"\n"
            "    cutoff = datetime.utcnow() - timedelta(days=days)\n"
            "    return await self.repository.get_updated_after(cutoff, skip, limit)\n"
        )

    if analysis['has_soft_delete']:
        methods.append(
            "async def soft_delete(self, id: Union[int, str, UUID], reason: Optional[str] = None) -> bool:\n"
            "    \"\"\"Soft delete entity\"\"\"\n"
            "    if reason:\n"
            "        logging.info(f\"Soft deleting {id}. Reason: {reason}\")\n"
            "    return await self.repository.soft_delete(id)\n"
        )
        methods.append(
            "async def restore(self, id: Union[int, str, UUID]) -> bool:\n"
            "    \"\"\"Restore soft-deleted entity\"\"\"\n"
            "    return await self.repository.restore(id)\n"
        )

    if analysis['has_user_relation']:
        methods.append(
            "async def get_for_user(self, user_id: Union[int, str, UUID], skip: int = 0, limit: int = 100) -> List[{class_name}]:\n"
            "    \"\"\"Get entities for user\"\"\"\n"
            "    return await self.repository.get_by_user_id(user_id, skip, limit)\n"
        )

    # Business logic methods
    methods.extend([
        "async def search(self, query: str, fields: List[str] = None, skip: int = 0, limit: int = 100) -> List[{class_name}]:\n"
        "    \"\"\"Search entities\"\"\"\n"
        "    if not fields:\n"
        "        fields = ['name']\n"
        "    search_conditions = [getattr(self.model, field).ilike(f'%{query}%') for field in fields]\n"
        "    return await self.repository.get_all(\n"
        "        skip=skip,\n"
        "        limit=limit,\n"
        "        filters={'or': search_conditions}\n"
        "    )\n",

        "async def get_stats(self) -> Dict[str, Any]:\n"
        "    \"\"\"Get statistics\"\"\"\n"
        "    stats = {'total_count': await self.repository.count()}\n"
    ])

    if analysis['has_status']:
        methods.append(
            "    stats['status_counts'] = {\n"
            "        status: await self.repository.count(filters={'status': status})\n"
            "        for status in ['active', 'inactive', 'pending']\n"
            "    }\n"
        )

    if analysis['has_user_relation']:
        methods.append(
            "    stats['user_distribution'] = await self._get_user_distribution_stats()\n"
        )

    methods.append("    return stats\n")

    methods.extend([
        "async def export_to_csv(self, filters: Optional[Dict[str, Any]] = None) -> str:\n"
        "    \"\"\"Export to CSV\"\"\"\n"
        "    import csv\n"
        "    from io import StringIO\n"
        "    output = StringIO()\n"
        "    writer = csv.writer(output)\n"
        "    writer.writerow(['id', 'name'])\n"
        "    instances = await self.repository.get_all(filters=filters)\n"
        "    for instance in instances:\n"
        "        writer.writerow([instance.id, instance.name])\n"
        "    return output.getvalue()\n",

        "async def import_from_csv(self, csv_data: str) -> int:\n"
        "    \"\"\"Import from CSV\"\"\"\n"
        "    import csv\n"
        "    from io import StringIO\n"
        "    reader = csv.DictReader(StringIO(csv_data))\n"
        "    items = [row for row in reader]\n"
        "    instances = await self.repository.bulk_create(items)\n"
        "    return len(instances)\n"
    ])

    # Combine all parts
    file_content = "\n".join(imports) + "\n" + class_def + "\n".join(methods)
    return file_content

def generate_controller(table_name: str, class_name: str) -> str:
    """Generate Flask-RESTX controller for a table"""
    return f"""from flask_restx import Namespace, Resource, fields, reqparse
from typing import List, Optional
from domain.{table_name}.{table_name}_entity import {class_name}
from domain.{table_name}.{table_name}_service_interface import IAsync{class_name}Service
from api.dtos.{table_name}_dto import {class_name}Create, {class_name}Update, {class_name}Response
from api.mappers.{table_name}_mapper import {table_name}_mapper
from api.validations.{table_name}_validation_schemas import validate_{table_name}_create, validate_{table_name}_update
from shared.utils.auth import token_required
from shared.utils.logger import logger

# Initialize namespace
api = Namespace('{table_name}', description='{class_name} operations')

# Request parsers
pagination_parser = reqparse.RequestParser()
pagination_parser.add_argument('page', type=int, help='Page number', default=1)
pagination_parser.add_argument('per_page', type=int, help='Items per page', default=10)

# Model definitions
{table_name}_create_model = api.model('{class_name}Create', {{
    'name': fields.String(required=True, description='{table_name} name'),
    'description': fields.String(description='{table_name} description'),
    'status': fields.String(description='{table_name} status', enum=['active', 'inactive', 'pending'])
}})

{table_name}_update_model = api.model('{class_name}Update', {{
    'name': fields.String(description='{table_name} name'),
    'description': fields.String(description='{table_name} description'),
    'status': fields.String(description='{table_name} status', enum=['active', 'inactive', 'pending'])
}})

{table_name}_response_model = api.model('{class_name}Response', {{
    'id': fields.String(description='{table_name} ID'),
    'name': fields.String(description='{table_name} name'),
    'description': fields.String(description='{table_name} description'),
    'status': fields.String(description='{table_name} status'),
    'created_at': fields.DateTime(description='Creation timestamp'),
    'updated_at': fields.DateTime(description='Last update timestamp')
}})

def initialize_controller(service: IAsync{class_name}Service):
    \"\"\"Initialize controller with service dependency\"\"\"

    @api.route('/')
    class {class_name}List(Resource):
        @api.doc('list_{table_name}s')
        @api.expect(pagination_parser)
        @api.marshal_list_with({table_name}_response_model)
        @token_required
        async def get(self):
            \"\"\"List all {table_name}s\"\"\"
            try:
                args = pagination_parser.parse_args()
                skip = (args['page'] - 1) * args['per_page']
                limit = args['per_page']

                results = await service.get_all(skip=skip, limit=limit)
                return [{table_name}_mapper.to_dto(item) for item in results]
            except Exception as e:
                logger.error(f"Error getting {table_name}s: {{str(e)}}")
                api.abort(400, str(e))

        @api.doc('create_{table_name}')
        @api.expect({table_name}_create_model)
        @api.marshal_with({table_name}_response_model, code=201)
        @token_required
        async def post(self):
            \"\"\"Create a new {table_name}\"\"\"
            try:
                data = api.payload
                validated_data = validate_{table_name}_create(data)
                entity = {table_name}_mapper.to_entity(validated_data)
                result = await service.create(entity.to_dict())
                return {table_name}_mapper.to_dto(result), 201
            except Exception as e:
                logger.error(f"Error creating {table_name}: {{str(e)}}")
                api.abort(400, str(e))

    @api.route('/<string:id>')
    @api.param('id', 'The {table_name} identifier')
    @api.response(404, '{class_name} not found')
    class {class_name}Resource(Resource):
        @api.doc('get_{table_name}')
        @api.marshal_with({table_name}_response_model)
        @token_required
        async def get(self, id):
            \"\"\"Get a {table_name} given its identifier\"\"\"
            try:
                result = await service.get_by_id(id)
                if not result:
                    api.abort(404, f"{class_name} not found")
                return {table_name}_mapper.to_dto(result)
            except Exception as e:
                logger.error(f"Error getting {table_name} {{id}}: {{str(e)}}")
                api.abort(400, str(e))

        @api.doc('update_{table_name}')
        @api.expect({table_name}_update_model)
        @api.marshal_with({table_name}_response_model)
        @token_required
        async def put(self, id):
            \"\"\"Update a {table_name} given its identifier\"\"\"
            try:
                data = api.payload
                validated_data = validate_{table_name}_update(data)
                entity = {table_name}_mapper.to_entity(validated_data)
                result = await service.update(id, entity.to_dict())
                if not result:
                    api.abort(404, f"{class_name} not found")
                return {table_name}_mapper.to_dto(result)
            except Exception as e:
                logger.error(f"Error updating {table_name} {{id}}: {{str(e)}}")
                api.abort(400, str(e))

        @api.doc('delete_{table_name}')
        @api.response(204, '{class_name} deleted')
        @token_required
        async def delete(self, id):
            \"\"\"Delete a {table_name} given its identifier\"\"\"
            try:
                success = await service.delete(id)
                if not success:
                    api.abort(404, f"{class_name} not found")
                return '', 204
            except Exception as e:
                logger.error(f"Error deleting {table_name} {{id}}: {{str(e)}}")
                api.abort(400, str(e))

    return api
"""

def generate_validation_schemas(table_name: str, class_name: str) -> str:
    """Generate validation schemas for a table"""
    return f"""from enum import Enum
from pydantic import BaseModel, validator
from typing import Optional
from datetime import datetime

class {class_name}Status(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"

class {class_name}Base(BaseModel):
    \"\"\"Base schema for {table_name}\"\"\"
    pass

class {class_name}Create({class_name}Base):
    \"\"\"Schema for creating {table_name}\"\"\"
    name: str
    description: Optional[str] = None
    status: {class_name}Status = {class_name}Status.ACTIVE

    @validator('name')
    def validate_name(cls, v):
        if len(v) < 3:
            raise ValueError("Name must be at least 3 characters")
        return v

class {class_name}Update({class_name}Base):
    \"\"\"Schema for updating {table_name}\"\"\"
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[{class_name}Status] = None

class {class_name}Response({class_name}Base):
    \"\"\"Response schema for {table_name}\"\"\"
    id: str
    name: str
    description: Optional[str] = None
    status: {class_name}Status
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

def validate_{table_name}_create(data: {class_name}Create) -> {class_name}Create:
    \"\"\"Validate {table_name} creation data\"\"\"
    return data

def validate_{table_name}_update(data: {class_name}Update) -> {class_name}Update:
    \"\"\"Validate {table_name} update data\"\"\"
    return data
"""

def generate_mapper(table_name: str, class_name: str) -> str:
    """Generate mapper for a table"""
    return f"""from domain.{table_name}.{table_name}_entity import {class_name}
from api.dtos.{table_name}_dto import {class_name}Create, {class_name}Update, {class_name}Response
from typing import Union

class {class_name}Mapper:
    \"\"\"Mapper for {class_name} between entity and DTOs\"\"\"

    @staticmethod
    def to_dto(entity: {class_name}) -> {class_name}Response:
        \"\"\"Convert entity to response DTO\"\"\"
        return {class_name}Response(
            id=str(entity.id),
            name=entity.name,
            description=entity.description,
            status=entity.status,
            created_at=entity.created_at,
            updated_at=entity.updated_at
        )

    @staticmethod
    def to_entity(dto: Union[{class_name}Create, {class_name}Update]) -> {class_name}:
        \"\"\"Convert DTO to entity\"\"\"
        return {class_name}(
            name=dto.name,
            description=dto.description,
            status=dto.status
        )

    @staticmethod
    def update_entity(entity: {class_name}, dto: {class_name}Update) -> {class_name}:
        \"\"\"Update entity from DTO\"\"\"
        if dto.name is not None:
            entity.name = dto.name
        if dto.description is not None:
            entity.description = dto.description
        if dto.status is not None:
            entity.status = dto.status
        return entity

{table_name}_mapper = {class_name}Mapper()
"""

def generate_sito(table_name: str, class_name: str) -> str:
    """Generate SITO (Service, Interface, Transformer, Operation) file"""
    return f"""from typing import List, Optional
from domain.{table_name}.{table_name}_entity import {class_name}
from domain.{table_name}.{table_name}_service_interface import IAsync{class_name}Service
from infrastructure.repositories.{table_name}.{table_name}_repository import {class_name}Repository
from api.mappers.{table_name}_mapper import {table_name}_mapper
from shared.utils.logger import logger

class {class_name}Service(IAsync{class_name}Service):
    \"\"\"Service implementation for {class_name}\"\"\"

    def __init__(self):
        self.repository = {class_name}Repository()

    async def get_by_id(self, id: str) -> Optional[{class_name}]:
        \"\"\"Get {table_name} by ID\"\"\"
        try:
            return await self.repository.get_by_id(id)
        except Exception as e:
            logger.error(f"Error getting {table_name} by ID: {{str(e)}}")
            raise

    async def get_all(self, skip: int = 0, limit: int = 100) -> List[{class_name}]:
        \"\"\"Get all {table_name}s\"\"\"
        try:
            return await self.repository.get_all(skip=skip, limit=limit)
        except Exception as e:
            logger.error(f"Error getting all {table_name}s: {{str(e)}}")
            raise

    async def create(self, data: {class_name}) -> {class_name}:
        \"\"\"Create new {table_name}\"\"\"
        try:
            return await self.repository.create({table_name}_mapper.to_entity(data))
        except Exception as e:
            logger.error(f"Error creating {table_name}: {{str(e)}}")
            raise

    async def update(self, id: str, data: {class_name}) -> Optional[{class_name}]:
        \"\"\"Update {table_name}\"\"\"
        try:
            return await self.repository.update(id, {table_name}_mapper.to_entity(data))
        except Exception as e:
            logger.error(f"Error updating {table_name}: {{str(e)}}")
            raise

    async def delete(self, id: str) -> bool:
        \"\"\"Delete {table_name}\"\"\"
        try:
            return await self.repository.delete(id)
        except Exception as e:
            logger.error(f"Error deleting {table_name}: {{str(e)}}")
            raise
"""

def generate_namespace(controllers: List[str]) -> str:
    """Generate namespace file with all controllers"""
    imports = "\n".join([f"from api.controllers.{controller}_controller import initialize_controller as {controller}_api"
                        for controller in controllers])
    api_registrations = "\n".join([f"api.add_namespace({controller}_api({controller.capitalize()}Service()), '/{controller}')"
                                 for controller in controllers])

    return f"""from flask_restx import Api
{imports}
from application.services.{controllers[0]}_service import {controllers[0].capitalize()}Service

api = Api(
    title='Hexagonal Architecture API',
    version='1.0',
    description='API built with Hexagonal Architecture pattern',
    doc='/docs'
)

{api_registrations}
"""

def generate_database_file():
    """Generate the database configuration file"""
    db_file_path = os.path.join(PROJECT_DIR, "infrastructure", "database", "db.py")
    os.makedirs(os.path.dirname(db_file_path), exist_ok=True)

    content = """import logging
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, AsyncEngine
from sqlalchemy.orm import sessionmaker, declarative_base
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional
from sqlalchemy import text
from config.config import settings
import asyncio

logger = logging.getLogger(__name__)

Base = declarative_base()

class AsyncDatabase:
    def __init__(self):
        self.engine: Optional[AsyncEngine] = None
        self.async_session_factory: Optional[sessionmaker] = None
        self._initialized = False

    async def init_db(self):
        \"\"\"Initialize async database connection with proper connection pooling\"\"\"
        if self._initialized:
            return

        try:
            self.engine = create_async_engine(
                settings.DATABASE_URL,
                pool_size=settings.DB_POOL_SIZE,
                max_overflow=settings.DB_MAX_OVERFLOW,
                pool_recycle=settings.DB_POOL_RECYCLE,
                pool_pre_ping=True,
                pool_timeout=30,
                connect_args={
                    "command_timeout": 10,
                    "server_settings": {
                        "application_name": settings.APP_NAME,
                        "statement_timeout": "15000"
                    }
                },
                echo=settings.DEBUG
            )

            # Verify connection with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    async with self.engine.begin() as conn:
                        await conn.execute(text("SELECT 1"))
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    await asyncio.sleep(1 * (attempt + 1))
                    logger.warning(f"Database connection attempt {attempt + 1} failed, retrying...")
                    continue

            self.async_session_factory = sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False
            )

            self._initialized = True
            logger.info("Async database initialized successfully")

        except Exception as e:
            logger.critical("Async database initialization failed", exc_info=True)
            raise

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        \"\"\"Async context manager for database sessions with proper error handling\"\"\"
        if not self._initialized:
            raise RuntimeError("Database not initialized. Call init_db() first.")

        session = self.async_session_factory()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error("Database operation failed", exc_info=True)
            raise
        finally:
            await session.close()

    async def close(self):
        \"\"\"Close all database connections\"\"\"
        if self.engine:
            await self.engine.dispose()
            self._initialized = False
            logger.info("Async database connections closed")

# Singleton async database instance
async_db = AsyncDatabase()
"""
    with open(db_file_path, "w") as f:
        f.write(content)

def generate_hexagonal_architecture():
    """Generate all hexagonal architecture components"""
    engine = create_engine(DATABASE_URL)
    Base = automap_base()
    Base.prepare(engine, reflect=True)

    controllers = []

    for class_name, table_class in Base.classes.items():
        table_name = table_class.__table__.name
        class_name = class_name.capitalize()
        controllers.append(table_name)
        analysis = analyze_table(table_class)

        # Generate domain layer
        domain_folder = os.path.join(PROJECT_DIR, "domain", table_name)
        os.makedirs(domain_folder, exist_ok=True)

        # Entity
        entity_content = generate_entity_file(table_name, class_name, table_class, analysis)
        with open(os.path.join(domain_folder, f"{table_name}_entity.py"), "w") as f:
            f.write(entity_content)

        # Repository interface
        repo_interface_content = generate_repository_interface(table_name, class_name, analysis)
        with open(os.path.join(domain_folder, f"{table_name}_repository_interface.py"), "w") as f:
            f.write(repo_interface_content)

        # Service interface
        service_interface_content = generate_service_interface(table_name, class_name, analysis)
        with open(os.path.join(domain_folder, f"{table_name}_service_interface.py"), "w") as f:
            f.write(service_interface_content)

        # Generate infrastructure layer (repository implementation)
        repo_impl_folder = os.path.join(PROJECT_DIR, "infrastructure", "repositories", table_name)
        os.makedirs(repo_impl_folder, exist_ok=True)

        repo_impl_content = generate_repository_implementation(table_name, class_name, analysis)
        with open(os.path.join(repo_impl_folder, f"{table_name}_repository.py"), "w") as f:
            f.write(repo_impl_content)

        # Generate application layer (service implementation)
        service_impl_folder = os.path.join(PROJECT_DIR, "application", "services", table_name)
        os.makedirs(service_impl_folder, exist_ok=True)

        service_impl_content = generate_service_implementation(table_name, class_name, analysis)
        with open(os.path.join(service_impl_folder, f"{table_name}_service.py"), "w") as f:
            f.write(service_impl_content)

        # Generate API layer
        api_folder = os.path.join(PROJECT_DIR, "api", table_name)
        os.makedirs(api_folder, exist_ok=True)

        # Controller
        controller_content = generate_controller(table_name, class_name)
        with open(os.path.join(api_folder, f"{table_name}_controller.py"), "w") as f:
            f.write(controller_content)

        # DTOs
        dto_content = generate_validation_schemas(table_name, class_name)
        with open(os.path.join(api_folder, f"{table_name}_dto.py"), "w") as f:
            f.write(dto_content)

        # Mapper
        mapper_content = generate_mapper(table_name, class_name)
        with open(os.path.join(api_folder, f"{table_name}_mapper.py"), "w") as f:
            f.write(mapper_content)

        # SITO
        sito_content = generate_sito(table_name, class_name)
        with open(os.path.join(api_folder, f"{table_name}_sito.py"), "w") as f:
            f.write(sito_content)

    # Generate namespace file
    namespace_content = generate_namespace(controllers)
    with open(os.path.join(PROJECT_DIR, "api", "namespaces.py"), "w") as f:
        f.write(namespace_content)

    # Generate database configuration
    generate_database_file()

def main():
    print("Starting hexagonal architecture generation...")

    print("Creating project structure...")
    create_folders_and_files()

    print("Generating SQLAlchemy models...")
    generate_sqlalchemy_models()

    print("Generating hexagonal architecture components...")
    generate_hexagonal_architecture()

    print(" Hexagonal architecture generation completed successfully!")

if __name__ == "__main__":
    main()
