# Rust Onion Architecture for Enterprise ROAE
# Awase Khirni Syed Copyright 2025 B ORI Inc. Canada. All Rights Reserved.


# Project root
mkdir roae

# Root files
touch roae/.env
touch roae/.gitignore
touch roae/Cargo.toml
touch roae/diesel.toml
touch roae/README.md

# Top-level dirs
mkdir -p roae/scripts
mkdir -p roae/migrations

# Define business modules
MODULES="examples sales finance corporate"

# Create layered modular structure
for module in $MODULES; do
  # Domain layer: models, repository traits, domain services
  mkdir -p roae/src/domain/$module/{models,repositories,services}
  touch roae/src/domain/$module/mod.rs
  touch roae/src/domain/$module/models/mod.rs
  touch roae/src/domain/$module/repositories/mod.rs
  touch roae/src/domain/$module/services/mod.rs

  # Infrastructure layer: DB schema, concrete repositories
  mkdir -p roae/src/infrastructure/$module/{repositories,schema}
  touch roae/src/infrastructure/$module/mod.rs
  touch roae/src/infrastructure/$module/repositories/mod.rs
  touch roae/src/infrastructure/$module/schema/mod.rs

  # Application services (orchestration layer)
  mkdir -p roae/src/application_services/$module
  touch roae/src/application_services/$module/mod.rs

  # API handlers (Actix web)
  mkdir -p roae/src/api/handlers/$module
  touch roae/src/api/handlers/$module/mod.rs

  # Tests
  mkdir -p roae/src/tests/$module
  touch roae/src/tests/$module/mod.rs
done

# Root source modules
touch roae/src/lib.rs
touch roae/src/main.rs

# Top-level mod.rs files for each layer
touch roae/src/domain/mod.rs
touch roae/src/infrastructure/mod.rs
touch roae/src/application_services/mod.rs
touch roae/src/api/mod.rs
touch roae/src/api/handlers/mod.rs
touch roae/src/tests/mod.rs

# Infrastructure core (shared DB connection, pool, etc.)
touch roae/src/infrastructure/db.rs

# Scripts
touch roae/scripts/run_postgres.sh

# Migrations (Diesel)
# Directory exists; Diesel will populate with versioned folders
