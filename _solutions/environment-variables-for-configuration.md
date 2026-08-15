---
title: Environment Variables for Configuration
description: Control configuration settings via environment variables
category:
- Operations
- Code
problems:
- configuration-chaos
- hardcoded-values
- deployment-environment-inconsistencies
- configuration-drift
- environment-variable-issues
- secret-management-problems
- complex-deployment-process
layout: solution
related_solutions:
- slug: externalized-configuration
  similarity: 0.85
- slug: platform-independent-configuration-management
  similarity: 0.75
- slug: platform-independent-configuration-files
  similarity: 0.75
- slug: virtual-development-environments
  similarity: 0.7
- slug: secure-configuration
  similarity: 0.7
- slug: environment-parity
  similarity: 0.7
---

## Description

Environment variables for configuration is the practice of externalizing values that differ between deployment targets — database URLs, API keys, feature flags, service endpoints — into the process environment rather than compiling or bundling them into the application artifact, following the twelve-factor app principle that configuration should vary by deployment while code does not. Legacy applications frequently hardcode such values directly in source files or maintain a separate, checked-in configuration file per environment, which both couples the build to a specific target and risks leaking production credentials into version control. Reading configuration from environment variables at startup, with validation that fails fast when required values are missing, decouples the build from the deployment target: the same artifact can move from development through staging to production unchanged. This is particularly valuable during legacy modernization because it removes one of the recurring causes of environment-specific defects and creates the seam needed for containerized or cloud-native deployment, where injecting environment variables is the native configuration mechanism. The approach has limits, however: it handles flat key-value settings well but becomes awkward for hierarchical configuration, and because any process in the same environment can typically read these variables, secrets still need additional protection such as a dedicated vault.

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Identify all configuration values that differ between environments: database URLs, API keys, feature flags, service endpoints
- Replace hardcoded configuration values and environment-specific config files with environment variable lookups
- Provide sensible defaults for development environments so the application works without explicit configuration
- Use a configuration library that supports environment variables with fallback to config files for backward compatibility
- Document all required environment variables with their purpose, format, and example values
- Validate environment variables at application startup to fail fast with clear error messages if required values are missing
- Use .env files for local development while deploying with actual environment variables in production

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Enables the same application artifact to run in any environment without rebuilding
- Separates configuration from code, following twelve-factor app principles
- Simplifies secret management by keeping sensitive values out of source control
- Makes configuration changes possible without redeployment
- Works naturally with containerization and cloud platform configuration mechanisms

**Costs and Risks:**
- Environment variables are flat key-value pairs, making complex hierarchical configuration awkward
- Typos in variable names cause silent failures unless validation is implemented
- Large numbers of environment variables become difficult to manage without tooling
- Environment variables are visible to all processes in the same environment, posing a security risk for secrets
- Legacy applications with deeply embedded configuration loading may require significant refactoring

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy Java application used separate properties files for each environment (dev.properties, staging.properties, prod.properties), committed to the repository with production database credentials. The team migrated to environment variable-based configuration using Spring's property resolution, which reads environment variables with fallback to a default properties file. They added startup validation that checked for all required variables and logged clear messages for missing ones. Production secrets were moved to a vault service and injected as environment variables by the deployment platform. This eliminated the security risk of credentials in source control and allowed the operations team to change database endpoints without developer involvement or code deployments.
