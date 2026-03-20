---
title: Environment Variables for Configuration
description: Control configuration settings via environment variables
category:
- Operations
- Code
quality_tactics_url: https://qualitytactics.de/en/portability/environment-variables-for-configuration
problems:
- configuration-chaos
- hardcoded-values
- deployment-environment-inconsistencies
- configuration-drift
- environment-variable-issues
- secret-management-problems
- complex-deployment-process
layout: solution
---

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

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy Java application used separate properties files for each environment (dev.properties, staging.properties, prod.properties), committed to the repository with production database credentials. The team migrated to environment variable-based configuration using Spring's property resolution, which reads environment variables with fallback to a default properties file. They added startup validation that checked for all required variables and logged clear messages for missing ones. Production secrets were moved to a vault service and injected as environment variables by the deployment platform. This eliminated the security risk of credentials in source control and allowed the operations team to change database endpoints without developer involvement or code deployments.
