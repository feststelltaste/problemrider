---
title: Secret Management
description: Securely managing application secrets using dedicated vaults and rotation policies
category:
- Security
- Operations
quality_tactics_url: https://qualitytactics.de/en/security/secret-management/
problems:
- secret-management-problems
- hardcoded-values
- environment-variable-issues
- configuration-chaos
- data-protection-risk
- insecure-data-transmission
- authentication-bypass-vulnerabilities
- error-message-information-disclosure
layout: solution
---

## How to Apply ◆

> Legacy systems routinely store credentials in source code, flat configuration files, or shared wikis — replacing those practices with centralized secret management is the highest-impact security step a modernization effort can take.

- Audit the codebase and its full git history with tools like TruffleHog or GitGuardian to locate all secrets already committed; assume every found secret is compromised and rotate it immediately after migrating it to the vault.
- Introduce a dedicated secret management tool — HashiCorp Vault, AWS Secrets Manager, or Azure Key Vault — as infrastructure before touching the application, so the migration destination exists and is stable.
- Migrate secrets service by service rather than all at once; start with the most sensitive credentials (production database passwords, payment API keys) and work outward to lower-risk secrets.
- Use environment variable bridges such as Kubernetes External Secrets or `envconsul` for legacy services that cannot be quickly refactored to call the vault API directly — this decouples migration of the storage layer from migration of the application code.
- Establish automated rotation schedules from day one; for legacy databases that have had the same password for years, treat the first forced rotation as a drill to confirm that all consumers are reading credentials dynamically.
- Encode least-privilege access policies in the vault: each service can read only the secrets it genuinely needs, preventing a single compromised component from exposing all credentials.
- Add pre-commit hooks with `detect-secrets` or `git-secrets` to prevent re-introduction of hardcoded credentials by developers accustomed to the old pattern.
- Plan for vault unavailability from the start — legacy systems often lack graceful degradation; implement encrypted in-memory caching of retrieved secrets with short TTLs so a brief vault outage does not cause an immediate production failure.

## Tradeoffs ⇄

> Secret management eliminates the most common credential-exposure vectors in legacy systems, but it introduces the vault itself as a new critical infrastructure dependency that must be operated with high reliability.

**Benefits:**

- Removes credentials from source code, configuration files, and CI/CD pipeline logs where they accumulate invisibly over years of legacy development.
- Enables automated rotation, eliminating the operational fear of changing credentials that are hardcoded in dozens of places — a paralysis common in long-lived systems.
- Provides a complete audit trail of secret access, supporting compliance audits (PCI DSS, HIPAA, SOC 2) that legacy systems often fail due to lack of any such record.
- Short-lived dynamic credentials reduce the blast radius of a compromised service account — credentials expire in minutes rather than remaining valid for years.
- Centralizes governance so that when staff leave or a vendor key is compromised, revocation and rotation can happen in one place rather than requiring a hunt through every configuration file.

**Costs and Risks:**

- The vault becomes a hard dependency for application startup; if it is unavailable during deployment, services cannot initialize — this is a new category of outage risk that did not exist when credentials were baked into config files.
- Migrating a large legacy codebase with dozens of services and hundreds of credentials is a multi-month effort that competes with feature work and cannot be done entirely over a weekend.
- Development teams unfamiliar with vault APIs need to learn new patterns; the temptation to revert to environment variables or config files must be actively managed through code review and pre-commit hooks.
- A vault compromise exposes all secrets simultaneously — the concentration of credentials that makes management easier also makes the vault a high-value target requiring its own rigorous protection.
- Legacy systems may have credentials shared across multiple applications with no clear ownership; untangling this sharing to apply per-service policies requires careful analysis before migration can proceed.

## How It Could Be

> The following scenarios illustrate how secret management surfaces in real legacy system modernization efforts.

A financial services company running a ten-year-old Java EE application found database passwords in plaintext inside `application.properties` files committed to a Subversion repository. When they began migrating the repository to Git and making it accessible to a wider team, a security review revealed that the production Oracle credentials had been in version control since the initial project commit. The team deployed HashiCorp Vault, migrated the credentials, and used `envconsul` to inject secrets as environment variables so the aging application could consume them without code changes. Rotating the Oracle password — something that had not been done in eight years — was then completed without incident because only the vault entry needed updating.

A government agency with dozens of independently deployed batch processing scripts discovered that each script had its own hardcoded API key for a third-party data provider. When the provider changed their key management policy and revoked old keys, the agency needed three days to identify and update every affected script. After the incident, they adopted AWS Secrets Manager and rewrote the scripts to fetch the API key at runtime. The next forced rotation took fifteen minutes: update one entry in Secrets Manager, and all scripts pick up the new key automatically on their next run.

A retail company operating a shared monolithic application across multiple business units had a single shared database user with full read-write access used by every component. When a developer accidentally logged the connection string during a debugging session and the log file was later included in a support package, the company had to treat the credential as compromised. Rotating a credential embedded in that many places caused a four-hour coordination effort. The incident drove the adoption of Azure Key Vault with per-service managed identities, so each component now has its own credential with only the permissions it needs — limiting the damage any future exposure could cause.
