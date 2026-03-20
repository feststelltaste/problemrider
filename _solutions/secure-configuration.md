---
title: Secure Configuration
description: Deliver and operate systems with secure default settings
category:
- Security
- Operations
quality_tactics_url: https://qualitytactics.de/en/security/secure-configuration
problems:
- configuration-chaos
- configuration-drift
- deployment-environment-inconsistencies
- inadequate-configuration-management
- secret-management-problems
- error-message-information-disclosure
- legacy-configuration-management-chaos
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Create a configuration baseline that documents all security-relevant settings for each environment
- Automate configuration deployment using infrastructure-as-code tools to prevent manual drift
- Remove or disable all unnecessary services, ports, and default accounts from production systems
- Implement configuration scanning tools that detect deviations from the secure baseline
- Separate secrets from configuration files and store them in dedicated secret management systems
- Version-control configuration templates and require review for any changes
- Conduct regular configuration audits comparing running systems against the documented baseline

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Eliminates common attack vectors caused by misconfiguration
- Ensures consistency across development, staging, and production environments
- Provides auditability and traceability for configuration changes
- Reduces the time to deploy new environments securely

**Costs and Risks:**
- Legacy systems may have undocumented configuration dependencies that break when hardened
- Automating configuration for systems not designed for it can require significant tooling investment
- Strict configuration management can slow down troubleshooting when developers need temporary relaxed settings
- Some legacy components may not support externalized or automated configuration

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

An e-commerce company operating a legacy .NET application discovered during a security audit that their production servers had different configurations than staging, including enabled remote debugging and verbose error pages on two of five production nodes. The team created Ansible playbooks defining the secure configuration baseline and applied them across all environments. Automated weekly scans detected any drift within 24 hours. Configuration-related security findings in subsequent audits dropped from 11 to one.
