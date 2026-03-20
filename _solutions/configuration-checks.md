---
title: Configuration Checks
description: Document and regularly review security-relevant settings
category:
- Security
- Operations
quality_tactics_url: https://qualitytactics.de/en/security/configuration-checks
problems:
- configuration-drift
- configuration-chaos
- inadequate-configuration-management
- legacy-configuration-management-chaos
- regulatory-compliance-drift
- deployment-environment-inconsistencies
- secret-management-problems
layout: solution
---

## How to Apply ◆

> Legacy systems accumulate security misconfigurations over years of ad-hoc changes, patches, and personnel turnover. Configuration checks systematically document expected security settings and verify them regularly against actual system state.

- Create a security configuration baseline that documents all security-relevant settings: firewall rules, TLS versions, cipher suites, authentication parameters, logging levels, file permissions, database access controls, and service account privileges.
- Implement automated configuration scanning using tools like CIS Benchmarks, OpenSCAP, or custom scripts that compare actual system configuration against the documented baseline and report deviations.
- Schedule regular configuration audits (at minimum monthly) that run automated scans and produce reports of configuration drift from the security baseline. Treat deviations as findings that require investigation and remediation.
- Verify that default credentials have been changed for all system components, including databases, message brokers, administrative consoles, and monitoring tools. Legacy systems frequently retain factory-default passwords on internal components.
- Check that unnecessary services, ports, and features are disabled. Legacy systems often run services that were enabled during initial setup or debugging and never removed, expanding the attack surface unnecessarily.
- Review file and directory permissions to ensure that configuration files, log files, and data directories are only accessible to the users and processes that need them. Legacy systems commonly use overly permissive file permissions.
- Integrate configuration checks into the deployment pipeline so that new deployments are automatically verified against the security baseline before going live.

## Tradeoffs ⇄

> Configuration checks provide systematic visibility into security-relevant settings and catch drift before it becomes a vulnerability, but they require baseline definition and ongoing maintenance.

**Benefits:**

- Detects security misconfigurations before they can be exploited by comparing actual settings against a known-good baseline.
- Prevents configuration drift by identifying changes that were made without following the change management process.
- Supports compliance audits by providing documented evidence that security settings meet required standards.
- Reduces the risk of human error during system changes by automatically verifying that security-critical settings remain correct.

**Costs and Risks:**

- Creating the initial configuration baseline for a legacy system requires significant effort to identify and document all security-relevant settings.
- Automated scanning tools may produce false positives that require investigation, consuming security team time.
- The baseline must be updated whenever legitimate configuration changes are made, creating an ongoing maintenance burden.
- Configuration checks may not cover application-level security settings in custom legacy code, focusing primarily on infrastructure and platform configuration.

## How It Could Be

> The following scenarios illustrate how configuration checks detect and prevent security issues in legacy systems.

A legacy web application server runs with TLS 1.0 and 1.1 enabled alongside TLS 1.2, because the original configuration was never updated when these older protocols were deprecated. A vulnerability scanner discovers that the server is susceptible to the POODLE attack through TLS 1.0. The team implements a configuration baseline that specifies TLS 1.2 as the minimum protocol version with a defined set of strong cipher suites. Automated configuration checks run weekly and compare the actual TLS configuration against this baseline. When a system administrator reinstalls the web server software after a hardware migration and the default configuration re-enables TLS 1.0, the next configuration check flags the deviation within 24 hours, and the configuration is corrected before the system is exposed to the vulnerability.

A legacy database server has been running with the default administrative password for its management console for over five years. The console is accessible from the internal network, and anyone with network access can log in as the database administrator. A configuration check audit identifies this issue along with 15 other default-credential findings across the legacy infrastructure. The team rotates all default passwords, implements a quarterly credential rotation policy, and adds automated checks that verify no component is running with known default credentials. The configuration check report becomes a standard item in the quarterly security review, ensuring that new installations and upgrades do not reintroduce default credentials.
