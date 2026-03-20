---
title: Least Privilege
description: Equip users and processes with only the minimal necessary rights
category:
- Security
quality_tactics_url: https://qualitytactics.de/en/security/least-privilege
problems:
- authorization-flaws
- data-protection-risk
- authentication-bypass-vulnerabilities
- regulatory-compliance-drift
- password-security-weaknesses
- poorly-defined-responsibilities
layout: solution
---

## How to Apply ◆

> Legacy systems commonly grant excessive permissions to users, service accounts, and processes — often because it was easier than determining the minimum required access. The principle of least privilege restricts every entity to only the permissions necessary for its specific function.

- Audit all user accounts and their permissions. Identify accounts with administrative or elevated privileges and verify that each privilege is justified by the account's current role. Legacy systems often have dozens of accounts with full administrative access.
- Review service account permissions and reduce them to the minimum required. Legacy application service accounts often run as root/administrator or have full database access when they only need access to specific tables or operations.
- Implement database-level least privilege by creating application-specific database users with granular permissions. A reporting service should have read-only access; a transaction processor should have read-write access only to transaction-related tables.
- Remove default accounts and permissions that ship with legacy software, databases, and middleware. Default accounts with known passwords are a primary attack vector.
- Implement just-in-time (JIT) privilege elevation for administrative tasks: administrators use standard accounts for daily work and elevate to privileged accounts only when performing administrative operations, with automatic expiration.
- Apply least privilege to file system permissions: application processes should only have access to the directories they need, configuration files should be readable only by the application user, and log directories should be writable only by the logging process.
- Review and restrict network-level access so that each component can only communicate with the specific endpoints it needs, rather than having unrestricted network access within the legacy system's segment.

## Tradeoffs ⇄

> Least privilege limits the damage from compromised accounts and reduces the attack surface, but it requires detailed analysis of actual access needs and ongoing maintenance.

**Benefits:**

- Limits the blast radius of compromised credentials — an attacker who compromises a restricted account can only access the resources that account is permitted to reach.
- Reduces the risk of accidental damage from administrative errors by ensuring that routine operations run with limited permissions.
- Supports compliance with security standards and regulations that mandate access control based on business need.
- Makes security auditing more effective by creating a clear, documented mapping between roles and permissions.

**Costs and Risks:**

- Determining the minimum required permissions for legacy applications requires extensive testing, as undocumented dependencies on elevated permissions are common.
- Reducing permissions too aggressively can break functionality, particularly in legacy systems where the actual permission requirements are not well documented.
- Least privilege requires ongoing enforcement as the system evolves — new features may require new permissions, and old permissions may need to be revoked.
- Just-in-time privilege elevation adds friction to administrative workflows and requires supporting infrastructure.

## Examples

> The following scenarios illustrate how least privilege reduces risk in legacy systems.

A legacy web application runs under a service account that has full administrative access to the SQL Server database, including the ability to create and drop tables, modify schema, and access all databases on the server. When a SQL injection vulnerability is exploited, the attacker uses the application's database permissions to enumerate all databases, extract data from the HR database (which the application does not use), and drop audit tables to cover their tracks. After implementing least privilege, the application's database user has SELECT, INSERT, and UPDATE permissions on only the 12 tables it actually uses, with no access to other databases, no DDL permissions, and no ability to modify audit tables. When a subsequent SQL injection vulnerability is discovered, the attacker's access is confined to the application's own tables, and the audit trail remains intact, enabling rapid detection and response.

A legacy Linux application server has 15 user accounts, 8 of which have sudo access with no restrictions (equivalent to root). Investigation reveals that 6 of these users added themselves to the sudoers file years ago for a one-time debugging task and never removed the access. The team implements a least-privilege model: sudo access is removed from all 6 unnecessary accounts, the remaining 2 administrative accounts have sudo access restricted to specific commands needed for their roles, and a JIT privilege elevation system requires administrators to request temporary elevated access with a business justification and automatic 4-hour expiration. Root login is disabled entirely, and all privileged actions are logged to a centralized, tamper-evident audit system.
