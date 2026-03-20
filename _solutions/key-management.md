---
title: Key Management
description: Establish procedures for the secure generation, distribution, and storage of cryptographic keys
category:
- Security
- Operations
quality_tactics_url: https://qualitytactics.de/en/security/key-management
problems:
- secret-management-problems
- insecure-data-transmission
- data-protection-risk
- password-security-weaknesses
- configuration-chaos
- regulatory-compliance-drift
layout: solution
---

## How to Apply ◆

> Legacy systems often store cryptographic keys in configuration files, source code, or shared directories with insufficient access controls. Proper key management ensures that keys are generated securely, distributed safely, stored appropriately, rotated regularly, and destroyed when no longer needed.

- Inventory all cryptographic keys in use: encryption keys, signing keys, API keys, SSH keys, TLS private keys, and database encryption keys. Document their purpose, location, age, and who has access to each.
- Migrate keys from insecure storage (source code, config files, environment variables in plaintext, shared drives) to a dedicated key management system (KMS) or hardware security module (HSM) that provides access control, auditing, and tamper protection.
- Implement automated key rotation on a defined schedule. Encryption keys should be rotated at least annually, with the ability to perform emergency rotation within hours if a compromise is suspected.
- Generate keys using cryptographically secure random number generators with appropriate key lengths for the algorithm in use. Many legacy keys were generated with insufficient entropy or outdated key sizes.
- Implement the principle of least privilege for key access: each application or service should have access only to the specific keys it needs, and no human should have access to production encryption keys without a documented, audited process.
- Establish key escrow or key recovery procedures for critical encryption keys so that encrypted data remains accessible even if the primary key holder is unavailable. Document the recovery process and test it periodically.
- Define key lifecycle procedures: generation, distribution, activation, use, rotation, deactivation, archival, and destruction. Each phase should have documented procedures and access controls.

## Tradeoffs ⇄

> Proper key management protects the foundation of all cryptographic operations, but it introduces operational complexity and requires dedicated infrastructure.

**Benefits:**

- Prevents key compromise from being the single point of failure that defeats all encryption, signing, and authentication controls.
- Enables key rotation without application downtime by supporting multiple active key versions during transition periods.
- Provides audit trails for key access and usage, supporting compliance requirements and forensic investigation.
- Reduces the risk of key loss causing permanent data loss by establishing backup and recovery procedures.

**Costs and Risks:**

- Key management infrastructure (KMS, HSM) requires investment, expertise, and high availability — a failed KMS can make encrypted data inaccessible.
- Key rotation requires application changes to support multiple key versions and graceful transition between old and new keys.
- Migrating keys from legacy storage to a KMS requires careful handling during the transition to avoid both key exposure and key loss.
- Overly complex key management processes can lead to workarounds where developers bypass the system and embed keys directly in code.

## How It Could Be

> The following scenarios illustrate how key management protects legacy system cryptography.

A legacy payment processing system uses a single AES encryption key to protect stored credit card numbers. The key is hardcoded in the application's configuration file, which is stored in version control. Every developer who has ever worked on the project has had access to this key, and it has never been rotated in 8 years. The team migrates the encryption key to a cloud KMS with access restricted to the production service account. They implement key rotation by adding a key version identifier to each encrypted record, allowing the system to decrypt with the old key and re-encrypt with the new key during a rolling migration. Going forward, the key is rotated automatically every 90 days, and the KMS audit log shows exactly which service accessed the key and when. No human has direct access to the plaintext key material.

A legacy email system uses PGP encryption for secure customer communications. The private key is stored on a single server's filesystem, protected only by file permissions. When the server's hard drive fails, the private key is lost, and the team discovers that no backup exists — all previously encrypted emails from customers are now permanently unreadable. After rebuilding the system with a new key pair, the team implements key management procedures: private keys are stored in a hardware security module with automatic backup to a geographically separate HSM, key recovery requires two-person authorization (split knowledge), and a key escrow copy is sealed and stored in a physical safe with documented access procedures. A quarterly test verifies that the escrowed key can successfully decrypt test data.
