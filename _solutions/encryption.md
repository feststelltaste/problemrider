---
title: Encryption
description: Encrypt data during transmission and storage
category:
- Security
quality_tactics_url: https://qualitytactics.de/en/security/encryption
problems:
- insecure-data-transmission
- data-protection-risk
- password-security-weaknesses
- regulatory-compliance-drift
- secret-management-problems
- error-message-information-disclosure
layout: solution
---

## How to Apply ◆

> Legacy systems frequently transmit and store sensitive data without encryption, relying on network perimeter security or physical access controls that are insufficient for modern threat models. Implementing encryption protects data both in transit and at rest.

- Audit all data transmission paths in the legacy system and identify unencrypted channels: HTTP connections, unencrypted database connections, plaintext file transfers (FTP), unencrypted inter-service communication, and plaintext email with sensitive attachments.
- Enable TLS 1.2 or later for all network communication. Disable older protocols (SSL 3.0, TLS 1.0, TLS 1.1) and configure strong cipher suites. For legacy systems that cannot support modern TLS, implement a TLS-terminating reverse proxy in front of the legacy component.
- Implement encryption at rest for databases containing sensitive data. Use Transparent Data Encryption (TDE) for database-level encryption or column-level encryption for specific sensitive fields (credit card numbers, Social Security numbers, health records).
- Encrypt backup files and archives, particularly when stored on removable media or transferred to off-site storage. Unencrypted backups are a common source of data breaches.
- Implement key management that separates encryption keys from encrypted data. Store keys in dedicated key management systems (KMS), hardware security modules (HSMs), or secret management services, never alongside the data they protect.
- Encrypt configuration files and environment variables that contain sensitive values (database passwords, API keys, service account credentials). Legacy systems commonly store these in plaintext.
- Implement field-level encryption for the most sensitive data elements, ensuring that even database administrators and system operators cannot access plaintext values without explicit key access.

## Tradeoffs ⇄

> Encryption protects data confidentiality even when other security controls fail, but it adds computational overhead, key management complexity, and can complicate debugging and monitoring.

**Benefits:**

- Protects data confidentiality even if the network is compromised, storage media is stolen, or database access controls are bypassed.
- Meets regulatory requirements (PCI DSS, HIPAA, GDPR) that mandate encryption of sensitive data in transit and at rest.
- Provides defense in depth — even if an attacker gains access to the system, encrypted data remains protected without the decryption keys.
- Enables safe use of cloud storage and third-party infrastructure for legacy system data by ensuring data remains protected regardless of the storage provider's security.

**Costs and Risks:**

- Encryption adds CPU overhead for every data operation, which can impact performance on legacy hardware that was not sized for cryptographic processing.
- Key management failures (lost keys, compromised keys, unavailable KMS) can result in permanent data loss or widespread exposure.
- Encrypted data cannot be searched, indexed, or processed without decryption, which may require application changes and impacts query performance.
- Debugging and monitoring become more difficult when data is encrypted, as log analysis and data inspection require additional steps.

## How It Could Be

> The following scenarios illustrate how encryption protects legacy system data.

A legacy HR system transmits employee salary data between the main application server and a reporting server over an unencrypted HTTP connection on the internal network. A network security audit reveals that anyone with access to the internal network can capture this traffic using standard packet sniffing tools. The team implements TLS for the connection between the two servers, deploys a reverse proxy with TLS termination in front of the legacy application (which does not natively support HTTPS), and adds TDE to the reporting database. Additionally, they encrypt the nightly data export files that are transferred to the payroll provider, replacing an unencrypted FTP transfer with SFTP. The entire network now carries only encrypted traffic, and a subsequent penetration test confirms that captured network packets reveal no readable sensitive data.

A legacy customer database stores credit card numbers in plaintext to support a recurring billing feature built 10 years ago. A PCI DSS audit identifies this as a critical finding requiring immediate remediation. The team implements column-level encryption using AES-256 for the credit card number field, stores encryption keys in a dedicated KMS with access restricted to the billing service account, and modifies the application to decrypt values only at the point of use during billing operations. They also implement tokenization for display purposes — the application shows only the last four digits of the card number in the user interface. The tokenized display and encrypted storage reduce the PCI DSS compliance scope from the entire application to just the billing service, simplifying future audits significantly.
