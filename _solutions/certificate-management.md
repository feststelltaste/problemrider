---
title: Certificate Management
description: Managing X.509 certificate lifecycles including PKI, revocation, and pinning
category:
- Security
- Operations
quality_tactics_url: https://qualitytactics.de/en/security/certificate-management
problems:
- insecure-data-transmission
- system-outages
- configuration-drift
- secret-management-problems
- operational-overhead
- poor-operational-concept
- regulatory-compliance-drift
layout: solution
---

## How to Apply ◆

> Legacy systems often use certificates that are manually managed, expired, or self-signed without proper lifecycle tracking. Certificate management establishes systematic processes for the entire certificate lifecycle from issuance to renewal and revocation.

- Inventory all certificates in use across the legacy system: TLS/SSL certificates for web servers, mutual TLS certificates for service-to-service communication, code signing certificates, and client authentication certificates. Record their expiration dates, issuers, and locations.
- Implement automated certificate monitoring that alerts at least 30, 14, and 7 days before any certificate expires. Certificate expiration is one of the most common causes of unexpected legacy system outages.
- Automate certificate renewal using protocols like ACME (Let's Encrypt) where possible. For internal certificates, implement automated issuance through an internal Certificate Authority with programmatic enrollment.
- Establish a certificate revocation process for compromised keys. Configure applications to check Certificate Revocation Lists (CRLs) or use OCSP (Online Certificate Status Protocol) to verify certificate validity.
- Implement certificate pinning for critical service-to-service connections where man-in-the-middle attacks are a concern. Pin to the public key rather than the full certificate to simplify rotation.
- Store private keys securely using hardware security modules (HSMs), dedicated secret management systems, or encrypted key stores. Never store private keys in source code repositories, configuration files, or shared file systems.
- Document the certificate architecture including the chain of trust, renewal procedures, and emergency revocation procedures so that certificate issues can be resolved quickly by any qualified team member.

## Tradeoffs ⇄

> Proper certificate management prevents outages from expired certificates and strengthens transport security, but it adds operational complexity and requires automation investment.

**Benefits:**

- Eliminates certificate expiration as a cause of system outages by providing visibility into certificate lifecycles and automated renewal.
- Strengthens transport security by ensuring certificates are properly issued, validated, and revoked when compromised.
- Supports compliance with security standards that require proper PKI management and certificate handling.
- Reduces emergency response burden by converting certificate issues from urgent incidents to routine operational procedures.

**Costs and Risks:**

- Automating certificate management for legacy systems that use custom SSL configurations or embedded certificate stores can be technically challenging.
- Certificate pinning, while more secure, makes certificate rotation more complex and can cause outages if not coordinated correctly.
- Internal PKI infrastructure (Certificate Authority, CRL distribution, OCSP responder) introduces additional components that must be maintained and kept highly available.
- Over-rotation of certificates without proper testing can break integrations with external systems that cache or pin the previous certificate.

## How It Could Be

> The following scenarios illustrate how certificate management prevents outages and strengthens security in legacy systems.

A legacy e-commerce platform experiences an unexpected outage at 2 AM on a Saturday when its TLS certificate expires. The certificate was manually installed three years ago by an engineer who has since left the company, and no one was tracking its expiration. Customers see browser security warnings and abandon purchases for 6 hours until an engineer manually obtains and installs a new certificate. After the incident, the team deploys a certificate monitoring system that discovers 23 additional certificates across the legacy infrastructure, three of which are expiring within 30 days. They implement automated renewal via ACME for public-facing certificates and automated alerting for internal certificates that require manual renewal. Over the following year, zero certificate-related outages occur.

A legacy banking system uses mutual TLS for communication between its core banking engine and the payment gateway. The certificates are self-signed with a 10-year validity period and use 1024-bit RSA keys, which are considered insecure by current standards. The team implements a phased certificate migration: they deploy a new internal CA with 2048-bit RSA keys (with a plan to move to ECDSA), issue new certificates for both services, configure a transition period where both old and new certificates are accepted, and then remove the old certificates after verifying all connections use the new ones. They also implement automated certificate rotation on a 90-day cycle, ensuring that compromised keys have limited exposure windows.
