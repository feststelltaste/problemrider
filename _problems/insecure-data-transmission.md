---
title: Insecure Data Transmission
description: Sensitive data transmitted without proper encryption or security controls,
  exposing it to interception and unauthorized access.
category:
- Security
- Security
related_problems:
- slug: silent-data-corruption
  similarity: 0.55
solutions:
- secret-management
- security-hardening-process
- authentication
- checksums
- prepared-statements
- privacy-by-design
- secure-protocols
- service-mesh
layout: problem
---

## Description

Insecure data transmission occurs when sensitive information is sent over networks without adequate encryption or security controls, making it vulnerable to interception, eavesdropping, and man-in-the-middle attacks. This includes transmitting data over unencrypted channels, using weak encryption methods, or failing to properly validate secure connections.

## Indicators ⟡

- Sensitive data transmitted over HTTP instead of HTTPS
- Applications accepting invalid or self-signed SSL certificates
- Weak encryption algorithms or protocols used for data transmission
- Authentication credentials sent in plain text
- Personal or financial information transmitted without encryption

## Symptoms ▲

- [Data Protection Risk](data-protection-risk.md)
<br/>  Transmitting data without encryption directly creates regulatory and legal data protection risks.
- [Authentication Bypass Vulnerabilities](authentication-bypass-vulnerabilities.md)
<br/>  Unencrypted transmission of credentials enables interception and replay attacks that bypass authentication.
- [Silent Data Corruption](silent-data-corruption.md)
<br/>  Man-in-the-middle attacks on unencrypted channels can modify data in transit without detection.
- [Regulatory Compliance Drift](regulatory-compliance-drift.md)
<br/>  Insecure data transmission causes the system to fall out of compliance with security regulations like PCI-DSS and GDPR.
- [Customer Dissatisfaction](customer-dissatisfaction.md)
<br/>  Security breaches resulting from insecure transmission erode customer trust and satisfaction.
## Causes ▼

- [Obsolete Technologies](obsolete-technologies.md)
<br/>  Legacy systems using outdated protocols may lack support for modern encryption standards.
- [Insufficient Design Skills](insufficient-design-skills.md)
<br/>  Developers lacking security design knowledge may fail to implement proper encryption for data in transit.
- [Configuration Chaos](configuration-chaos.md)
<br/>  Poor configuration management can lead to SSL/TLS being misconfigured or disabled in certain environments.
## Detection Methods ○

- **Network Traffic Analysis:** Monitor network communications for unencrypted sensitive data
- **SSL/TLS Configuration Testing:** Test encryption implementation and certificate validation
- **Mixed Content Detection:** Identify HTTPS pages loading HTTP resources
- **Protocol Analysis:** Analyze which encryption protocols and cipher suites are used
- **Certificate Validation Testing:** Test application behavior with invalid certificates

## Examples

An e-commerce website collects credit card information over HTTPS but submits it to the payment processor over HTTP. While the initial form appears secure to users, the actual payment data is transmitted in plain text, making it vulnerable to interception. Network analysis reveals that credit card numbers, expiration dates, and CVV codes are visible to anyone monitoring network traffic. Another example involves a mobile banking application that validates SSL certificates during development but disables certificate validation in production to avoid connectivity issues with load balancers. This makes the application vulnerable to man-in-the-middle attacks where attackers can intercept and modify banking transactions by presenting fake certificates that the application accepts without validation.