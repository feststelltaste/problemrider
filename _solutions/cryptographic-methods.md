---
title: Cryptographic Methods
description: Use proven and standardized algorithms and protocols for cryptographic functions
category:
- Security
quality_tactics_url: https://qualitytactics.de/en/security/cryptographic-methods
problems:
- insecure-data-transmission
- password-security-weaknesses
- data-protection-risk
- regulatory-compliance-drift
- secret-management-problems
- authentication-bypass-vulnerabilities
layout: solution
---

## How to Apply ◆

> Legacy systems often use deprecated cryptographic algorithms (DES, MD5, SHA-1, RC4) or custom-built encryption schemes that provide false security. Modernizing cryptographic methods ensures that data protection relies on proven, vetted algorithms.

- Audit all cryptographic usage in the legacy system: password hashing, data encryption at rest and in transit, digital signatures, token generation, and random number generation. Identify which algorithms and key sizes are in use.
- Replace deprecated algorithms with current standards: AES-256 for symmetric encryption, RSA-2048+ or ECDSA P-256+ for asymmetric operations, SHA-256 or SHA-3 for hashing, and bcrypt/Argon2id for password hashing.
- Use established cryptographic libraries (OpenSSL, libsodium, Bouncy Castle) rather than custom implementations. Even well-known algorithms can be implemented incorrectly, leading to vulnerabilities that are difficult to detect.
- Implement proper random number generation using cryptographically secure pseudorandom number generators (CSPRNGs) for all security-sensitive operations: session tokens, API keys, nonces, initialization vectors, and password salts.
- Migrate from ECB (Electronic Codebook) mode to authenticated encryption modes like AES-GCM or ChaCha20-Poly1305 that provide both confidentiality and integrity protection. ECB mode, commonly found in legacy systems, reveals patterns in encrypted data.
- Plan and execute crypto-agility: design the system so that cryptographic algorithms can be replaced without major code changes. This prepares for future algorithm deprecation and eventual post-quantum cryptography migration.
- Ensure all cryptographic operations use proper key derivation, padding, and initialization vector handling. Many legacy vulnerabilities stem from these implementation details rather than the core algorithm itself.

## Tradeoffs ⇄

> Standardized cryptographic methods provide well-vetted data protection that withstands known attacks, but migration from legacy algorithms requires careful planning to avoid data loss or access disruption.

**Benefits:**

- Protects data against known attacks that exploit weaknesses in deprecated algorithms, maintaining confidentiality and integrity.
- Ensures compliance with current security standards and regulations that mandate specific cryptographic requirements.
- Leverages decades of cryptographic research and peer review rather than relying on security through obscurity.
- Enables interoperability with modern systems and protocols that require current cryptographic standards.

**Costs and Risks:**

- Migrating encrypted data from legacy algorithms to modern ones requires decrypting with the old algorithm and re-encrypting with the new one, creating a window of exposure.
- Stronger algorithms may have higher computational overhead, impacting performance on legacy hardware.
- Cryptographic migration can break integrations with external systems that expect the legacy algorithm or data format.
- Incorrect implementation of even strong algorithms (wrong mode, predictable IVs, improper padding) can negate their security benefits.

## Examples

> The following scenarios illustrate how upgrading cryptographic methods strengthens legacy system security.

A legacy healthcare system stores patient Social Security numbers encrypted with DES (56-bit key), which was standard when the system was built in 1998. Modern hardware can brute-force DES in hours. The team implements a rolling migration: they add a new AES-256-GCM encrypted column, write a batch process that decrypts each value with DES and re-encrypts it with AES-256-GCM, and updates the application to read from the new column. The migration runs during maintenance windows over two weeks, processing 2 million records. After verification, the DES-encrypted column is securely wiped. The application is also updated to use a key management service rather than a hardcoded encryption key embedded in the source code.

A legacy banking application generates session tokens using Java's `Math.random()`, which is not cryptographically secure and produces predictable sequences. A security researcher demonstrates that by observing a few hundred tokens, they can predict future tokens with high accuracy, enabling session hijacking. The team replaces `Math.random()` with `SecureRandom` using the platform's native CSPRNG, and increases the token length from 32 bits to 256 bits. They also add token binding to the client's TLS session to prevent token replay from different connections. After the fix, penetration testing confirms that token prediction is computationally infeasible.
