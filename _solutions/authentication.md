---
title: Authentication
description: Verify the identity of users and systems
category:
- Security
quality_tactics_url: https://qualitytactics.de/en/security/authentication
problems:
- authentication-bypass-vulnerabilities
- password-security-weaknesses
- session-management-issues
- data-protection-risk
- authorization-flaws
- error-message-information-disclosure
- insecure-data-transmission
layout: solution
---

## How to Apply ◆

> Legacy systems frequently rely on outdated authentication mechanisms — plaintext passwords, weak hashing algorithms, or custom-built authentication logic with known vulnerabilities. Modernizing authentication is a foundational step toward securing any legacy system.

- Audit the existing authentication mechanism to identify weaknesses: plaintext or weakly hashed passwords (MD5, SHA-1 without salt), hardcoded credentials, session tokens with predictable patterns, and authentication bypass paths.
- Replace custom authentication implementations with well-tested authentication libraries or frameworks. Legacy systems often contain hand-rolled authentication code with subtle vulnerabilities that standard libraries have already addressed.
- Implement strong password hashing using bcrypt, scrypt, or Argon2id with appropriate work factors. Migrate existing password hashes by re-hashing on next successful login — users authenticate with the old hash, and their password is immediately re-hashed with the modern algorithm.
- Add multi-factor authentication (MFA) for administrative accounts and sensitive operations. Even if the full user base cannot immediately adopt MFA, protecting privileged accounts eliminates the highest-risk authentication targets.
- Implement account lockout or progressive delays after failed login attempts to prevent brute-force attacks. Ensure lockout policies do not enable denial-of-service by locking out legitimate users — use CAPTCHA or temporary delays rather than permanent lockout.
- Secure session management by generating cryptographically random session tokens, setting appropriate expiration times, and invalidating sessions on logout and password change. Use secure, HttpOnly, SameSite cookie attributes.
- Eliminate generic error messages in authentication flows ("invalid username or password" rather than "username not found" or "incorrect password") to prevent user enumeration.

## Tradeoffs ⇄

> Strong authentication prevents unauthorized access and is a prerequisite for all other security controls, but it adds friction to the user experience and complexity to system integration.

**Benefits:**

- Prevents unauthorized access by verifying that users and systems are who they claim to be, forming the foundation of all access control.
- Protects against credential-based attacks (brute force, credential stuffing, phishing) through modern hashing, MFA, and lockout policies.
- Provides accountability by linking every action in the system to an authenticated identity.
- Enables compliance with security standards and regulations that mandate strong authentication controls.

**Costs and Risks:**

- Stronger authentication adds user friction (MFA steps, password complexity requirements, session timeouts), which can reduce adoption and productivity.
- Migrating legacy authentication mechanisms requires careful planning to avoid locking out users during the transition.
- Integration with external systems that rely on the legacy authentication mechanism may break when authentication is modernized.
- Account lockout mechanisms can be exploited for denial-of-service if not implemented with rate limiting rather than hard lockouts.

## How It Could Be

> The following scenarios illustrate how authentication modernization addresses security gaps in legacy systems.

A legacy enterprise application stores user passwords as unsalted MD5 hashes in the database. A database backup is accidentally exposed, and an attacker uses rainbow tables to crack 85% of the passwords within hours. The team implements a migration strategy: they add a new bcrypt password column to the user table. When a user logs in successfully with their MD5-hashed password, the system transparently re-hashes their plaintext password with bcrypt and stores the new hash. Users who have not logged in within 90 days are forced to reset their passwords. After six months, 95% of active accounts have been migrated to bcrypt, and the legacy MD5 column is dropped. The team also adds MFA via TOTP for all administrative accounts, preventing future credential compromises from granting administrative access.

A legacy supply chain management system uses session IDs that are sequential integers, making session hijacking trivial for anyone who can observe or guess a valid session ID. The team replaces the session management with cryptographically random 256-bit session tokens, sets session expiration to 30 minutes of inactivity, and implements session binding to the client's IP address and user agent. They also add a session invalidation endpoint that the application calls on logout. After deployment, the security team's penetration test confirms that session prediction and hijacking are no longer feasible, and the authenticated session is properly terminated on logout.
