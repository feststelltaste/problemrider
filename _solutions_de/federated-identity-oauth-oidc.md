---
title: Federated Identity (OAuth/OIDC)
description: Delegation der Authentifizierung an vertrauenswürdige externe Identitätsanbieter.
category:
- Security
- Architecture
problems:
- authentication-bypass-vulnerabilities
- password-security-weaknesses
- session-management-issues
- data-protection-risk
- vendor-lock-in
- difficult-developer-onboarding
- technology-lock-in
layout: solution
lang: de
en_slug: federated-identity
related_solutions:
- slug: authentication
  similarity: 0.75
- slug: two-factor-authentication
  similarity: 0.75
- slug: secret-management
  similarity: 0.7
- slug: cryptographic-methods
  similarity: 0.7
- slug: authorization
  similarity: 0.7
- slug: api-security
  similarity: 0.65
---

## Description

Federated Identity delegiert die Authentifizierung an einen spezialisierten, vertrauenswürdigen externen Identitätsanbieter mittels OAuth 2.0 oder OpenID Connect, statt dass jede Anwendung ihre eigene Nutzerdatenbank, Passwortspeicherung und Login-Logik pflegt, typischerweise durch Platzierung eines Authentifizierungs-Proxys vor der Legacy-Anwendung, der nicht authentifizierte Anfragen abfängt, zum Identitätsanbieter umleitet und die Ansprüche des zurückgegebenen Tokens in das bestehende Nutzer- und Autorisierungsmodell der Anwendung übersetzt. Legacy-Systeme sammeln häufig über viele Jahre gewachsenen eigenen, maßgeschneiderten Authentifizierungscode an, oft mit schwachem Passwort-Hashing, ohne Mehrfaktor-Authentifizierung und mit inkonsistenter Durchsetzung über die vielen Anwendungen hinweg, die eine Organisation betreibt, jede mit einem separaten, isolierten Satz an Zugangsdaten. Die Einführung eines Authentifizierungs-Proxys oder einer Middleware-Schicht erlaubt es, Federation zu übernehmen, ohne den Authentifizierungscode der Legacy-Anwendung direkt zu ändern, und weil Passwortspeicherung und MFA-Durchsetzung zu einem einzigen, zweckgebauten Identitätsanbieter wandern, beseitigt dies eine ganze Kategorie von Sicherheitsrisiko, die zuvor in jeder Legacy-Anwendung separat korrekt repliziert werden musste. Die dadurch entstehende Abhängigkeit ist jedoch beträchtlich: Der Identitätsanbieter wird zu einem Single Point of Failure für die Authentifizierung über jede föderierte Anwendung hinweg, die Integration von OAuth/OIDC mit Legacy-eigener Authentifizierung kann erhebliche Middleware-Entwicklung erfordern, und Legacy-Autorisierungslogik, die eng an das alte Nutzermodell gekoppelt ist, kann die zugrundeliegende Identitätsmigration selbst zu einem langen Projekt machen.

## How to Apply ◆

> Legacy-Systeme pflegen häufig eigene Nutzerdatenbanken mit maßgeschneiderter Authentifizierungslogik, was Sicherheitsrisiken durch Passwortspeicherung, inkonsistente Authentifizierungsrichtlinien und die Last der Identitätsverwaltung schafft. Federated Identity delegiert die Authentifizierung an spezialisierte, vertrauenswürdige Identitätsanbieter.

- Bewerten Sie den aktuellen Authentifizierungsmechanismus des Legacy-Systems und identifizieren Sie einen Migrationspfad zu OAuth 2.0 oder OpenID Connect (OIDC). Viele Legacy-Systeme nutzen formularbasierten Login mit Session-Cookies, der mit einem Authentifizierungs-Proxy umhüllt werden kann, der den OAuth/OIDC-Flow abwickelt.
- Stellen Sie einen Identitätsanbieter (IdP) bereit — entweder eine organisatorische SSO-Lösung (Azure AD, Okta, Keycloak) oder einen selbst gehosteten OIDC-Anbieter — und konfigurieren Sie ihn als maßgebliche Quelle für Nutzeridentitäten.
- Implementieren Sie einen Authentifizierungs-Proxy oder eine Middleware-Schicht vor der Legacy-Anwendung, der nicht authentifizierte Anfragen abfängt, zum IdP umleitet und die zurückgegebenen Tokens verarbeitet. Dieser Ansatz ermöglicht Federation, ohne den Authentifizierungscode der Legacy-Anwendung zu ändern.
- Bilden Sie föderierte Identitätsattribute (Rollen, Gruppen, E-Mail) auf das interne Nutzermodell des Legacy-Systems ab. Diese Zuordnungsschicht übersetzt zwischen den Token-Ansprüchen des IdP und der Autorisierungsstruktur der Legacy-Anwendung.
- Implementieren Sie Token-Validierung für den API-Zugriff: Ersetzen Sie Legacy-API-Key- oder Basic-Auth-Mechanismen durch OAuth-2.0-Bearer-Tokens, die gegen den Token-Introspection-Endpunkt des IdP oder durch Verifikation der JWT-Signaturen validiert werden.
- Planen Sie eine Nutzermigration von der Legacy-Nutzerdatenbank zum föderierten Identitätsanbieter. Dies kann schrittweise geschehen — erlauben Sie Login über beide Mechanismen während einer Übergangsphase, deaktivieren Sie dann die Legacy-Authentifizierung.
- Implementieren Sie Single Sign-Out, damit das Beenden einer Session am IdP auch Sessions in der Legacy-Anwendung beendet und verwaiste Sessions verhindert.

## Tradeoffs ⇄

> Federated Identity zentralisiert die Authentifizierung bei einem spezialisierten Anbieter, was Sicherheit und Nutzererfahrung verbessert, aber eine Abhängigkeit vom Identitätsanbieter einführt und Integrationsarbeit erfordert.

**Vorteile:**

- Beseitigt die Notwendigkeit, Passwörter im Legacy-System zu speichern und zu verwalten, was eine wesentliche Kategorie von Sicherheitsrisiko entfernt.
- Ermöglicht Single Sign-On (SSO) über das Legacy-System und andere Anwendungen hinweg, was die Nutzererfahrung verbessert und Passwortmüdigkeit verringert.
- Zentralisiert Authentifizierungsrichtlinien (Passwortkomplexität, MFA-Anforderungen, Sperrrichtlinien) beim Identitätsanbieter und sichert konsistente Durchsetzung.
- Vereinfacht Nutzer-Provisionierung und -Deprovisionierung durch Verwaltung von Identitäten an einem einzigen Ort statt in jeder Legacy-Anwendung separat.

**Kosten und Risiken:**

- Der Identitätsanbieter wird zu einer kritischen Abhängigkeit — ist er nicht verfügbar, können sich Nutzer nicht beim Legacy-System authentifizieren.
- Die Integration von OAuth/OIDC mit Legacy-Anwendungen, die maßgeschneiderte Authentifizierung nutzen, kann erhebliche Middleware-Entwicklung erfordern.
- Token-Handling führt neue Sicherheitsüberlegungen ein (Token-Speicherung, Refresh-Token-Rotation, Token-Widerruf), die das Team korrekt umsetzen muss.
- Legacy-Anwendungen mit eingebetteten Nutzerdatenbanken haben möglicherweise Autorisierungslogik, die eng an ihr Nutzermodell gekoppelt ist, was die Identitätsmigration komplex macht.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie Federated Identity die Authentifizierung von Legacy-Systemen modernisiert.

Ein Unternehmen betreibt 12 Legacy-Anwendungen, jede mit eigener Nutzerdatenbank und Login-Seite. Nutzer pflegen separate Zugangsdaten für jede Anwendung, was zu weitverbreiteter Passwort-Wiederverwendung, häufigen Passwort-Reset-Anfragen und inkonsistenter Authentifizierungssicherheit über Anwendungen hinweg führt. Das Team stellt Keycloak als zentralisierten OIDC-Identitätsanbieter bereit und konfiguriert Authentifizierungs-Proxys vor jeder Legacy-Anwendung. Nutzer authentifizieren sich nun einmal über Keycloak und erhalten Zugriff auf alle Anwendungen via SSO. Die Nutzerdatenbanken der Legacy-Anwendungen bleiben für die Autorisierungszuordnung erhalten, speichern aber keine Passwörter mehr. MFA wird zentral bei Keycloak für alle Anwendungen durchgesetzt. Passwort-Reset-Anfragen sinken um 85 %, und die konsistente MFA-Anforderung beseitigt die Credential-Stuffing-Angriffe, die zuvor auf die schwächste Anwendung zielten.

Eine Legacy-SaaS-Plattform verwaltet 50.000 Nutzerkonten mit einem maßgeschneiderten Authentifizierungssystem, das Passwörter als gesalzene SHA-1-Hashes speichert und keine MFA-Fähigkeit bietet. Enterprise-Kunden verlangen SAML/OIDC-Federation, damit ihre Mitarbeiter Unternehmenszugangsdaten nutzen können. Das Team integriert die Plattform mit einem OIDC-kompatiblen Identitätsbroker, der sowohl SAML-Federation für Enterprise-Kunden als auch Social Login für Einzelnutzer unterstützt. Enterprise-Kunden verbinden ihre Unternehmensidentitätsanbieter, wodurch die Notwendigkeit separater Plattform-Zugangsdaten entfällt. Einzelnutzer können zu Social Login migrieren oder weiterhin lokale Konten nutzen (die auf bcrypt-Hashing aufgerüstet werden). Die Angriffsfläche der Plattform schrumpft erheblich, da sie für föderierte Nutzer keine Passwörter mehr verwaltet.
