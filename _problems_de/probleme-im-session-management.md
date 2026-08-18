---
title: Probleme im Session-Management
description: Schlechte Session-Handhabung schafft Sicherheitslücken durch Session-Hijacking,
  Session-Fixation oder unsachgemäßes Lebenszyklusmanagement.
category:
- Security
related_problems:
- slug: secret-management-problems
  similarity: 0.65
- slug: authentication-bypass-vulnerabilities
  similarity: 0.6
- slug: password-security-weaknesses
  similarity: 0.6
- slug: cross-site-scripting-vulnerabilities
  similarity: 0.55
- slug: authorization-flaws
  similarity: 0.55
- slug: inadequate-error-handling
  similarity: 0.55
solutions:
- security-hardening-process
- authentication
- role-based-access-control
- secure-session-management
- security-policies-for-users
- federated-identity
- two-factor-authentication
- encryption
- security-tests
- penetration-tests
- security-monitoring
layout: problem
lang: de
en_slug: session-management-issues
---

## Description

Probleme im Session-Management treten auf, wenn Anwendungen Nutzersitzungen unsachgemäß handhaben, was Sicherheitslücken schafft, die es Angreifern erlauben, legitime Nutzersitzungen zu kapern, Session-Fixation-Angriffe durchzuführen oder schwaches Session-Lebenszyklusmanagement auszunutzen. Schlechtes Session-Management kann zu unbefugtem Zugriff, Datendiebstahl und Kompromittierung von Nutzerkonten führen.

## Indicators ⟡

- Nutzer können ohne Einschränkung gleichzeitig von mehreren Orten eingeloggt sein
- Session-Tokens bleiben nach Logout oder Passwortänderungen gültig
- Session-Identifikatoren sind vorhersehbar oder unzureichend zufällig
- Sessions laufen nicht angemessen ab oder haben exzessive Timeouts
- Session-Daten werden unsicher gespeichert oder unverschlüsselt übertragen

## Symptoms ▲

- [Schwachstellen zur Umgehung der Authentifizierung](schwachstellen-zur-umgehung-der-authentifizierung.md)
<br/>  Schwaches Session-Management erlaubt es Angreifern, Sessions zu kapern oder zu fälschen, was die Authentifizierung effektiv umgeht.
- [Autorisierungsfehler](autorisierungsfehler.md)
<br/>  Schlechte Session-Handhabung kann es Nutzern erlauben, Privilegien zu eskalieren oder auf Sessions anderer Nutzer zuzugreifen, was Autorisierungsfehler schafft.
- [Datenschutzrisiko](datenschutzrisiko.md)
<br/>  Kompromittierte Sessions legen Nutzerdaten und sensible Informationen unbefugtem Zugriff durch Session-Hijacking offen.
- [Erosion des Nutzervertrauens](erosion-des-nutzervertrauens.md)
<br/>  Sicherheitsverletzungen durch Session-Hijacking untergraben das Vertrauen der Nutzer in die Fähigkeit des Systems, ihre Konten zu schützen.

## Causes ▼

- [Unerfahrene Entwickler](unerfahrene-entwickler.md)
<br/>  Entwickler ohne Sicherheitserfahrung implementieren möglicherweise vorhersehbare Session-Tokens, überspringen Verschlüsselung oder vernachlässigen ordentliches Session-Lebenszyklusmanagement.
- [Legacy-Code ohne Tests](legacy-code-ohne-tests.md)
<br/>  Session-Management-Code in Legacy-Systemen ohne Testabdeckung macht es riskant, Schwachstellen zu beheben oder Session-Handhabungspraktiken zu aktualisieren.
- [Unzureichendes Testen](unzureichendes-testen.md)
<br/>  Unzureichendes Sicherheitstesten versäumt es, Session-Management-Schwachstellen wie vorhersehbare Tokens oder fehlende Invalidierung zu identifizieren.

## Detection Methods ○

- **Session-Sicherheitstests:** Testen der Stärke, des Lebenszyklus und der Sicherheitsattribute von Session-Tokens
- **Session-Hijacking-Simulation:** Versuch, Sessions über verschiedene Angriffsvektoren zu kapern
- **Session-Speicheranalyse:** Überprüfung, wie und wo Session-Daten gespeichert und übertragen werden
- **Test gleichzeitiger Sessions:** Testen des Verhaltens mit mehreren gleichzeitigen Sessions
- **Session-Timeout- und Invalidierungstest:** Verifikation ordentlichen Session-Ablaufs und -Bereinigung

## Examples

Eine Online-Banking-Anwendung generiert Session-Tokens mithilfe eines einfachen inkrementierenden Zählers, was Session-IDs vorhersehbar macht. Ein Angreifer kann gültige Session-Tokens erraten, indem er sequenzielle Zahlen ausprobiert, und Zugriff auf Banking-Sessions anderer Nutzer erlangen. Die Anwendung versäumt es außerdem, Sessions beim Logout zu invalidieren, was es Angreifern mit Zugriff auf Session-Tokens erlaubt, Konten weiter zu nutzen, selbst nachdem sich legitime Nutzer ausgeloggt haben. Ein weiteres Beispiel betrifft eine E-Commerce-Website, die den Authentifizierungsstatus des Nutzers in einem clientseitigen Cookie ohne Verschlüsselung oder Signierung speichert. Nutzer können den Cookie-Wert modifizieren, um ihren Authentifizierungsstatus zu ändern oder andere Nutzer zu imitieren. Zusätzlich fehlt den Session-Cookies das Secure-Flag, was ihre Übertragung über unverschlüsselte Verbindungen erlaubt, wo sie von Angreifern abgefangen werden können.
