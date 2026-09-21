---
title: Autorisierungsfehler
description: Unzureichende Zugriffskontrollmechanismen erlauben Nutzern, Aktionen
  auszuführen oder auf Ressourcen zuzugreifen, die über ihre vorgesehenen Berechtigungen
  hinausgehen.
category:
- Code
- Security
related_problems:
- slug: authentication-bypass-vulnerabilities
  similarity: 0.75
- slug: authorization-role-explosion
  similarity: 0.6
- slug: error-message-information-disclosure
  similarity: 0.55
- slug: session-management-issues
  similarity: 0.55
- slug: password-security-weaknesses
  similarity: 0.55
- slug: sql-injection-vulnerabilities
  similarity: 0.55
solutions:
- security-hardening-process
- abuse-case-definition
- api-security
- audit-trail-management
- authentication
- authorization
- authorization-concept
- red-teaming
- role-based-access-control
- secure-session-management
- security-by-design
- security-tests-by-external-parties
- data-flow-control
- defense-lines
- domain-based-authorization-concept
- least-privilege
- network-segmentation
- penetration-tests
- threat-modeling
- trust-boundaries
- two-factor-authentication
- zero-trust-architecture
- role-model-rationalization
layout: problem
lang: de
en_slug: authorization-flaws
---

## Description

Autorisierungsfehler entstehen, wenn Zugriffskontrollmechanismen es versäumen, Nutzeraktionen und Ressourcenzugriffe entsprechend den vorgesehenen Berechtigungen ordentlich einzuschränken. Diese Schwachstellen erlauben Nutzern, unbefugte Operationen auszuführen, auf eingeschränkte Daten zuzugreifen oder ihre Rechte über das erlaubte Maß hinaus auszuweiten, was potenziell die Systemsicherheit und Datenintegrität gefährdet.

## Indicators ⟡

- Nutzer können auf Ressourcen zugreifen oder Aktionen außerhalb ihrer zugewiesenen Rollen ausführen
- Horizontale Rechteausweitung ermöglicht Zugriff auf Daten anderer Nutzer
- Vertikale Rechteausweitung ermöglicht es Nutzern, administrative Rechte zu erlangen
- Zugriffskontrollentscheidungen werden clientseitig statt serverseitig getroffen
- Inkonsistente Durchsetzung von Berechtigungen über verschiedene Systemkomponenten hinweg

## Symptoms ▲

- [Datenschutzrisiko](datenschutzrisiko.md)
<br/>  Fehlerhafte Autorisierung erlaubt unbefugten Zugriff auf sensible Daten und schafft erhebliche Datenschutzrisiken.
- [Regulatorische Compliance-Drift](regulatorische-compliance-drift.md)
<br/>  Autorisierungsfehler verletzen Compliance-Anforderungen an die Zugriffskontrolle und drängen das System aus der regulatorischen Konformität.
- [Erosion des Nutzervertrauens](erosion-des-nutzervertrauens.md)
<br/>  Nutzer verlieren das Vertrauen, wenn sie entdecken, dass andere aufgrund von Autorisierungsfehlern auf ihre Daten zugreifen können.
- [Rechtsstreitigkeiten](rechtsstreitigkeiten.md)
<br/>  Autorisierungsfehler, die unbefugten Zugriff auf sensible Daten erlauben, können rechtliche Schritte betroffener Parteien auslösen.

## Causes ▼

- [Unzureichende Fehlerbehandlung](unzureichende-fehlerbehandlung.md)
<br/>  Schlechte Fehlerbehandlung kann Autorisierungsprüfungen stillschweigend umgehen und unbefugten Zugriff ermöglichen.
- [Komplexe und unklare Logik](komplexe-und-unklare-logik.md)
<br/>  Komplexe Autorisierungslogik enthält mit höherer Wahrscheinlichkeit Fehler, die unbeabsichtigten Zugriff erlauben.
- [Unzureichendes Testen](unzureichendes-testen.md)
<br/>  Ohne gründliches Testen der Autorisierung bleiben Zugriffskontrollfehler unentdeckt, bis sie ausgenutzt werden.
- [Unerfahrene Entwickler](unerfahrene-entwickler.md)
<br/>  Entwickler ohne Sicherheitserfahrung setzen Autorisierungsprüfungen möglicherweise unvollständig oder fehlerhaft um.

## Detection Methods ○

- **Zugriffskontrolltests:** Testen aller geschützten Ressourcen und Funktionen auf ordentliche Autorisierung
- **Tests zur Rechteausweitung:** Versuch, Rechte über verschiedene Angriffsvektoren auszuweiten
- **Rollenbasierte Zugriffstests:** Verifikation, dass Rollenzuweisungen den Zugriff ordentlich einschränken
- **Tests auf direkte Objektreferenzen:** Testen der Manipulation von Objektkennungen, um auf unbefugte Ressourcen zuzugreifen
- **Review der Autorisierung auf Funktionsebene:** Überprüfung aller administrativen und sensiblen Funktionen auf ordentliche Zugriffskontrolle

## Examples

Eine Projektmanagement-Anwendung erlaubt Nutzern, Projektdetails über URLs wie `/project/123` einzusehen. Nutzer entdecken, dass sie die Projekt-ID ändern können, um auf jedes Projekt im System zuzugreifen, einschließlich vertraulicher Projekte, die sie nicht sehen sollten. Die Anwendung authentifiziert Nutzer, versäumt es aber zu prüfen, ob sie die Berechtigung haben, auf das spezifisch angeforderte Projekt zuzugreifen, was jedem erlaubt, beliebige Projektdaten einzusehen. Ein weiteres Beispiel betrifft ein Content-Management-System, bei dem normale Nutzer auf administrative Funktionen zugreifen können, indem sie direkt zu Admin-URLs navigieren. Während die Benutzeroberfläche Admin-Menüpunkte vor normalen Nutzern verbirgt, prüft die Serverseite die Nutzerrollen vor der Ausführung administrativer Operationen nicht, was Rechteausweitung durch direkte URL-Manipulation ermöglicht.
