---
title: Informationspreisgabe durch Fehlermeldungen
description: Fehlermeldungen geben sensible Systeminformationen preis, die Angreifer
  nutzen können, um Systemarchitektur und Schwachstellen zu verstehen.
category:
- Code
- Communication
- Security
related_problems:
- slug: sql-injection-vulnerabilities
  similarity: 0.6
- slug: authentication-bypass-vulnerabilities
  similarity: 0.6
- slug: secret-management-problems
  similarity: 0.55
- slug: authorization-flaws
  similarity: 0.55
- slug: log-injection-vulnerabilities
  similarity: 0.55
- slug: inadequate-error-handling
  similarity: 0.55
solutions:
- secret-management
- security-hardening-process
- authentication
- authorization
- secure-by-default
- secure-configuration
- data-flow-control
- encryption
- negative-testing
- output-encoding
layout: problem
lang: de
en_slug: error-message-information-disclosure
---

## Description

Informationspreisgabe durch Fehlermeldungen entsteht, wenn Anwendungen sensible technische Informationen durch Fehlermeldungen, Stack-Traces oder Debug-Ausgaben preisgeben, die Angreifern helfen können, Systemarchitektur, Datenbankschemata, Dateipfade oder interne Anwendungslogik zu verstehen. Diese Informationen können genutzt werden, um gezieltere Angriffe zu erstellen oder spezifische Schwachstellen zu identifizieren.

## Indicators ⟡

- Datenbank-Fehlermeldungen geben Tabellennamen, Spaltennamen oder Abfragestruktur preis
- Stack-Traces legen interne Dateipfade, Klassennamen oder Systemarchitektur offen
- Fehlermeldungen enthalten Details zur Systemkonfiguration oder Versionsinformationen
- Debug-Informationen werden Endnutzern in Produktionsumgebungen angezeigt
- Fehlerantworten geben die Existenz oder Nicht-Existenz von Ressourcen preis

## Symptoms ▲

- [Schwachstellen zur Umgehung der Authentifizierung](schwachstellen-zur-umgehung-der-authentifizierung.md)
<br/>  Preisgegebene Details zur Systemarchitektur und interne Pfade helfen Angreifern, Authentifizierungsschwächen zu identifizieren, die ausgenutzt werden können.
- [SQL-Injection-Schwachstellen](sql-injection-schwachstellen.md)
<br/>  Offengelegte Datenbankschema-Informationen wie Tabellennamen und Spaltennamen ermöglichen es Angreifern, gezielte SQL-Injection-Angriffe zu erstellen.

## Causes ▼

- [Unzureichende Fehlerbehandlung](unzureichende-fehlerbehandlung.md)
<br/>  Schlechte Fehlerbehandlung gibt rohe Ausnahmen und Stack-Traces an Nutzer weiter, statt bereinigte Fehlermeldungen anzuzeigen.
- [Probleme bei der Logging-Konfiguration](probleme-bei-der-logging-konfiguration.md)
<br/>  Falsch konfigurierte Logging-Level in Produktionsumgebungen verursachen, dass Informationen auf Debug-Ebene Endnutzern angezeigt werden.
- [Unerfahrene Entwickler](unerfahrene-entwickler.md)
<br/>  Entwickler ohne Sicherheitsbewusstsein erkennen möglicherweise nicht, dass detaillierte Fehlermeldungen in der Produktion ein Sicherheitsrisiko darstellen.
- [Unzureichendes Testen](unzureichendes-testen.md)
<br/>  Fehlende Sicherheitstests für Fehlerbedingungen bedeuten, dass Informationspreisgabe durch Fehlermeldungen vor der Produktion unentdeckt bleibt.

## Detection Methods ○

- **Sicherheitsüberprüfung von Fehlermeldungen:** Überprüfung aller Fehlermeldungen auf Preisgabe sensibler Informationen
- **Produktionsfehler-Tests:** Testen von Fehlerbedingungen in produktionsähnlichen Umgebungen
- **Fehlerantwort-Analyse:** Analyse von Fehlerantworten auf Informationen, die Angreifern helfen könnten
- **Sicherheitstests für Informationspreisgabe:** Testen verschiedener Fehlerbedingungen auf Informationslecks
- **Audit des Fehlerbehandlungscodes:** Überprüfung des Fehlerbehandlungscodes auf angemessene Informationsfilterung

## Examples

Das Login-Formular einer Webanwendung zeigt detaillierte Datenbank-Fehlermeldungen an, wenn SQL-Abfragen fehlschlagen, und legt das vollständige Datenbankschema offen, einschließlich Tabellennamen wie "users", "admin_accounts" und "payment_info" zusammen mit Spaltennamen wie "password_hash" und "credit_card_number". Angreifer können diese Informationen nutzen, um SQL-Injection-Angriffe zu erstellen, die auf bestimmte Tabellen und Spalten abzielen. Ein weiteres Beispiel betrifft einen Datei-Upload-Dienst, der vollständige Java-Stack-Traces anzeigt, wenn die Dateiverarbeitung fehlschlägt, und dabei interne Anwendungsarchitektur, Bibliotheksversionen und Dateisystempfade wie "/opt/app/uploads/processing/temp/" offenlegt. Diese Informationen helfen Angreifern, die Systemstruktur zu verstehen und potenzielle Angriffsvektoren wie Directory Traversal oder abhängigkeitsspezifische Schwachstellen zu identifizieren.
