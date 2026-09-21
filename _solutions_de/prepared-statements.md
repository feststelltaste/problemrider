---
title: Prepared Statements
description: Nutzung parametrisierter Abfragen zur Verhinderung von
  SQL-Injection.
category:
- Security
- Database
problems:
- sql-injection-vulnerabilities
- buffer-overflow-vulnerabilities
- inadequate-error-handling
- poor-documentation
- legacy-code-without-tests
- insufficient-testing
- insecure-data-transmission
layout: solution
lang: de
en_slug: prepared-statements
related_solutions:
- slug: secure-programming-interfaces
  similarity: 0.7
- slug: secure-coding-guidelines
  similarity: 0.7
- slug: database-abstraction
  similarity: 0.7
- slug: object-relational-mapping-orm
  similarity: 0.7
- slug: web-application-firewall
  similarity: 0.7
- slug: secure-protocols
  similarity: 0.65
---

## Description

Prepared Statements — parametrisierte Abfragen, bei denen die SQL-Struktur fest ist und nutzergelieferte Werte separat statt in den Abfrage-String verkettet übergeben werden — sind die Standardverteidigung gegen SQL-Injection, weil der Datenbanktreiber Parameter strikt als Daten behandelt, nie als ausführbare SQL-Syntax, unabhängig davon, welche Zeichen sie enthalten. Sie in eine Legacy-Codebasis einzuführen bedeutet im Allgemeinen, jede Abfragekonstruktionsstelle auf String-Verkettungs- oder Interpolationsmuster zu prüfen, jede durch die parametrisierte API des jeweiligen Datenbanktreibers zu ersetzen und statische Analyseregeln hinzuzufügen, damit neu geschriebener Code dasselbe verwundbare Muster nicht still wieder einführen kann. Dies ist ein hochprioritäres Anliegen speziell in Legacy-Systemen, weil ältere Codebasen — oft geschrieben, bevor parametrisierte Abfrage-APIs idiomatisch waren, oder von aufeinanderfolgenden Entwicklern gewartet, die mit den ursprünglichen Konventionen nicht vertraut waren — dazu neigen, große Mengen roher, verketteter SQL-Anweisungen anzusammeln, manchmal in die Hunderte gehend über eine einzelne Anwendung hinweg. Über die Schließung der Injection-Schwachstelle selbst hinaus haben Prepared Statements einen sekundären Nutzen für Legacy-Systeme unter Performance-Druck: Da die Abfragestruktur fest und wiederverwendbar ist, kann die Datenbank ihren Ausführungsplan über Aufrufe hinweg zwischenspeichern, was häufig die Abfrageperformance als Nebeneffekt des Sicherheitsfixes verbessert. Die hauptsächliche praktische Schwierigkeit ist, dass eine kleine Anzahl dynamischer Abfragemuster — variable Tabellen- oder Spaltennamen — nicht direkt parametrisiert werden können und einen separaten Allowlisting-Ansatz erfordern, und gespeicherte Prozeduren, die SQL dynamisch über Konstrukte wie `EXEC` oder `sp_executesql` bauen, brauchen ihren eigenen Behebungspfad.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Prüfen Sie die Codebasis auf rohe SQL-String-Verkettungs- oder Interpolationsmuster und katalogisieren Sie jede Abfragekonstruktionsstelle
- Ersetzen Sie string-verkettete Abfragen durch parametrisierte Prepared Statements mittels der Datenbanktreiber-API Ihrer Sprache
- Führen Sie eine ORM- oder Query-Builder-Schicht ein, die parametrisierte Abfragen standardmäßig für neue Codepfade erzwingt
- Fügen Sie statische Analyseregeln hinzu, um rohe SQL-Verkettung in Code-Reviews und CI-Pipelines zu markieren
- Erstellen Sie wiederverwendbare Datenzugriffsfunktionen oder Repository-Klassen, die die Nutzung von Prepared Statements kapseln
- Aktualisieren Sie Legacy-gespeicherte Prozeduren, die SQL dynamisch mit `EXEC` oder `sp_executesql` bauen, um korrekte Parametrisierung zu nutzen
- Schreiben Sie Integrationstests, die SQL-Injection-Payloads versuchen, um zu verifizieren, dass Prepared Statements korrekt angewendet werden

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Beseitigt die häufigste und gefährlichste Klasse von Injection-Schwachstellen
- Verbessert Query-Plan-Caching und Datenbankperformance durch Statement-Wiederverwendung
- Vereinfacht Code, indem Abfragestruktur von Datenwerten getrennt wird
- Reduziert das Risiko stiller Datenkorruption durch fehlgeformte Eingaben

**Kosten und Risiken:**
- Die Migration großer Legacy-Codebasen mit Tausenden roher Abfragen erfordert erheblichen Aufwand
- Manche dynamischen Abfragen (z. B. variable Spalten- oder Tabellennamen) können nicht vollständig parametrisiert werden und brauchen Allowlisting
- Mit Prepared Statements nicht vertraute Entwickler könnten Workarounds einführen, die Schutzmaßnahmen umgehen
- ORM-Einführung kann eigene Komplexität und Performance-Overhead in bestimmten Szenarien einführen

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Finanzdienstleistungsunternehmen entdeckte während eines Penetrationstests, dass seine Legacy-PHP-Anwendung über 300 Instanzen direkter SQL-String-Verkettung enthielt. Das Team ersetzte diese systematisch durch PDO-Prepared-Statements über einen sechswöchigen Sprint, beginnend mit den kritischsten Zahlungsverarbeitungsabfragen. Nach der Migration bestätigte ein Folge-Penetrationstest null SQL-Injection-Befunde, und das Datenbankteam beobachtete eine 15%ige Verbesserung der Abfrageantwortzeiten aufgrund besseren Query-Plan-Cachings.
