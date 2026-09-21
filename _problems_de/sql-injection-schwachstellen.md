---
title: SQL-Injection-Schwachstellen
description: Unzureichende Bereinigung von Eingaben erlaubt es Angreifern, bösartigen
  SQL-Code einzuschleusen, was potenziell die Datenbanksicherheit und Datenintegrität
  kompromittiert.
category:
- Database
- Security
related_problems:
- slug: cross-site-scripting-vulnerabilities
  similarity: 0.7
- slug: log-injection-vulnerabilities
  similarity: 0.65
- slug: error-message-information-disclosure
  similarity: 0.6
- slug: authentication-bypass-vulnerabilities
  similarity: 0.55
- slug: authorization-flaws
  similarity: 0.55
- slug: session-management-issues
  similarity: 0.5
solutions:
- security-hardening-process
- abuse-case-definition
- api-security
- prepared-statements
- red-teaming
- secure-coding-guidelines
- secure-programming-interfaces
- security-tests
- canonicalization
- defense-lines
- dynamic-code-analysis
- fuzz-testing
- input-validation
- negative-testing
- output-encoding
- penetration-tests
- secure-software
- static-code-analysis
- web-application-firewall
layout: problem
lang: de
en_slug: sql-injection-vulnerabilities
---

## Description

SQL-Injection-Schwachstellen treten auf, wenn Anwendungen es versäumen, Nutzereingaben ordentlich zu bereinigen, bevor sie in SQL-Abfragen genutzt werden, was Angreifern erlaubt, bösartigen SQL-Code einzuschleusen, der Datenbankoperationen manipulieren kann. Diese Schwachstellen können zu unbefugtem Datenzugriff, Datenmodifikation, Datenlöschung oder vollständiger Datenbankkompromittierung führen, was sie zu einem der kritischsten Sicherheitsrisiken für Webanwendungen macht.

## Indicators ⟡

- Nutzereingaben werden direkt in SQL-Abfragestrings verkettet
- Datenbankabfragen werden dynamisch ohne Parametrisierung konstruiert
- Fehlermeldungen offenbaren Datenbankstruktur oder Abfragedetails
- Anwendungen nutzen Datenbankkonten mit exzessiven Privilegien
- Eingabevalidierung fehlt oder ist unzureichend für SQL-Abfragekontexte

## Symptoms ▲

- [Systemausfälle](systemausfaelle.md)
<br/>  Destruktive SQL-Injection-Angriffe wie DROP TABLE können Systemausfälle verursachen, indem sie kritische Daten zerstören.

## Causes ▼

- [Inkonsistente Coding-Standards](inkonsistente-coding-standards.md)
<br/>  Ohne Coding-Standards, die parametrisierte Abfragen vorschreiben, nutzen Entwickler möglicherweise String-Verkettung für SQL.
- [Unzureichendes Code-Review](unzureichendes-code-review.md)
<br/>  Ohne Code-Review bleiben unsichere Muster der Abfragekonstruktion unentdeckt, bevor sie die Produktion erreichen.
- [Legacy-Code ohne Tests](legacy-code-ohne-tests.md)
<br/>  Älterer Code, der geschrieben wurde, bevor Sicherheits-Best-Practices etabliert waren, enthält oft SQL-Injection-Schwachstellen.

## Detection Methods ○

- **Eingabevalidierungstests:** Testen aller Eingabefelder auf SQL-Injection-Angriffsvektoren
- **Automatisiertes Sicherheitsscanning:** Nutzung von Sicherheitsscannern zur Identifikation von SQL-Injection-Schwachstellen
- **Code-Review für Abfragekonstruktion:** Überprüfung des gesamten Datenbankabfrage-Konstruktionscodes
- **Datenbankfehleranalyse:** Analyse von Fehlermeldungen auf Informationspreisgabe
- **Penetrationstests:** Durchführung manueller Tests für komplexe SQL-Injection-Szenarien

## Examples

Ein Login-Formular konstruiert SQL-Abfragen durch direktes Einfügen von Nutzereingaben: `SELECT * FROM users WHERE username = '` + username + `' AND password = '` + password + `'`. Ein Angreifer gibt `admin'--` als Benutzernamen ein, was die Abfrage `SELECT * FROM users WHERE username = 'admin'--' AND password = ''` erzeugt. Das `--` kommentiert die Passwortprüfung aus, was den Login als Admin ohne Kenntnis des Passworts erlaubt. Ein weiteres Beispiel betrifft eine Produktsuche, die Abfragen wie `SELECT * FROM products WHERE name LIKE '%` + searchTerm + `%'` erstellt. Ein Angreifer gibt `'; DROP TABLE products; --` ein, was die ursprüngliche Abfrage beendet und einen destruktiven Befehl ausführt, der potenziell die gesamte Produkttabelle löscht. Die Nutzung parametrisierter Abfragen würde beide Angriffe verhindern, indem Nutzereingaben als Daten statt als ausführbarer Code behandelt werden.
