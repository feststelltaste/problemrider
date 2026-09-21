---
title: Log-Injection-Schwachstellen
description: Unbereinigte Benutzereingaben in Log-Nachrichten erlauben es Angreifern,
  bösartigen Inhalt einzuschleusen, der die Log-Integrität kompromittieren oder Log-Verarbeitungssysteme
  ausnutzen kann.
category:
- Code
- Security
related_problems:
- slug: sql-injection-vulnerabilities
  similarity: 0.65
- slug: logging-configuration-issues
  similarity: 0.6
- slug: cross-site-scripting-vulnerabilities
  similarity: 0.55
- slug: insufficient-audit-logging
  similarity: 0.55
- slug: error-message-information-disclosure
  similarity: 0.55
- slug: log-spam
  similarity: 0.55
solutions:
- observability-and-monitoring
- security-hardening-process
- canonicalization
- input-validation
- logging-and-monitoring
- output-encoding
- logging-guidelines
- secure-coding-guidelines
- code-reviews
- static-analysis-and-linting
layout: problem
lang: de
en_slug: log-injection-vulnerabilities
---

## Description

Log-Injection-Schwachstellen treten auf, wenn Anwendungen unbereinigte Benutzereingaben in Log-Nachrichten einbeziehen, was es Angreifern erlaubt, bösartigen Inhalt einzuschleusen, der Log-Dateien beschädigen, Log-Verarbeitungssysteme ausnutzen oder falsche Log-Einträge erzeugen kann. Dies kann zu Log-Manipulation, Denial-of-Service-Angriffen auf Logging-Systeme oder zur Ausnutzung von Log-Analysewerkzeugen führen.

## Indicators ⟡

- Benutzereingaben werden direkt ohne Bereinigung in Log-Nachrichten aufgenommen
- Log-Einträge enthalten unerwartete Formatierungszeichen oder Escape-Sequenzen
- Log-Verarbeitungssysteme erleben Fehler beim Parsen bestimmter Log-Einträge
- Log-Dateien enthalten verdächtige oder fehlgeformte Einträge
- Benutzer können den Log-Inhalt über Eingabefelder beeinflussen

## Symptoms ▲

- [Unzureichendes Audit-Logging](unzureichendes-audit-logging.md)
<br/>  Eingeschleuste gefälschte Log-Einträge beschädigen Audit-Trails, was legitimes Audit-Logging unzuverlässig und nicht vertrauenswürdig macht.
- [Datenschutzrisiko](datenschutzrisiko.md)
<br/>  Log-Injection kann genutzt werden, um Angriffsspuren zu verbergen oder Inhalt einzuschleusen, der Systeme kompromittiert, die diese Logs verarbeiten, was Datenschutzrisiken schafft.
- [Systemausfälle](systemausfaelle.md)
<br/>  Eingeschleuste Format-Strings oder bösartiger Inhalt können Log-Verarbeitungssysteme zum Absturz bringen, was Service-Unterbrechungen verursacht.
- [Debugging-Schwierigkeiten](debugging-schwierigkeiten.md)
<br/>  Beschädigte oder manipulierte Log-Dateien machen es extrem schwierig, echte Probleme zu diagnostizieren, wenn gefälschte Einträge echte Log-Daten verdecken.

## Causes ▼

- [Unzureichende Fehlerbehandlung](unzureichende-fehlerbehandlung.md)
<br/>  Schlechte Praktiken der Fehlerbehandlung, die unbereinigte Benutzereingaben in Log-Nachrichten ausgeben, schaffen Möglichkeiten für Injection.
- [Probleme bei der Logging-Konfiguration](probleme-bei-der-logging-konfiguration.md)
<br/>  Unsachgemäß konfiguriertes Logging, das keine Eingabebereinigung oder strukturierte Logging-Formate erzwingt, ermöglicht Injection-Angriffe.
- [Unerfahrene Entwickler](unerfahrene-entwickler.md)
<br/>  Entwickler, die mit bewährten Sicherheitspraktiken nicht vertraut sind, erkennen möglicherweise nicht, dass Benutzereingaben in Log-Nachrichten bereinigt werden müssen.

## Detection Methods ○

- **Log-Inhaltsanalyse:** Regelmäßige Analyse von Log-Dateien auf verdächtige oder fehlgeformte Einträge
- **Eingabevalidierungstests:** Testen des Logging-Verhaltens mit verschiedenen bösartigen Eingabemustern
- **Sicherheitstests der Log-Verarbeitung:** Testen von Log-Analysewerkzeugen mit potenziell bösartigen Log-Einträgen
- **Log-Integritätsüberwachung:** Überwachung von Logs auf Anzeichen von Manipulation oder Beschädigung
- **Überprüfung der Benutzereingabebereinigung:** Überprüfung, wie Benutzereingaben im Logging-Code gehandhabt werden

## Examples

Eine Webanwendung protokolliert fehlgeschlagene Anmeldeversuche einschließlich des angegebenen Benutzernamens: `Log.info("Failed login attempt for user: " + username)`. Ein Angreifer gibt einen Benutzernamen ein, der Zeilenumbrüche und gefälschte Log-Einträge enthält: `"admin\n[INFO] Successful login for user: admin\n[INFO] Admin privileges granted"`. Dies erzeugt falsche Log-Einträge, die den Anschein erwecken, dass der Admin-Benutzer sich erfolgreich angemeldet und Rechte erhalten hat, was möglicherweise den eigentlichen Angriff verbirgt. Ein weiteres Beispiel betrifft eine E-Commerce-Anwendung, die Benutzersuchanfragen protokolliert. Ein Angreifer schleust Log-Format-Strings in das Suchfeld ein, was dazu führt, dass das Logging-System abstürzt, wenn es versucht, Formatspezifizierer wie `%s%s%s%n` zu verarbeiten, die zusätzliche Parameter erwarten, was effektiv einen Denial-of-Service gegen die Logging-Infrastruktur erzeugt.
