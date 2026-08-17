---
title: Pufferüberlauf-Schwachstellen
description: Programme schreiben Daten über die Grenzen zugewiesener Speicherpuffer
  hinaus, was zu Sicherheitslücken und Systeminstabilität führt.
category:
- Code
- Security
related_problems:
- slug: stack-overflow-errors
  similarity: 0.65
- slug: integer-overflow-underflow
  similarity: 0.65
- slug: null-pointer-dereferences
  similarity: 0.65
- slug: memory-leaks
  similarity: 0.5
- slug: sql-injection-vulnerabilities
  similarity: 0.5
- slug: error-message-information-disclosure
  similarity: 0.5
solutions:
- security-hardening-process
- abuse-case-definition
- prepared-statements
- secure-coding-guidelines
- canonicalization
- defense-lines
- dynamic-code-analysis
- fuzz-testing
- input-validation
- negative-testing
- penetration-tests
- secure-software
- static-code-analysis
layout: problem
lang: de
en_slug: buffer-overflow-vulnerabilities
---

## Description

Pufferüberlauf-Schwachstellen entstehen, wenn ein Programm mehr Daten in einen Puffer schreibt, als dieser aufnehmen kann, wodurch die überschüssigen Daten benachbarte Speicherstellen überschreiben. Dies kann Daten beschädigen, die Anwendung zum Absturz bringen oder von Angreifern ausgenutzt werden, um schädlichen Code auszuführen. Pufferüberläufe sind besonders gefährlich, weil sie genutzt werden können, um die Systemsicherheit zu kompromittieren, indem Rücksprungadressen, Funktionszeiger oder andere kritische Programmdaten überschrieben werden.

## Indicators ⟡

- Anwendungsabstürze mit Segmentation Faults oder Zugriffsverletzungen
- Speicherkorruptions-Symptome wie unerwartete Variablenwerte
- Sicherheits-Scanning-Werkzeuge melden potenzielle Pufferüberlauf-Schwachstellen
- Abstürze treten bei der Verarbeitung bestimmter Eingabemuster oder -größen auf
- Stack-Traces zeigen Korruption oder unerwartete Funktionsaufrufsequenzen

## Symptoms ▲

- [Erhöhte Fehlerraten](erhoehte-fehlerraten.md)
<br/>  Pufferüberläufe verursachen sporadische Abstürze und Datenkorruption, die sich als schwer reproduzierbare, intermittierende Fehler äußern.
- [Kaskadierende Ausfälle](kaskadierende-ausfaelle.md)
<br/>  Ein Pufferüberlauf-Absturz in einem gemeinsam genutzten Service kann Ausfälle in abhängigen Komponenten auslösen.
- [Datenschutzrisiko](datenschutzrisiko.md)
<br/>  Pufferüberlauf-Schwachstellen können ausgenutzt werden, um auf sensible Daten im Speicher zuzugreifen, was direkt Datenschutzrisiken schafft.

## Causes ▼

- [Inkonsistente Coding-Standards](inkonsistente-coding-standards.md)
<br/>  Ohne Coding-Standards, die Grenzprüfungen und sichere String-Funktionen vorschreiben, bleiben unsichere Pufferoperationen bestehen.
- [Unzureichendes Testen](unzureichendes-testen.md)
<br/>  Fehlendes Testen von Grenzbedingungen und Fuzz-Testing lässt Pufferüberlauf-Fehler in die Produktion gelangen.
- [Unzureichende Code-Reviews](unzureichende-code-reviews.md)
<br/>  Code-Reviews, die nicht gezielt auf Speichersicherheitsprobleme prüfen, lassen Pufferüberlauf-Schwachstellen durchrutschen.
- [Unerfahrene Entwickler](unerfahrene-entwickler.md)
<br/>  Entwickler ohne Kenntnisse über Speichersicherheit und sichere Programmierpraktiken schreiben mit höherer Wahrscheinlichkeit Code mit Pufferüberlauf-Schwachstellen.

## Detection Methods ○

- **Statische Analysewerkzeuge:** Nutzung von Werkzeugen, die potenzielle Pufferüberlauf-Schwachstellen im Quellcode identifizieren können
- **Dynamische Analyse:** Laufzeitwerkzeuge wie AddressSanitizer, die Pufferüberläufe während der Ausführung erkennen
- **Fuzzing:** Automatisiertes Testen mit zufälligen oder fehlerhaften Eingaben, um Pufferüberlaufbedingungen auszulösen
- **Code-Review:** Manuelle Überprüfung mit Fokus auf Speicherverwaltung und Grenzprüfung
- **Penetrationstests:** Sicherheitstests, die gezielt auf die Ausnutzung von Pufferüberläufen abzielen
- **Speicherschutzwerkzeuge:** Nutzung von Werkzeugen wie Valgrind zur Erkennung von Speicherfehlern während der Entwicklung

## Examples

Ein C-Programm nutzt die Funktion strcpy, um Nutzereingaben in ein Character-Array fester Größe zu kopieren, ohne die Eingabelänge zu prüfen. Wenn ein Nutzer eine Eingabe bereitstellt, die länger als die Puffergröße ist, überschreibt strcpy benachbarten Stack-Speicher, was möglicherweise lokale Variablen oder die Funktionsrücksprungadresse korrumpiert. Ein Angreifer kann dies ausnutzen, indem er Eingaben erstellt, die die Rücksprungadresse mit der Adresse schädlichen Codes überschreiben, wodurch effektiv die Programmausführung übernommen wird. Ein weiteres Beispiel betrifft einen Netzwerkdienst, der Paketdaten in einen Puffer fester Größe liest, ohne das Paketgrößenfeld zu validieren. Schädliche Pakete mit falschen Größeninformationen können den Dienst dazu bringen, über Puffergrenzen hinaus zu schreiben, was möglicherweise Remote-Code-Ausführung ermöglicht oder Dienstabstürze verursacht, die Denial-of-Service-Angriffe ermöglichen.
