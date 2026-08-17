---
title: Schwer testbarer Code
description: Komponenten lassen sich aufgrund enger Kopplung, globaler Abhängigkeiten
  oder komplexer Voraussetzungen nicht leicht isoliert testen.
category:
- Code
- Testing
related_problems:
- slug: difficult-to-understand-code
  similarity: 0.75
- slug: debugging-difficulties
  similarity: 0.65
- slug: difficult-code-reuse
  similarity: 0.65
- slug: legacy-code-without-tests
  similarity: 0.65
- slug: difficult-code-comprehension
  similarity: 0.65
- slug: tight-coupling-issues
  similarity: 0.65
solutions:
- test-coverage-strategy
- abstracted-file-system-access
- automated-tests
- hexagonal-architecture
- test-driven-development-tdd
- database-abstraction
- dependency-injection
- dependency-injection-container
- characterization-tests
- dependency-breaking-techniques
layout: problem
lang: de
en_slug: difficult-to-test-code
---

## Description

Schwer testbarer Code bezeichnet Softwarekomponenten, die aufgrund architektonischer Probleme, Abhängigkeiten oder Design-Entscheidungen nicht leicht oder wirksam mit Unit-Tests versehen werden können. Dieser Code erfordert typischerweise komplexe Setup-Prozeduren, hängt von externen Systemen ab oder hat so viele gegenseitige Abhängigkeiten, dass eine Isolation zum Testen unpraktikabel wird. Wenn Code schwer zu testen ist, überspringen Entwickler das Schreiben von Tests oft ganz, was zu verringertem Vertrauen in Codeänderungen und einer höheren Wahrscheinlichkeit von Fehlern führt.

## Indicators ⟡
- Unit-Tests erfordern umfangreiches Setup oder Mock-Konfigurationen
- Tests benötigen Zugriff auf Datenbanken, Dateisysteme oder externe Dienste, um zu laufen
- Einfache Funktionen erfordern das Testen ganzer Anwendungs-Workflows
- Entwickler überspringen häufig das Schreiben von Tests, weil sie zu kompliziert sind
- Die Testausführung ist aufgrund komplexer Abhängigkeiten langsam

## Symptoms ▲

- [Schlechte Testabdeckung](schlechte-testabdeckung.md)
<br/>  Wenn Code schwer zu testen ist, überspringen Entwickler das Schreiben von Tests, was zu geringer Testabdeckung führt.
- [Legacy-Code ohne Tests](legacy-code-ohne-tests.md)
<br/>  Code, der schwer zu testen ist, häuft sich schrittweise zu einer großen ungetesteten Legacy-Codebasis an.
- [Hohe Rate an neu eingeführten Fehlern](hohe-rate-an-neu-eingefuehrten-fehlern.md)
<br/>  Ohne ausreichende Tests als Sicherheitsnetz führen Änderungen häufig neue Fehler ein.
- [Angst vor Veränderung](angst-vor-veraenderung.md)
<br/>  Entwickler zögern, ungetesteten Code zu ändern, weil sie nicht verifizieren können, dass ihre Änderungen nichts brechen.
- [Regressionsfehler](regressionsfehler.md)
<br/>  Ohne Tests, die Regressionen abfangen, tauchen zuvor behobene Fehler wieder auf, wenn Code geändert wird.
- [Erhöhter manueller Testaufwand](erhoehter-manueller-testaufwand.md)
<br/>  Wenn automatisiertes Testen unpraktikabel ist, greifen Teams auf teures und langsames manuelles Testen zurück.

## Causes ▼

- [Probleme durch enge Kopplung](probleme-durch-enge-kopplung.md)
<br/>  Eng gekoppelte Komponenten können nicht für Unit-Tests isoliert werden, was komplexes Setup der gesamten Abhängigkeitskette erfordert.
- [Globaler Zustand und Nebeneffekte](globaler-zustand-und-nebeneffekte.md)
<br/>  Globaler Zustand und versteckte Nebeneffekte machen es unmöglich, Komponenten isoliert mit vorhersehbaren Ergebnissen zu testen.
- [God-Object-Antipattern](god-object-antipattern.md)
<br/>  God Objects mit vielen Verantwortlichkeiten und Abhängigkeiten erfordern umfangreiches Mocking, um selbst einfache Funktionalität zu testen.
- [Monolithische Funktionen und Klassen](monolithische-funktionen-und-klassen.md)
<br/>  Große Funktionen, die viele Dinge tun, erfordern das Testen ganzer Workflows statt einzelner Verhaltensweisen.
- [Hohe Kopplung und geringe Kohäsion](hohe-kopplung-und-geringe-kohaesion.md)
<br/>  Hohe Kopplung und geringe Kohäsion machen Code direkt schwer testbar, weil Komponenten nicht isoliert werden können.

## Detection Methods ○
- **Testabdeckungsanalyse:** Geringe Abdeckung in bestimmten Modulen deutet oft auf Testschwierigkeiten hin
- **Testkomplexitätsmetriken:** Messung der Anzahl der Setup-Schritte oder Mock-Objekte, die für Tests erforderlich sind
- **Entwickler-Feedback:** Befragung von Entwicklern, welche Teile der Codebasis am schwierigsten zu testen sind
- **Testausführungszeit:** Beobachtung, welche Tests aufgrund von Setup-Komplexität am längsten dauern
- **Abhängigkeitsanalyse:** Nutzung von Werkzeugen zur Identifikation von Komponenten mit den meisten externen Abhängigkeiten

## Examples

Eine Zahlungsabwicklungsfunktion verbindet sich direkt mit einem Zahlungs-Gateway, schreibt in eine Datenbank, sendet E-Mail-Benachrichtigungen und aktualisiert mehrere globale Konfigurationsobjekte. Um diese Funktion zu testen, müssten Entwickler eine Testdatenbank einrichten, die Zahlungs-Gateway-API mocken, einen E-Mail-Server konfigurieren und alle globalen Zustandsobjekte mit korrekten Werten initialisieren. Die Komplexität dieses Setups bedeutet, dass Entwickler entweder das Testen der Funktion ganz überspringen oder Integrationstests schreiben, die langsam und brüchig sind. Ein weiteres Beispiel betrifft ein Berichtserstellungsmodul, das vom aktuellen Datum abhängt, aus mehreren Datenbanktabellen liest, auf Dateien aus dem Dateisystem zugreift und drei unterschiedliche Web-Services aufruft. Das Testen eines einzelnen Aspekts der Berichtserstellung erfordert das Mocken oder Einrichten all dieser Abhängigkeiten, was es unpraktikabel macht, fokussierte Unit-Tests zu schreiben, die spezifische Geschäftslogik verifizieren.
