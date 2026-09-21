---
title: Defensive Programmierpraktiken
description: Entwickler schreiben übermäßig ausführlichen Code, exzessive Kommentare
  oder unnötige defensive Logik, um erwarteter Kritik im Code-Review zuvorzukommen.
category:
- Code
- Process
- Team
related_problems:
- slug: fear-of-conflict
  similarity: 0.7
- slug: inadequate-code-reviews
  similarity: 0.7
- slug: superficial-code-reviews
  similarity: 0.7
- slug: review-process-avoidance
  similarity: 0.65
- slug: nitpicking-culture
  similarity: 0.65
- slug: review-process-breakdown
  similarity: 0.65
solutions:
- clean-code
- design-by-contract
- solid-principles
- characterization-tests
- internal-technical-coaching
- preparatory-refactoring
- code-reading-sessions
- lightweight-design-review
- automated-tests
- code-reviews
layout: problem
lang: de
en_slug: defensive-coding-practices
---

## Description

Defensive Programmierpraktiken entstehen, wenn Entwickler ihren Programmierstil nicht ändern, um Funktionalität oder Wartbarkeit zu verbessern, sondern um erwarteter Kritik im Code-Review zuvorzukommen. Dies umfasst das Schreiben unnötig ausführlichen Codes, das Hinzufügen exzessiver Kommentare, um jede Entscheidung zu rechtfertigen, die Umsetzung übermäßig defensiver Fehlerbehandlung oder die Wahl konservativer Ansätze, die weniger effizient, aber schwerer zu kritisieren sind. Während etwas defensive Programmierung nützlich ist, stellt dieses Problem Programmierentscheidungen dar, die von der Angst vor Review-Feedback statt von technischem Verdienst getrieben werden.

## Indicators ⟡

- Code enthält weit mehr Kommentare als nötig, oft mit Erklärungen offensichtlicher Operationen
- Entwickler wählen weniger effiziente, aber "sicherere" Implementierungen, um Review-Debatten zu vermeiden
- Variablennamen werden übermäßig lang und beschreibend, um Kritik an der Benennung zu verhindern
- Code enthält unnötige Fehlerbehandlung für unmögliche Szenarien
- Entwickler erwähnen, dass sie Code speziell ändern, um Review-Kommentare zu vermeiden

## Symptoms ▲

- [Erhöhte kognitive Last](erhoehte-kognitive-last.md)
<br/>  Übermäßig ausführlicher Code mit exzessiven Kommentaren und unnötiger defensiver Logik erhöht den mentalen Aufwand, der nötig ist, um die Codebasis zu verstehen.
- [Langsame Entwicklungsgeschwindigkeit](langsame-entwicklungsgeschwindigkeit.md)
<br/>  Das Schreiben und Warten unnötig ausführlichen und defensiven Codes braucht mehr Zeit als das Schreiben sauberer, fokussierter Implementierungen.
- [Verringerte Häufigkeit von Code-Einreichungen](verringerte-haeufigkeit-von-code-einreichungen.md)
<br/>  Entwickler verbringen zusätzliche Zeit damit, defensiven Code hinzuzufügen, um Review-Kritik zuvorzukommen, was ihre Einreichungen verzögert.

## Causes ▼

- [Nitpicking-Kultur](nitpicking-kultur.md)
<br/>  Wenn sich Reviews auf kleinere Details konzentrieren, lernen Entwickler, triviale Bedenken vorbeugend durch übermäßig ausführlichen und defensiven Code anzugehen.
- [Perfektionistische Review-Kultur](perfektionistische-review-kultur.md)
<br/>  Eine Kultur, die perfekten Code durch Reviews verlangt, treibt Entwickler dazu, exzessive defensive Maßnahmen hinzuzufügen, um Kritik zu vermeiden.
- [Schuldzuweisungskultur](schuldzuweisungskultur.md)
<br/>  Wenn Fehler bestraft werden, schreiben Entwickler übermäßig vorsichtigen Code, um jede mögliche Kritik oder Schuldzuweisung zu minimieren.
- [Undefinierte Code-Stil-Richtlinien](undefinierte-code-stil-richtlinien.md)
<br/>  Ohne klare Coding-Standards können Entwickler nicht vorhersehen, was Reviewer kritisieren werden, was dazu führt, dass sie ihre Entscheidungen übermäßig dokumentieren und verteidigen.

## Detection Methods ○

- **Code-Komplexitätsanalyse:** Vergleich der Code-Komplexität vor und nach Review-Erfahrungen
- **Kommentardichte-Bewertung:** Messung der Kommentar-zu-Code-Verhältnisse und Bewertung der Notwendigkeit von Kommentaren
- **Performance-Auswirkungsbewertung:** Bewertung, ob defensive Praktiken die Systemperformance beeinträchtigen
- **Entwickler-Verhaltensumfragen:** Sammlung von Feedback zu den Motivationen für Programmierentscheidungen
- **Nachverfolgung der Code-Stil-Entwicklung:** Beobachtung, wie sich Programmiermuster als Reaktion auf Review-Feedback ändern

## Examples

Ein Entwickler, der zuvor umfangreiches Feedback zur Variablenbenennung erhalten hat, beginnt, extrem lange, beschreibende Namen wie `userAuthenticationTokenValidationResult` statt `authResult` zu nutzen, was den Code schwerer lesbar macht, aber hofft, Kritik an der Benennung zu verhindern. Er fügt auch Kommentare für jede Zeile hinzu, die offensichtliche Operationen erklären, wie `// Zähler um 1 erhöhen` und `// Prüfen, ob Nutzer existiert`, um gründliche Dokumentation zu demonstrieren. Der resultierende Code ist doppelt so lang wie nötig und tatsächlich schwerer verständlich, trotz der "Verbesserungen". Ein weiteres Beispiel betrifft einen Entwickler, der dreifach verschachtelte Fehlerbehandlung für Szenarien umsetzt, die realistisch nicht auftreten können, weil ein vorheriger Reviewer seinen Fehlerbehandlungsansatz infrage gestellt hat. Er fügt Validierung für unmögliche Bedingungen und defensive Prüfungen hinzu, die nie auslösen, was die Code-Logik erheblich verkompliziert und die Performance beeinträchtigt, alles um potenzielle Kritik an unzureichender Fehlerbehandlung zu vermeiden.
