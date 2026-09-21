---
title: Wirkungslosigkeit automatisierter Werkzeuge
description: Eine Situation, in der automatisierte Werkzeuge wie Linter und Formatter
  aufgrund der Inkonsistenz der Codebasis nicht wirksam sind.
category:
- Code
- Process
related_problems:
- slug: inconsistent-coding-standards
  similarity: 0.65
- slug: inconsistent-behavior
  similarity: 0.65
- slug: tool-limitations
  similarity: 0.6
- slug: inconsistent-codebase
  similarity: 0.6
- slug: difficult-code-reuse
  similarity: 0.6
- slug: undefined-code-style-guidelines
  similarity: 0.6
solutions:
- static-analysis-and-linting
- code-quality-gates
- code-conventions
- development-workflow-automation
- code-metrics
- code-hotspot-analysis
- fast-feedback-loops
- team-retrospectives
- code-review-guidelines
- communities-of-practice
layout: problem
lang: de
en_slug: automated-tooling-ineffectiveness
---

## Description
Wirkungslosigkeit automatisierter Werkzeuge ist eine Situation, in der automatisierte Werkzeuge wie Linter und Formatter aufgrund der Inkonsistenz der Codebasis nicht wirksam sind. Dies ist ein verbreitetes Problem in Teams, die keine klaren Coding-Standards haben. Die Wirkungslosigkeit automatisierter Werkzeuge kann zu einer Reihe von Problemen führen, darunter eine sinkende Codequalität, eine steigende Anzahl von Fehlern und eine allgemeine Verlangsamung des Entwicklungsprozesses.

## Indicators ⟡
- Die automatisierten Werkzeuge melden ständig eine große Anzahl von Verstößen.
- Entwickler ignorieren die von den automatisierten Werkzeugen gemeldeten Verstöße.
- Die automatisierten Werkzeuge sind nicht in der Lage, alle Verstöße automatisch zu beheben.
- Die automatisierten Werkzeuge werden nicht konsistent von allen Entwicklern genutzt.

## Symptoms ▲

- [Geringere Codequalität](geringere-codequalitaet.md)
<br/>  Ohne wirksame automatisierte Werkzeuge zur Fehlererkennung sinkt die allgemeine Codequalität.
- [Erhöhtes Risiko für Fehler](erhoehtes-risiko-fuer-fehler.md)
<br/>  Wirkungslose Linter und Analysewerkzeuge übersehen häufige Programmierfehler, was das Fehlerrisiko erhöht.
- [Stildiskussionen in Code-Reviews](stildiskussionen-in-code-reviews.md)
<br/>  Wenn automatisierte Formatter wirkungslos sind, müssen Stilunstimmigkeiten manuell in Code-Reviews geklärt werden.
- [Erhöhte manuelle Arbeit](erhoehte-manuelle-arbeit.md)
<br/>  Wenn automatisierte Werkzeuge ihre Aufgabe nicht erfüllen können, müssen Entwickler Prüfungen manuell durchführen, die automatisiert sein sollten.
- [Inkonsistente Codebasis](inkonsistente-codebasis.md)
<br/>  Wenn automatisierte Werkzeuge wirkungslos sind, können sie Konsistenz nicht durchsetzen, wodurch die Codebasis inkonsistent bleibt oder wird.

## Causes ▼

- [Inkonsistente Coding-Standards](inkonsistente-coding-standards.md)
<br/>  Ohne konsistente Coding-Standards können automatisierte Werkzeuge nicht wirksam konfiguriert werden.
- [Gemischte Coding-Stile](gemischte-coding-stile.md)
<br/>  Eine Codebasis mit gemischten Stilen erzeugt überwältigende Werkzeug-Verstöße, was Entwickler dazu bringt, die Werkzeuge zu ignorieren.
- [Undefinierte Code-Stil-Richtlinien](undefinierte-code-stil-richtlinien.md)
<br/>  Ohne definierte Stilrichtlinien fehlt eine Grundlage für die Konfiguration automatisierter Werkzeuge.
- [Werkzeugeinschränkungen](werkzeugeinschraenkungen.md)
<br/>  Die Werkzeuge selbst können Einschränkungen haben, die sie daran hindern, die Komplexität oder die Muster der Codebasis zu bewältigen.

## Detection Methods ○
- **Analyse der Ausgabe automatisierter Werkzeuge:** Suche nach einer großen Anzahl von Verstößen.
- **Team-Umfragen:** Befragung von Entwicklern, ob sie die automatisierten Werkzeuge konsistent nutzen.
- **Retrospektiven:** Nutzung von Retrospektiven zur Identifikation von Problemen mit den automatisierten Werkzeugen.

## Examples
Ein Team hat einen Linter für sein Projekt konfiguriert. Der Linter meldet jedoch ständig eine große Anzahl von Verstößen. Die Entwickler ignorieren die Verstöße, weil es so viele davon gibt. Infolgedessen ist der Linter wirkungslos, und die Codebasis ist inkonsistent. Dies führt zu einer Reihe von Problemen, einschließlich sinkender Codequalität und einer steigenden Anzahl von Fehlern.
