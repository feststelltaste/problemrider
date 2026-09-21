---
title: Undefinierte Code-Stil-Richtlinien
description: Dem Team fehlen klare, vereinbarte Coding-Standards, was zu subjektivem
  stilistischem Feedback und inkonsistentem Code führt.
category:
- Code
- Process
related_problems:
- slug: inconsistent-codebase
  similarity: 0.8
- slug: inconsistent-coding-standards
  similarity: 0.75
- slug: style-arguments-in-code-reviews
  similarity: 0.75
- slug: inconsistent-naming-conventions
  similarity: 0.7
- slug: mixed-coding-styles
  similarity: 0.7
- slug: inadequate-code-reviews
  similarity: 0.65
solutions:
- static-analysis-and-linting
- code-conventions
- compatibility-standards
- secure-coding-guidelines
- security-policies-for-development
- style-guide
- communities-of-practice
- clean-code
- code-reviews
- code-quality-gates
- code-review-guidelines
- quality-ratchet
- automated-code-migration
- large-scale-refactoring
layout: problem
lang: de
en_slug: undefined-code-style-guidelines
---

## Description
Wenn einem Projekt klar definierte und dokumentierte Code-Stil-Richtlinien fehlen, sind Entwickler auf sich selbst gestellt, was in einer chaotischen und inkonsistenten Codebasis resultiert. Dieses Problem geht über bloße Ästhetik hinaus; es beeinflusst Lesbarkeit, Wartbarkeit und die Leichtigkeit, mit der neue Entwickler eingearbeitet werden können. Ohne einen zu befolgenden Standard werden Code-Reviews subjektiv und zeitaufwendig, wobei sie sich auf triviale Stilfragen statt substantielle Logik konzentrieren. Die Etablierung und Durchsetzung eines konsistenten Stils ist eine fundamentale Praxis für jedes gesunde Softwareprojekt.

## Indicators ⟡
- Es gibt keinen Stilguide für das Projekt.
- Das Team hat einen Stilguide, aber er wird nicht durchgesetzt.
- Es gibt häufige Diskussionen über Stil in Code-Reviews.
- Die Codebasis ist eine Mischung verschiedener Stile.

## Symptoms ▲

- [Inkonsistente Codebasis](inkonsistente-codebasis.md)
<br/>  Ohne definierte Stilrichtlinien wendet jeder Entwickler seine eigene Formatierung und Konventionen an, was zu inkonsistentem Code führt.
- [Stildiskussionen in Code-Reviews](stildiskussionen-in-code-reviews.md)
<br/>  Ohne einen autoritativen Stilguide arten Code-Reviews in subjektive Debatten über Stilpräferenzen aus.
- [Gemischte Coding-Stile](gemischte-coding-stile.md)
<br/>  Das Fehlen vereinbarter Standards führt direkt zu einer Codebasis mit widersprüchlicher Formatierung und Namenskonventionen.
- [Ineffizienz im Code-Review](ineffizienz-im-code-review.md)
<br/>  Ohne klare Stilregeln verbringen Reviewer exzessive Zeit mit subjektiven Stilfragen statt substantiellem Logik-Review.
- [Inkonsistente Namenskonventionen](inkonsistente-namenskonventionen.md)
<br/>  Ohne Richtlinien für Namenskonventionen wählen Entwickler ihre eigenen Benennungsmuster, was Verwirrung in der Codebasis schafft.

## Causes ▼

- [Fehlende Eigenverantwortung und Rechenschaftspflicht](fehlende-eigenverantwortung-und-rechenschaftspflicht.md)
<br/>  Niemand übernimmt Verantwortung für die Etablierung und Pflege von Code-Stil-Standards für das Team.
- [Auswirkung von Team-Fluktuation](auswirkung-von-team-fluktuation.md)
<br/>  Häufige Teamfluktuation macht es schwierig, konsistente Coding-Standards zu etablieren und zu pflegen, während neue Mitglieder unterschiedliche Gewohnheiten mitbringen.
- [Zeitdruck](zeitdruck.md)
<br/>  Unter Zeitdruck wird die Etablierung und Dokumentation von Coding-Standards als nicht essentiell angesehen und unbegrenzt aufgeschoben.

## Detection Methods ○

- **Code-Review-Analyse:** Beobachtung der Häufigkeit und Art stilbezogener Kommentare in Pull Requests.
- **Codebasis-Audit:** Manuelle Inspektion verschiedener Teile der Codebasis zur Identifikation stilistischer Variationen.
- **Entwicklerbefragungen/Interviews:** Befragung von Entwicklern zu ihrem Verständnis von Coding-Standards und jeglicher Verwirrung, die sie erleben.
- **Versuch, einen Linter/Formatter auszuführen:** Das Ausführen eines Linters oder Formatters ohne Konfigurationsdatei wird das Fehlen definierter Regeln hervorheben.

## Examples
Ein neues Feature wird von zwei verschiedenen Entwicklern entwickelt. Einer nutzt Tabs zur Einrückung, der andere Leerzeichen. Einer bevorzugt `camelCase` für alle Variablen, der andere `snake_case`. Wenn ihr Code gemerged wird, ist die resultierende Datei ein Durcheinander widersprüchlicher Stile, was sie schwer zu lesen und zu warten macht. In einem anderen Fall bricht während eines Code-Reviews eine Debatte darüber aus, ob eine Funktion `getUserData` oder `get_user_data` genannt werden sollte. Ohne klare Richtlinie ist die Diskussion subjektiv und unproduktiv, was wertvolle Review-Zeit verschwendet. Klare und konsistent angewendete Code-Stil-Richtlinien sind fundamental für eine gesunde Codebasis. Sie reduzieren kognitive Last, verbessern Lesbarkeit, erleichtern Zusammenarbeit und ermöglichen effektive Nutzung automatisierter Werkzeuge, was entscheidend für die Aufrechterhaltung von Qualität bei Legacy-System-Modernisierungsbemühungen ist.
