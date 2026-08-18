---
title: Verworrene Querschnittsbelange
description: Eine Situation, in der Querschnittsbelange wie Logging, Sicherheit und
  Transaktionen eng mit der Geschäftslogik gekoppelt sind.
category:
- Architecture
- Code
related_problems:
- slug: tight-coupling-issues
  similarity: 0.6
- slug: deployment-coupling
  similarity: 0.6
- slug: mixed-coding-styles
  similarity: 0.55
- slug: spaghetti-code
  similarity: 0.55
- slug: difficult-code-reuse
  similarity: 0.55
- slug: high-coupling-low-cohesion
  similarity: 0.55
solutions:
- incremental-refactoring
- modularization-and-bounded-contexts
- aspect-oriented-programming-aop
- high-cohesion
- layered-architecture
- change-impact-analysis
- separation-of-concerns
- code-hotspot-analysis
- preparatory-refactoring
- lightweight-design-review
layout: problem
lang: de
en_slug: tangled-cross-cutting-concerns
---

## Description
Verworrene Querschnittsbelange ist eine Situation, in der Querschnittsbelange wie Logging, Sicherheit und Transaktionen eng mit der Geschäftslogik gekoppelt sind. Dies ist ein häufiges Problem in monolithischen Architekturen, wo es keine klare Trennung der Zuständigkeiten gibt. Verworrene Querschnittsbelange können zu einer Reihe von Problemen führen, einschließlich Code-Duplizierung, Problemen durch enge Kopplung und schwer testbarem Code.

## Indicators ⟡
- Derselbe Code für Logging, Sicherheit oder Transaktionen wird an mehreren Stellen wiederholt.
- Es ist nicht möglich, die Implementierung eines Querschnittsbelangs zu ändern, ohne die Geschäftslogik zu beeinflussen.
- Es ist nicht möglich, die Geschäftslogik zu testen, ohne auch die Querschnittsbelange zu testen.
- Der Code ist schwer zu verstehen und zu warten.

## Symptoms ▲

- [Code-Duplizierung](code-duplizierung.md)
<br/>  Querschnittslogik wie Logging und Sicherheit wird in jede Komponente kopiert, statt zentralisiert zu werden.
- [Schwer testbarer Code](schwer-testbarer-code.md)
<br/>  Geschäftslogik, die mit Querschnittsbelangen verwoben ist, kann nicht isoliert getestet werden, ohne auch Logging, Sicherheit usw. auszuüben.
- [Wellenwirkung von Änderungen](wellenwirkung-von-aenderungen.md)
<br/>  Das Ändern eines Querschnittsbelangs wie Logging erfordert Modifikationen über alle Komponenten hinweg, in die er eingebettet ist.
- [Hohe Wartungskosten](hohe-wartungskosten.md)
<br/>  Die Wartung von Querschnittslogik, die über die Codebasis verstreut ist, erfordert unverhältnismäßigen Aufwand für jede Änderung.
- [Komplexe und unklare Logik](komplexe-und-unklare-logik.md)
<br/>  Geschäftslogik wird schwer zu verstehen, wenn sie mit Transaktionsmanagement, Sicherheitsprüfungen und Logging-Code verschachtelt ist.

## Causes ▼

- [Hohe Kopplung und geringe Kohäsion](hohe-kopplung-und-geringe-kohaesion.md)
<br/>  Schlechte Trennung der Zuständigkeiten auf architektonischer Ebene führt dazu, dass Querschnittslogik direkt in Geschäftskomponenten eingebettet wird.
- [Feature-Creep ohne Refactoring](feature-creep-ohne-refactoring.md)
<br/>  Das kontinuierliche Hinzufügen von Features ohne Umstrukturierung führt dazu, dass Querschnittsbelange graduell in die Geschäftslogik gemischt werden.
- [Unerfahrene Entwickler](unerfahrene-entwickler.md)
<br/>  Entwickler ohne Erfahrung mit Mustern der Trennung von Zuständigkeiten betten Querschnittslogik direkt in Geschäftscode ein.

## Detection Methods ○
- **Code-Reviews:** Suche nach Code, in dem Querschnittsbelange mit der Geschäftslogik vermischt sind.
- **Statische Analysewerkzeuge:** Nutzung von Werkzeugen zur Identifikation duplizierten Codes und anderer Code-Smells.
- **Architekturdiagramme:** Erstellung eines Diagramms der Systemarchitektur zur Identifikation, wo sich die Querschnittsbelange befinden.

## Examples
Ein Unternehmen hat eine große, monolithische E-Commerce-Anwendung. Die Anwendung hat mehrere verschiedene Services, einschließlich eines Produktkatalogs, eines Warenkorbs und eines Zahlungsgateways. Der Code für Logging, Sicherheit und Transaktionen ist in allen Services dupliziert. Dies macht es schwierig, die Implementierung eines Querschnittsbelangs zu ändern, und es macht es auch schwierig, die Geschäftslogik isoliert zu testen. Infolgedessen ist der Code schwer zu warten, und die Codequalität ist schlecht.
