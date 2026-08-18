---
title: Übermäßige Abhängigkeit von Utility-Klassen
description: Die exzessive Nutzung von Utility-Klassen mit statischen Methoden kann
  zu einem prozeduralen Programmierstil und fehlendem ordentlichem objektorientiertem
  Design führen.
category:
- Architecture
- Code
related_problems:
- slug: misunderstanding-of-oop
  similarity: 0.75
- slug: procedural-background
  similarity: 0.65
- slug: difficult-code-reuse
  similarity: 0.65
- slug: poor-encapsulation
  similarity: 0.65
- slug: insufficient-design-skills
  similarity: 0.6
- slug: inefficient-code
  similarity: 0.6
solutions:
- incremental-refactoring
- dependency-breaking-techniques
- domain-driven-design
- solid-principles
- high-cohesion
- separation-of-concerns
- domain-modeling
- lightweight-design-review
- code-reading-sessions
- preparatory-refactoring
- automated-code-migration
- large-scale-refactoring
layout: problem
lang: de
en_slug: over-reliance-on-utility-classes
---

## Description
Eine übermäßige Abhängigkeit von Utility-Klassen ist ein häufiges Designproblem in der objektorientierten Programmierung. Es tritt auf, wenn ein Team eine große Anzahl von Utility-Klassen mit statischen Methoden erstellt. Dies kann zu einem prozeduralen Programmierstil und fehlendem ordentlichem objektorientiertem Design führen. Eine übermäßige Abhängigkeit von Utility-Klassen ist oft ein Zeichen für ein Missverständnis der Prinzipien objektorientierter Programmierung.

## Indicators ⟡
- Die Codebasis ist voller Utility-Klassen.
- Die Codebasis ist voller statischer Methoden.
- Die Codebasis nutzt keine Vererbung oder Polymorphie.
- Die Codebasis ist schwer zu verstehen und zu warten.

## Symptoms ▲

- [Schwer testbarer Code](schwer-testbarer-code.md)
<br/>  Statische Utility-Methoden schaffen harte Abhängigkeiten, die nicht leicht gemockt oder ersetzt werden können, was Unit-Testing erschwert.
- [Hohe Kopplung und geringe Kohäsion](hohe-kopplung-und-geringe-kohaesion.md)
<br/>  Utility-Klassen schaffen implizite Abhängigkeiten über die Codebasis hinweg, da viele Komponenten von gemeinsam genutzten statischen Methoden abhängen, was die Kopplung erhöht.
- [Schwierige Code-Wiederverwendung](schwierige-code-wiederverwendung.md)
<br/>  Prozedurale Utility-Klassen bündeln nicht zusammenhängende Methoden, was es schwierig macht, spezifische Funktionalität wiederzuverwenden, ohne unnötige Abhängigkeiten mitzuziehen.
- [Code-Duplizierung](code-duplizierung.md)
<br/>  Wenn Utility-Klassen unhandlich werden, erstellen Entwickler neue Utility-Methoden statt bestehende zu finden, was zu duplizierter Logik führt.
- [Übermäßige Klassengröße](uebermaessige-klassengroesse.md)
<br/>  Utility-Klassen neigen dazu, unbegrenzt zu wachsen, während Entwickler mehr statische Methoden hinzufügen, und werden zu aufgeblähten Auffangbehältern.

## Causes ▼

- [Missverständnis der Objektorientierung](missverstaendnis-der-objektorientierung.md)
<br/>  Entwickler, die objektorientierte Designprinzipien nicht verstehen, verfallen standardmäßig darauf, statische Utility-Methoden zu erstellen, statt ordentlicher Objekte mit Verhalten.
- [Prozeduraler Hintergrund](prozeduraler-hintergrund.md)
<br/>  Entwickler mit prozeduralem Programmierhintergrund tendieren natürlicherweise zu statischen Utility-Funktionen statt objektorientiertem Design.
- [Bequemlichkeitsgetriebene Entwicklung](bequemlichkeitsgetriebene-entwicklung.md)
<br/>  Das Hinzufügen einer statischen Methode zu einer Utility-Klasse ist der schnellste und einfachste Ansatz, selbst wenn ordentliches OOP-Design angemessener wäre.

## Detection Methods ○
- **Code-Reviews:** Code-Reviews sind eine großartige Methode, um eine übermäßige Abhängigkeit von Utility-Klassen zu identifizieren.
- **Statische Analyse:** Nutzung statischer Analysewerkzeuge zur Identifikation von Klassen mit einer großen Anzahl statischer Methoden.
- **Abhängigkeitsanalyse:** Analyse der Abhängigkeiten zwischen den Komponenten des Systems zur Identifikation von Bereichen hoher Kopplung.
- **Code-Abdeckung:** Messung der Testabdeckung Ihrer Tests. Eine niedrige Codeabdeckung kann ein Zeichen für eine übermäßige Abhängigkeit von Utility-Klassen sein.

## Examples
Ein Unternehmen hat eine Codebasis, die voller Utility-Klassen ist. Die Klassen haben Namen wie `StringUtils`, `DateUtils` und `FileUtils`. Die Klassen enthalten eine große Anzahl statischer Methoden. Die Codebasis ist schwer zu verstehen und zu warten. Das Unternehmen muss schließlich ein Team erfahrener objektorientierter Entwickler einstellen, um das gesamte System neu zu schreiben.
