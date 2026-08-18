---
title: Prozeduraler Hintergrund
description: Entwickler mit einem Hintergrund in prozeduraler Programmierung kämpfen
  möglicherweise damit, sich an objektorientiertes Denken anzupassen, was zur Entstehung
  von Code im prozeduralen Stil in einer objektorientierten Sprache führt.
category:
- Architecture
- Team
related_problems:
- slug: procedural-programming-in-oop-languages
  similarity: 0.7
- slug: misunderstanding-of-oop
  similarity: 0.7
- slug: over-reliance-on-utility-classes
  similarity: 0.65
- slug: insufficient-design-skills
  similarity: 0.6
- slug: difficult-code-reuse
  similarity: 0.55
- slug: undefined-code-style-guidelines
  similarity: 0.55
solutions:
- architecture-reviews
- solid-principles
- technical-skills-development
- pair-and-mob-programming
- refactoring-katas
- clean-code
- domain-driven-design
- code-reviews
layout: problem
lang: de
en_slug: procedural-background
---

## Description
Ein prozeduraler Hintergrund kann ein bedeutendes Hindernis beim Schreiben guten objektorientierten Codes sein. Entwickler, die gewohnt sind, prozedural zu denken, kämpfen möglicherweise damit, sich an objektorientiertes Denken anzupassen. Dies kann zur Entstehung von Code im prozeduralen Stil in einer objektorientierten Sprache führen, der schwer zu warten und weiterzuentwickeln sein kann. Ein prozeduraler Hintergrund ist ein häufiges Problem in der Softwarebranche, und es kann schwierig sein, es anzugehen.

## Indicators ⟡
- Die Codebasis ist voller statischer Methoden.
- Die Codebasis ist voller Utility-Klassen.
- Die Codebasis nutzt keine Vererbung oder Polymorphie.
- Die Codebasis ist schwer zu verstehen und zu warten.

## Symptoms ▲

- [Prozedurale Programmierung in OOP-Sprachen](prozedurale-programmierung-in-oop-sprachen.md)
<br/>  Entwickler mit prozeduraler Ausbildung schreiben natürlicherweise Code im prozeduralen Stil, selbst in OOP-Sprachen.
- [Übermäßige Abhängigkeit von Utility-Klassen](uebermaessige-abhaengigkeit-von-utility-klassen.md)
<br/>  Prozedurales Denken führt dazu, Funktionen in Utility-Klassen zu gruppieren, statt Verhalten in Objekten zu kapseln.
- [Missverständnis der Objektorientierung](missverstaendnis-der-objektorientierung.md)
<br/>  Entwickler, die in prozeduralen Paradigmen ausgebildet sind, wenden OOP-Konzepte wie Vererbung und Polymorphie möglicherweise falsch an.
- [Schlechte Kapselung](schlechte-kapselung.md)
<br/>  Prozedurale Gewohnheiten führen dazu, Daten offenzulegen und von Verhalten zu trennen, was Kapselungsprinzipien verletzt.
- [Schwierige Code-Wiederverwendung](schwierige-code-wiederverwendung.md)
<br/>  Prozedurale Codestrukturen sind schwerer über Kontexte hinweg wiederzuverwenden im Vergleich zu gut designten OOP-Komponenten.

## Causes ▼

- [Lücken in der Kompetenzentwicklung](luecken-in-der-kompetenzentwicklung.md)
<br/>  Entwicklern fehlt Schulung in OOP-Prinzipien, sodass sie ihre prozeduralen Programmiergewohnheiten beibehalten.

## Detection Methods ○
- **Code-Reviews:** Code-Reviews sind eine großartige Methode zur Identifikation von Code im prozeduralen Stil.
- **Statische Analyse:** Nutzung statischer Analysewerkzeuge zur Identifikation von Code, der objektorientierten Designprinzipien nicht folgt.
- **Entwicklerbefragungen:** Befragung von Entwicklern zu ihrem Vertrauen in ihre objektorientierten Design-Fähigkeiten.
- **Architektur-Bewertungen:** Durchführung einer Bewertung der Systemarchitektur zur Identifikation von Design-Mängeln.

## Examples
Ein Unternehmen stellt ein Team von Entwicklern mit Hintergrund in prozeduraler Programmierung ein. Das Team hat die Aufgabe, eine neue Webanwendung in einer objektorientierten Sprache zu bauen. Das Team kämpft damit, sich an objektorientiertes Denken anzupassen, und erstellt ein System, das schlecht designt und schwer zu warten ist. Das Unternehmen muss schließlich ein Team erfahrener objektorientierter Entwickler einstellen, um das gesamte System neu zu schreiben.
