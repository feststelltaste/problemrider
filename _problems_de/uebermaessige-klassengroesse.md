---
title: Übermäßige Klassengröße
description: Klassen werden übermäßig groß und komplex, was sie schwer verständlich,
  wartbar und testbar macht.
category:
- Architecture
- Code
related_problems:
- slug: bloated-class
  similarity: 0.8
- slug: monolithic-functions-and-classes
  similarity: 0.7
- slug: god-object-anti-pattern
  similarity: 0.7
- slug: large-pull-requests
  similarity: 0.65
- slug: increased-cognitive-load
  similarity: 0.65
- slug: large-estimates-for-small-changes
  similarity: 0.6
solutions:
- incremental-refactoring
- code-hotspot-analysis
- dependency-breaking-techniques
- solid-principles
- clean-code
- high-cohesion
- separation-of-concerns
- code-metrics
layout: problem
lang: de
en_slug: excessive-class-size
---

## Description
Übermäßige Klassengröße ist ein Code Smell, bei dem eine Klasse so groß geworden ist, dass sie schwer zu handhaben ist. Große Klassen häufen oft zu viele Verantwortlichkeiten an und verletzen dabei das Single-Responsibility-Prinzip. Diese Komplexität macht den Code schwerer zu lesen, zu testen und zu warten, was die Wahrscheinlichkeit von Fehlern erhöht und die Entwicklung verlangsamt.

## Indicators ⟡
- Klassen mit hoher Zeilenanzahl (z. B. über 500 oder 1000 Zeilen).
- Eine einzelne Klasse, die häufig von mehreren Entwicklern aus unterschiedlichen Gründen geändert wird.
- Schwierigkeit, die Klasse prägnant zu benennen, weil sie zu viele Dinge tut.
- Die Klasse hat eine große Anzahl von Methoden und Instanzvariablen.

## Symptoms ▲

- [Schwer testbarer Code](schwer-testbarer-code.md)
<br/>  Große Klassen mit vielen Verantwortlichkeiten und Abhängigkeiten sind extrem schwer isoliert zu testen.
- [Erhöhte kognitive Last](erhoehte-kognitive-last.md)
<br/>  Entwickler müssen die gesamte große Klasse im Arbeitsgedächtnis behalten, um sicher Änderungen vorzunehmen, was die mentale Belastung erhöht.
- [Hohe Kopplung und geringe Kohäsion](hohe-kopplung-und-geringe-kohaesion.md)
<br/>  Übergroße Klassen mischen typischerweise unzusammenhängende Verantwortlichkeiten, was zu geringer Kohäsion und hoher Kopplung zu vielen anderen Komponenten führt.
- [Erhöhtes Risiko für Fehler](erhoehtes-risiko-fuer-fehler.md)
<br/>  Die Komplexität großer Klassen macht es wahrscheinlicher, dass Änderungen unbeabsichtigte Nebeneffekte und Defekte einführen.
- [Wellenwirkung von Änderungen](wellenwirkung-von-aenderungen.md)
<br/>  Änderungen an einer übermäßig großen Klasse betreffen viele unterschiedliche Funktionalitäten, was kaskadierende Modifikationen im gesamten System verursacht.
- [Große Pull Requests](grosse-pull-requests.md)
<br/>  Änderungen an großen Klassen erzeugen tendenziell große Pull Requests, weil die Klasse viele Belange gleichzeitig betrifft.
- [Schwer verständliche Codebasis](schwer-verstaendliche-codebasis.md)
<br/>  Übermäßig große Klassen sind von Natur aus schwer verständlich.

## Causes ▼

- [Feature-Creep ohne Refactoring](feature-creep-ohne-refactoring.md)
<br/>  Das kontinuierliche Hinzufügen von Features ohne Refactoring führt dazu, dass Klassen im Laufe der Zeit immer mehr Verantwortlichkeiten aufnehmen.
- [Refactoring-Vermeidung](refactoring-vermeidung.md)
<br/>  Das Vermeiden von Refactoring bedeutet, dass übergroße Klassen nie in kleinere, fokussiertere Komponenten aufgeteilt werden.
- [Missverständnis der Objektorientierung](missverstaendnis-der-objektorientierung.md)
<br/>  Fehlendes Verständnis der SOLID-Prinzipien, besonders Single Responsibility, führt dazu, dass Entwickler alle verwandte Logik in einer Klasse anhäufen.
- [Kurzfristiger Fokus](kurzfristiger-fokus.md)
<br/>  Die Priorisierung schneller Feature-Lieferung über Codestruktur führt dazu, dass Entwickler zu bestehenden Klassen hinzufügen, statt ordentliche Abstraktionen zu entwerfen.
- [Fehlende Eigenverantwortung und Rechenschaftspflicht](fehlende-eigenverantwortung-und-rechenschaftspflicht.md)
<br/>  Ohne klare Code-Eigenverantwortung übernimmt niemand die Verantwortung für die Aufrechterhaltung von Klassengrenzen, was unkontrolliertes Wachstum von Klassen erlaubt.

## Detection Methods ○
- **Code-Metrik-Werkzeuge:** Nutzung statischer Analysewerkzeuge zur Messung von Klassengröße, zyklomatischer Komplexität und anderen Metriken.
- **Code-Reviews:** Regelmäßige Überprüfung von Code auf große Klassen und Klassen mit mehreren Verantwortlichkeiten.
- **Verantwortlichkeitsanalyse:** Analyse der Methoden und Eigenschaften einer Klasse, um festzustellen, ob sie eine einzige, gut definierte Verantwortlichkeit hat.

## Examples
In einer großen E-Commerce-Anwendung beginnt eine Klasse namens `Product` damit, Produktinformationen wie Name, Preis und Beschreibung zu verwalten. Im Laufe der Zeit wird sie geändert, um auch Bestandsverwaltung, Lieferantendetails, Kundenbewertungen und Rabattberechnungen zu handhaben. Die Klasse wächst auf Tausende Codezeilen an, und eine Änderung an der Rabattlogik riskiert, Bestandsaktualisierungen zu brechen. Dies ist ein klassisches Beispiel für übermäßige Klassengröße, bei dem eine einzelne Klasse zu einem "God Object" geworden ist, was das gesamte System brüchig und schwer wartbar macht.
