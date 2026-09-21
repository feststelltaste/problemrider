---
title: Aufgeblähte Klasse
description: Eine Klasse, die so groß geworden ist, dass sie schwer zu verstehen,
  zu warten und zu testen ist.
category:
- Code
related_problems:
- slug: excessive-class-size
  similarity: 0.8
- slug: monolithic-functions-and-classes
  similarity: 0.65
- slug: god-object-anti-pattern
  similarity: 0.65
- slug: uncontrolled-codebase-growth
  similarity: 0.6
- slug: feature-bloat
  similarity: 0.6
- slug: poorly-defined-responsibilities
  similarity: 0.55
solutions:
- incremental-refactoring
- code-metrics
- high-cohesion
- code-hotspot-analysis
- dependency-breaking-techniques
- solid-principles
- separation-of-concerns
- preparatory-refactoring
- clean-code
- code-reading-sessions
layout: problem
lang: de
en_slug: bloated-class
---

## Description
Eine aufgeblähte Klasse ist eine Klasse, die im Laufe der Zeit zu viele Verantwortlichkeiten angehäuft hat. Sie beginnt oft als kleine, gut entworfene Klasse, wächst aber mit dem Hinzufügen neuer Features in Größe und Komplexität. Dies macht sie schwer zu verstehen, zu warten und zu testen. Aufgeblähte Klassen sind ein verbreiteter Code Smell und ein Zeichen für technische Schulden.

## Indicators ⟡
- Eine Klasse mit einer großen Anzahl von Methoden und Eigenschaften.
- Eine Klasse, die schwer zu benennen ist, weil sie zu viele Dinge tut.
- Eine Klasse, die häufig von mehreren Entwicklern aus unterschiedlichen Gründen geändert wird.
- Eine Klasse, die schwer isoliert zu testen ist.

## Symptoms ▲

- [Schwer verständliche Codebasis](schwer-verstaendliche-codebasis.md)
<br/>  Übergroße Klassen mit zu vielen Verantwortlichkeiten werden für Entwickler extrem schwer zu verstehen und nachzuvollziehen.
- [Regressionsfehler](regressionsfehler.md)
<br/>  Das Ändern eines Teils einer aufgeblähten Klasse bricht häufig nicht verwandte Funktionalität innerhalb derselben Klasse.
- [Merge-Konflikte](merge-konflikte.md)
<br/>  Mehrere Entwickler, die an unterschiedlichen Features innerhalb derselben aufgeblähten Klasse arbeiten, erzeugen häufig Merge-Konflikte.
- [Hohe technische Schulden](hohe-technische-schulden.md)
<br/>  Aufgeblähte Klassen sind eine eindeutige Form technischer Schulden.
- [Langsame Feature-Entwicklung](langsame-feature-entwicklung.md)
<br/>  Aufgeblähte Klassen mit vielen Verantwortlichkeiten verlangsamen die Feature-Entwicklung, weil Entwickler die gesamte Klasse verstehen müssen.

## Causes ▼

- [Inkonsistente Coding-Standards](inkonsistente-coding-standards.md)
<br/>  Ohne Standards, die Single Responsibility und Klassengrößenbegrenzungen durchsetzen, wachsen Klassen unkontrolliert im Laufe der Zeit.
- [Feature-Creep](feature-creep.md)
<br/>  Das kontinuierliche Hinzufügen von Features ohne Refactoring führt dazu, dass sich Verantwortlichkeiten in bestehenden Klassen anhäufen.
- [Refactoring-Vermeidung](refactoring-vermeidung.md)
<br/>  Das Vermeiden des Aufwands, Klassen in kleinere, fokussierte Komponenten aufzuteilen, lässt die Aufblähung unkontrolliert fortschreiten.
- [Zeitdruck](zeitdruck.md)
<br/>  Unter Termindruck fügen Entwickler Funktionalität zu bestehenden Klassen hinzu, statt neue ordentlich zu entwerfen.

## Detection Methods ○
- **Code-Metrik-Werkzeuge:** Nutzung von Werkzeugen zur Messung von Klassengröße, Methodenanzahl und zyklomatischer Komplexität.
- **Code-Reviews:** Suche nach Klassen, die schwer zu verstehen und zu überprüfen sind.
- **Statische Analysewerkzeuge:** Nutzung von Werkzeugen zur Identifikation von Code Smells wie großen Klassen und langen Methoden.

## Examples
Eine `User`-Klasse in einer Social-Media-Anwendung, die für alles verantwortlich ist, von Authentifizierung und Autorisierung über Profilverwaltung bis hin zur News-Feed-Generierung und dem Versenden von Benachrichtigungen. Die Klasse hat über 50 Methoden und 1000 Codezeilen. Wenn ein Entwickler eine Änderung an der News-Feed-Generierungslogik vornehmen möchte, muss er darauf achten, die Authentifizierungslogik nicht zu brechen. Es ist auch sehr schwierig, Unit-Tests für die Klasse zu schreiben, weil sie so viele Abhängigkeiten hat. Infolgedessen ist die Entwicklung langsam und fehleranfällig.
