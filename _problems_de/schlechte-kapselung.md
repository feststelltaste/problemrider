---
title: Schlechte Kapselung
description: Daten und das Verhalten, das auf diese Daten wirkt, sind nicht in einer
  einzigen, kohäsiven Einheit gebündelt, was zu fehlender Datenkapselung und hohem
  Kopplungsgrad führt.
category:
- Architecture
- Code
related_problems:
- slug: high-coupling-low-cohesion
  similarity: 0.7
- slug: over-reliance-on-utility-classes
  similarity: 0.65
- slug: misunderstanding-of-oop
  similarity: 0.6
- slug: tight-coupling-issues
  similarity: 0.6
- slug: poorly-defined-responsibilities
  similarity: 0.6
- slug: incomplete-knowledge
  similarity: 0.55
solutions:
- incremental-refactoring
- dependency-breaking-techniques
- solid-principles
- separation-of-concerns
- high-cohesion
- facades
- abstraction
- lightweight-design-review
- code-reading-sessions
- preparatory-refactoring
layout: problem
lang: de
en_slug: poor-encapsulation
---

## Description
Schlechte Kapselung ist ein häufiges Designproblem in der objektorientierten Programmierung. Es tritt auf, wenn Daten und das Verhalten, das auf diese Daten wirkt, nicht in einer einzigen, kohäsiven Einheit gebündelt sind. Dies kann zu einer Reihe von Problemen führen, einschließlich fehlender Datenkapselung, hohem Kopplungsgrad und einem System, das schwer zu verstehen und zu warten ist. Schlechte Kapselung ist oft ein Zeichen für mangelndes Verständnis der Prinzipien objektorientierten Designs.

## Indicators ⟡
- Klassen haben eine große Anzahl öffentlicher Felder.
- Klassen haben eine große Anzahl von Gettern und Settern.
- Daten werden zwischen einer großen Anzahl verschiedener Objekte herumgereicht.
- Es ist schwierig zu verstehen, wie Daten im System genutzt werden.

## Symptoms ▲

- [Probleme durch enge Kopplung](probleme-durch-enge-kopplung.md)
<br/>  Das direkte Offenlegen interner Daten schafft Abhängigkeiten zwischen Komponenten, was sie eng koppelt.
- [Wellenwirkung von Änderungen](wellenwirkung-von-aenderungen.md)
<br/>  Wenn interner Zustand öffentlich zugänglich ist, pflanzen sich Änderungen an Datenstrukturen durch die gesamte Codebasis fort.
- [Schwer testbarer Code](schwer-testbarer-code.md)
<br/>  Ohne Kapselung sind Komponenten schwer isoliert zu testen, weil sie von gemeinsam genutztem veränderlichem Zustand abhängen und ihn offenlegen.
- [Hohe Wartungskosten](hohe-wartungskosten.md)
<br/>  Schlechte Kapselung macht das System teurer in der Wartung, weil jede interne Änderung externe Konsumenten brechen kann.
- [Brüchige Codebasis](bruechige-codebasis.md)
<br/>  Fehlende Datenkapselung macht die Codebasis fragil, da viele Komponenten von Implementierungsdetails abhängen, die sich ändern könnten.

## Causes ▼

- [Missverständnis der Objektorientierung](missverstaendnis-der-objektorientierung.md)
<br/>  Entwickler, die objektorientierte Prinzipien nicht verstehen, versäumen es, Daten und Verhalten ordentlich zu kapseln.
- [Prozedurale Programmierung in OOP-Sprachen](prozedurale-programmierung-in-oop-sprachen.md)
<br/>  Das Anwenden prozeduralen Denkens auf OOP-Sprachen resultiert in Datenstrukturen ohne zugehöriges Verhalten.
- [Unerfahrene Entwickler](unerfahrene-entwickler.md)
<br/>  Weniger erfahrene Entwickler verfallen oft standardmäßig auf öffentliche Felder und Getter/Setter, ohne die Vorteile der Kapselung zu verstehen.
- [Termindruck](termindruck.md)
<br/>  Unter Zeitdruck nehmen Entwickler Abkürzungen, indem sie Felder öffentlich machen, statt ordentliche Schnittstellen zu designen.

## Detection Methods ○
- **Code-Reviews:** Code-Reviews sind eine großartige Methode zur Identifikation von Kapselungsproblemen.
- **Statische Analyse:** Nutzung statischer Analysewerkzeuge zur Identifikation von Klassen mit einer großen Anzahl öffentlicher Felder oder Getter/Setter.
- **Abhängigkeitsanalyse:** Analyse der Abhängigkeiten zwischen den Komponenten des Systems zur Identifikation von Bereichen hoher Kopplung.
- **Code-Abdeckung:** Messung der Testabdeckung Ihrer Tests. Eine niedrige Codeabdeckung kann ein Zeichen für schlechte Kapselung sein.

## Examples
Eine Klasse hat ein öffentliches Feld, auf das eine große Anzahl anderer Klassen zugreift. Dies ist ein Beispiel für schlechte Kapselung. Das Problem könnte gelöst werden, indem das Feld privat gemacht wird und eine öffentliche Methode zum Zugriff darauf bereitgestellt wird. Dies würde die Implementierungsdetails der Klasse verbergen und es einfacher machen, die Klasse in Zukunft zu ändern, ohne andere Teile des Systems zu brechen.
