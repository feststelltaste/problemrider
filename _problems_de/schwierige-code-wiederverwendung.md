---
title: Schwierige Code-Wiederverwendung
description: Code lässt sich schwer in anderen Kontexten wiederverwenden, weil er
  nicht modular und wiederverwendbar gestaltet ist.
category:
- Architecture
- Code
related_problems:
- slug: difficult-to-understand-code
  similarity: 0.7
- slug: difficult-code-comprehension
  similarity: 0.7
- slug: difficult-to-test-code
  similarity: 0.65
- slug: brittle-codebase
  similarity: 0.65
- slug: code-duplication
  similarity: 0.65
- slug: partial-bug-fixes
  similarity: 0.65
solutions:
- modularization-and-bounded-contexts
- abstraction
- abstraction-layers
- api-first-development
- bridges
- cross-platform-frameworks
- design-tokens
- facades
- fluent-interfaces
- modulith
- dependency-injection
layout: problem
lang: de
en_slug: difficult-code-reuse
---

## Description
Schwierige Code-Wiederverwendung ist ein verbreitetes Problem in der Softwareentwicklung. Sie entsteht, wenn es schwierig ist, Code in unterschiedlichen Kontexten wiederzuverwenden, weil er nicht modular und wiederverwendbar gestaltet ist. Dies kann zu einer Reihe von Problemen führen, einschließlich Code-Duplizierung, einem hohen Grad an Kopplung und einem System, das schwer zu warten und weiterzuentwickeln ist. Schwierige Code-Wiederverwendung ist oft ein Zeichen für mangelnde Erfahrung mit Software-Design-Prinzipien und -Mustern.

## Indicators ⟡
- Die Codebasis ist voller duplizierter Code.
- Die Komponenten des Systems sind eng gekoppelt.
- Es ist schwierig, eine Komponente aus dem System zu extrahieren und in einem anderen Kontext wiederzuverwenden.
- Das Team erfindet ständig das Rad neu.

## Symptoms ▲

- [Code-Duplizierung](code-duplizierung.md)
<br/>  Wenn Code nicht wiederverwendet werden kann, kopieren Entwickler ähnliche Implementierungen, was zu dupliziertem Code im gesamten System führt.
- [Erhöhte Entwicklungskosten](erhoehte-entwicklungskosten.md)
<br/>  Dieselbe Funktionalität wiederholt zu bauen, statt sie wiederzuverwenden, erhöht Entwicklungszeit und -kosten.
- [Inkonsistentes Verhalten](inkonsistentes-verhalten.md)
<br/>  Mehrere Implementierungen ähnlicher Funktionalität weichen unweigerlich im Laufe der Zeit auseinander, was inkonsistentes Verhalten verursacht.
- [Hohe Wartungskosten](hohe-wartungskosten.md)
<br/>  Die Wartung mehrerer Kopien ähnlichen Codes vervielfacht den Aufwand für Fehlerbehebungen und Aktualisierungen.
- [Langsame Feature-Entwicklung](langsame-feature-entwicklung.md)
<br/>  Die Unfähigkeit, bestehende Komponenten wiederzuverwenden, bedeutet, dass jedes neue Feature den Bau gemeinsamer Funktionalität von Grund auf erfordert.

## Causes ▼

- [Probleme durch enge Kopplung](probleme-durch-enge-kopplung.md)
<br/>  Eng gekoppelter Code kann nicht extrahiert und in unterschiedlichen Kontexten wiederverwendet werden, weil er von zu vielen anderen Komponenten abhängt.
- [Hohe Kopplung und geringe Kohäsion](hohe-kopplung-und-geringe-kohaesion.md)
<br/>  Komponenten, die viele Verantwortlichkeiten mischen und stark voneinander abhängen, können nicht unabhängig wiederverwendet werden.
- [God-Object-Antipattern](god-object-antipattern.md)
<br/>  God Objects, die zu viel Funktionalität enthalten, können nicht wiederverwendet werden, weil konsumierender Code alle Verantwortlichkeiten des Objekts übernehmen muss.
- [Monolithische Funktionen und Klassen](monolithische-funktionen-und-klassen.md)
<br/>  Große monolithische Komponenten bündeln zu viel Funktionalität zusammen, was es unmöglich macht, nur die benötigten Teile wiederzuverwenden.

## Detection Methods ○
- **Code-Duplizierungsanalyse:** Nutzung statischer Analysewerkzeuge zur Identifikation duplizierten Codes.
- **Abhängigkeitsanalyse:** Analyse der Abhängigkeiten zwischen den Komponenten des Systems zur Identifikation von Bereichen hoher Kopplung.
- **Code-Reviews:** Code-Reviews sind ein guter Weg, um Gelegenheiten zur Code-Wiederverwendung zu identifizieren.
- **Komponentenbibliotheks-Audit:** Audit der Komponentenbibliothek des Teams, um zu sehen, ob sie effektiv genutzt wird.

## Examples
Ein Unternehmen hat mehrere unterschiedliche Webanwendungen. Jede Anwendung hat ihre eigene Implementierung eines Nutzerauthentifizierungssystems. Dies ist ein Beispiel für schwierige Code-Wiederverwendung. Das Problem könnte gelöst werden, indem eine einzige, wiederverwendbare Nutzerauthentifizierungskomponente geschaffen wird, die von allen Webanwendungen des Unternehmens genutzt werden kann. Dies würde Code-Duplizierung reduzieren, die Wartbarkeit verbessern und es einfacher machen, neue Features zum Nutzerauthentifizierungssystem hinzuzufügen.
