---
title: Feature-Creep ohne Refactoring
description: Die kontinuierliche Ergänzung neuer Features in einer Codebasis, ohne
  sich die Zeit zu nehmen, das Design zu refaktorieren und zu verbessern.
category:
- Code
- Process
related_problems:
- slug: feature-creep
  similarity: 0.85
- slug: refactoring-avoidance
  similarity: 0.7
- slug: uncontrolled-codebase-growth
  similarity: 0.7
- slug: slow-feature-development
  similarity: 0.7
- slug: scope-creep
  similarity: 0.65
- slug: high-technical-debt
  similarity: 0.65
solutions:
- incremental-refactoring
- performance-budgets
- improvement-budget
- definition-of-done
- technical-debt-backlog
- code-hotspot-analysis
- clean-code
- code-quality-gates
- preparatory-refactoring
layout: problem
lang: de
en_slug: feature-creep-without-refactoring
---

## Description
Feature-Creep ohne Refactoring ist der Prozess, kontinuierlich neue Features in eine Codebasis einzufügen, ohne sich die Zeit zu nehmen, das Design zu refaktorieren und zu verbessern. Dies führt zu einer schrittweisen Verschlechterung der Codebasis, wodurch sie zunehmend schwerer zu warten und zu erweitern ist. Es ist ein verbreitetes Problem in der Softwareentwicklung und wird oft durch den Wunsch angetrieben, neue Features so schnell wie möglich auszuliefern.

## Indicators ⟡
- Die Codebasis wird zunehmend komplexer und schwerer verständlich.
- Es dauert immer länger, neue Features hinzuzufügen.
- Die Anzahl der Fehler nimmt zu.
- Entwickler werden zunehmend frustriert von der Codebasis.

## Symptoms ▲

- [Hohe technische Schulden](hohe-technische-schulden.md)
<br/>  Das Hinzufügen von Features ohne Refactoring häuft direkt Design- und Implementierungsabkürzungen an, die die langfristigen Kosten erhöhen.
- [Zunehmende Brüchigkeit](zunehmende-bruechigkeit.md)
<br/>  Jedes ohne Refactoring hinzugefügte Feature macht die Codebasis brüchiger, da neuer Code auf ein zunehmend instabiles Fundament geschichtet wird.
- [Langsame Feature-Entwicklung](langsame-feature-entwicklung.md)
<br/>  Die sich verschlechternde Codebasis macht jedes weitere Feature schwerer und langsamer umzusetzen, während die Komplexität wächst.
- [Erhöhte Fehleranzahl](erhoehte-fehleranzahl.md)
<br/>  Ohne Refactoring zur Erhaltung der Codequalität führt jedes neue Feature mit höherer Wahrscheinlichkeit zu Defekten.
- [Schwer verständliche Codebasis](schwer-verstaendliche-codebasis.md)
<br/>  Die Codebasis wird zunehmend schwerer zu verstehen, während Features hinzugefügt werden, ohne das zugrunde liegende Design zu verbessern.
- [Spaghetticode](spaghetticode.md)
<br/>  Kontinuierliche Feature-Ergänzungen ohne strukturelle Verbesserung schaffen verworrenen, unstrukturierten Code, der kaum noch zu warten ist.

## Causes ▼

- [Termindruck](termindruck.md)
<br/>  Der Druck, Features schnell auszuliefern, lässt keine Zeit für Refactoring oder Designverbesserung.
- [Kurzfristiger Fokus](kurzfristiger-fokus.md)
<br/>  Das Management priorisiert die unmittelbare Feature-Auslieferung über die langfristige Codegesundheit und verschiebt Refactoring-Arbeit ständig.
- [Feature-Fabrik](feature-fabrik.md)
<br/>  Eine Organisationskultur, die auf Feature-Ausstoß-Metriken fixiert ist, entmutigt das Investieren von Zeit in Nicht-Feature-Arbeit wie Refactoring.
- [Unsichtbarkeit technischer Schulden](unsichtbarkeit-technischer-schulden.md)
<br/>  Wenn technische Schulden für Stakeholder nicht sichtbar sind, gibt es keine Unterstützung dafür, Zeit für Refactoring neben der Feature-Entwicklung einzuplanen.

## Detection Methods ○
- **Code-Metrik-Werkzeuge:** Nutzung von Werkzeugen zur Messung von Codekomplexität, Klassengröße und anderen Metriken.
- **Code-Reviews:** Achten auf Code, der schwer zu verstehen und zu überprüfen ist.
- **Statische Analysewerkzeuge:** Nutzung von Werkzeugen zur Identifikation von Code Smells wie großen Klassen und langen Methoden.

## Examples
Ein Startup baut eine neue Social-Media-Anwendung. Das Team steht unter hohem Druck, so schnell wie möglich neue Features auszuliefern. Sie fügen der Codebasis ständig neue Features hinzu, ohne sich die Zeit zu nehmen, sie zu refaktorieren. Infolgedessen wird die Codebasis immer komplexer und schwerer zu warten. Das Team erlebt eine Verlangsamung der Entwicklungsgeschwindigkeit, und die Anzahl der Fehler steigt. Wenn sie nicht anfangen, sich Zeit für Refactoring zu nehmen, werden sie irgendwann einen Punkt erreichen, an dem es unmöglich ist, neue Features hinzuzufügen.
