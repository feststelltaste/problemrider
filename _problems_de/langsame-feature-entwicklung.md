---
title: Langsame Feature-Entwicklung
description: Das Tempo der Entwicklung und Lieferung neuer Features ist konsequent
  langsam, oft aufgrund der Komplexität und Brüchigkeit der bestehenden Codebasis.
category:
- Code
- Process
related_problems:
- slug: slow-development-velocity
  similarity: 0.8
- slug: delayed-value-delivery
  similarity: 0.75
- slug: inefficient-development-environment
  similarity: 0.7
- slug: large-feature-scope
  similarity: 0.7
- slug: long-lived-feature-branches
  similarity: 0.7
- slug: feature-creep-without-refactoring
  similarity: 0.7
solutions:
- architecture-roadmap
- development-workflow-automation
- code-generation
- microservices
- standard-software
- feature-driven-development
- delivery-performance-metrics
- fast-feedback-loops
- explicit-extension-points
- variant-consolidation
layout: problem
lang: de
en_slug: slow-feature-development
---

## Description
Langsame Feature-Entwicklung ist die konsequente Unfähigkeit eines Entwicklungsteams, neue Funktionalität zeitnah zu liefern. Dies ist ein häufiges und frustrierendes Problem sowohl für Entwickler als auch für Stakeholder. Es ist oft ein Symptom tieferliegender Probleme innerhalb der Codebasis und des Entwicklungsprozesses. Wenn es Monate dauert, ein Feature zu liefern, das Wochen hätte dauern sollen, ist dies ein klares Zeichen dafür, dass das Team von einem Erbe vergangener Entscheidungen zurückgehalten wird.

## Indicators ⟡
- Das Team schafft es konsequent nicht, seine eigenen Schätzungen für die Feature-Lieferung einzuhalten.
- Stakeholder fragen konstant nach Updates zum Status längst überfälliger Features.
- Der Backlog des Teams wächst viel schneller, als er abgearbeitet wird.
- Es gibt ein allgemeines Gefühl von Frustration und Ungeduld sowohl vonseiten des Geschäfts als auch des Entwicklungsteams.

## Symptoms ▲

- [Verpasste Termine](verpasste-termine.md)
<br/>  Langsame Feature-Entwicklung verursacht direkt, dass Liefertermine verpasst werden.
- [Verzögerte Wertlieferung](verzoegerte-wertlieferung.md)
<br/>  Wenn Features zu lange zum Bauen brauchen, wird Geschäftswert spät geliefert, was den Wettbewerbsvorteil verringert.

## Causes ▼

- [Hohe technische Schulden](hohe-technische-schulden.md)
<br/>  Technische Schulden in der Codebasis zwingen Entwickler, exzessive Zeit damit zu verbringen, bestehende Probleme zu umgehen, bevor sie neue Features implementieren.
- [Brüchige Codebasis](bruechige-codebasis.md)
<br/>  Eine fragile Codebasis erfordert umfangreiches Testen und Vorsicht bei jeder Änderung, was die Feature-Entwicklung erheblich verlangsamt.
- [Spaghetticode](spaghetticode.md)
<br/>  Verworrener Code macht es extrem schwierig zu verstehen, wo und wie neue Funktionalität sicher hinzugefügt werden kann.
- [Schlechte Dokumentation](schlechte-dokumentation.md)
<br/>  Ohne Dokumentation müssen Entwickler die Codebasis zurückentwickeln, bevor sie Features hinzufügen können.
- [Hohe Kopplung und geringe Kohäsion](hohe-kopplung-und-geringe-kohaesion.md)
<br/>  Eng gekoppelter Code bedeutet, dass das Hinzufügen eines Features in einem Bereich Änderungen über viele nicht verwandte Komponenten hinweg erfordert.
- [Anhäufung von Workarounds](anhaeufung-von-workarounds.md)
<br/>  Entwickler müssen das bestehende Geflecht von Workarounds verstehen und umgehen, bevor sie sicher neue Funktionalität hinzufügen können, was die Lieferung verlangsamt.
- [Architektonische Fehlpassung](architektonische-fehlpassung.md)
<br/>  Features, die nicht zur Architektur passen, brauchen aufgrund der Notwendigkeit umfangreicher Anpassungen viel länger zur Implementierung.

## Detection Methods ○
- **Zykluszeit:** Messung der Zeit, die ein Feature von der Idee bis zur Produktion braucht. Eine lange Zykluszeit ist ein klarer Indikator für langsame Feature-Entwicklung.
- **Durchlaufzeit:** Messung der Zeit, die ein Feature braucht, um geliefert zu werden, nachdem es angefragt wurde. Eine lange Durchlaufzeit ist ein Zeichen dafür, dass das Team nicht responsiv auf die Bedürfnisse des Geschäfts ist.
- **Durchsatz:** Messung der Anzahl der Features, die das Team in einem gegebenen Zeitraum liefern kann. Ein niedriger Durchsatz ist ein Zeichen dafür, dass das Team nicht produktiv ist.
- **Stakeholder-Zufriedenheitsbefragungen:** Befragung von Stakeholdern zu ihrer Zufriedenheit mit der Geschwindigkeit der Feature-Lieferung. Ihr Feedback kann eine wertvolle Informationsquelle sein.

## Examples
Ein Unternehmen möchte ein neues Feature zu seinem Flaggschiff-Produkt hinzufügen. Das Feature ist relativ einfach, aber das Entwicklungsteam schätzt, dass es sechs Monate zur Implementierung braucht. Der Grund für die lange Schätzung ist, dass das Produkt auf einer Legacy-Codebasis aufgebaut ist, die schwer zu verstehen und zu modifizieren ist. Das Team muss viel Zeit damit verbringen, den bestehenden Code zurückzuentwickeln und umfangreiche Tests zu schreiben, um sicherzustellen, dass nichts kaputtgeht. Infolgedessen verpasst das Unternehmen eine wichtige Marktchance, und seine Wettbewerber schaffen es, ein ähnliches Feature zuerst zu launchen.
