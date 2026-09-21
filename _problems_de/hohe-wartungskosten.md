---
title: Hohe Wartungskosten
description: Ein unverhältnismäßig großer Teil des Entwicklungsbudgets und -aufwands
  wird durch die Wartung des bestehenden Systems verbraucht, statt neuen Wert zu
  schaffen.
category:
- Business
- Code
related_problems:
- slug: maintenance-overhead
  similarity: 0.8
- slug: maintenance-cost-increase
  similarity: 0.75
- slug: increased-cost-of-development
  similarity: 0.7
- slug: high-technical-debt
  similarity: 0.7
- slug: large-estimates-for-small-changes
  similarity: 0.65
- slug: operational-overhead
  similarity: 0.6
solutions:
- strangler-fig-pattern
- technical-debt-backlog
- api-deprecation-policy
- cross-platform-frameworks
- design-tokens
- failover-cluster
- redundancy
- regular-maintenance-and-updates
- serverless-computing
- site-reliability-engineering-sre
- standard-software
- strategic-code-deletion
- deprecation-strategy
- feature-usage-measurement
- total-cost-of-ownership-transparency
- application-portfolio-inventory
- system-decommissioning
- baseline-measurement
- cost-of-delay
- modernization-options-comparison
- risk-quantification
- value-hierarchy
- customization-cost-attribution
- variant-consolidation
- explicit-extension-points
- fit-to-standard-principle
- retention-and-disposal-policy
layout: problem
lang: de
en_slug: high-maintenance-costs
---

## Description
Hohe Wartungskosten sind ein verbreitetes Problem bei Legacy-Systemen. Während ein System altert, wird es zunehmend teurer, es zu warten. Dies liegt daran, dass die Codebasis komplexer wird, die Technologie veraltet und die ursprünglichen Entwickler das Unternehmen verlassen. Irgendwann können die Kosten für die Wartung des Systems so hoch werden, dass es wirtschaftlich nicht mehr tragfähig ist. An diesem Punkt steht das Unternehmen vor einer schwierigen Wahl: entweder in ein kostspieliges Modernisierungsprojekt investieren oder weiterhin Geld in ein sterbendes System stecken.

## Indicators ⟡
- Das Entwicklungsteam verbringt mehr als 50 % seiner Zeit mit Wartungsaufgaben.
- Das Unternehmen verschiebt ständig neue Projekte, weil es sich nicht leisten kann, gleichzeitig das alte System zu warten und neue zu bauen.
- Die Kosten für die Behebung eines Fehlers sind oft höher als die Kosten des ursprünglichen Features.
- Das Geschäft zögert, Änderungen am System zu genehmigen, wegen der hohen Kosten und Risiken.

## Symptoms ▲

- [Unfähigkeit zu innovieren](unfaehigkeit-zu-innovieren.md)
<br/>  Wenn der Großteil des Budgets von Wartung verbraucht wird, haben Teams keine Kapazität, neue Technologien zu erkunden oder neue Features zu bauen.
- [Kundenunzufriedenheit](kundenunzufriedenheit.md)
<br/>  Fehlende neue Features und langsame Reaktion auf Änderungsanfragen frustrieren Kunden, während Wettbewerber Verbesserungen liefern.
- [Hohe Fluktuation](hohe-fluktuation.md)
<br/>  Entwickler werden frustriert, weil sie hauptsächlich an der Wartung alternder Systeme arbeiten statt neue Dinge zu bauen, was sie dazu bringt, zu kündigen.
- [Wartungslähmung](wartungslaehmung.md)
<br/>  Wenn Wartungskosten das Budget dominieren, gerät das System in einen Zustand, in dem sinnvolle Verbesserungen unmöglich werden.

## Causes ▼

- [Hohe technische Schulden](hohe-technische-schulden.md)
<br/>  Angehäufte Design-Abkürzungen und Codequalitätsprobleme machen jede Änderung teurer und zeitaufwendiger.
- [Probleme durch enge Kopplung](probleme-durch-enge-kopplung.md)
<br/>  Hohe Kopplung zwischen Komponenten bedeutet, dass Änderungen in einem Bereich sich durch das gesamte System ausbreiten, was den Wartungsaufwand vervielfacht.
- [Schlechte Dokumentation](schlechte-dokumentation.md)
<br/>  Ohne ordentliche Dokumentation verbringen Entwickler übermäßig viel Zeit damit, Systemverhalten zu verstehen, bevor sie Änderungen vornehmen können.

## Detection Methods ○
- **Total-Cost-of-Ownership-Analyse:** Berechnung der Gesamtkosten für Besitz und Wartung des Systems über seine Lebensdauer. Dies gibt ein klares Bild der finanziellen Auswirkung des Systems.
- **Verhältnis von Wartung zu Neuentwicklung:** Nachverfolgung des Prozentsatzes des Entwicklungsbudgets, der für Wartung im Vergleich zu Neuentwicklung ausgegeben wird. Ein hohes Verhältnis ist ein klares Zeichen für ein Problem.
- **Analyse der Fehlerbehebungskosten:** Analyse der Kosten für die Behebung von Fehlern über die Zeit. Steigende Kosten sind ein Zeichen dafür, dass das System zunehmend schwerer zu warten wird.
- **Geschäftswertbewertung:** Bewertung des Geschäftswerts, den das System liefert. Wenn die Kosten für die Wartung des Systems größer sind als der Wert, den es liefert, ist es Zeit, seine Außerbetriebnahme in Betracht zu ziehen.

## Examples
Ein großes Finanzinstitut betreibt sein Kernbankensystem auf einem Mainframe, der über 30 Jahre alt ist. Das System ist in COBOL geschrieben, und es wird zunehmend schwieriger und teurer, Entwickler zu finden, die in der Sprache versiert sind. Das Unternehmen gibt jährlich Millionen von Dollar aus, nur um das System am Laufen zu halten. Es kann nicht in neue, innovative Produkte investieren, weil alle Ressourcen in der Wartung des alten Systems gebunden sind. Das Unternehmen steckt in einer schwierigen Position fest: Es weiß, dass es sein System modernisieren muss, hat aber Angst vor den Kosten und dem Risiko eines so großen Projekts.
