---
title: Unzureichende Anforderungserhebung
description: Unzureichende Analyse und Dokumentation von Anforderungen führt dazu,
  Lösungen zu bauen, die tatsächliche Bedürfnisse nicht erfüllen.
category:
- Process
- Testing
related_problems:
- slug: requirements-ambiguity
  similarity: 0.65
- slug: feature-gaps
  similarity: 0.6
- slug: poor-planning
  similarity: 0.6
- slug: knowledge-gaps
  similarity: 0.6
- slug: no-continuous-feedback-loop
  similarity: 0.6
- slug: poor-documentation
  similarity: 0.6
solutions:
- evolutionary-requirements-development
- requirements-analysis
- stakeholder-feedback-loops
- abuse-case-definition
- acceptance-tests
- behavior-driven-development-bdd
- business-process-modeling
- business-quality-scenarios
- compatibility-requirements
- on-site-customer
- performance-budgets
- personas
- requirements-traceability-matrix
- security-requirements-definition
- specification-by-example
- story-mapping
- subject-matter-reviews
- user-stories
- functional-gap-analysis
- definition-of-ready
- regular-stakeholder-demonstrations
- domain-immersion
- exploratory-testing
- attribute-usage-analysis
- fit-to-standard-principle
layout: problem
lang: de
en_slug: inadequate-requirements-gathering
---

## Description

Unzureichende Anforderungserhebung tritt auf, wenn Teams mit der Entwicklung beginnen, ohne ausreichend zu verstehen, zu analysieren oder zu dokumentieren, was gebaut werden muss. Dies kann das Überstürzen der Anforderungsanalyse, das Versäumnis, die richtigen Stakeholder einzubinden, das Übersehen von Randfällen oder das Nicht-Validieren von Annahmen über Nutzerbedürfnisse umfassen. Schlechte Anforderungserhebung führt zu Lösungen, die die tatsächlichen Probleme nicht angehen, was kostspielige Nacharbeit erfordert und möglicherweise scheitert, Geschäftswert zu liefern.

## Indicators ⟡

- Die Entwicklung beginnt mit vagen oder allgemein gehaltenen Anforderungen
- Wichtige Stakeholder sind nicht in die Anforderungsdefinition eingebunden
- Anforderungsdokumente sind unvollständig oder mehrdeutig
- Randfälle und Fehlerbedingungen werden nicht berücksichtigt
- Nutzer-Workflows und Geschäftsprozesse werden nicht gründlich verstanden

## Symptoms ▲

- [Implementierungs-Nacharbeit](implementierungs-nacharbeit.md)
<br/>  Features müssen neu gebaut werden, wenn sich das anfängliche Verständnis aufgrund unzureichender Anforderungsanalyse als falsch erweist.
- [Fehlausgerichtete Liefergegenstände](fehlausgerichtete-liefergegenstaende.md)
<br/>  Gelieferte Features entsprechen nicht den Erwartungen der Stakeholder, weil Anforderungen vorab nicht ordentlich verstanden wurden.
- [Funktionslücken](funktionsluecken.md)
<br/>  Wichtige Funktionalität fehlt, weil sie während der Anforderungserhebung nie identifiziert wurde.
- [Scope Creep](scope-creep.md)
<br/>  Fehlende Anforderungen werden während der Entwicklung entdeckt, was den Projektumfang kontinuierlich über die ursprünglichen Schätzungen hinaus ausweitet.
- [Budgetüberschreitungen](budgetueberschreitungen.md)
<br/>  Nacharbeit und Umfangsausweitung durch schlechte Anforderungen treiben Kosten über die ursprünglichen Budgets hinaus.
- [Annahmenbasierte Entwicklung](annahmenbasierte-entwicklung.md)
<br/>  Entwickler treffen Annahmen darüber, was Nutzer brauchen, statt Anforderungen durch ordentliche Analyse zu validieren.

## Causes ▼

- [Kommunikationslücke zwischen Stakeholdern und Entwicklern](kommunikationsluecke-zwischen-stakeholdern-und-entwicklern.md)
<br/>  Schlechte Kommunikation zwischen Stakeholdern und Entwicklern verhindert wirksame Anforderungserhebung und -validierung.
- [Zeitdruck](zeitdruck.md)
<br/>  Druck, schnell mit der Entwicklung zu beginnen, führt dazu, dass Teams gründliche Anforderungsanalyse überstürzen oder überspringen.
- [Feedback-Isolation](feedback-isolation.md)
<br/>  Teams, die sich nicht regelmäßig mit Stakeholdern und Nutzern austauschen, verpassen kritische Anforderungen und Kontext.

## Detection Methods ○

- **Bewertung der Anforderungsqualität:** Bewertung von Vollständigkeit, Klarheit und Testbarkeit der Anforderungen
- **Analyse der Stakeholder-Abdeckung:** Bewertung, ob alle relevanten Stakeholder zu den Anforderungen beigetragen haben
- **Häufigkeit von Änderungsanfragen:** Nachverfolgung, wie oft sich Anforderungen während der Entwicklung ändern
- **Ergebnisse der Nutzerabnahmetests:** Messung, wie gut gelieferte Lösungen den Nutzererwartungen entsprechen
- **Nacharbeits-Prozentsatz:** Berechnung des Prozentsatzes des Entwicklungsaufwands, der für Nacharbeit aufgrund von Anforderungsproblemen aufgewendet wird

## Examples

Ein Entwicklungsteam wird gebeten, ein Kundensupport-Ticketsystem zu bauen, und erhält allgemein gehaltene Anforderungen wie "Kundenprobleme verfolgen" und "Tickets Support-Agenten zuweisen". Ohne tiefere Analyse bauen sie ein einfaches System mit Ticketerstellung, -zuweisung und Statusaktualisierungen. Als sie das System demonstrieren, zeigen Support-Manager auf, dass sie komplexe Routing-Regeln basierend auf Kundenstufen, Integration mit mehreren Kommunikationskanälen, SLA-Verfolgung, Eskalationsverfahren und Reporting-Fähigkeiten benötigen, die in den ursprünglichen Anforderungen nicht erwähnt wurden. Das einfache System, das sie gebaut haben, kann diese Bedürfnisse nicht erfüllen und muss erheblich neu gestaltet werden. Ein weiteres Beispiel betrifft ein E-Commerce-Team, das eine Produktempfehlungs-Engine basierend auf der Anforderung "verwandte Produkte anzeigen" baut. Sie implementieren einen einfachen Algorithmus basierend auf Produktkategorien, entdecken aber später, dass das Geschäft tatsächlich personalisierte Empfehlungen basierend auf Nutzerverhalten, Kaufhistorie, saisonalen Trends und Bestandsniveaus braucht. Der einfache kategoriebasierte Ansatz liefert wenig Geschäftswert und muss vollständig durch ein anspruchsvolleres System ersetzt werden.
