---
title: Unsichtbarkeit technischer Schulden
description: Die Auswirkung technischer Schulden ist für nicht-technische Stakeholder
  nicht sichtbar, was es schwer macht, ihre Behebung zu rechtfertigen und Ressourcen
  für Verbesserungen bereitzustellen.
category:
- Communication
- Management
- Process
related_problems:
- slug: high-technical-debt
  similarity: 0.75
- slug: difficulty-quantifying-benefits
  similarity: 0.65
- slug: complex-and-obscure-logic
  similarity: 0.65
- slug: test-debt
  similarity: 0.65
- slug: accumulated-decision-debt
  similarity: 0.6
- slug: delayed-issue-resolution
  similarity: 0.6
solutions:
- technical-debt-backlog
- business-metrics
- code-metrics
- compatibility-measurement
- risk-analysis
- security-metrics
- security-relevant-metrics
- code-hotspot-analysis
- total-cost-of-ownership-transparency
- workaround-registry
- baseline-measurement
- benefits-realization-tracking
- value-hierarchy
- cost-of-delay
- risk-quantification
- technical-debt-assessment
- debt-classification
- debt-remediation-estimation
- debt-accrual-analysis
- attribute-usage-analysis
- customization-cost-attribution
- customization-under-version-control
- role-model-rationalization
layout: problem
lang: de
en_slug: invisible-nature-of-technical-debt
---

## Description

Die Unsichtbarkeit technischer Schulden tritt auf, wenn die Kosten und Auswirkungen angehäufter technischer Abkürzungen, schlechter Designentscheidungen und Wartungsaufwands für nicht-technische Stakeholder, die Ressourcenzuweisungsentscheidungen treffen, nicht ersichtlich sind. Diese Unsichtbarkeit macht es schwierig, das Aufwenden von Zeit und Ressourcen für technische Verbesserungen zu rechtfertigen, was zu fortgesetzter Anhäufung technischer Schulden und letztlicher Systemverschlechterung führt.

## Indicators ⟡

- Das Management stellt den Wert von Refactoring oder technischer Verbesserungsarbeit infrage
- Diskussionen über technische Schulden stoßen bei Geschäfts-Stakeholdern nicht auf Resonanz
- Verbesserungsvorschläge werden aufgrund fehlenden sichtbaren Kundennutzens abgelehnt
- Entwicklungsteams haben Schwierigkeiten zu erklären, warum Wartungsaufgaben wichtig sind
- Geschäftsentscheidungen priorisieren sichtbare Features über unsichtbare Infrastrukturverbesserungen

## Symptoms ▲

- [Hohe technische Schulden](hohe-technische-schulden.md)
<br/>  Wenn Stakeholder technische Schulden nicht sehen können, weisen sie keine Ressourcen zu ihrer Behebung zu, was ihr Wachstum verursacht.
- [Widerstand gegen Veränderung](widerstand-gegen-veraenderung.md)
<br/>  Das Management widersteht vorgeschlagenen technischen Verbesserungen, weil die Kosten der Untätigkeit nicht sichtbar sind.
- [Fehler bei der Ressourcenzuweisung](fehler-bei-der-ressourcenzuweisung.md)
<br/>  Ressourcen werden sichtbaren Features zugewiesen statt unsichtbaren, aber kritischen technischen Verbesserungen.
- [Wartungslähmung](wartungslaehmung.md)
<br/>  Ohne Stakeholder-Unterstützung für die Behebung technischer Schulden stockt die Wartungsarbeit, und das System wird schwerer zu ändern.
- [Entwicklerfrustration und Burnout](entwicklerfrustration-und-burnout.md)
<br/>  Entwickler werden frustriert, wenn ihre Bedenken zu technischen Schulden durchgängig vom Management abgetan werden.
- [Zunehmende technische Abkürzungen](zunehmende-technische-abkuerzungen.md)
<br/>  Wenn es kein Budget für die Behebung von Schulden gibt, greifen Entwickler auf mehr Abkürzungen zurück, was das Problem verstärkt.
- [Schwierigkeiten beim Quantifizieren von Nutzen](schwierigkeiten-beim-quantifizieren-von-nutzen.md)
<br/>  Wenn technische Schulden für Stakeholder unsichtbar sind, wird es noch schwerer, den Nutzen ihrer Behebung zu quantifizieren.
- [Kurzfristiger Fokus](kurzfristiger-fokus.md)
<br/>  Wenn technische Schulden für Entscheidungsträger unsichtbar sind, neigen sie natürlich zu kurzfristigen Prioritäten mit sichtbaren, messbaren Ergebnissen, da es kein sichtbares Signal gibt, das sie vor den Kosten der Vernachlässigung warnt.

## Causes ▼

- [Kommunikationslücke zwischen Stakeholdern und Entwicklern](kommunikationsluecke-zwischen-stakeholdern-und-entwicklern.md)
<br/>  Schlechte Kommunikation zwischen Entwicklern und Geschäfts-Stakeholdern verhindert die wirksame Erklärung der Auswirkungen technischer Schulden.
- [Feature-Fabrik](feature-fabrik.md)
<br/>  Eine Kultur, die nur Feature-Ausstoß wertschätzt, macht Nicht-Feature-technische Arbeit unsichtbar und unterbewertet.

## Detection Methods ○

- **Stakeholder-Verständnis-Umfragen:** Bewertung, wie gut nicht-technische Stakeholder die Auswirkungen technischer Schulden verstehen
- **Entscheidungsmuster-Analyse:** Nachverfolgung, wie technische Verbesserungsvorschläge aufgenommen und priorisiert werden
- **Analyse der Kommunikationswirksamkeit:** Überwachung, ob technische Bedenken erfolgreich an Geschäfts-Stakeholder kommuniziert werden
- **Ressourcenzuweisungs-Review:** Analyse, welcher Prozentsatz der Ressourcen technischen Verbesserungen gewidmet ist
- **Auswirkungskorrelationsanalyse:** Messung der Korrelation zwischen technischen Schulden und Geschäftskennzahlen über die Zeit

## Examples

Ein Entwicklungsteam weiß, dass sein Datenbankdesign Performance-Probleme verursacht und die Implementierung neuer Features erschwert, aber als es ein 6-wöchiges Datenbankmodernisierungsprojekt vorschlägt, lehnt das Management es ab, weil Kunden sich nicht über das aktuelle System beschweren. Das Team hat Schwierigkeiten zu erklären, dass das schlechte Datenbankdesign die gesamte Entwicklung um 30 % verlangsamt und letztlich Skalierbarkeitsprobleme verursachen wird, aber diese Auswirkungen sind in vierteljährlichen Geschäftsberichten nicht sichtbar. Ein weiteres Beispiel betrifft eine mobile App, bei der technische Schulden Abstürze und Akkuverbrauchsprobleme verursachen, aber die geschäftliche Auswirkung wird "Nutzerverhalten" statt technischen Problemen zugeschrieben, sodass Ressourcen weiterhin neuen Features zugewiesen werden, während die zugrunde liegenden technischen Probleme die App zunehmend instabil machen.
