---
title: Anstieg der Wartungskosten
description: Die für die Wartung, Unterstützung und Aktualisierung von Softwaresystemen
  erforderlichen Ressourcen wachsen über die Zeit und verbrauchen zunehmende Anteile
  der Entwicklungsbudgets.
category:
- Code
- Management
- Performance
related_problems:
- slug: increased-cost-of-development
  similarity: 0.8
- slug: high-maintenance-costs
  similarity: 0.75
- slug: maintenance-overhead
  similarity: 0.7
- slug: increasing-brittleness
  similarity: 0.65
- slug: quality-degradation
  similarity: 0.6
- slug: high-technical-debt
  similarity: 0.6
solutions:
- technical-debt-backlog
- standard-software
- code-hotspot-analysis
- improvement-budget
- total-cost-of-ownership-transparency
- incremental-refactoring
- feature-usage-measurement
- strategic-code-deletion
- value-stream-mapping
- system-decommissioning
- baseline-measurement
- cost-of-delay
- debt-accrual-analysis
- quality-ratchet
- technical-debt-assessment
- continuous-dependency-updates
- automated-code-migration
- duplication-detection
- customization-cost-attribution
- variant-consolidation
layout: problem
lang: de
en_slug: maintenance-cost-increase
---

## Description

Anstieg der Wartungskosten tritt auf, wenn die Ressourcen, die erforderlich sind, um Softwaresysteme betriebsbereit zu halten, Fehler zu beheben und Modifikationen vorzunehmen, über die Zeit erheblich wachsen. Dieser Anstieg übertrifft oft das Hinzufügen neuer Funktionalität, was bedeutet, dass Organisationen immer mehr ihrer Entwicklungsbudgets für die Wartung bestehender Systeme statt für die Schaffung neuen Werts ausgeben. Der Trend deutet auf sich anhäufende technische Schulden und sich verschlechternde Systemgesundheit hin.

## Indicators ⟡

- Steigender Prozentsatz des Entwicklungsbudgets, der für Wartung statt neue Features ausgegeben wird
- Fehlerbehebungszeit steigt für ähnliche Arten von Problemen
- Einfache Änderungen erfordern mehr Aufwand und Tests als erwartet
- Mehr Entwickler werden benötigt, um dieselbe Funktionalität zu warten
- Support-Kosten wachsen schneller als Nutzerbasis oder Systemnutzung

## Symptoms ▲

- [Budgetüberschreitungen](budgetueberschreitungen.md)
<br/>  Wachsende Wartungskosten verbrauchen mehr Budget als geplant, was zu Kostenüberschreitungen in Entwicklungsprojekten führt.
- [Verringerte Innovation](verringerte-innovation.md)
<br/>  Wenn Wartung den Großteil des Budgets verbraucht, bleibt wenig übrig, um in neue Features und Innovation zu investieren.
- [Wettbewerbsnachteil](wettbewerbsnachteil.md)
<br/>  Ressourcen, die durch eskalierende Wartungskosten verbraucht werden, können nicht in Wettbewerbsfeatures investiert werden, was die Marktposition untergräbt.

## Causes ▼

- [Hohe technische Schulden](hohe-technische-schulden.md)
<br/>  Angehäufte technische Schulden machen jede Änderung teurer, da Entwickler um angehäufte Abkürzungen und schlechtes Design herumarbeiten müssen.
- [Zunehmende Brüchigkeit](zunehmende-bruechigkeit.md)
<br/>  Eine brüchige Codebasis erfordert sorgfältigere und zeitaufwendigere Änderungen, was die Kosten jeder Wartungsaufgabe in die Höhe treibt.
- [Code-Duplizierung](code-duplizierung.md)
<br/>  Duplizierter Code vervielfacht den Wartungsaufwand, da dieselbe Korrektur oder Änderung an mehreren Stellen angewendet werden muss.
- [Veraltete Technologien](veraltete-technologien.md)
<br/>  Die Wartung von Systemen auf veralteten Technologien ist teuer aufgrund knapper Expertise und fehlender Anbieterunterstützung.
- [Anhäufung von Workarounds](anhaeufung-von-workarounds.md)
<br/>  Jede nachfolgende Änderung muss um bestehende Workarounds navigieren, sodass der Wartungsaufwand wächst, während sich das Netz temporärer Fixes ausdehnt.

## Detection Methods ○

- **Kostenzuordnungs-Nachverfolgung:** Überwachung des Prozentsatzes der Entwicklungsressourcen, die für Wartung versus neue Entwicklung ausgegeben werden
- **Zeitanalyse von Wartungsaufgaben:** Nachverfolgung, wie lange ähnliche Wartungsaufgaben über die Zeit dauern
- **Defektlösungs-Metriken:** Messung von Zeit und Aufwand, die zur Behebung von Fehlern ähnlicher Komplexität erforderlich sind
- **Total-Cost-of-Ownership-Bewertung:** Berechnung vollständiger Lebenszykluskosten einschließlich Wartung
- **Ressourcennutzungsanalyse:** Analyse, wie die Zeit des Entwicklungsteams zwischen Wartung und neuer Arbeit aufgeteilt wird

## Examples

Ein Unternehmen entdeckt, dass 80 % seines Entwicklungsbudgets jetzt für die Wartung einer 10 Jahre alten E-Commerce-Plattform ausgegeben werden, wobei nur 20 % für neue Features und Verbesserungen übrig bleiben. Was früher einfache Änderungen waren, erfordert jetzt Wochen an Aufwand aufgrund komplexer Abhängigkeiten und veralteter Technologie. Das Wartungsteam ist von 2 auf 8 Entwickler gewachsen, nur um das System am Laufen zu halten, während der Wettbewerbsdruck neue Fähigkeiten verlangt, die aufgrund von Ressourcenbeschränkungen nicht geliefert werden können. Ein weiteres Beispiel betrifft ein Finanzsystem, bei dem routinemäßige Wartungsaufgaben, die früher Stunden dauerten, jetzt aufgrund angehäufter technischer Schulden Tage dauern, und die Kosten der Wartung des Legacy-Systems übersteigen die Kosten der Entwicklung eines modernen Ersatzes.
