---
title: Erhöhte Entwicklungskosten
description: Die Kosten für die Behebung von Fehlern und die Wartung schlechten Codes
  sind erheblich höher als die anfängliche Vermeidung von Problemen.
category:
- Business
- Code
- Process
related_problems:
- slug: maintenance-cost-increase
  similarity: 0.8
- slug: high-technical-debt
  similarity: 0.7
- slug: debugging-difficulties
  similarity: 0.7
- slug: high-maintenance-costs
  similarity: 0.7
- slug: increased-risk-of-bugs
  similarity: 0.7
- slug: lower-code-quality
  similarity: 0.7
solutions:
- architecture-roadmap
- development-workflow-automation
- regression-testing
- code-generation
- standard-software
- feature-usage-measurement
- total-cost-of-ownership-transparency
- system-decommissioning
- baseline-measurement
- cost-of-delay
- value-hierarchy
- benefits-realization-tracking
- customization-cost-attribution
- variant-consolidation
- explicit-extension-points
- fit-to-standard-principle
layout: problem
lang: de
en_slug: increased-cost-of-development
---

## Description

Erhöhte Entwicklungskosten treten auf, wenn die Gesamtausgaben für den Bau und die Wartung von Software aufgrund von Qualitätsproblemen, technischen Schulden oder ineffizienten Prozessen erheblich höher werden als nötig. Dies folgt dem Prinzip, dass die Behebung von Problemen exponentiell teurer wird, je später sie im Entwicklungszyklus entdeckt werden. Wenn Systeme technische Schulden und Qualitätsprobleme anhäufen, wird jede nachfolgende Änderung teurer, was einen sich verstärkenden Effekt auf die Entwicklungskosten schafft.

## Indicators ⟡
- Entwicklungsschätzungen steigen durchgängig für ähnliche Arten von Arbeit
- Fehlerbehebung verbraucht einen unverhältnismäßigen Anteil der Entwicklungsressourcen
- Einfache Änderungen erfordern umfangreiches Testen und Risikominderung
- Notfall-Fixes und Produktionssupport erfordern erhebliche Überstunden
- Die Entwicklungsgeschwindigkeit sinkt, während Teamgröße und Kosten steigen

## Symptoms ▲

- [Erhöhte Time-to-Market](erhoehte-time-to-market.md)
<br/>  Höhere Entwicklungskosten korrelieren mit langsamerer Lieferung, da mehr Ressourcen für Wartung statt neue Features aufgewendet werden.
- [Sinkende Geschäftskennzahlen](sinkende-geschaeftskennzahlen.md)
<br/>  Steigende Entwicklungskosten verringern die Fähigkeit, in Features zu investieren, die Geschäftswachstum vorantreiben, was wichtige Kennzahlen beeinträchtigt.

## Causes ▼

- [Hohe technische Schulden](hohe-technische-schulden.md)
<br/>  Angehäufte technische Schulden machen jede Änderung teurer, da Entwickler Abkürzungen und schlechtes Design umgehen müssen.
- [Erhöhte Fehleranzahl](erhoehte-fehleranzahl.md)
<br/>  Mehr Fehler bedeuten mehr Zeit und Geld, die für Debugging und Behebung aufgewendet werden, was die Entwicklungskosten direkt erhöht.
- [Anstieg der Wartungskosten](anstieg-der-wartungskosten.md)
<br/>  Steigende Wartungslast verbraucht Entwicklungsbudget, das sonst in produktive Feature-Arbeit fließen würde.
- [Zunehmende Brüchigkeit](zunehmende-bruechigkeit.md)
<br/>  Eine brüchige Codebasis erfordert umfangreiches Testen und Risikominderung selbst für einfache Änderungen, was die Kosten in die Höhe treibt.
- [Architektonische Fehlpassung](architektonische-fehlpassung.md)
<br/>  Das Umgehen architektonischer Einschränkungen erhöht die Kosten für die Implementierung neuer Features erheblich.

## Detection Methods ○
- **Kosten-pro-Feature-Tracking:** Beobachtung der Gesamtkosten zur Lieferung ähnlicher Features über die Zeit
- **Verhältnis von Wartung zu Entwicklung:** Nachverfolgung, welcher Prozentsatz der Ressourcen in Wartung vs. Neuentwicklung fließt
- **Analyse der Fehlerbehebungskosten:** Berechnung der Gesamtkosten für die Behebung von Fehlern im Vergleich zur Feature-Entwicklung
- **Geschwindigkeit vs. Teamgröße:** Vergleich des Entwicklungsoutputs mit Teamgröße und Kosten über die Zeit
- **Bewertung der Auswirkung technischer Schulden:** Quantifizierung, wie technische Schulden Entwicklungsschätzungen beeinflussen

## Examples

Ein Legacy-E-Commerce-System hat über fünf Jahre erhebliche technische Schulden angehäuft. Was ursprünglich 2 Wochen und 10.000 $ an Entwicklungskosten brauchte, um eine neue Zahlungsmethode hinzuzufügen, braucht jetzt 8 Wochen und 40.000 $, weil Entwickler architektonische Einschränkungen umgehen, mehrere miteinander verbundene Module aktualisieren und umfangreiche Tests durchführen müssen, um bestehende Funktionalität nicht zu brechen. Das Unternehmen berechnet, dass es 70 % seines Entwicklungsbudgets für Wartung und Behebung technischer Schulden ausgibt, wobei nur 30 % für neue Features bleiben, die Umsatz generieren könnten. Ein einfacher Fehler, dessen Behebung 2 Stunden gedauert hätte, wenn er während der Entwicklung erfasst worden wäre, erfordert jetzt 2 Wochen Untersuchung, Fixes über mehrere Komponenten hinweg und umfangreiche Regressionstests, weil er in Produktion entdeckt wurde. Ein weiteres Beispiel betrifft eine Gesundheitsanwendung, bei der schlechte anfängliche Architekturentscheidungen bedeuten, dass das Hinzufügen von HIPAA-Compliance-Features die Modifikation der gesamten Datenzugriffsschicht erfordert. Was ein einmonatiges Projekt hätte sein sollen, wird zu einem sechsmonatigen Aufwand, der 500.000 $ kostet, weil das System nicht mit Sicherheit und Compliance im Blick entworfen wurde. Die Kosten für die nachträgliche Sicherheitsintegration sind zehnmal höher, als es gewesen wäre, sie von Anfang an korrekt zu bauen.
