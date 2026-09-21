---
title: Qualitätsverschlechterung
description: Die Systemqualität nimmt über die Zeit ab, aufgrund angehäufter technischer
  Schulden, Abkürzungen und unzureichender Qualitätspraktiken.
category:
- Code
- Process
related_problems:
- slug: gradual-performance-degradation
  similarity: 0.75
- slug: lower-code-quality
  similarity: 0.7
- slug: increasing-brittleness
  similarity: 0.7
- slug: inconsistent-quality
  similarity: 0.7
- slug: information-decay
  similarity: 0.7
- slug: increased-bug-count
  similarity: 0.65
solutions:
- definition-of-done
- code-metrics
- compatibility-as-error
- compatibility-standards
- code-quality-gates
- fitness-functions
- defect-triage-process
- delivery-performance-metrics
- exploratory-testing
- baseline-measurement
- benefits-realization-tracking
- quality-ratchet
- debt-accrual-analysis
- code-hotspot-analysis
- duplication-detection
layout: problem
lang: de
en_slug: quality-degradation
---

## Description

Qualitätsverschlechterung tritt auf, wenn Softwaresysteme über die Zeit einen stetigen Rückgang bei Zuverlässigkeit, Wartbarkeit, Performance oder Nutzbarkeit erleben. Diese Verschlechterung resultiert typischerweise aus angehäuften technischen Schulden, überhasteten Entwicklungspraktiken, unzureichendem Testen und fehlender systematischer Qualitätspflege. Anders als isolierte Qualitätsprobleme stellt dies einen systemischen Rückgang dar, der mehrere Aspekte des Systems betrifft.

## Indicators ⟡

- Fehlerberichte nehmen über die Zeit trotz laufenden Entwicklungsaufwands zu
- Die Systemperformance nimmt allmählich ohne klare Ursache ab
- Code wird zunehmend schwierig zu modifizieren und zu verstehen
- Die Nutzerzufriedenheit mit Systemzuverlässigkeit und Nutzbarkeit nimmt ab
- Mehr Zeit wird relativ zu neuen Features für Wartung und Fehlerbehebungen aufgewendet

## Symptoms ▲

- [Zunehmende Brüchigkeit](zunehmende-bruechigkeit.md)
<br/>  Angehäufte Qualitätsprobleme machen das System fragil, wo kleine Änderungen unerwartete Fehlschläge verursachen.
- [Schleichende Performance-Verschlechterung](schleichende-performance-verschlechterung.md)
<br/>  Die Systemperformance verschlechtert sich stetig, während sich Qualitätsprobleme verstärken und Ineffizienzen anhäufen.
- [Unzufriedenheit der Stakeholder](unzufriedenheit-der-stakeholder.md)
<br/>  Abnehmende Zuverlässigkeit und Nutzbarkeit untergraben das Vertrauen von Nutzern und Stakeholdern in das System.
- [Kundenunzufriedenheit](kundenunzufriedenheit.md)
<br/>  Während sich verschlechterte Qualität durch Fehler, Inkonsistenzen und Unzuverlässigkeit direkt für Nutzer zeigt, werden Kunden mit dem Produkt unzufrieden.

## Causes ▼

- [Kurzfristiger Fokus](kurzfristiger-fokus.md)
<br/>  Die anhaltende Priorisierung sofortiger Lieferung durch das Management lässt keine Zeit für Refactoring, Testinvestition oder das Angehen von Code-Gesundheit, was Qualität über die Zeit erodieren lässt.
- [Qualitätskompromisse](qualitaetskompromisse.md)
<br/>  Das absichtliche Senken von Qualitätsstandards schafft die Abkürzungen und Schulden, die allmählichen Qualitätsrückgang antreiben.
- [Unzureichendes Testen](unzureichendes-testen.md)
<br/>  Unzureichendes Testen erlaubt es Defekten, sich unentdeckt anzuhäufen, was zu systematischer Qualitätserosion beiträgt.
- [Geringere Codequalität](geringere-codequalitaet.md)
<br/>  Schlechte Codequalitätspraktiken verstärken sich über die Zeit, was eine Abwärtsspirale der Wartbarkeit schafft.
- [Informationsverfall](informationsverfall.md)
<br/>  Veraltete Dokumentation und verlorenes Wissen führen zu inkorrekten Annahmen, die die Qualität weiter verschlechtern.

## Detection Methods ○

- **Qualitätstrendanalyse:** Nachverfolgung von Qualitätsmetriken über die Zeit zur Identifikation von Verschlechterungsmustern
- **Fehlerratenüberwachung:** Überwachung von Fehlererkennungs- und Lösungsraten über Releases hinweg
- **Performance-Baseline-Vergleich:** Vergleich aktueller Performance mit historischen Baselines
- **Code-Qualitätsmetriken:** Nachverfolgung von Codekomplexität, Wartbarkeit und Testabdeckungstrends
- **Nutzerzufriedenheitsbefragungen:** Regelmäßige Bewertung von Nutzererfahrung und Zufriedenheit

## Examples

Ein Customer-Relationship-Management-System, das zwei Jahre lang gut funktionierte, beginnt häufige Abstürze, langsame Antwortzeiten und Dateninkonsistenzen zu erleben. Untersuchung offenbart, dass schnelle Feature-Ergänzungen ohne entsprechendes Refactoring eine komplexe, brüchige Codebasis geschaffen haben, in der kleine Änderungen unvorhersehbare Auswirkungen haben. Das Team verbringt 70 % seiner Zeit mit dem Beheben von Fehlern und der Wartung bestehender Funktionalität statt der Entwicklung neuer Features. Ein weiteres Beispiel betrifft eine E-Commerce-Plattform, bei der Checkout-Erfolgsraten über sechs Monate allmählich von 99 % auf 85 % sinken, aufgrund angehäufter Integrationsprobleme, Datenbank-Performance-Probleme und ungelöster Randfälle, die sich über die Zeit verstärken.
