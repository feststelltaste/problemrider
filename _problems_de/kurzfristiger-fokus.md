---
title: Kurzfristiger Fokus
description: Das Management priorisiert unmittelbare Feature-Lieferung über langfristige
  Code-Gesundheit und architektonische Verbesserungen, was Nachhaltigkeitsprobleme
  schafft.
category:
- Management
- Process
related_problems:
- slug: increased-technical-shortcuts
  similarity: 0.65
- slug: high-technical-debt
  similarity: 0.65
- slug: unrealistic-deadlines
  similarity: 0.6
- slug: time-pressure
  similarity: 0.6
- slug: feature-factory
  similarity: 0.55
- slug: slow-development-velocity
  similarity: 0.55
solutions:
- technical-debt-backlog
- security-culture
- error-budgets
- explicit-prioritization-framework
- improvement-budget
- total-cost-of-ownership-transparency
- outcome-based-goal-setting
- delivery-performance-metrics
- feature-usage-measurement
- pilot-projects
- baseline-measurement
- benefits-realization-tracking
- cost-of-delay
- executive-sponsorship
- value-hierarchy
- debt-classification
- debt-accrual-analysis
- customization-cost-attribution
layout: problem
lang: de
en_slug: short-term-focus
---

## Description

Kurzfristiger Fokus tritt auf, wenn organisatorische Entscheidungsfindung konsequent unmittelbare Liefergegenstände und schnelle Erfolge über langfristige Nachhaltigkeit, Codequalität und architektonische Gesundheit priorisiert. Dieser Managementansatz führt zu sich anhäufenden technischen Schulden, abnehmender Systemwartbarkeit und schließlich Produktivitätsverschlechterung, während die Kosten der Wartung schlecht designter Systeme über die Zeit steigen.

## Indicators ⟡

- Alle Entwicklungszeit wird der Feature-Lieferung zugewiesen, ohne Zeit für Verbesserungsarbeit
- Vorschläge für technische Schulden und Refactoring werden konsequent abgelehnt oder verschoben
- Das Management misst Erfolg primär an der Geschwindigkeit der Feature-Lieferung statt an der Systemgesundheit
- Langfristige architektonische Planung ist minimal oder nicht existent
- Qualitätsverbesserungsinitiativen werden als nicht-essentieller Overhead angesehen

## Symptoms ▲

- [Hohe technische Schulden](hohe-technische-schulden.md)
<br/>  Das konsequente Wählen schneller Lösungen über ordentliche Technik häuft technische Schulden an, die sich über die Zeit verstärken.
- [Zunehmende technische Abkürzungen](zunehmende-technische-abkuerzungen.md)
<br/>  Der Druck, Features sofort zu liefern, treibt Entwickler dazu, mehr Abkürzungen zu nehmen und Schnelllösungen statt ordentlicher Lösungen zu implementieren.
- [Qualitätsverschlechterung](qualitaetsverschlechterung.md)
<br/>  Die Systemqualität verschlechtert sich stetig, da keine Zeit für Refactoring, Verbesserung oder das Angehen von Code-Gesundheitsproblemen zugewiesen wird.
- [Stagnierende Architektur](stagnierende-architektur.md)
<br/>  Die Architektur entwickelt sich nie weiter, weil langfristige Verbesserungsarbeit dauerhaft zugunsten unmittelbarer Feature-Lieferung depriorisiert wird.
- [Entwicklerfrustration und Burnout](entwicklerfrustration-und-burnout.md)
<br/>  Entwickler werden frustriert und ausgebrannt, wenn ihre Anfragen nach Zeit zur Behebung von Qualitätsproblemen konsequent abgelehnt werden.
- [Langsame Entwicklungsgeschwindigkeit](langsame-entwicklungsgeschwindigkeit.md)
<br/>  Während sich angehäufte Schulden und unbehandelte Ineffizienzen vermehren, sinkt die Entwicklungsgeschwindigkeit systematisch, weil jede Änderung mehr Aufwand erfordert, um sicher implementiert zu werden.
- [Schwierigkeiten beim Quantifizieren von Nutzen](schwierigkeiten-beim-quantifizieren-von-nutzen.md)
<br/>  Kurzfristiges Denken schafft ein Umfeld, in dem nur unmittelbar messbare Ergebnisse geschätzt werden, was es schwieriger macht, langfristige technische Vorteile zu quantifizieren.
- [Termindruck](termindruck.md)
<br/>  Die anhaltende Priorisierung unmittelbarer Lieferung durch das Management führt dazu, dass es aggressive Termine als primären Hebel zur Durchsetzung dieser Priorität setzt und beibehält.

## Causes ▼

- [Marktdruck](marktdruck.md)
<br/>  Wettbewerbliche Marktkräfte schaffen Dringlichkeit, Features schnell zu liefern, was das Management dazu drängt, unmittelbare Lieferung über Nachhaltigkeit zu priorisieren.
- [Unsichtbarkeit technischer Schulden](unsichtbarkeit-technischer-schulden.md)
<br/>  Wenn die Kosten technischer Schulden für Entscheidungsträger nicht sichtbar sind, fehlt ihnen das Gegensignal, das nötig ist, um die Priorisierung langfristiger Gesundheit über unmittelbare Lieferung zu rechtfertigen, was einen kurzfristigen Fokus verstärkt.

## Detection Methods ○

- **Ressourcenzuweisungsanalyse:** Verfolgung des Prozentsatzes der Entwicklungszeit, der für Verbesserung vs. neue Features aufgewendet wird
- **Trendanalyse technischer Schulden:** Überwachung, ob technische Schulden über die Zeit zu- oder abnehmen
- **Verfolgung der Entwicklungskosten:** Messung, ob sich Entwicklungsgeschwindigkeit und -kosten in nachhaltige Richtungen entwickeln
- **Analyse von Managemententscheidungen:** Überprüfung, wie Verbesserungsvorschläge im Vergleich zu Feature-Anfragen priorisiert werden
- **Entwicklerzufriedenheitsbefragungen:** Bewertung der Teamzufriedenheit mit der Fähigkeit, Codequalität zu erhalten

## Examples

Ein Softwareunternehmen lehnt konsequent Vorschläge ab, sein 10 Jahre altes Authentifizierungssystem zu modernisieren, weil dies 3 Monate ohne unmittelbar kundensichtbare Vorteile dauern würde. Stattdessen fügt es weiterhin Feature-Patches hinzu, die die Beschränkungen umgehen, und verbringt geschätzte 15 % der Entwicklungszeit mit authentifizierungsbezogenen Workarounds und Wartung. Über zwei Jahre kostet dieser Ansatz mehr Entwicklungszeit, als die Modernisierung erfordert hätte, während die fundamentalen Probleme ungelöst bleiben. Ein weiteres Beispiel betrifft eine E-Commerce-Plattform, bei der das Management jedes Quartal die Hinzufügung neuer Produktfeatures priorisiert, aber nie Zeit zuweist, um Performance-Probleme anzugehen. Die Website wird zunehmend langsamer, was zunehmend komplexe Caching-Strategien und Infrastrukturausgaben erfordert, die letztlich mehr kosten, als architektonische Verbesserungen gekostet hätten.
