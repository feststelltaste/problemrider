---
title: Error Budgets
description: Quantifizierung akzeptabler Unzuverlässigkeit als Balance zwischen Feature-Geschwindigkeit
  und Zuverlässigkeit.
category:
- Management
- Process
problems:
- quality-compromises
- short-term-focus
- deployment-risk
- high-defect-rate-in-production
- competing-priorities
- constant-firefighting
- blame-culture
- micromanagement-culture
layout: solution
lang: de
en_slug: error-budgets
related_solutions:
- slug: chaos-engineering
  similarity: 0.8
- slug: service-level-indicators
  similarity: 0.8
- slug: incident-management
  similarity: 0.8
- slug: secure-software
  similarity: 0.8
- slug: site-reliability-engineering-sre
  similarity: 0.8
- slug: error-reporting-and-analysis
  similarity: 0.75
---

## Description

Ein Error Budget quantifiziert die maximale Menge an Unzuverlässigkeit, die ein System über einen Zeitraum ansammeln darf, berechnet als das Komplement eines Service Level Objective — zum Beispiel lässt ein 99,9-Prozent-Verfügbarkeits-SLO ein monatliches Error Budget von etwa 43 Minuten Ausfallzeit übrig. Statt jeden Vorfall als einseitiges Versagen zu behandeln, das um jeden Preis minimiert werden muss, formuliert das Error Budget Zuverlässigkeit als eine Ressource neu, die ausgegeben werden kann: Solange der Verbrauch innerhalb des Budgets bleibt, ist das Team frei, Feature-Geschwindigkeit zu priorisieren, und sobald es erschöpft ist, schreibt die Richtlinie vor, dass sich die Arbeit auf Stabilität verschiebt. Dies ist besonders nützlich in Legacy-Systemen, die in einem Zyklus aus Feature-Auslieferung, Vorfallverursachung und dann reaktivem Feuerlöschen gefangen sind, weil es subjektive, oft politische Argumente darüber, „wie zuverlässig zuverlässig genug ist", durch eine objektive, vorab vereinbarte Schwelle ersetzt, die sowohl Engineering- als auch Produkt-Stakeholder in Echtzeit sich erschöpfen sehen können. Der Mechanismus funktioniert jedoch nur, wenn das zugrunde liegende Monitoring Zuverlässigkeit genau messen kann und die SLOs realistisch kalibriert sind; gut gemacht verwandelt es die Spannung zwischen Geschwindigkeit und Stabilität in einen gemanagten Tradeoff statt in eine Schuldquelle.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Definieren Sie Service Level Objectives (SLOs) für kritische Systemfunktionen basierend auf Geschäftsanforderungen (z. B. 99,9 Prozent Verfügbarkeit)
- Berechnen Sie das Error Budget als Umkehrung des SLO (z. B. 0,1 Prozent erlaubte Ausfallzeit pro Monat)
- Implementieren Sie Monitoring, das tatsächliche Zuverlässigkeit gegen das SLO verfolgt und verbleibendes Error Budget in Echtzeit zeigt
- Etablieren Sie Richtlinien dafür, was passiert, wenn das Error Budget erschöpft ist (z. B. Feature-Releases einfrieren, sich auf Zuverlässigkeitsarbeit konzentrieren)
- Nutzen Sie die Error-Budget-Verbrauchsrate, um datengetriebene Entscheidungen über Release-Geschwindigkeit vs. Stabilitätsinvestition zu treffen
- Überprüfen Sie Error Budgets monatlich mit sowohl Engineering- als auch Produkt-Stakeholdern, um Ausrichtung aufrechtzuerhalten
- Beginnen Sie mit einigen kritischen Services und erweitern Sie die Praxis, während das Team Erfahrung sammelt

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Bietet einen objektiven Rahmen zum Ausbalancieren von Feature-Entwicklung gegen Zuverlässigkeitsarbeit
- Eliminiert subjektive Argumente darüber, wann in Stabilität vs. Features investiert werden soll
- Macht Zuverlässigkeitskosten für Produkt- und Geschäfts-Stakeholder sichtbar
- Schafft natürliche Anreize für Teams, in Zuverlässigkeit zu investieren, bevor das Error Budget erschöpft ist
- Erkennt an, dass perfekte Zuverlässigkeit weder erreichbar noch wünschenswert ist

**Kosten und Risiken:**
- Erfordert ausgereiftes Monitoring und Observability, um Zuverlässigkeit genau zu messen
- Error-Budget-Richtlinien können sich strafend anfühlen, wenn sie nicht konstruktiv gerahmt werden
- SLOs könnten falsch gesetzt werden, entweder zu streng (blockiert alle Entwicklung) oder zu lax (keine echte Einschränkung)
- Kultureller Widerstand von Teams, die an quantifizierte Zuverlässigkeitsziele nicht gewöhnt sind
- Das Manipulieren der Kennzahlen ist möglich, wenn Messpunkte nicht umfassend sind

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Eine Legacy-SaaS-Plattform war in einem Zyklus gefangen, in dem das Team schnell Features auslieferte, was Produktionsvorfälle verursachte, dann Wochen mit Feuerlöschen verbrachte, bevor es zu Features zurückkehrte. Das Team führte Error Budgets mit einem 99,9-Prozent-Verfügbarkeits-SLO für ihre API ein. Im ersten Monat verbrauchte eine Reihe von Vorfällen bis zur zweiten Woche 80 Prozent des monatlichen Error Budgets. Gemäß Richtlinie stoppte das Team Feature-Arbeit und verbrachte die verbleibenden zwei Wochen mit Zuverlässigkeitsverbesserungen: Hinzufügen von Circuit Breakern, Beheben von Connection-Pool-Lecks und Verbesserung der Deployment-Rollback-Geschwindigkeit. Im nächsten Monat wurde das Error Budget kaum angetastet, und das Team lieferte mehr Features als in jedem vorherigen Monat, weil sie nicht durch Vorfälle unterbrochen wurden.
