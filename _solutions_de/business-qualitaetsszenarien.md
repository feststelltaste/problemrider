---
title: Business-Qualitätsszenarien
description: Spezifikation und Verifikation von Qualitätsanforderungen durch geschäftsgetriebene
  Szenarien.
category:
- Requirements
- Testing
problems:
- requirements-ambiguity
- quality-blind-spots
- inadequate-requirements-gathering
- difficulty-quantifying-benefits
- stakeholder-developer-communication-gap
- reduced-feature-quality
layout: solution
lang: de
en_slug: business-quality-scenarios
related_solutions:
- slug: behavior-driven-development-bdd
  similarity: 0.7
- slug: business-metrics
  similarity: 0.7
- slug: user-stories
  similarity: 0.65
- slug: specification-by-example
  similarity: 0.65
- slug: acceptance-tests
  similarity: 0.65
- slug: requirements-analysis
  similarity: 0.65
---

## Description

Business-Qualitätsszenarien formulieren nicht-funktionale Anforderungen — Verfügbarkeit, Performance, Zuverlässigkeit — in einem konkreten Stimulus-Antwort-Format, das festlegt, wer oder was eine Bedingung auslöst, was das System als Reaktion tut, und welches messbare Ergebnis als akzeptabel gilt, wobei die spezifischen Zahlen aus echten Geschäftsanliegen statt generischen Engineering-Zielen abgeleitet werden. Der Mechanismus zwingt vage Bestrebungen wie „das System sollte schnell und zuverlässig sein" in testbare Aussagen wie eine definierte Antwortzeit unter einer definierten gleichzeitigen Last, die als Teil einer Test-Suite automatisiert und objektiv geprüft statt subjektiv diskutiert werden können. Dies ist besonders wichtig in der Legacy-Modernisierung, weil Qualitätsanforderungen in älteren Systemen häufig nie formuliert wurden, was Architekten dazu zwingt, akzeptable Performance- oder Verfügbarkeitsziele zu raten, wenn sie entscheiden, wo knapper Modernisierungsaufwand investiert werden soll, ohne Möglichkeit zu erkennen, ob eine vorgeschlagene Änderung tatsächlich ein geschäftsrelevantes Anliegen adressiert oder nur eine technische Präferenz. Szenarien direkt aus Geschäftsereignissen abzuleiten — Monatsendverarbeitungslast, Failover-Zeit während eines Datenbankausfalls — bindet architektonische Entscheidungen an tatsächliche Geschäftsauswirkung und gibt Modernisierungsarbeit konkrete, priorisierte Akzeptanzkriterien statt eines ergebnisoffenen Qualitätsverbesserungsmandats. Die laufenden Kosten sind, dass Szenarien fortgesetzte Zusammenarbeit mit Geschäfts-Stakeholdern erfordern, um aktuell zu bleiben, und nicht jedes Qualitätsattribut übersetzt sich natürlich in ein geschäftsskaliertes Szenario, sodass manche technischen Qualitäten schwerer auf diese Weise auszudrücken bleiben.

## How to Apply ◆

- Definieren Sie Qualitätsszenarien mit dem Stimulus-Antwort-Format: wer/was löst das Szenario aus, was geschieht, und welche messbare Antwort wird erwartet.
- Leiten Sie Szenarien aus echten Geschäftsanliegen ab (z. B. „Wenn 500 Nutzer während eines Verkaufsereignisses gleichzeitig Bestellungen aufgeben, müssen 99 % der Bestellungen innerhalb von 3 Sekunden abgeschlossen sein").
- Priorisieren Sie Szenarien basierend auf Geschäftsauswirkung und nutzen Sie sie, um architektonische Entscheidungen in der Legacy-Modernisierung zu leiten.
- Automatisieren Sie die Verifikation von Qualitätsszenarien, wo möglich, und integrieren Sie sie in Performance- und Integrations-Test-Suiten.
- Überprüfen und aktualisieren Sie Qualitätsszenarien, während sich Geschäftsanforderungen weiterentwickeln.
- Nutzen Sie Qualitätsszenarien, um nicht-funktionale Anforderungen in Begriffen zu kommunizieren, die Geschäfts-Stakeholder verstehen.

## Tradeoffs ⇄

**Vorteile:**
- Übersetzt abstrakte Qualitätsanforderungen in konkrete, testbare und geschäftsrelevante Szenarien.
- Bietet klare Akzeptanzkriterien für nicht-funktionale Anforderungen, die oft vage bleiben.
- Hilft, architektonische Investitionen zu priorisieren, indem Qualitätsattribute an Geschäftswert gebunden werden.

**Kosten:**
- Die Definition bedeutsamer Qualitätsszenarien erfordert Zusammenarbeit zwischen Geschäfts- und technischen Teams.
- Nicht alle Qualitätsattribute lassen sich leicht als Geschäftsszenarien ausdrücken.
- Automatisierte Verifikation von Qualitätsszenarien könnte spezialisierte Testinfrastruktur erfordern.
- Szenarien brauchen regelmäßige Überprüfung, um mit sich entwickelnden Geschäftsbedürfnissen abgestimmt zu bleiben.

## How It Could Be

Eine Legacy-Banking-Anwendung muss strenge Verfügbarkeits- und Performance-Anforderungen erfüllen, aber diese sind nur als vage Aussagen wie „das System sollte schnell und zuverlässig sein" formuliert. Das Team arbeitet mit Geschäfts-Stakeholdern zusammen, um konkrete Qualitätsszenarien zu definieren: „Während der Monatsendverarbeitung, wenn 200 gleichzeitige Nutzer Kontostandsberichte ausführen, muss jeder Bericht innerhalb von 5 Sekunden abgeschlossen sein" und „Wenn die primäre Datenbank ausfällt, muss das System innerhalb von 30 Sekunden ohne Datenverlust auf den Standby umschalten." Diese Szenarien leiten die Modernisierungsbemühung, indem sie klarmachen, welche Qualitätsverbesserungen Geschäftswert liefern und welche nur technische Präferenzen sind. Das Team baut automatisierte Tests, die diese Szenarien in Staging-Umgebungen vor jedem Release verifizieren.
