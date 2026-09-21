---
title: Architekturdokumentation
description: Erstellung und Pflege detaillierter Dokumentation der Softwarearchitektur.
category:
- Architecture
- Communication
problems:
- poor-documentation
- legacy-system-documentation-archaeology
- difficult-developer-onboarding
- knowledge-silos
- implicit-knowledge
- stagnant-architecture
- tacit-knowledge
- difficult-code-comprehension
- extended-research-time
- information-fragmentation
layout: solution
lang: de
en_slug: architecture-documentation
related_solutions:
- slug: architecture-decision-records
  similarity: 0.85
- slug: documentation-as-code
  similarity: 0.85
- slug: living-documentation
  similarity: 0.8
- slug: api-documentation
  similarity: 0.8
- slug: architecture-roadmap
  similarity: 0.8
- slug: architecture-governance
  similarity: 0.75
---

## Description

Architekturdokumentation ist eine bewusst gepflegte, strukturierte Beschreibung der tatsächlichen Architektur eines Systems — typischerweise unter Nutzung eines leichtgewichtigen Standardformats wie arc42 oder C4, das Kontext, Container, Komponenten und Schlüsselentscheidungen abdeckt —, gehalten präzise genug, um als verlässliche Grundlage für Modernisierungsentscheidungen zu dienen. In Legacy-Systemen existiert Architekturdokumentation entweder gar nicht oder beschreibt eine Version des Systems, die Jahre veraltet ist, weil die ursprünglichen Design-Dokumente nie aktualisiert wurden, während sich das System durch unzählige inkrementelle Änderungen weiterentwickelte, was neue Entwickler dazu zwingt, ein Verständnis des Systems durch Code-Archäologie und Flurgespräche zu rekonstruieren statt ein Dokument zu lesen. Nützliche Dokumentation für ein solches System zu produzieren bedeutet, die Architektur so zu dokumentieren, wie sie heute tatsächlich ist, nicht wie sie ursprünglich beabsichtigt war, da ungenaue Dokumentation Leser aktiv in die Irre führt und schlimmer ist, als gar keine zu haben. Das wertvollste und am häufigsten fehlende Artefakt ist üblicherweise ein High-Level-Kontextdiagramm, das die externen Integrationen und Datenflüsse des Legacy-Systems zeigt, ergänzt durch Architecture Decision Records, die die Begründung sowohl historischer als auch Modernisierungsentscheidungen erfassen, sodass geklärte Fragen nicht stillschweigend erneut aufgeworfen oder rückgängig gemacht werden. Diese Dokumentation zusammen mit dem Code in der Versionskontrolle statt in einem separaten Wiki zu speichern und periodische Reviews zu planen, ist es, was sie davon abhält, wieder in denselben veralteten, irreführenden Zustand zu verfallen, in dem sie begann. Der Gewinn ist eine gemeinsame Referenz, die die Onboarding-Zeit dramatisch verringert und Auswirkungsanalysen für vorgeschlagene Änderungen unterstützt, aber Dokumentation allein stoppt keinen architektonischen Verfall — sie muss mit Governance und Durchsetzung gepaart werden, um über die Zeit vertrauenswürdig zu bleiben.

## How to Apply ◆

> In Legacy-Systemen existiert Architekturdokumentation oft nicht oder spiegelt eine Version des Systems von vor Jahren wider — die Erstellung akkurater, lebendiger Dokumentation ist essenziell, um informierte Modernisierungsentscheidungen zu ermöglichen.

- Dokumentieren Sie die Architektur so, wie sie tatsächlich ist, nicht wie sie designt wurde — Legacy-Systeme weichen fast immer von ihrem ursprünglichen Design ab, und ungenaue Dokumentation ist schlimmer als keine.
- Nutzen Sie ein leichtgewichtiges, standardisiertes Format wie arc42 oder C4 zur Strukturierung der Dokumentation und fokussieren Sie sich auf die für das Team relevantesten Sichten: Kontext, Container, Komponenten und Schlüsselentscheidungen.
- Beginnen Sie mit einem High-Level-Kontextdiagramm, das die externen Integrationen, Datenflüsse und Nutzergruppen des Legacy-Systems zeigt — dies ist oft das wertvollste und am häufigsten fehlende Stück Dokumentation.
- Dokumentieren Sie architektonische Entscheidungen und ihre Begründung mithilfe von Architecture Decision Records (ADRs), besonders für Entscheidungen, die während der Modernisierung getroffen wurden.
- Speichern Sie Architekturdokumentation zusammen mit dem Code in der Versionskontrolle, sodass sie sich mit dem System weiterentwickelt, statt in einem separaten Wiki zu verrotten.
- Halten Sie Dokumentation minimal, aber akkurat — ein paar gut gepflegte Diagramme sind wertvoller als Hunderte von Seiten, die niemand liest oder aktualisiert.
- Planen Sie regelmäßige Dokumentations-Reviews (vierteljährlich oder nach größeren Änderungen), um Drift zwischen Dokumentation und Realität zu verhindern.

## Tradeoffs ⇄

> Architekturdokumentation bietet essenzielles gemeinsames Verständnis, erfordert aber laufenden Pflegeaufwand, um wertvoll zu bleiben.

**Vorteile:**

- Ermöglicht es neuen Teammitgliedern, die Struktur des Legacy-Systems zu verstehen, ohne Monate an Code-Archäologie und Flurgesprächen.
- Bietet eine gemeinsame Referenz für Modernisierungsplanung und macht es möglich, Änderungen in Bezug auf architektonische Komponenten statt einzelner Dateien zu diskutieren.
- Erfasst die Begründung hinter architektonischen Entscheidungen und verhindert, dass zukünftige Teams geklärte Fragen erneut aufwerfen oder unbeabsichtigt beabsichtigte Designentscheidungen rückgängig machen.
- Unterstützt Auswirkungsanalysen für vorgeschlagene Änderungen, indem gezeigt wird, wie Komponenten miteinander und mit externen Systemen in Beziehung stehen.

**Kosten und Risiken:**

- Dokumentation, die nicht gepflegt wird, wird irreführend, während sich das System weiterentwickelt, und erzeugt falsches Vertrauen in falsche Informationen.
- Die Erstellung initialer Dokumentation für ein großes Legacy-System ohne bestehende Dokumentation erfordert erheblichen Reverse-Engineering-Aufwand.
- Teams könnten übermäßig in detaillierte Dokumentation investieren, die schnell veraltet, statt eine kleinere Menge hochwertiger Dokumente zu pflegen.
- Dokumentation allein verhindert keinen architektonischen Verfall — sie muss mit Governance- und Durchsetzungsmechanismen kombiniert werden.

## How It Could Be

> Das folgende Szenario veranschaulicht die Auswirkung von Architekturdokumentation auf das Verständnis von Legacy-Systemen.

Ein Medienunternehmen erwarb einen Konkurrenten und erbte eine Legacy-Content-Management-Plattform ohne Architekturdokumentation. Neue Entwickler, die mit der Pflege des Systems betraut wurden, verbrachten durchschnittlich drei Monate, bevor sie zuversichtlich Änderungen vornehmen konnten, und selbst dann verursachten sie regelmäßig unerwartete Nebeneffekte, weil sie die versteckten Integrationspunkte des Systems nicht verstanden. Ein Senior-Entwickler verbrachte sechs Wochen damit, ein C4-Modell zu erstellen, das die 4 obersten Container, 23 Komponenten und 12 externen Integrationen des Systems dokumentierte, zusammen mit ADRs für die 15 wichtigsten Designentscheidungen. Diese Dokumentation verringerte die Einarbeitungszeit neuer Entwickler von drei Monaten auf drei Wochen und halbierte die Rate integrationsbezogener Vorfälle. Die Dokumentation deckte außerdem zwei ungenutzte externe Integrationen auf, die noch Ressourcen verbrauchten, welche das Team umgehend außer Betrieb nahm.
