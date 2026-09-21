---
title: Walking Skeleton
description: Entwicklung eines minimalen, lauffähigen Systems mit den
  zentralen Architekturideen.
category:
- Architecture
- Process
problems:
- implementation-starts-without-design
- modernization-strategy-paralysis
- analysis-paralysis
- strangler-fig-pattern-failures
- immature-delivery-strategy
- complex-deployment-process
- procrastination-on-complex-tasks
- incomplete-projects
- large-feature-scope
layout: solution
lang: de
en_slug: walking-skeleton
related_solutions:
- slug: strangler-fig-pattern
  similarity: 0.75
- slug: architecture-roadmap
  similarity: 0.75
- slug: prototypes
  similarity: 0.7
- slug: tracer-bullets
  similarity: 0.7
- slug: prototyping
  similarity: 0.7
- slug: architecture-decision-records
  similarity: 0.7
---

## Description

Ein Walking Skeleton ist eine minimale, aber vollständig funktionsfähige, durchgängige Implementierung der Kernarchitektur eines Systems — UI, Geschäftslogik, Persistenz und Deployment umspannend —, gebaut, um zu beweisen, dass der architektonische Ansatz in der Praxis tatsächlich funktioniert, bevor irgendein echter Feature-Aufwand darauf investiert wird. Anders als ein zum Verwerfen bestimmter Prototyp bleibt das Skelett deploybar und wird bewusst gezüchtet, indem "Fleisch" zu seinen bewährten Knochen hinzugefügt wird, was echte Funktionalität inkrementell implementiert, sobald die zugrunde liegende Architektur bereits validiert wurde. Dies adressiert ein spezifisches und häufiges Fehlermuster in der Legacy-Modernisierung: Projekte, die in ausgedehnten Designphasen stecken bleiben und monatelang Diagramme und Spezifikationen produzieren, ohne eine einzige Zeile funktionierenden Codes, weil das wahrgenommene Risiko und Ausmaß des Ersatzes des Legacy-Systems Lähmung statt Handlung auslöst. Indem das Team gezwungen wird, sofort etwas Echtes und Deploybares zu bauen, auch wenn es funktional fast nichts tut, entlässt der Walking Skeleton früh die größten architektonischen Unbekannten — ob die Deployment-Pipeline tatsächlich funktioniert, ob die gewählten Komponenten genuin integrieren können —, während die Kosten des Falsch-Liegens noch klein sind. Es gibt Stakeholdern auch ein greifbares, laufendes System, auf das sie weit früher reagieren können, als ein traditioneller Big-Design-Upfront-Ansatz es erlauben würde, obwohl das Team sich gegen zwei vorhersagbare Fehlermuster wappnen muss: vorzeitig echte Features hinzuzufügen, bevor die Architektur bewiesen ist, und Stakeholder, die die notwendigerweise minimale Qualität des Skeletts mit der beabsichtigten Qualität des fertigen Produkts verwechseln.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Identifizieren Sie die dünnstmögliche durchgängige Scheibe durch das System, die alle wichtigen architektonischen Komponenten ausübt
- Bauen Sie eine minimale, aber vollständig funktionsfähige Version, die UI, Geschäftslogik, Datenpersistenz und Deployment einschließt
- Nutzen Sie den Walking Skeleton, um die Deployment-Pipeline, Infrastruktur und Integrationspunkte früh zu validieren
- Priorisieren Sie das Beweisen architektonischer Risiken über das Liefern von Features im anfänglichen Skelett
- Iterieren Sie über das Skelett, indem Sie Fleisch hinzufügen: implementieren Sie echte Features inkrementell auf der bewährten Architektur
- Nutzen Sie bei der Modernisierung eines Legacy-Systems den Walking Skeleton, um die Zielarchitektur zu beweisen, bevor Features migriert werden
- Halten Sie das Skelett jederzeit deploybar, um einen funktionierenden Referenzpunkt aufrechtzuerhalten

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Validiert architektonische Annahmen und Deployment-Pipelines vor erheblicher Investition
- Bietet früh ein greifbares, laufendes System, das Stakeholder sehen und mit dem sie interagieren können
- Bringt Integrationsprobleme zwischen Komponenten in der frühestmöglichen Phase zutage
- Reduziert das Risiko, grundlegende architektonische Fehler spät im Projekt zu entdecken

**Kosten und Risiken:**
- Das Skelett könnte zu dünn sein, um bestimmte architektonische Herausforderungen zu offenbaren
- Stakeholder könnten den minimalen Prototyp fälschlich als endgültige Produktqualität interpretieren
- Erfordert Disziplin, das Skelett minimal zu halten, statt vorzeitig Features hinzuzufügen
- Könnte sichtbare Feature-Lieferung kurzfristig verzögern

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Eine Regierungsbehörde plante, ein Legacy-Genehmigungsverarbeitungssystem zu modernisieren, indem sie von einem Mainframe zu einer Cloud-nativen Architektur migrierte. Vorherige Modernisierungsversuche waren nach Monaten des Designs ohne funktionierenden Code ins Stocken geraten. Diesmal baute das Team einen Walking Skeleton: einen einzelnen Genehmigungstyp, der durch ein React-Frontend, eine Spring-Boot-API, eine PostgreSQL-Datenbank und eine Kubernetes-Deployment-Pipeline floss. Das Skelett verarbeitete genau einen Genehmigungstyp mit minimaler Geschäftslogik, aber es bewies, dass die Architektur durchgängig funktionierte und dass die Deployment-Pipeline Änderungen zuverlässig liefern konnte. Mit den entlassenen architektonischen Risiken begann das Team zuversichtlich, die verbleibenden 30 Genehmigungstypen auf das bewährte Fundament zu migrieren.
