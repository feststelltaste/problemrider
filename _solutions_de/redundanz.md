---
title: Redundanz
description: Mehrfache Instanzen kritischer Komponenten oder Systeme.
category:
- Architecture
- Operations
problems:
- single-points-of-failure
- system-outages
- cascade-failures
- capacity-mismatch
- deployment-risk
- high-maintenance-costs
layout: solution
lang: de
en_slug: redundancy
related_solutions:
- slug: redundant-data-storage
  similarity: 0.85
- slug: failover-cluster
  similarity: 0.8
- slug: high-availability-architectures
  similarity: 0.8
- slug: failover-mechanisms
  similarity: 0.8
- slug: data-replication
  similarity: 0.8
- slug: resilience
  similarity: 0.8
---

## Description

Redundanz ist die bewusste Duplizierung kritischer Komponenten, Dienste oder Infrastruktur, sodass der Ausfall einer Instanz nicht das gesamte System zu Fall bringt. Statt sich auf einen einzelnen Anwendungsserver, eine Datenbank oder einen Netzwerkpfad zu verlassen, betreiben redundante Architekturen mehrere äquivalente Instanzen parallel, mit einem Failover- oder Lastverteilungsmechanismus, der Traffic von jeder Instanz weglenkt, die nicht mehr verfügbar wird. Das Konzept gilt auf jeder Ebene — Hardware, Netzwerk, Daten und Anwendung — und kann als Active-Active-Konfigurationen implementiert werden, die Last kontinuierlich teilen, oder als Active-Passive-Konfigurationen, die einen Standby bereithalten, um zu übernehmen. In Legacy-Systemen ist Redundanz oft der schnellste Weg, Single Points of Failure zu beseitigen, die nie hinterfragt wurden, als das System klein war und seine Verfügbarkeitsanforderungen bescheiden waren, weil sie häufig auf eine bestehende Architektur aufgeschichtet werden kann, ohne die Anwendungslogik selbst neu zu schreiben. Sie zählt besonders während der Modernisierung, weil ein Legacy-System, das inkrementelle Änderung durchläuft, stärker Ausfällen ausgesetzt ist als ein stabiles, und Redundanz die Sicherheitsmarge bietet, die nötig ist, um das Geschäft am Laufen zu halten, während die zugrunde liegende Architektur umgeformt wird. Der Zielkonflikt ist, dass Redundanz Kapital- und Betriebskosten gegen reduziertes Risiko eintauscht, und ihr Schutz ist nur so gut wie die Vielfalt und das Testen der redundanten Pfade selbst.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Identifizieren Sie alle Single Points of Failure in der Legacy-System-Architektur und priorisieren Sie sie nach Geschäftsauswirkung
- Stellen Sie redundante Instanzen kritischer Anwendungskomponenten hinter Load Balancern bereit
- Implementieren Sie Datenbankreplikation mit automatischem Failover für Datenpersistenzschichten
- Stellen Sie sicher, dass redundante Komponenten über verschiedene Fehlerdomänen (Racks, Zonen, Regionen) verteilt bereitgestellt werden
- Testen Sie, dass redundante Komponenten tatsächlich Last übernehmen können, durch regelmäßiges Simulieren von Primärausfällen
- Vermeiden Sie Common-Mode-Ausfälle, indem Sie wo praktikabel unterschiedliche Implementierungen oder Konfigurationen nutzen
- Überwachen Sie alle redundanten Instanzen, um sicherzustellen, dass Standby-Komponenten gesund und bereit bleiben

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Beseitigt Single Points of Failure, die in Legacy-Architekturen üblich sind
- Ermöglicht Wartung und Upgrades ohne Ausfallzeit
- Erhöht die gesamte Systemkapazität durch Active-Active-Konfigurationen
- Bietet Versicherung gegen Hardwareausfälle und Infrastrukturprobleme

**Kosten und Risiken:**
- Verdoppelt oder verdreifacht Infrastrukturkosten für redundante Komponenten
- Zustandssynchronisierung zwischen redundanten Instanzen fügt Komplexität hinzu
- Nie getestete redundante Komponenten könnten ausfallen, wenn sie tatsächlich gebraucht werden
- Legacy-Anwendungen unterstützen möglicherweise keine Multi-Instanz-Bereitstellung ohne Modifikation

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Eine Kommunalverwaltung betrieb ihr Bürgerdienste-Portal auf einem einzelnen Legacy-Anwendungsserver und einem einzelnen Datenbankserver. Ein Festplattenausfall am Datenbankserver verursachte einen dreitägigen Ausfall, während Daten von Bandbackups wiederhergestellt wurden. Nach diesem Vorfall stellte das Team redundante Datenbankserver mit synchroner Replikation, redundante Anwendungsserver hinter einem Load Balancer und redundante Netzwerkpfade bereit. Die Investition erhöhte die Infrastrukturkosten um 120 %, aber der nächste Hardwareausfall wurde transparent mit automatischem Failover und null bürgerseitiger Auswirkung gehandhabt.
