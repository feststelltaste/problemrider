---
title: Hochverfügbarkeitsarchitekturen
description: Für maximale Verfügbarkeit und Fehlertoleranz konzipierte Architekturen.
category:
- Architecture
- Operations
problems:
- single-points-of-failure
- system-outages
- cascade-failures
- stagnant-architecture
- capacity-mismatch
- monolithic-architecture-constraints
- technical-architecture-limitations
layout: solution
lang: de
en_slug: high-availability-architectures
related_solutions:
- slug: redundancy
  similarity: 0.8
- slug: failover-cluster
  similarity: 0.8
- slug: data-replication
  similarity: 0.8
- slug: resilience
  similarity: 0.75
- slug: failover-mechanisms
  similarity: 0.75
- slug: load-balancing
  similarity: 0.75
---

## Description

Hochverfügbarkeitsarchitektur ist eine Menge von Designentscheidungen — Redundanz auf jeder Ebene, Datenreplikation über Knoten oder Regionen hinweg, zustandslose Anwendungsschichten und automatisiertes Failover — die darauf abzielen, ein explizites Verfügbarkeitsziel wie 99,9 % oder 99,99 % zu erreichen, statt dass Verfügbarkeit als beiläufiges Nebenprodukt daraus entsteht, wie ein System zufällig gebaut ist. Sie beginnt mit einer Verfügbarkeitsanforderungsanalyse, die das Ziel für jeden Dienst festlegt, weil das Streben nach höherer Verfügbarkeit steil abnehmende Erträge und entsprechend steil steigende Kosten hat, während sich das Ziel 100 % nähert, was das Ziel selbst zu einer geschäftlichen statt einer rein technischen Entscheidung macht. Legacy-Systeme sind überproportional Verfügbarkeitsrisiken ausgesetzt, weil sie häufig um ein einziges Rechenzentrum, eine einzige Datenbankinstanz oder Sitzungszustand im Server-Speicher herum gebaut wurden — architektonische Entscheidungen, die unter den Annahmen ihrer Ära Sinn ergaben, aber jetzt Single Points of Failure ohne Redundanz darstellen, auf die zurückgegriffen werden könnte. Hochverfügbarkeit in einem solchen System zu erreichen erfordert typischerweise, die Sitzungsverwaltung zustandslos zu refaktorieren und Datenreplikationsstrategien einzuführen, bevor überhaupt redundante Infrastruktur hinzugefügt werden kann, da das bloße Duplizieren einer zustandsbehafteten Legacy-Anwendung an sich keine Failover-Fähigkeit bietet. Die Belohnung für diese beträchtliche Investition ist ein System, das Infrastrukturausfälle — ein verlorenes Rechenzentrum, eine ausgefallene Verfügbarkeitszone — ohne kundensichtbare Auswirkung übersteht, was oft der Unterschied zwischen Legacy-Systemen ist, die moderne regulatorische oder vertragliche Verfügbarkeitsverpflichtungen erfüllen, und solchen, die es nicht tun.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Führen Sie eine Verfügbarkeitsanforderungsanalyse durch, um die Ziel-Betriebszeit (z. B. 99,9 % vs. 99,99 %) für jeden Dienst zu bestimmen
- Beseitigen Sie Single Points of Failure, indem Sie Redundanz auf jeder Ebene hinzufügen: Rechenleistung, Speicher, Netzwerk und Lastverteilung
- Implementieren Sie Datenreplikationsstrategien (synchron für kritische Daten, asynchron für weniger kritische) über Verfügbarkeitszonen hinweg
- Entwerfen Sie zustandslose Anwendungsschichten, wo möglich, sodass jede Instanz jede Anfrage bearbeiten kann
- Nutzen Sie geografische Verteilung oder Multi-Region-Deployments für Disaster-Recovery-Szenarien
- Etablieren Sie automatisiertes Failover mit Gesundheitsüberwachung, um die Wiederherstellungszeit zu minimieren
- Planen und führen Sie regelmäßige Disaster-Recovery-Übungen durch, um die Architektur unter echten Ausfallbedingungen zu validieren

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Verringert ungeplante Ausfallzeit und deren geschäftliche Auswirkung dramatisch
- Ermöglicht Wartung und Upgrades ohne Dienstunterbrechung
- Bietet eine widerstandsfähige Grundlage für geschäftskritische Legacy-Dienste
- Unterstützt regulatorische Compliance-Anforderungen für Dienstverfügbarkeit

**Kosten und Risiken:**
- Erheblich höhere Infrastruktur- und Betriebskosten
- Erhöhte architektonische Komplexität, die spezialisierte Fähigkeiten zur Verwaltung erfordert
- Datenkonsistenzherausforderungen bei Multi-Knoten- oder Multi-Region-Setups
- Legacy-Anwendungen benötigen möglicherweise erhebliches Refactoring, um zustandslos oder clusterfähig zu werden
- Abnehmende Erträge, während sich Verfügbarkeitsziele 100 % nähern

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Bankinstitut, das ein Legacy-Transaktionsverarbeitungssystem in einem einzigen Rechenzentrum betrieb, erlebte einen vierstündigen Ausfall wegen eines Stromausfalls, was zu regulatorischer Prüfung und Kundenabwanderung führte. Das Team gestaltete die Architektur mit Active-Active-Clustern über zwei Rechenzentren neu, mit synchroner Datenbankreplikation für Transaktionsdaten und automatisiertem DNS-Failover. Während die Migration acht Monate dauerte und die Refaktorierung der Sitzungsverwaltungsschicht erforderte, überstand das System anschließend einen vollständigen Rechenzentrums-Netzwerkausfall ohne jegliche kundensichtbare Auswirkung und erreichte das von Regulierern vorgeschriebene 99,99-%-Verfügbarkeitsziel.
