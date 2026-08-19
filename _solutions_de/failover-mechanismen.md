---
title: Failover-Mechanismen
description: Automatischer Wechsel zu redundanten Komponenten im Fehlerfall.
category:
- Architecture
- Operations
problems:
- single-points-of-failure
- system-outages
- cascade-failures
- slow-incident-resolution
- unpredictable-system-behavior
- service-timeouts
- constant-firefighting
layout: solution
lang: de
en_slug: failover-mechanisms
related_solutions:
- slug: failover-cluster
  similarity: 0.85
- slug: retry
  similarity: 0.8
- slug: redundancy
  similarity: 0.8
- slug: resilience
  similarity: 0.8
- slug: chaos-engineering
  similarity: 0.75
- slug: rollback-mechanisms
  similarity: 0.75
---

## Description

Failover-Mechanismen sind die breitere Menge an Techniken — auf Infrastrukturebene durch Load Balancer, DNS-Failover oder Container-Orchestrierung, und auf Anwendungsebene durch Datenbankverbindungs-Failover, Umleitung von Message Queues oder Circuit Breaker —, die Verkehr oder Arbeit automatisch von einer ausgefallenen Komponente zu einer gesunden redundanten umleiten, ohne dass ein Mensch den Ausfall bemerken und manuell eingreifen muss. Legacy-Systemen fehlt davon oft alles: Eine abgebrochene Broker-Verbindung oder eine ausgefallene Abhängigkeit bedeutete historisch einen manuellen Neustart und einen echten Ausfall, weil das ursprüngliche Design annahm, die betreffende Komponente würde einfach weiter funktionieren. Automatisches Failover genau an den Stellen hinzuzufügen, an denen Legacy-Komponenten bekanntermaßen ausfallen — mit sorgfältig abgestimmten Schwellenwerten, damit vorübergehende Ausschläge keine unnötigen Umschaltungen auslösen, und Circuit Breakern neben dem Failover, um zu verhindern, dass der Wechsel selbst kaskadiert —, verwandelt das, was früher einen Bereitschaftsingenieur und zwanzig Minuten Ausfallzeit erforderte, in eine automatische Fünf-Sekunden-Wiederherstellung. Der Mechanismus muss bewusst durch Chaos Engineering oder geplante Übungen erprobt werden, um überhaupt Vertrauen zu haben, dass er bei einem echten Ausfall funktioniert, da Failover-Logik, die außerhalb eines Tests noch nie tatsächlich ausgelöst wurde, selbst eine häufige Quelle von Überraschungen ist, und asynchrone Replikation hinter einem Failover-Pfad birgt immer ein gewisses Risiko von Datenkonsistenzlücken während des Wechsels.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Kartieren Sie alle kritischen Pfade durch das Legacy-System und identifizieren Sie, wo der Ausfall einer einzelnen Komponente den Dienst anhalten würde
- Implementieren Sie automatisches Failover auf Infrastrukturebene mittels Load Balancern, DNS-Failover oder Container-Orchestrierung
- Konfigurieren Sie Failover auf Anwendungsebene für Datenbankverbindungen, Message Queues und externe Serviceaufrufe
- Definieren Sie Failover-Schwellenwerte sorgfältig, um vorzeitiges Umschalten wegen vorübergehender Netzwerkprobleme zu vermeiden
- Implementieren Sie Circuit Breaker neben Failover, um kaskadierende Ausfälle während des Wechsels zu verhindern
- Führen Sie Chaos-Engineering-Übungen oder geplante Failover-Tests durch, um zu validieren, dass die Mechanismen unter echten Bedingungen funktionieren
- Überwachen Sie Failover-Ereignisse und alarmieren Sie darauf, damit Teams Ursachen untersuchen können

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Verringert die mittlere Wiederherstellungszeit beim Ausfall von Komponenten
- Verhindert nutzerseitig sichtbare Ausfälle bei vorübergehenden Infrastrukturproblemen
- Schafft Vertrauen für das Deployment von Änderungen an Legacy-Systemen
- Automatisiert Wiederherstellungsmaßnahmen, die zuvor manuellen Eingriff erforderten

**Kosten und Risiken:**
- Failover-Logik fügt Komplexität hinzu, die selbst ausfallen oder sich unerwartet verhalten kann
- Datenkonsistenzrisiken während des Wechsels, wenn die Replikation asynchron ist
- Das Verdecken zugrundeliegender Probleme kann notwendige architektonische Korrekturen verzögern
- Erfordert laufendes Testen, um sicherzustellen, dass Failover-Pfade funktionsfähig bleiben

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Logistikunternehmen verließ sich auf ein Legacy-Messaging-System, das gelegentlich Verbindungen zum primären Broker verlor. Ingenieure fügten einen automatischen Failover-Mechanismus hinzu, der die Nichtverfügbarkeit des Brokers innerhalb von fünf Sekunden erkannte und Nachrichten zu einem Standby-Broker umleitete. Dies beseitigte die zuvor üblichen 20-minütigen Ausfälle, die einen manuellen Neustart der Messaging-Pipeline erforderten, und gab dem Betriebsteam Raum, ein richtiges Upgrade der Messaging-Infrastruktur zu planen.
