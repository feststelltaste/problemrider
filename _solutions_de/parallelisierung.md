---
title: Parallelisierung
description: Gleichzeitige Ausführung mehrerer Berechnungen oder Aufgaben.
category:
- Performance
- Architecture
problems:
- slow-application-performance
- bottleneck-formation
- scaling-inefficiencies
- long-build-and-test-times
- slow-database-queries
- insufficient-worker-capacity
- growing-task-queues
- atomic-operation-overhead
- false-sharing
- memory-barrier-inefficiency
layout: solution
lang: de
en_slug: parallelization
related_solutions:
- slug: distributed-processing
  similarity: 0.85
- slug: pipelining
  similarity: 0.8
- slug: reactive-programming
  similarity: 0.8
- slug: distributed-caching
  similarity: 0.8
- slug: lazy-loading
  similarity: 0.75
- slug: data-replication
  similarity: 0.75
---

## Description

Parallelisierung zerlegt unabhängige Arbeitseinheiten, sodass sie gleichzeitig über mehrere Kerne, Threads oder Prozesse ausgeführt werden, statt nacheinander. Viele Legacy-Systeme wurden als Single-Thread-Batch-Jobs oder Request-Handler zu einer Zeit geschrieben, als Hardware der Engpass war statt der Fähigkeit der Software, sie zu nutzen, sodass sie moderne Multi-Core-Kapazität nahezu vollständig ungenutzt lassen. Zu identifizieren, welche Teile eines Workloads wirklich unabhängig sind — Datensätze in einem Batch, Anfragen in einer Warteschlange, Tests in einer Suite —, und sie gleichzeitig auszuführen, kann für genau diese Workloads nahezu lineare Beschleunigungen produzieren, ohne zu ändern, was der Code tatsächlich berechnet. Der Haken ist, dass Legacy-Code häufig versteckten gemeinsamen veränderlichen Zustand trägt, und die Parallelisierung um diesen herum ohne vorherige Beseitigung oder korrekte Synchronisierung dieses Zustands führt Race Conditions und Deadlocks ein, die weit schwerer zu diagnostizieren sind als das Performance-Problem, das sie lösen sollte.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Profilieren Sie die Anwendung, um CPU- oder I/O-gebundene Engpässe zu identifizieren, die von gleichzeitiger Ausführung profitieren könnten
- Zerlegen Sie unabhängige Aufgaben (z. B. Batch-Verarbeitung, Berichtserstellung, Datenimporte) in parallelisierbare Einheiten
- Verwenden Sie Thread-Pools, Worker-Prozesse oder asynchrone I/O-Frameworks, die zu Sprache und Laufzeitumgebung passen
- Stellen Sie sicher, dass gemeinsamer Zustand korrekt synchronisiert oder beseitigt wird, um Race Conditions und Deadlocks zu verhindern
- Beginnen Sie mit unproblematisch parallelen Workloads (z. B. Verarbeitung unabhängiger Datensätze), bevor Sie sich voneinander abhängigen Aufgaben zuwenden
- Parallelisieren Sie Build- und Testpipelines, um Feedback-Schleifenzeiten während der Entwicklung zu reduzieren
- Überwachen Sie Thread- und Prozessauslastung, um den Grad der Parallelität richtig zu dimensionieren

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Kann nahezu lineare Beschleunigungen für Workloads liefern, die sich sauber in unabhängige Einheiten zerlegen lassen
- Nutzt moderne Multi-Core-Hardware besser, die Legacy-Single-Thread-Code unterauslastet
- Reduziert die Ende-zu-Ende-Verarbeitungszeit für Batch-Jobs und Datenpipelines

**Kosten und Risiken:**
- Führt Nebenläufigkeitsfehler (Race Conditions, Deadlocks) ein, die schwer zu reproduzieren und zu debuggen sind
- Legacy-Code mit globalem Zustand oder gemeinsamen veränderlichen Daten erfordert erhebliche Umgestaltung, um sicher parallelisiert zu werden
- Parallelität fügt Komplexität zu Fehlerbehandlung, Wiederholungslogik und Ergebnisaggregation hinzu
- Kann Ressourcenkonkurrenz (Speicher, Datenbankverbindungen, I/O) erhöhen, wenn nicht ordnungsgemäß verwaltet
- Abnehmende Erträge jenseits eines bestimmten Parallelitätsgrads aufgrund von Synchronisierungsaufwand

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Die Batch-Verarbeitung eines Finanzinstituts zum Geschäftsschluss dauerte über sechs Stunden, wenn sie sequenziell durch Kontoabgleich, Zinsberechnung und Berichtserstellung lief. Die Analyse zeigte, dass diese drei Prozesse auf unabhängigen Datenpartitionen arbeiteten. Das Team parallelisierte jeden Prozess über Kontobereiche mittels eines Worker-Pools und führte die drei Prozesse zudem gleichzeitig aus, wo Datenabhängigkeiten dies erlaubten. Die gesamte Batch-Verarbeitungszeit fiel auf 90 Minuten, deutlich innerhalb des nächtlichen Wartungsfensters, ohne jegliche Änderungen an der Geschäftslogik selbst.
