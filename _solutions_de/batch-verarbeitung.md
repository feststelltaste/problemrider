---
title: Batch-Verarbeitung
description: Sammlung und gemeinsame Verarbeitung mehrerer Jobs.
category:
- Performance
- Operations
problems:
- slow-application-performance
- high-number-of-database-queries
- high-database-resource-utilization
- growing-task-queues
- gradual-performance-degradation
- excessive-disk-io
- interrupt-overhead
- unoptimized-file-access
- long-running-database-transactions
- long-running-transactions
layout: solution
lang: de
en_slug: batch-processing
related_solutions:
- slug: distributed-processing
  similarity: 0.8
- slug: pipelining
  similarity: 0.8
- slug: parallelization
  similarity: 0.75
- slug: streaming
  similarity: 0.75
- slug: transactions
  similarity: 0.75
- slug: lazy-loading
  similarity: 0.75
---

## Description

Batch-Verarbeitung gruppiert viele einzelne Operationen — Datenbankschreibvorgänge, API-Aufrufe, Dateioperationen — in eine einzelne gesammelte Einheit, die gemeinsam ausgeführt wird, wodurch feste Kosten pro Operation wie Verbindungsaufbau, Transaktions-Overhead und Netzwerk-Roundtrips über viele Elemente amortisiert werden, statt sie einmal pro Element zu zahlen. Der Mechanismus tauscht Latenz gegen Durchsatz: Einzelne Elemente warten, bis sich ein Batch füllt oder ein Zeitfenster verstreicht, aber die Gesamtkosten der Verarbeitung des ganzen Sets sinken erheblich im Vergleich zur separaten Handhabung jedes einzelnen. Dies passt natürlich zu Legacy-Systemen, die ursprünglich gebaut wurden, um einen Datensatz nach dem anderen zu verarbeiten, und seither weit über die Transaktionsvolumina hinaus gedrängt wurden, für die ihr Pro-Element-Design jemals gedacht war, was dazu führt, dass die Datenbank oder das nachgelagerte System den Großteil ihrer Kapazität für Pro-Aufruf-Overhead statt tatsächlicher Arbeit ausgibt. Die Einführung von Batching erfordert üblicherweise keine Neuarchitektur der Kernlogik des Legacy-Systems — sie erfordert die Identifikation, welche einzeln verarbeiteten Operationen sicher gesammelt und neu geordnet werden können, und den Ersatz von Einzelzeilen-Datenbankaufrufen durch Bulk-Äquivalente, die die Datenbank bereits unterstützt, die der Legacy-Code aber nie genutzt hat. Der Tradeoff ist, dass ein Fehler jetzt einen gesamten Batch betrifft statt eines einzelnen Elements, sodass Neustartbarkeit und Teilwiederholungshandhabung notwendige Designbelange werden, die ein rein pro-elementbasiertes System nie adressieren musste.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Identifizieren Sie Operationen, die Elemente einzeln verarbeiten, aber gruppiert werden könnten: Datenbankeinfügungen, API-Aufrufe, Dateischreibvorgänge
- Sammeln Sie Elemente in Batches angemessener Größe basierend auf Speicherbeschränkungen und Verarbeitungszeitanforderungen
- Nutzen Sie Bulk-Datenbankoperationen (Batch-Einfügungen, Bulk-Updates) statt einzelner Zeilenoperationen
- Implementieren Sie Batch-Fenster für nicht zeitkritische Operationen zur Verarbeitung in verkehrsarmen Stunden
- Fügen Sie Monitoring hinzu, um Batch-Größen, Verarbeitungszeiten und Fehlerraten zu verfolgen
- Designen Sie Batch-Prozesse so, dass sie vom Fehlerpunkt statt vom Anfang neu gestartet werden können
- Erwägen Sie Micro-Batching für Nahezu-Echtzeit-Anforderungen, wo volle Batch-Fenster zu langsam sind

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Verringert dramatisch den Pro-Element-Overhead wie Verbindungsaufbau, Transaktionsmanagement und Netzwerk-Roundtrips
- Verbessert den Durchsatz, indem feste Kosten über viele Elemente amortisiert werden
- Verringert die Last auf nachgelagerte Systeme, indem Anfragemuster geglättet werden
- Ermöglicht effiziente Nutzung von Bulk-APIs und Datenbankoperationen

**Kosten und Risiken:**
- Führt Latenz für einzelne Elemente ein, die auf die Füllung des Batches warten müssen
- Batch-Fehler betreffen mehrere Elemente und erfordern robuste Fehlerbehandlung und Teilwiederholungslogik
- Die Abstimmung der Batch-Größe erfordert Experimentieren, um Durchsatz und Latenz auszubalancieren
- Legacy-Systeme unterstützen möglicherweise keine Bulk-Operationen, was Workarounds erfordert

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Legacy-Bestandsmanagementsystem aktualisierte Lagerbestände durch Ausführung einzelner UPDATE-Anweisungen für jede Verkaufstransaktion und verarbeitete über 50.000 einzelne Datenbankaufrufe während Spitzenstunden. Das Team führte Batch-Verarbeitung ein, die Lagerbestandsupdates in Gruppen von 500 sammelte und sie alle 5 Sekunden als Bulk-UPDATE-Anweisungen ausführte. Die Datenbanklast sank um über 90 %, und die freigesetzten Ressourcen erlaubten dem System, wachsende Transaktionsvolumina ohne Hardware-Upgrades zu handhaben. Die leichte Verzögerung bei Lagerbestandsupdates war akzeptabel, weil das Geschäft bereits mit einer Toleranz für geringfügige Bestandsdiskrepanzen operierte.
