---
title: Lazy Evaluation
description: Laden und Verarbeiten von Daten nur bei Bedarf.
category:
- Performance
- Code
problems:
- slow-application-performance
- excessive-object-allocation
- high-client-side-resource-consumption
- memory-leaks
- gradual-performance-degradation
- lazy-loading
layout: solution
lang: de
en_slug: lazy-evaluation
related_solutions:
- slug: lazy-loading
  similarity: 0.95
- slug: predictive-loading
  similarity: 0.8
- slug: progressive-loading
  similarity: 0.75
- slug: distributed-caching
  similarity: 0.75
- slug: connection-pooling
  similarity: 0.75
- slug: parallelization
  similarity: 0.75
---

## Description

Lazy Evaluation verschiebt die Berechnung oder das Laden eines Wertes bis zu dem Moment, in dem er tatsächlich gebraucht wird, statt ihn eifrig zu berechnen, sobald er deklariert oder konstruiert wird. Der Mechanismus nimmt typischerweise die Form eines Proxys, eines Suppliers oder Thunks, eines Generators oder einer lazy initialisierten ORM-Assoziation an, die den ersten Zugriff abfängt und erst dann die teure Arbeit — Datenbankabfrage, Objektkonstruktion oder Berechnung — ausführt, das Ergebnis danach zwischenspeichert oder verwirft, je nachdem. Legacy-Systeme greifen häufig standardmäßig zu eifriger Initialisierung, weil sie zur Schreibzeit einfacher zu durchdenken ist: Ganze Objektgraphen, Sammlungen oder Konfigurationsbäume werden im Voraus geladen, unabhängig davon, ob ein gegebener Codepfad sie je nutzen wird, was mit wachsendem Datenvolumen über die Jahre zunehmend teuer wird, während der eifrig ladende Code selbst nie überarbeitet wird. Lazy Evaluation auf solchen Code anzuwenden verschiebt Kosten von „immer, ob gebraucht oder nicht" zu „nur wenn tatsächlich genutzt", was besonders effektiv in Legacy-Systemen ist, wo ein großer Anteil vorgeladener Daten selten genutzten Features oder Randfällen dient. Der Zielkonflikt, der in einem Legacy-Kontext am meisten zählt, ist, dass Lazy Evaluation vorhersagbare, vorab anfallende Latenz gegen Latenz eintauscht, die unvorhersehbar beim ersten Zugriff auftaucht, was sich als neue, schwer diagnostizierbare Verlangsamungen zeigen kann, sofern das Team dies nicht explizit berücksichtigt, besonders rund um das N+1-Abfrageproblem bei lazy geladenen ORM-Beziehungen.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Identifizieren Sie eifrig geladene Daten, die häufig ungenutzt sind: vorgeladene Sammlungen, verbundene Beziehungen, berechnete Felder
- Ersetzen Sie eifrige Initialisierung durch Lazy Proxies oder Supplier-Muster, die Berechnung bis zum ersten Zugriff verschieben
- Implementieren Sie Lazy Loading für ORM-Beziehungen, die nicht in jedem Anwendungsfall gebraucht werden
- Nutzen Sie Generatoren oder Streams statt ganze Sammlungen zur Verarbeitung im Speicher zu materialisieren
- Wenden Sie Paginierung und virtuelles Scrollen im Frontend an, statt ganze Datensätze zu laden
- Seien Sie vorsichtig beim N+1-Problem: Nutzen Sie Batch-Fetching oder explizites Eager Loading, wo Lazy Loading übermäßige Abfragen verursacht
- Profilieren Sie, um zu verifizieren, dass Lazy Evaluation die Performance im jeweiligen spezifischen Fall tatsächlich verbessert

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Verringert Startzeit und Speicherverbrauch, indem Arbeit bis zum tatsächlichen Bedarf verschoben wird
- Beseitigt Berechnung und Datenladen für Codepfade, die nie ausgeführt werden
- Verbessert die gefühlte Performance, indem Initialisierungskosten über die Zeit verteilt werden
- Ermöglicht die Arbeit mit Datensätzen größer als der verfügbare Speicher durch Streaming

**Kosten und Risiken:**
- Kann Latenz auf unerwartete Momente verschieben, was nutzersichtbare Verzögerungen beim ersten Zugriff verursacht
- Lazy geladene ORM-Beziehungen können N+1-Abfrageprobleme auslösen, wenn nicht sorgfältig verwaltet
- Debugging wird schwerer, weil Initialisierung zu unvorhersehbaren Zeitpunkten geschieht
- Thread-Sicherheit von Lazy-Initialisierung erfordert sorgfältige Implementierung in nebenläufigen Umgebungen

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Legacy-Content-Management-System lud eifrig alle Metadaten, verwandte Dokumente und Zugriffskontrolllisten für jedes Dokument, wenn eine Ordnerliste angezeigt wurde. Ein Ordner mit 200 Dokumenten löste über 1.000 Datenbankabfragen aus und lud mehrere hundert Megabyte Daten in den Speicher, obwohl Nutzer nur mit wenigen Dokumenten gleichzeitig interagierten. Das Team änderte die Ordnerliste, sodass nur Dokumenttitel und -daten geladen werden, wobei Metadaten und Beziehungen lazy geladen werden, wenn ein Nutzer auf ein bestimmtes Dokument klickt. Die Antwortzeit der Ordnerliste sank von 8 Sekunden auf 300 Millisekunden, und der Server-Speicherverbrauch beim Durchsuchen von Ordnern sank um über 80 %.
