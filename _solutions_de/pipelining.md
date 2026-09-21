---
title: Pipelining
description: Gleichzeitige Ausführung sequenzieller Verarbeitungsschritte.
category:
- Performance
- Architecture
problems:
- slow-application-performance
- bottleneck-formation
- growing-task-queues
- long-build-and-test-times
- scaling-inefficiencies
- work-queue-buildup
layout: solution
lang: de
en_slug: pipelining
related_solutions:
- slug: parallelization
  similarity: 0.8
- slug: distributed-processing
  similarity: 0.8
- slug: batch-processing
  similarity: 0.8
- slug: streaming
  similarity: 0.75
- slug: asynchronous-processing
  similarity: 0.7
- slug: reactive-programming
  similarity: 0.7
---

## Description

Pipelining zerlegt einen sequenziellen Verarbeitungs-Workflow in diskrete, durch Warteschlangen verbundene Stufen, sodass jede Stufe gleichzeitig an einem anderen Datenelement arbeiten kann, statt dass der gesamte Workflow wartet, bis ein Element vollständig jede Stufe durchlaufen hat, bevor der nächste beginnt. Legacy-Batch- und ETL-Jobs sind häufig als eine lange sequenzielle Kette gebaut — extrahieren, dann transformieren, dann laden —, wobei jede Phase leerläuft, während sie darauf wartet, dass die vorherige vollständig abschließt, obwohl die Stufen sich eindeutig überlappen könnten. Die Umstrukturierung eines solchen Workflows in eine explizite Pipeline erhöht den Gesamtdurchsatz, indem jede Stufe kontinuierlich beschäftigt gehalten wird, und macht zudem die Engpassstufe sichtbar und unabhängig skalierbar, statt in einem monolithischen Prozess verborgen zu sein. Die zusätzliche Komplexität zeigt sich in der Fehlerbehandlung: Ein Fehler mitten in einer Pipeline ist schwerer nachzuvollziehen als ein Fehler in einem einzelnen sequenziellen Skript, und Backpressure zwischen Stufen muss bewusst gemanagt werden, sonst überwältigt ein schneller Produzent einen langsamen Konsumenten.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Identifizieren Sie sequenzielle Verarbeitungs-Workflows, bei denen die Ausgabe einer Stufe in die nächste einfließt (z. B. Extract-Transform-Load, Anfrageverarbeitungsketten)
- Zerlegen Sie den Workflow in diskrete Stufen, die gleichzeitig an unterschiedlichen Datenelementen arbeiten können
- Verbinden Sie Stufen mit begrenzten Warteschlangen oder Kanälen, um Backpressure zu managen und Speichererschöpfung zu verhindern
- Stellen Sie sicher, dass jede Stufe unabhängig skalierbar ist, sodass Engpassstufen mehr Ressourcen erhalten können
- Implementieren Sie Überwachung für Durchsatz und Warteschlangentiefe jeder Stufe, um zu identifizieren, welche Stufen den Gesamtdurchsatz begrenzen
- Beginnen Sie damit, die sequenziellsten und zeitaufwendigsten Workflows zu pipelinen, und erweitern Sie dann auf andere Bereiche

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Erhöht den Gesamtdurchsatz durch Überlappung der Ausführung sequenzieller Schritte
- Macht Engpassstufen sichtbar und unabhängig adressierbar
- Verbessert die Ressourcennutzung, indem alle Verarbeitungseinheiten gleichzeitig beschäftigt gehalten werden
- Ermöglicht Streaming-Verarbeitung großer Datensätze, ohne alles in den Speicher zu laden

**Kosten und Risiken:**
- Fügt Komplexität in der Fehlerbehandlung hinzu, wenn Fehler mitten in der Pipeline auftreten
- Das Debuggen von Problemen wird schwerer, wenn Daten durch mehrere gleichzeitige Stufen fließen
- Backpressure-Management ist kritisch; ohne es können schnelle Produzenten langsame Konsumenten überwältigen
- Legacy-Code mit eng gekoppelten sequenziellen Schritten erfordert erhebliche Umgestaltung für die Pipeline-Bildung
- Erhöht die Latenz für einzelne Elemente, selbst während der Durchsatz steigt

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Data-Warehousing-System verarbeitete nächtliche Datenfeeds, indem es sequenziell Daten aus Quellsystemen extrahierte, sie durch Geschäftsregeln transformierte und in das Warehouse lud. Jede Phase wartete, bis die vorherige vollständig abgeschlossen war, was zu einem 10-Stunden-Verarbeitungsfenster führte. Das Team strukturierte die Pipeline so um, dass Extraktion, Transformation und Laden gleichzeitig an unterschiedlichen Datenbatches arbeiteten. Sobald der erste Batch extrahiert war, begann die Transformation daran, während die Extraktion am nächsten Batch fortgesetzt wurde. Diese Überlappung reduzierte die gesamte Verarbeitungszeit auf unter 4 Stunden, komfortabel innerhalb des nächtlichen Fensters, selbst während die Datenvolumina weiter wuchsen.
