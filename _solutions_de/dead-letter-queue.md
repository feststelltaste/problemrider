---
title: Dead Letter Queue
description: Weiterleitung fehlgeschlagener Nachrichten an eine dedizierte Warteschlange
  zur späteren Prüfung und Wiederverarbeitung, statt sie zu verlieren.
category:
- Architecture
- Operations
problems:
- silent-data-corruption
- inadequate-error-handling
- cascade-failures
- monitoring-gaps
- task-queues-backing-up
- increased-error-rates
layout: solution
lang: de
en_slug: dead-letter-queue
related_solutions:
- slug: retry
  similarity: 0.75
- slug: failover-mechanisms
  similarity: 0.7
- slug: error-reporting-and-analysis
  similarity: 0.7
- slug: error-handling
  similarity: 0.7
- slug: rate-limiting
  similarity: 0.65
- slug: heartbeat
  similarity: 0.65
---

## Description

Eine Dead Letter Queue ist ein dediziertes Ziel für Nachrichten, die eine Verarbeitungspipeline nach einer definierten Anzahl von Wiederholungsversuchen nicht erfolgreich handhaben konnte, wobei die ursprüngliche Payload zusammen mit Fehlerdetails, Wiederholungszähler und Zeitstempel erfasst wird, statt die Nachricht zu verwerfen oder sie die dahinterliegende Pipeline blockieren zu lassen. Dies gibt asynchroner Verarbeitung einen expliziten Fehlerpfad: Statt dass eine einzelne Poison-Nachricht die gesamte Warteschlange stoppt oder ein vorübergehender nachgelagerter Ausfall still Daten verwirft, werden fehlgeschlagene Nachrichten zu einem separaten Ort umgeleitet, wo sie geprüft, diagnostiziert und wiedergegeben werden können, sobald das zugrunde liegende Problem behoben ist. Dies ist besonders wichtig in Legacy-nachrichtengetriebenen Systemen, die häufig mit Fehlerbehandlung gebaut wurden, die darauf hinauslief, einen Fehler zu protokollieren und die Nachricht zu verwerfen — ein Ansatz, der bei jedem nachgelagerten Ausfall still Daten verliert und keine Spur davon hinterlässt, was verloren ging, bis ein Kunde oder Prüfer nach einer Transaktion fragt, die nie stattfand. Die Einführung einer Dead Letter Queue verwandelt diese stillen Verluste in einen dauerhaften, überprüfbaren Rückstau, und sie mit Überwachung der Warteschlangentiefe zu kombinieren verwandelt Verarbeitungsfehler in ein operatives Signal, auf das das Team handeln kann, statt in ein Rätsel, das viel später entdeckt wird. Weil das Wiedergeben einer Dead-Lettered-Nachricht die ursprünglich fehlgeschlagene Verarbeitung erneut auslöst, muss das Zielsystem sichere Wiederverarbeitung vertragen — eine Idempotenzanforderung, die Dead Letter Queues aufdecken statt erzeugen, und eine, die Legacy-Systeme, die ohne diese Garantie gebaut wurden, adressieren müssen, bevor Replay mit Vertrauen genutzt werden kann.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Identifizieren Sie alle Nachrichtenwarteschlangen und asynchronen Verarbeitungspipelines im Legacy-System
- Konfigurieren Sie Dead Letter Queues für jede Verarbeitungswarteschlange, die Nachrichten weiterleiten, die nach einer definierten Anzahl von Wiederholungsversuchen fehlschlagen
- Beziehen Sie die ursprüngliche Nachrichten-Payload, Fehlerdetails, Wiederholungszähler und Zeitstempel in Dead-Letter-Einträge ein
- Bauen Sie Überwachung und Alarmierung auf die Tiefe der Dead Letter Queue, um Verarbeitungsfehler früh zu erkennen
- Erstellen Sie Werkzeuge zur Prüfung von Dead-Letter-Nachrichten, zur Fehlerdiagnose und zum Wiedergeben nach Korrekturen
- Definieren Sie Aufbewahrungsrichtlinien für Dead-Letter-Nachrichten basierend auf regulatorischen und geschäftlichen Anforderungen
- Implementieren Sie automatisierte Klassifizierung von Dead-Letter-Nachrichten, um wiederkehrende Fehlermuster zu identifizieren

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Verhindert Datenverlust durch vorübergehende Fehler oder Verarbeitungsprobleme
- Bietet ein Diagnosewerkzeug zum Verständnis, warum Nachrichten fehlschlagen
- Entkoppelt Fehlerbehandlung von der Hauptverarbeitungspipeline und hält sie sauber
- Ermöglicht Nachrichtenwiedergabe nach Fehlerbehebungen, ohne vorgelagerte Systeme erneut auszulösen
- Verhindert, dass Poison-Nachrichten die Hauptverarbeitungswarteschlange blockieren

**Kosten und Risiken:**
- Dead Letter Queues erfordern Überwachung; unbeaufsichtigte Warteschlangen können unbegrenzt wachsen
- Wiedergegebene Nachrichten können Nebeneffekte verursachen, wenn das System nicht idempotent ist
- Fügt für jede Nachrichtenwarteschlange Infrastruktur und operative Komplexität hinzu
- Veraltete Dead-Letter-Nachrichten können ungültig werden, wenn sich das System seit dem ursprünglichen Fehler geändert hat

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Legacy-Auftragsabwicklungssystem verarbeitete Nachrichten aus einer RabbitMQ-Warteschlange. Wenn eine Nachricht wegen Datenvalidierungsfehlern oder nachgelagerten Serviceausfällen nicht verarbeitet werden konnte, wurde die Nachricht mit nur einem Logeintrag verworfen. Während eines Zahlungsgateway-Ausfalls gingen 2.400 Bestellungen dauerhaft verloren. Nach diesem Vorfall fügte das Team allen Verarbeitungsstufen Dead Letter Queues hinzu. Fehlgeschlagene Nachrichten wurden mit vollständigem Fehlerkontext an DLQs weitergeleitet. Ein einfaches Web-Dashboard erlaubte Operatoren, Dead-Letter-Nachrichten zu prüfen, zu filtern und wiederzugeben. Als drei Monate später ein ähnliches Zahlungsgateway-Problem auftrat, wurden alle 1.800 betroffenen Bestellungen automatisch in der DLQ erfasst und erfolgreich wiederverarbeitet, sobald sich das Gateway erholte.
