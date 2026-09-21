---
title: Reaktive Programmierung
description: Entwicklung von Anwendungen, die auf Ereignisse reagieren und
  Datenströme verarbeiten.
category:
- Architecture
- Performance
problems:
- slow-application-performance
- thread-pool-exhaustion
- scaling-inefficiencies
- high-connection-count
- imperative-data-fetching-logic
- cascade-failures
layout: solution
lang: de
en_slug: reactive-programming
related_solutions:
- slug: parallelization
  similarity: 0.8
- slug: connection-pooling
  similarity: 0.75
- slug: distributed-caching
  similarity: 0.75
- slug: retry
  similarity: 0.75
- slug: lazy-loading
  similarity: 0.75
- slug: distributed-processing
  similarity: 0.75
---

## Description

Reaktive Programmierung strukturiert I/O-gebundenen Code um nicht blockierende, asynchrone Datenströme herum — unter Nutzung von Bibliotheken wie RxJava, Project Reactor oder RxJS —, sodass eine auf eine langsame nachgelagerte Antwort wartende Anfrage nicht mehr für die Dauer dieses Wartens einen dedizierten Thread belegt, und Backpressure-Mechanismen verhindern, dass schnelle Produzenten langsamere Konsumenten überwältigen. In einer Legacy-Thread-pro-Anfrage-Architektur adressiert dies einen spezifischen und häufigen Fehlermodus: Ein Thread-Pool fester Größe wird unter Last erschöpft, weil die meisten seiner Threads einfach blockiert auf Antworten von nachgelagerten Diensten warten, sodass dem System die Kapazität für neue Anfragen ausgeht, obwohl seine tatsächliche CPU- und Netzwerknutzung niedrig bleibt. Legacy-Systeme neigen dazu, diese Verwundbarkeit schrittweise anzusammeln, während im Laufe der Zeit mehr nachgelagerte Abhängigkeiten hinzugefügt werden, jede einen weiteren Punkt hinzufügend, an dem ein Thread blockiert gehalten werden kann, bis eine Verlangsamung in irgendeinem einzelnen nachgelagerten Dienst ausreicht, um zu einem Ausfall für das gesamte System zu kaskadieren. Reaktive Programmierung schrittweise einzuführen — an Integrationsgrenzen statt als vollständige Neuschreibung — erlaubt einem Legacy-System, weit mehr gleichzeitige Last mit einem viel kleineren, festen Pool von Event-Loop-Threads zu absorbieren, da Threads nicht mehr wartend statt arbeitend gebunden sind. Die Kosten sind eine echt steile Lernkurve für Teams, die an sequenziellen, imperativen Code gewöhnt sind, materiell komplexeres Debugging und Stack Traces, und das Risiko, dass die Vermischung reaktiver und blockierender Codepfade — leicht versehentlich während einer schrittweisen Migration geschehend — genau das Thread-Hungern wieder einführt, das die Migration beheben sollte.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Identifizieren Sie I/O-gebundene Komponenten, bei denen Threads die meiste Zeit wartend verbringen (Datenbankaufrufe, HTTP-Anfragen, Datei-I/O)
- Führen Sie reaktive Bibliotheken (RxJava, Project Reactor, RxJS) schrittweise an Integrationsgrenzen ein, statt ganze Anwendungen neu zu schreiben
- Wandeln Sie blockierende API-Aufrufe in nicht blockierende reaktive Streams um, beginnend mit den ressourcenbeschränktesten Endpunkten
- Nutzen Sie Backpressure-Mechanismen, um zu verhindern, dass schnelle Produzenten langsame Konsumenten überwältigen
- Refaktorieren Sie callback-lastigen Legacy-Code in komponierbare reaktive Pipelines für bessere Lesbarkeit und Fehlerbehandlung
- Schulen Sie das Team in reaktiven Konzepten vor der Einführung, da der Paradigmenwechsel ein anderes mentales Modell erfordert

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Handhabt weit mehr gleichzeitige Verbindungen mit weniger Threads und verbessert die Ressourceneffizienz
- Bietet eingebaute Backpressure-Behandlung zur Steuerung von Datenflussraten
- Macht das System widerstandsfähiger gegen langsame nachgelagerte Dienste durch nicht blockierende I/O
- Ermöglicht ereignisgetriebene Architekturen, die natürlich mit der Last skalieren

**Kosten und Risiken:**
- Steile Lernkurve für an imperative, sequenzielle Programmierung gewöhnte Teams
- Stack Traces und Debugging werden mit reaktiven Pipelines erheblich komplexer
- Die Vermischung reaktiven und blockierenden Codes kann subtile Performance-Probleme und Thread-Pool-Hungern verursachen
- Das Testen reaktiven Codes erfordert spezialisierte Muster und Werkzeuge
- Nicht alle Legacy-Bibliotheken und -Frameworks unterstützen nicht blockierenden Betrieb, was den Einführungsumfang einschränkt

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Legacy-API-Gateway verarbeitete Anfragen mittels eines Thread-pro-Anfrage-Modells mit einem Pool von 200 Threads. Als der Traffic wuchs, wurde der Pool während Spitzenzeiten häufig erschöpft, weil die meisten Threads darauf warteten, blockiert, auf Antworten von nachgelagerten Microservices. Das Team schrieb die Anfrage-Routing-Schicht des Gateways mittels Project Reactor neu und ersetzte blockierende HTTP-Aufrufe durch nicht blockierende WebClient-Operationen. Derselbe Server handhabte nun das Zehnfache der gleichzeitigen Verbindungen mit 50 Event-Loop-Threads, und das Kaskadenausfall-Problem verschwand, weil langsame nachgelagerte Dienste keine Gateway-Threads mehr verbrauchten.
