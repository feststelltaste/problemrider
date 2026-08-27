---
title: Rate Limiting
description: Steuerung eingehender Anfrageraten gegen Systemüberlastung bei
  Traffic-Spitzen.
category:
- Architecture
- Performance
problems:
- rate-limiting-issues
- capacity-mismatch
- system-outages
- cascade-failures
- slow-application-performance
- high-api-latency
- graphql-complexity-issues
- unbounded-data-structures
- work-queue-buildup
- task-queues-backing-up
layout: solution
lang: de
en_slug: rate-limiting
related_solutions:
- slug: load-shedding
  similarity: 0.8
- slug: retry
  similarity: 0.8
- slug: distributed-caching
  similarity: 0.8
- slug: load-balancing
  similarity: 0.8
- slug: connection-pooling
  similarity: 0.75
- slug: circuit-breaker
  similarity: 0.75
---

## Description

Rate Limiting begrenzt die Anzahl der Anfragen, die ein Client, API-Schlüssel oder Endpunkt innerhalb eines gegebenen Zeitfensters stellen darf, typischerweise an einem API-Gateway oder Reverse Proxy mittels Algorithmen wie Token Bucket oder Sliding Window durchgesetzt, wobei Anfragen jenseits des Limits mit einer informativen 429-Antwort abgelehnt werden, statt das Backend überwältigen zu lassen. Dies ist besonders relevant für Legacy-Systeme, weil sie häufig für eine feste, begrenzte Menge von Konsumenten und ein Lastprofil entworfen und dimensioniert wurden, das seither weit über die ursprünglichen Annahmen hinausgewachsen ist, ohne architektonischen Spielraum, um einen unerwarteten Anstieg von einem einzelnen fehlverhaltenden Client oder einer Integration zu absorbieren. Eine einzelne schlecht implementierte nachgelagerte Integration, die einen Legacy-Endpunkt hämmert, kann eine gemeinsame Ressource erschöpfen — einen Datenbank-Connection-Pool zum Beispiel — und die Erfahrung für jeden anderen Konsumenten desselben Legacy-Backends verschlechtern, ein Fehlermodus, den Rate Limiting von einem unkontrollierten, systemweiten Ausfall in eine vorhersagbare, isolierte Ablehnung allein des verursachenden Traffics verwandelt. Am Gateway platziert, schützt Rate Limiting das Legacy-System, ohne jegliche Änderung am Legacy-Code selbst zu erfordern, was zählt, weil dieser Code häufig der Teil des Systems ist, der am wenigsten sicher oder gut genug verstanden ist, um direkt modifiziert zu werden. Der Zielkonflikt ist, dass die Festlegung effektiver Limits ein genaues Verständnis des tatsächlichen nachhaltigen Durchsatzes des Legacy-Systems erfordert, und falsch gesetzte Limits schützen entweder das Backend nicht oder drosseln unnötig legitime Hochvolumen-Nutzung während echter Geschäftshöhepunkte.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Identifizieren Sie den maximal nachhaltigen Durchsatz für Legacy-System-Endpunkte durch Lasttests
- Implementieren Sie Rate Limits an der API-Gateway- oder Reverse-Proxy-Schicht, um Legacy-Backends zu schützen
- Verwenden Sie Token-Bucket- oder Sliding-Window-Algorithmen für glatte Ratendurchsetzung
- Konfigurieren Sie unterschiedliche Rate Limits pro Client, API-Schlüssel oder Endpunkt basierend auf Geschäftspriorität
- Geben Sie informative 429-Antworten (Too Many Requests) mit Retry-After-Headern zurück
- Implementieren Sie Rate Limiting für interne Service-zu-Service-Aufrufe, um Noisy-Neighbor-Probleme zu verhindern
- Überwachen Sie Rate-Limit-Treffer, um zwischen Missbrauch und legitimer Nachfrage zu unterscheiden, die Kapazitätserweiterung braucht

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Schützt Legacy-Systeme vor Traffic-Spitzen, die ihre Kapazität überschreiten
- Verhindert, dass ein einzelner Client oder eine Integration Systemressourcen monopolisiert
- Bietet eine vorhersagbare, kontrollierte Reaktion auf Überlastung statt unvorhersehbarer Fehler
- Ermöglicht faire Ressourcenteilung über mehrere Konsumenten von Legacy-Diensten hinweg

**Kosten und Risiken:**
- Legitime Hochvolumen-Nutzer könnten während Geschäftsspitzen gedrosselt werden
- Die Konfiguration von Rate Limits erfordert Verständnis der tatsächlichen Systemkapazität
- Falsch gesetzte Limits können entweder das System nicht schützen oder unnötig gültigen Traffic ablehnen
- Rate Limiting am Rand schützt nicht vor internen Verstärkungsmustern

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Legacy-ERP-System bot APIs, die von mehreren internen Anwendungen und Drittanbieter-Integrationen konsumiert wurden. Eine schlecht implementierte Integration eines Partners hämmerte wiederholt den Bestellabfrage-Endpunkt mit Tausenden von Anfragen pro Minute, was Erschöpfung des Datenbank-Connection-Pools verursachte, die alle Nutzer betraf. Durch die Bereitstellung von Rate Limiting am API-Gateway mit Kontingenten pro Client schützte das Team das Legacy-Backend vor Überlastung durch einzelne Konsumenten. Der Partner erhielt klare Rate-Limit-Dokumentation und passte seine Integration an, um Batch-Abfragen zu nutzen, was sein Anfragevolumen um 95 % reduzierte, während dieselben Daten abgerufen wurden.
