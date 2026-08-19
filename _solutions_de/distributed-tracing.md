---
title: Distributed Tracing
description: Verfolgung von Anfragen über Microservice-Grenzen hinweg mitsamt ihrer
  Performance-Auswirkung.
category:
- Operations
- Performance
problems:
- debugging-difficulties
- slow-incident-resolution
- monitoring-gaps
- microservice-communication-overhead
- cascade-failures
- slow-application-performance
layout: solution
lang: de
en_slug: distributed-tracing
related_solutions:
- slug: monitoring
  similarity: 0.75
- slug: continuous-performance-monitoring
  similarity: 0.75
- slug: observability-and-monitoring
  similarity: 0.75
- slug: logging
  similarity: 0.75
- slug: performance-measurements
  similarity: 0.75
- slug: service-level-indicators
  similarity: 0.75
---

## Description

Distributed Tracing heftet einen eindeutigen Trace-Identifikator an eine Anfrage in dem Moment, in dem sie ins System eintritt, und propagiert diesen Identifikator durch jeden nachgelagerten Service-Aufruf, jede Datenbankabfrage und jede Message-Queue-Interaktion, die die Anfrage berührt, wobei jeder Schritt als zeitgestempelte Spanne aufgezeichnet wird, die später zu einem einzigen Ende-zu-Ende-Bild dessen zusammengesetzt werden kann, was geschah und wie lange jeder Teil dauerte. Dies adressiert direkt einen blinden Fleck, der entsteht, wenn ein Legacy-Monolith in Microservices zerlegt wird: Die Logs jedes einzelnen Services können isoliert vollkommen normale Antwortzeiten zeigen, während die tatsächliche nutzerseitige Latenz oder der Fehler durch eine Interaktion zwischen mehreren Services verursacht wird, die kein einzelnes Service-Log offenbaren kann. Legacy-Modernisierungsbemühungen, die schrittweise Services aus einem Monolithen herausschneiden, sind besonders anfällig für dieses Problem, weil das resultierende System verteilte Komplexität hat, ohne noch passende verteilte Observability zu haben, was Teams unfähig lässt, mit Sicherheit zu sagen, welcher von mehreren Services für eine gegebene Verlangsamung verantwortlich ist. Legacy-Services für Tracing zu instrumentieren muss typischerweise schrittweise erfolgen, beginnend mit den häufigsten oder am häufigsten in Vorfälle verwickelten Anfragepfaden, da das Nachrüsten von Tracing über ein gesamtes System mit gemischten Technologien auf einmal selten praktikabel ist. Einmal etabliert, verwandeln Trace-Daten „das System fühlt sich irgendwo langsam an" in eine präzise, visualisierte Antwort darauf, genau welcher Service und welche Operation im kritischen Pfad verantwortlich ist, was die Untersuchungszeit im Vergleich zum manuellen Korrelieren separater Logdateien erheblich verkürzt.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Instrumentieren Sie Services mit einer Tracing-Bibliothek (OpenTelemetry, Jaeger, Zipkin), die Trace-Kontext über Servicegrenzen hinweg propagiert
- Injizieren Sie Trace-IDs am Systemeintrittspunkt und propagieren Sie sie durch alle nachgelagerten Aufrufe mittels Headern
- Erfassen Sie Spannen für bedeutende Operationen: HTTP-Aufrufe, Datenbankabfragen, Message-Queue-Interaktionen und Cache-Abfragen
- Setzen Sie ein Trace-Erfassungs- und Visualisierungs-Backend ein, um Trace-Daten zu speichern und abzufragen
- Fügen Sie Tracing schrittweise zu Legacy-Services hinzu, beginnend mit den Services, die an den häufigsten oder problematischsten Anfragepfaden beteiligt sind
- Nutzen Sie Trace-Daten, um Latenz-Engpässe zu identifizieren und den kritischen Pfad zu optimieren
- Setzen Sie Sampling-Raten angemessen, um Observability mit Speicher- und Performance-Kosten auszubalancieren

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Bietet Ende-zu-Ende-Sichtbarkeit in den Anfragefluss über Servicegrenzen hinweg
- Identifiziert präzise, welcher Service oder welche Operation für Latenz in verteilten Systemen verantwortlich ist
- Ermöglicht die Identifikation kaskadierender Fehlermuster und Abhängigkeitsengpässe
- Reduziert die mittlere Lösungszeit für Probleme verteilter Systeme erheblich

**Kosten und Risiken:**
- Instrumentierung fügt jeder verfolgten Operation kleinen Latenz- und Ressourcenoverhead hinzu
- Trace-Speicher kann schnell wachsen und bei hohem Traffic-Volumen teuer werden
- Unvollständige Instrumentierung (fehlende Spannen in manchen Services) erzeugt irreführende Traces
- Erfordert Teamschulung, um Trace-Daten effektiv zu interpretieren

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Unternehmen hatte einen Legacy-Monolithen teilweise in acht Microservices zerlegt. Als Nutzer intermittierend langsame Antworten meldeten, konnte das Team nicht bestimmen, welcher Service verantwortlich war, weil die Logs jedes Services isoliert normale Antwortzeiten zeigten. Nach dem Deployment von OpenTelemetry über alle Services zeigten Traces, dass ein bestimmter Anfragepfad sechs Services sequenziell durchquerte und der dritte Service in der Kette einen synchronen Datenbankaufruf machte, der gelegentlich wegen Lock Contention 5 Sekunden dauerte. Die Trace-Visualisierung machte den Engpass sofort offensichtlich, und das Team löste das Problem, indem es die Datenbankabfrage optimierte und einen Circuit Breaker hinzufügte.
