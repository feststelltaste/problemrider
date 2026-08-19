---
title: Nonstop Forwarding
description: Kontinuierliche Anfragenweiterleitung trotz Fehlern oder Ausfällen.
category:
- Architecture
problems:
- cascade-failures
- system-outages
- service-timeouts
- single-points-of-failure
- unpredictable-system-behavior
layout: solution
lang: de
en_slug: nonstop-forwarding
related_solutions:
- slug: failover-mechanisms
  similarity: 0.75
- slug: retry
  similarity: 0.75
- slug: failover-cluster
  similarity: 0.75
- slug: resilience
  similarity: 0.75
- slug: circuit-breaker
  similarity: 0.7
- slug: rate-limiting
  similarity: 0.7
---

## Description

Nonstop Forwarding ist ein architektonisches Muster, das die Control Plane, die Routing- und Konfigurationsentscheidungen berechnet, von der Data Plane trennt, die einzelne Anfragen mittels des zuletzt berechneten Zustands weiterleitet. Weil die Data Plane weiterhin mit zwischengespeicherter Routing-Information arbeitet, selbst während die Control Plane neu startet oder ausfällt, fließt der Verkehr weiter durch Ausfälle, die sonst jede laufende Anfrage unterbrechen würden. Diese Entkopplung zählt in Legacy-Systemen, in denen Routing-, Gateway- oder Orchestrierungskomponenten enge Kopplung zwischen Konfigurationslogik und Anfragebehandlung angesammelt haben, was jeden Control-Plane-Neustart zu einem sichtbaren Ausfall macht. Das Muster anzuwenden bedeutet typischerweise, eine persistente, unabhängig adressierbare Weiterleitungsschicht einzuführen, die Prozessneustarts überleben kann, was selbst eine nützliche Zwangsfunktion ist, um Legacy-Netzwerk- und Servicekomponenten zu entwirren, die nie mit dieser Trennung im Blick entworfen wurden.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Trennen Sie die Control Plane (Routing-Entscheidungen) von der Data Plane (Anfragenweiterleitung) in Legacy-Netzwerk- und Servicearchitekturen
- Konfigurieren Sie Weiterleitungskomponenten, damit sie Verkehr während Control-Plane-Ausfällen weiterhin mittels zuletzt bekannter guter Routing-Tabellen leiten
- Implementieren Sie Graceful-Restart-Fähigkeiten, damit Komponenten ihre Steuerlogik neu starten können, ohne den Datenfluss zu unterbrechen
- Nutzen Sie persistenten Weiterleitungszustand, der Prozessneustarts oder Failover-Ereignisse überlebt
- Testen Sie Nonstop-Forwarding-Szenarien, indem Sie Control-Plane-Ausfälle simulieren und die Auswirkung auf die Data Plane messen
- Wenden Sie dieses Muster auf Service-Mesh- oder API-Gateway-Ebenen an, um Legacy-Backend-Dienste zu schützen

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Bewahrt die Kontinuität des Anfrageflusses während Störungen der Verwaltungs- oder Control-Plane
- Verringert den Explosionsradius von Ausfällen in Routing- und Orchestrierungskomponenten
- Ermöglicht Zero-Downtime-Upgrades der Routing-Infrastruktur
- Verhindert kaskadierende Timeouts, wenn Steuerkomponenten neu starten

**Kosten und Risiken:**
- Veraltete Routing-Information während längerer Control-Plane-Ausfälle kann Verkehr zu entfernten oder ungesunden Knoten leiten
- Erhöht die Komplexität der Weiterleitungsschicht
- Das Debuggen von Routing-Problemen wird schwerer, wenn die Weiterleitung unabhängig von der Steuerung arbeitet
- Nicht alle Legacy-Architekturen können Control und Data Plane sauber trennen

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Das Legacy-API-Gateway eines Telekommunikationsunternehmens erforderte periodische Neustarts für Konfigurationsaktualisierungen, was kurze Verkehrsunterbrechungen verursachte, die Timeout-Fehler in nachgelagerten Systemen auslösten. Durch die Neugestaltung des Gateways, um seine Routing-Konfigurationsverwaltung von seiner Anfragen-Weiterleitungs-Engine zu trennen, konnten Aktualisierungen durch Graceful Restarts angewendet werden, bei denen die Weiterleitungs-Engine weiterhin Anfragen mittels zwischengespeicherter Routing-Regeln verarbeitete, während die Control Plane neu lud. Dies beseitigte die 10-15-sekündigen Verkehrsunterbrechungen, die kaskadierende Ausfälle in latenzsensitiven Legacy-Backend-Diensten verursacht hatten.
