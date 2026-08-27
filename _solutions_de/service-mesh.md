---
title: Service Mesh
description: Verwaltung von Traffic auf Infrastrukturebene mit transparenter
  Protokollübersetzung, mTLS und Routing.
category:
- Architecture
- Operations
problems:
- microservice-communication-overhead
- service-discovery-failures
- service-timeouts
- network-latency
- insecure-data-transmission
- monitoring-gaps
- cascade-failures
layout: solution
lang: de
en_slug: service-mesh
related_solutions:
- slug: containerization
  similarity: 0.75
- slug: strangler-fig-pattern
  similarity: 0.7
- slug: microservices-architecture
  similarity: 0.7
- slug: microservices
  similarity: 0.7
- slug: api-gateway
  similarity: 0.7
- slug: chaos-engineering
  similarity: 0.7
---

## Description

Ein Service Mesh ist eine Infrastrukturschicht, typischerweise implementiert als eine Reihe von Sidecar-Proxys, die neben jeder Service-Instanz bereitgestellt werden und den gesamten Netzwerkverkehr zwischen Diensten abfangen und verwalten, ohne Änderungen am Anwendungscode zu erfordern. Es zentralisiert Querschnitts-Kommunikationsbelange — mutual-TLS-Verschlüsselung, Retries, Timeouts, Circuit Breaking, Load Balancing und verteiltes Tracing —, die andernfalls redundant, und oft inkonsistent, innerhalb jedes Dienstes implementiert werden müssten. Diese Externalisierung ist besonders wertvoll für Legacy-Systeme, wo Inter-Service-Kommunikation häufig moderne Sicherheits- und Resilienzpraktiken vordatiert: Verbindungen könnten unverschlüsselt sein, Timeout- und Retry-Verhalten könnte hartcodiert oder gänzlich abwesend sein, und es gibt oft keine Sichtbarkeit darüber, wie Legacy-Komponenten tatsächlich miteinander sprechen, bis das Tracing eines Meshs den echten Abhängigkeitsgraphen offenlegt. Da das Mesh auf der Netzwerkebene statt innerhalb des Anwendungscodes operiert, kann es inkrementell um bestehende Legacy-Dienste herum eingeführt werden, wobei Protokollübersetzung, Traffic-Shaping und Sicherheitskontrollen als Wrapper statt als Neuschreibung hinzugefügt werden. Dieselbe Traffic-Shaping-Fähigkeit macht das Mesh zu einem praktischen Mechanismus für schrittweise Migration, da ein Prozentsatz des Traffics zu einem modernisierten Ersatzdienst geroutet werden kann, während der Rest weiterhin zur Legacy-Implementierung fließt, was es erlaubt, Verhalten unter echter Last zu validieren, bevor eine vollständige Umschaltung erfolgt.

## How to Apply ◆

- Setzen Sie ein Service Mesh (z. B. Istio, Linkerd) als Sidecar-Proxy-Schicht neben bestehenden Legacy-Diensten ein, um Traffic-Management zu erhalten, ohne Anwendungscode zu modifizieren.
- Aktivieren Sie mTLS zwischen Diensten, um Kommunikationskanäle zu sichern, die Legacy-Systeme möglicherweise unverschlüsselt gelassen haben.
- Nutzen Sie die Traffic-Routing-Fähigkeiten des Meshs, um Canary-Deployments und schrittweise Migration von Legacy- zu modernisierten Diensten zu implementieren.
- Konfigurieren Sie Retry-Richtlinien, Circuit Breaker und Timeouts auf Infrastrukturebene, um die Resilienz von Legacy-Service-Interaktionen zu verbessern.
- Nutzen Sie eingebaute Observability (verteiltes Tracing, Metriken), um Sichtbarkeit in Legacy-Service-Kommunikationsmuster zu gewinnen.
- Nutzen Sie Protokollübersetzungsfunktionen, um Legacy-Protokolle mit modernen zu verbinden, ohne Service-Code neu zu schreiben.

## Tradeoffs ⇄

**Vorteile:**
- Fügt Legacy-Diensten Sicherheit, Observability und Resilienz hinzu, ohne Codeänderungen zu erfordern.
- Ermöglicht schrittweise Traffic-Verlagerung während der Migration von Legacy- zu modernen Diensten.
- Bietet konsistente Traffic-Richtlinien über heterogene Legacy- und moderne Komponenten hinweg.
- Zentralisiert Querschnittsbelange wie Retries, Timeouts und Authentifizierung.

**Kosten:**
- Führt erhebliche Infrastrukturkomplexität und betrieblichen Overhead ein.
- Sidecar-Proxys fügen jedem Service-Aufruf Latenz und Ressourcenverbrauch hinzu.
- Debugging wird schwieriger, weil Anfragen durch zusätzliche Proxy-Schichten laufen.
- Erfordert Container-Orchestrierung (typischerweise Kubernetes), die Legacy-Umgebungen möglicherweise nicht haben.
- Steile Lernkurve für Betriebsteams, die mit Mesh-Konzepten nicht vertraut sind.

## How It Could Be

Eine E-Commerce-Plattform betreibt eine Mischung aus Legacy-Java-Diensten und neueren Microservices. Die Inter-Service-Kommunikation ist unzuverlässig, mit häufigen Timeouts und ohne Verschlüsselung. Das Team setzt Linkerd als Service Mesh ein, beginnend mit den kritischsten Kommunikationspfaden. Das Mesh bietet automatisch mTLS, Retries mit Backoff und detaillierte Latenzmetriken. Während einer nachfolgenden Migrationsphase nutzen sie Traffic Splitting, um 10 % der Anfragen zu einem neu geschriebenen Dienst zu routen, während 90 % noch zur Legacy-Version gehen, was sichere Validierung vor der vollständigen Umschaltung ermöglicht. Die Observability-Daten des Meshs offenbaren auch zuvor unbekannte Abhängigkeitsketten zwischen Legacy-Diensten.
