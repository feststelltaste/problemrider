---
title: Load Balancing
description: Verteilung der Arbeitslast auf mehrere Ressourcen.
category:
- Architecture
- Operations
problems:
- load-balancing-problems
- capacity-mismatch
- single-points-of-failure
- slow-application-performance
- scaling-inefficiencies
- system-outages
- high-api-latency
layout: solution
lang: de
en_slug: load-balancing
related_solutions:
- slug: distributed-caching
  similarity: 0.8
- slug: distributed-processing
  similarity: 0.8
- slug: rate-limiting
  similarity: 0.8
- slug: horizontal-scaling
  similarity: 0.75
- slug: data-replication
  similarity: 0.75
- slug: failover-cluster
  similarity: 0.75
---

## Description

Load Balancing verteilt eingehende Anfragen über mehrere Instanzen eines Dienstes oder einer Anwendung, mittels eines Algorithmus wie Round-Robin, Least-Connections oder gewichteter Verteilung, kombiniert mit Health Checks, die Verkehr von ausfallenden oder überlasteten Instanzen weg leiten. Mechanisch sitzt ein Load Balancer vor der Anwendungsschicht als einziger adressierbarer Einstiegspunkt und leitet jede Anfrage an die Backend-Instanz weiter, die zu diesem Zeitpunkt am besten positioniert ist, sie zu bearbeiten, was eine Flotte einzelner Server von außen wie einen einzigen zuverlässigen Dienst erscheinen lässt. Legacy-Anwendungen wurden häufig als einzelne Instanz gebaut und deployt, sowohl weil frühe Verkehrsvolumina nicht mehr erforderten als auch weil die Anwendung selbst mit In-Memory-Sitzungszustand geschrieben wurde, was das Laufen mehrerer Instanzen umständlich macht — sodass das System keinen Weg zu horizontaler Skalierung und keine Redundanz hat, was bedeutet, dass jede Verlangsamung oder jeder Absturz eines einzelnen Servers zu einem vollständigen Ausfall wird. Load Balancing vor ein solches System einzuführen bietet einen sofortigen Widerstandsfähigkeits- und Kapazitätsgewinn, selbst bevor die zugrundeliegende Anwendung refaktoriert wird, obwohl der volle Nutzen erst realisiert wird, sobald die Legacy-Anwendung zustandslos gemacht oder zu externalisierter Sitzungsspeicherung verschoben wird, da Sticky Sessions sonst als Übergangslösung benötigt werden, die einschränkt, wie gleichmäßig Last tatsächlich verteilt werden kann. Weil der Load Balancer selbst zu einem neuen kritischen Pfad wird, sobald er eingeführt wird, muss er redundant deployt werden, sonst verlagert er nur das Single-Point-of-Failure-Problem, das er lösen sollte.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Setzen Sie einen Load Balancer vor Legacy-Anwendungsinstanzen ein, um eingehende Anfragen zu verteilen
- Wählen Sie einen angemessenen Verteilungsalgorithmus (Round-Robin, Least Connections, gewichtet) basierend auf Workload-Merkmalen
- Konfigurieren Sie Health Checks, damit der Load Balancer Verkehr nur zu gesunden Instanzen leitet
- Refaktorieren Sie Legacy-Anwendungen, um zustandslos zu sein oder externe Sitzungsspeicher zu nutzen, um angemessene Lastverteilung zu ermöglichen
- Implementieren Sie Sticky Sessions als Übergangsmaßnahme für zustandsbehaftete Legacy-Anwendungen, die nicht sofort refaktoriert werden können
- Planen Sie Load-Balancer-Redundanz, um kein neues Single Point of Failure einzuführen
- Nutzen Sie Load-Balancing-Metriken, um Kapazitätsengpässe zu identifizieren und Skalierungsentscheidungen zu planen

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Verbessert die Systemverfügbarkeit, indem Last über mehrere Instanzen verteilt wird
- Ermöglicht horizontale Skalierung von Legacy-Anwendungsschichten
- Bietet einen natürlichen Integrationspunkt für Health Checking und Verkehrsmanagement
- Unterstützt Rolling Deployments und Canary Releases

**Kosten und Risiken:**
- Zustandsbehaftete Legacy-Anwendungen benötigen möglicherweise Session-Affinität, was die Verteilungseffektivität verringert
- Fügt Netzwerklatenz und einen potenziellen Ausfallpunkt hinzu, wenn nicht angemessen redundant
- Konfigurationskomplexität steigt mit SSL-Terminierung, Routing-Regeln und Rate Limiting
- Fehlkonfiguration des Load Balancers kann ungleichmäßige Verteilung oder abgebrochene Verbindungen verursachen

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Eine Online-Bildungsplattform betrieb ihr Legacy-Kursverwaltungssystem auf einem einzigen Server, der während Einschreibungsphasen regelmäßig nicht mehr reagierte. Durch das Deployment eines Load Balancers mit drei dahinterliegenden Anwendungsinstanzen verteilte das Team den Einschreibungsverkehr über alle Knoten. Sie nutzten externe Sitzungsspeicherung, um Sticky Sessions zu vermeiden, und konfigurierten Health Checks, die nicht reagierende Instanzen aus der Rotation entfernten. Spitzeneinschreibungsverkehr wurde reibungslos gehandhabt, und das Team konnte Rolling Deployments ohne jegliche Ausfallzeit durchführen.
