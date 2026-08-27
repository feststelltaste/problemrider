---
title: Standardisierte Protokolle
description: Wahl von Transport- und Messaging-Protokollen mit breiter
  Ökosystem-Unterstützung.
category:
- Architecture
- Dependencies
problems:
- poor-interfaces-between-applications
- technology-lock-in
- vendor-lock-in
- integration-difficulties
- obsolete-technologies
- microservice-communication-overhead
layout: solution
lang: de
en_slug: standardized-protocols
related_solutions:
- slug: standardized-interfaces
  similarity: 0.75
- slug: protocol-abstraction
  similarity: 0.75
- slug: standardized-data-formats
  similarity: 0.75
- slug: platform-independent-data-storage
  similarity: 0.7
- slug: schema-registry
  similarity: 0.7
- slug: service-mesh
  similarity: 0.7
---

## Description

Standardisierte Protokolle bedeutet die Wahl von Transport- und Messaging-Protokollen mit breiter, herstellerübergreifender Ökosystem-Unterstützung — HTTP/2, AMQP, MQTT, gRPC — anstelle proprietärer Protokolle, die nur das Tooling oder die Middleware eines einzigen Anbieters sprechen kann. Legacy-Systeme, besonders in industriellen oder Telekommunikationskontexten, kommunizieren oft mit einem Protokoll, das von Anfang an proprietär war oder das mit der Zeit zu einem De-facto-Lock-in-Mechanismus wurde, was bedeutet, dass die Fähigkeit der Organisation zu integrieren, zu überwachen oder auch nur das System am Laufen zu halten, vollständig vom fortgesetzten Wohlwollen, den Lizenzbedingungen und dem Support-Lebenszyklus eines einzigen Anbieters abhängt. Diese Abhängigkeit wird in dem Moment akut, in dem dieser Anbieter seine Preisgestaltung ändert, die Middleware einstellt oder es einfach schwieriger wird, Entwickler dafür zu finden, an welchem Punkt die Organisation entdeckt, dass sie keine echte Alternative hat. Die Migration zu einem standardisierten Protokoll — typischerweise über eine Übergangsprotokollbrücke oder einen Adapter, der vor dem Legacy-System platziert wird, sodass der Übergang inkrementell statt als einzelne riskante Umschaltung geschehen kann — stellt die Fähigkeit wieder her, aus einem breiten, wettbewerbsfähigen Ökosystem von Werkzeugen, Bibliotheken und verfügbarem technischen Talent zu wählen. Die entsprechenden Kosten sind der Entwicklungs- und Testaufwand der Migration selbst, der betriebliche Overhead des Betriebs einer Brücke während der Übergangsperiode und die Möglichkeit, dass ein standardisiertes Protokoll ein spezialisiertes Feature vermissen lässt, das das proprietäre bot, was vor der Verpflichtung zur Änderung bewertet werden muss.

## How to Apply ◆

- Inventarisieren Sie alle Kommunikationsprotokolle, die aktuell über die Legacy-Landschaft hinweg genutzt werden, und identifizieren Sie proprietäre oder veraltete.
- Wählen Sie weit unterstützte Protokolle (HTTP/2, AMQP, MQTT, gRPC) basierend auf den erforderlichen Kommunikationsmustern (Request-Response, Event-Streaming, Pub-Sub).
- Führen Sie Protokollbrücken oder Adapter ein, um Legacy-Systemen, die proprietäre Protokolle nutzen, zu erlauben, während einer Übergangsperiode mit Systemen zu kommunizieren, die Standardprotokolle nutzen.
- Migrieren Sie Legacy-Integrationen von proprietären zu standardisierten Protokollen inkrementell, beginnend mit den Verbindungen mit dem höchsten Traffic oder den problematischsten.
- Stellen Sie sicher, dass gewählte Protokolle von den Zielplattformen und Sprachen unterstützt werden, die in der gesamten Organisation genutzt werden.

## Tradeoffs ⇄

**Vorteile:**
- Breite Ökosystem-Unterstützung bedeutet leicht verfügbare Bibliotheken, Werkzeuge und Entwicklerwissen.
- Reduziert Vendor-Lock-in durch Vermeidung proprietärer Kommunikationsmechanismen.
- Vereinfacht die Integration mit externen Partnern und Drittanbieterdiensten.
- Macht es einfacher, Entwickler zu finden, die die Technologie verstehen.

**Kosten:**
- Die Migration von proprietären Protokollen erfordert Entwicklungsaufwand und sorgfältiges Testen.
- Standardisierte Protokolle könnten spezialisierte Features vermissen lassen, die proprietäre Protokolle boten.
- Der Betrieb von Protokollbrücken während des Übergangs fügt betriebliche Komplexität hinzu.
- Manche Legacy-Systeme könnten moderne Protokolle nicht ohne erhebliche Modifikation unterstützen.

## How It Could Be

Die Legacy-SCADA-Systeme eines Fertigungsunternehmens kommunizieren mit einem proprietären Binärprotokoll, das nur die Middleware eines Anbieters handhaben kann. Als der Anbieter die Lizenzgebühren erheblich erhöht, entscheidet sich das Team, zu MQTT für Gerät-zu-Server-Kommunikation und AMQP für Inter-Service-Messaging zu migrieren. Sie setzen Protokolladapter an der Grenze von Legacy-Systemen ein, die nicht sofort modifiziert werden können. Neue Dienste werden von Anfang an mit den Standardprotokollen gebaut. Innerhalb eines Jahres ist die Anbieterabhängigkeit für die meisten Kommunikationspfade beseitigt, und das Team kann aus mehreren Open-Source-Werkzeugen für Monitoring und Verwaltung ihrer Messaging-Infrastruktur wählen.
