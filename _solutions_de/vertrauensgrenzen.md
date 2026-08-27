---
title: Vertrauensgrenzen
description: Definition von Grenzen zwischen Systemen und Komponenten mit
  unterschiedlichen Vertrauensstufen.
category:
- Security
- Architecture
problems:
- architectural-mismatch
- monolithic-architecture-constraints
- system-integration-blindness
- authentication-bypass-vulnerabilities
- authorization-flaws
- poor-interfaces-between-applications
layout: solution
lang: de
en_slug: trust-boundaries
related_solutions:
- slug: zero-trust-architecture
  similarity: 0.75
- slug: network-segmentation
  similarity: 0.75
- slug: security-architecture-analysis
  similarity: 0.7
- slug: security-by-design
  similarity: 0.7
- slug: threat-modeling
  similarity: 0.7
- slug: secure-by-default
  similarity: 0.7
---

## Description

Eine Vertrauensgrenze ist eine explizit definierte Linie in der Topologie eines Systems, über die hinweg sich Daten oder Anfragen zwischen Komponenten bewegen, die unterschiedliche Vertrauensstufen rechtfertigen — öffentlich zugänglich versus intern, Legacy versus modern, nutzergesteuert versus systemgesteuert —, wobei Validierung, Authentifizierung und Autorisierung an jedem Punkt durchgesetzt werden, an dem diese Linie überquert wird. Die Definition dieser Grenzen macht eine implizite Annahme explizit: Statt dass Komponenten einander vertrauen, nur weil sie zufällig im selben Netzwerk sitzen, wird Vertrauen bewusst und nur dort gewährt, wo es gerechtfertigt wurde. Dies ist besonders relevant für Legacy-Systeme, weil viele von ihnen ursprünglich als Single-Server- oder eng geclusterte Deployments konzipiert wurden, bei denen das gesamte interne Netzwerk implizit vertraut wurde, eine Annahme, die still aufhörte zu gelten, während das System wuchs, über mehr Hosts verteilt wurde oder mit neuen Integrationen verbunden wurde — während sich Code und Infrastruktur weiterhin so verhielten, als hätte sich nichts geändert. Die Nachrüstung expliziter Vertrauensgrenzen in ein solches System bedeutet, seine tatsächliche Komponententopologie zu kartieren, zu identifizieren, wo Vertrauensannahmen nicht mehr der Realität entsprechen, und Authentifizierung, Validierung und Netzwerksegmentierung an diesen Kreuzungspunkten einzuführen, sodass sich eine Kompromittierung auf einer Seite nicht frei auf die andere ausbreiten kann. Der Nutzen ist ein eingedämmter Blast-Radius: Ein Angreifer, der in eine Vertrauenszone eindringt, muss weiterhin zusätzliche Kontrollen überwinden, um die nächste zu erreichen, statt sich lateral durch ein Netzwerk zu bewegen, das nie darauf ausgelegt war, dieser Art von Bewegung zu widerstehen.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Kartieren Sie die Komponententopologie des Legacy-Systems und identifizieren Sie, wo unterschiedliche Vertrauensstufen existieren oder existieren sollten
- Definieren Sie explizite Vertrauensgrenzen zwischen internen und externen Komponenten, zwischen nutzerseitigen und Backend-Diensten, und zwischen Legacy- und modernen Systemen
- Implementieren Sie Validierung, Authentifizierung und Autorisierung an jeder Vertrauensgrenzen-Kreuzung
- Stellen Sie sicher, dass Daten, die Vertrauensgrenzen überqueren, unabhängig von ihrer Quelle validiert und bereinigt werden
- Nutzen Sie Netzwerksegmentierung, um Vertrauensgrenzen auf Infrastrukturebene durchzusetzen
- Dokumentieren Sie Vertrauensannahmen für jede Grenze, sodass sie überprüft werden können, während sich das System weiterentwickelt
- Wenden Sie das Prinzip der geringsten Rechte an Vertrauensgrenzen an und gewähren Sie nur den minimal erforderlichen Zugriff

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Dämmt den Blast-Radius von Sicherheitsverletzungen ein, indem laterale Bewegung über Grenzen hinweg verhindert wird
- Macht implizite Vertrauensannahmen explizit und überprüfbar
- Bietet klare Punkte für die Implementierung von Sicherheitskontrollen und Monitoring
- Ermöglicht unabhängige Sicherheitsbewertung und Härtung jeder Vertrauenszone

**Kosten und Risiken:**
- Legacy-Systeme entwickelten sich oft ohne Vertrauensgrenzen, was Nachrüstung komplex macht
- Das Hinzufügen von Authentifizierung und Validierung an internen Grenzen führt Latenz und Komplexität ein
- Übermäßige Segmentierung kann betrieblichen Overhead schaffen und legitime grenzüberschreitende Kommunikation erschweren
- Die konsistente Durchsetzung von Vertrauensgrenzen erfordert laufende Governance

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Eine Legacy-Enterprise-Anwendung war über 15 Jahre von einem Single-Server-Deployment zu einem verteilten System gewachsen, aber die gesamte interne Kommunikation nutzte weiterhin unauthentifizierte, unverschlüsselte Verbindungen, weil das ursprüngliche Design ein vertrauenswürdiges Netzwerk annahm. Nach einem Sicherheitsvorfall, bei dem ein Angreifer einen kompromittierten Webserver nutzte, um direkt auf die Datenbank zuzugreifen, definierte das Team drei Vertrauenszonen: öffentlich zugänglich, Anwendungsschicht und Datenschicht. Sie implementierten mutual TLS zwischen den Zonen, fügten Eingabevalidierung an jeder Grenze hinzu, und setzten Netzwerkrichtlinien ein, die grenzüberschreitende Kommunikation auf nur notwendige Pfade beschränkten. Die Kompartimentierung stellte sicher, dass eine nachfolgende Web-Anwendungsschwachstelle nicht genutzt werden konnte, um die Datenschicht zu erreichen.
