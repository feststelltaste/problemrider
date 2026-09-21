---
title: Cloud-Native-Entwicklung
description: Entwicklung und Optimierung von Anwendungen speziell für Cloud-Umgebungen.
category:
- Architecture
- Operations
problems:
- scaling-inefficiencies
- monolithic-architecture-constraints
- technology-lock-in
- complex-deployment-process
- operational-overhead
- poor-system-environment
layout: solution
lang: de
en_slug: cloud-native-development
related_solutions:
- slug: containerization
  similarity: 0.75
- slug: serverless-computing
  similarity: 0.75
- slug: elastic-resource-utilization
  similarity: 0.75
- slug: containerized-databases
  similarity: 0.75
- slug: virtual-development-environments
  similarity: 0.7
- slug: horizontal-scaling
  similarity: 0.7
---

## Description

Cloud-Native-Entwicklung ist ein Ansatz zum Bau und Betrieb von Anwendungen, der um die Eigenschaften herum designt ist, die Cloud-Plattformen tatsächlich bieten — elastische Skalierung, verwaltete Infrastruktur, wegwerfbare und zustandslose Instanzen —, statt die Cloud einfach als anderen Ort zum Betrieb derselben Architektur zu behandeln, die für feste, dedizierte Server designt wurde. Es bedeutet typischerweise, Zustand, Konfiguration und Dateispeicher aus dem Anwendungsprozess selbst zu externalisieren, Muster wie die Twelve-Factor-App-Prinzipien zu übernehmen und sich auf verwaltete Services für Datenbanken, Warteschlangen und Caches zu verlassen, statt diese Infrastruktur von Hand zu betreiben. Für Legacy-Systeme, von denen viele um Annahmen wie persistenten lokalen Speicher, statische IP-Adressen oder langlebige Serverinstanzen herum architektiert wurden, die einmal für Spitzenlast bereitgestellt werden und dann meist untätig sind, erfordert Cloud-Native-Entwicklung, diese Annahmen bewusst zurückzudrehen, statt einfach die bestehende Binärdatei auf Cloud-Compute erneut zu deployen. Dies schrittweise zu tun, beispielsweise durch Strangler-Fig-Migrationen, die Komponenten eine nach der anderen vom Legacy-Monolithen abschälen, ist für diese Systeme generell sicherer als eine vollständige Neuschreibung zu versuchen. Der Gewinn ist, dass Infrastruktur nun skalieren kann, um tatsächlicher Nachfrage zu entsprechen, statt für einen Worst Case dimensioniert zu sein, der selten eintritt, und operative Last verschiebt sich von manuellem Kapazitätsmanagement zu automatisierten, verwalteten Services. Diese Verschiebung ist nicht risikofrei, da die resultierende Cloud-Native-Architektur typischerweise schwieriger nachzuvollziehen und zu debuggen ist als das monolithische Deployment, das sie ersetzt, und die Abhängigkeit von den verwalteten Services eines spezifischen Cloud-Anbieters kann eine neue Form desselben Vendor-Lock-ins wieder einführen, dem das Legacy-System zu entkommen versuchte.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Bewerten Sie, welche Legacy-Komponenten am meisten von Cloud-Native-Mustern profitieren (zustandslose Services, verwaltete Datenbanken, Auto-Scaling)
- Externalisieren Sie Konfiguration, Sitzungszustand und Dateispeicher von der Anwendung zu Cloud-verwalteten Services
- Übernehmen Sie schrittweise Twelve-Factor-App-Prinzipien: umgebungsbasierte Konfiguration, zustandslose Prozesse, wegwerfbare Instanzen
- Nutzen Sie verwaltete Services (Datenbanken, Nachrichtenwarteschlangen, Caches), um operative Last zu verringern
- Implementieren Sie Infrastructure as Code (Terraform, CloudFormation), um Umgebungen reproduzierbar zu machen
- Designen Sie für Fehler: Implementieren Sie Wiederholungen, Circuit Breaker und Health Checks unter der Annahme, dass Komponenten ausfallen werden
- Migrieren Sie schrittweise unter Nutzung von Strangler-Fig- oder Sidecar-Mustern, statt eine Big-Bang-Neuschreibung zu versuchen

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Ermöglicht elastische Skalierung, die der Nachfrage entspricht, ohne Überbereitstellung
- Verringert operative Last durch verwaltete Services und automatisierte Infrastruktur
- Verbessert Deployment-Geschwindigkeit und -Häufigkeit durch Cloud-Native-CI/CD-Pipelines
- Bietet eingebaute Hochverfügbarkeits- und Disaster-Recovery-Fähigkeiten

**Kosten und Risiken:**
- Cloud-Anbieter-Lock-in kann den Legacy-Technologie-Lock-in ersetzen, den es zu lösen versuchte
- Cloud-Native-Architekturen sind komplexer zu debuggen und zu überwachen als monolithische Deployments
- Kostenmanagement in der Cloud erfordert ständige Aufmerksamkeit, um unerwartete Rechnungen zu vermeiden
- Legacy-Anwendungen mit Annahmen über lokale Dateisysteme, statische IPs oder persistente Instanzen erfordern erhebliche Refaktorierung
- Kompetenzlücke im Team zwischen traditionellem Infrastrukturmanagement und Cloud-Native-Betrieb

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Das Legacy-Content-Management-System eines Medienunternehmens lief auf dedizierten Servern, die für Spitzenlast bereitgestellt waren, aber 80 Prozent der Zeit untätig waren. Das Team containerisierte die Anwendung, verschob den Sitzungszustand zu Redis und deployte auf Kubernetes mit Auto-Scaling-Richtlinien. Dateispeicher migrierte von lokaler Festplatte zu Cloud-Objektspeicher. Das System skalierte nun von 2 auf 20 Instanzen während Traffic-Spitzen durch viralen Content und skalierte während ruhiger Perioden zurück. Die Infrastrukturkosten sanken um 45 Prozent, trotz Handhabung höheren Spitzentraffics, und Deployments wechselten von monatlichen Wartungsfenstern zu mehrmals täglich.
