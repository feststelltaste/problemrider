---
title: Containerisierung
description: Kapselung von Anwendungen und ihren Abhängigkeiten in Containern.
category:
- Operations
- Architecture
problems:
- deployment-environment-inconsistencies
- configuration-drift
- dependency-version-conflicts
- complex-deployment-process
- poor-system-environment
- technology-stack-fragmentation
- deployment-risk
- development-disruption
- flaky-tests
- tool-limitations
- environment-variable-issues
- testing-complexity
- inadequate-configuration-management
- legacy-configuration-management-chaos
- testing-environment-fragility
layout: solution
lang: de
en_slug: containerization
related_solutions:
- slug: containerized-databases
  similarity: 0.85
- slug: emulation
  similarity: 0.8
- slug: cloud-native-development
  similarity: 0.75
- slug: virtual-development-environments
  similarity: 0.75
- slug: virtual-networks
  similarity: 0.75
- slug: dependency-injection
  similarity: 0.75
---

## Description

Containerisierung packt eine Anwendung zusammen mit ihrer exakten Laufzeit, Bibliotheken und Konfiguration in ein einzelnes, portables Image, das unabhängig vom zugrunde liegenden Host identisch läuft, was umgebungsspezifische Installationsskripte und manuelle Einrichtungsprozeduren durch eine deklarative Build-Definition ersetzt. Legacy-Anwendungen sind oft eng an eine spezifische Betriebssystemversion, eine bestimmte Menge installierter Bibliotheken oder manuelle Konfigurationsschritte gekoppelt, die einmal, vor Jahren, durchgeführt und nie vollständig dokumentiert wurden — eine Kopplung, die Routineereignisse wie eine Server-Hardware-Erneuerung oder ein OS-End-of-Life in existenzielles Risiko für die Anwendung verwandelt. Indem der gesamte Laufzeitabhängigkeitsbaum im Image erfasst wird, entkoppelt Containerisierung die Legacy-Anwendung vom Host, auf dem sie zufällig läuft, sodass sich dasselbe Image vom Laptop eines Entwicklers zum Staging zur Produktion und schließlich auf moderne Orchestrierungsinfrastruktur bewegen kann, ohne die „funktioniert auf meiner Maschine"-Diskrepanzen, die aus Umgebungs-Drift entstehen. Dieselbe Portabilität macht Containerisierung zu einem praktischen Ermöglicher schrittweiser Modernisierungsstrategien wie dem Strangler-Fig-Muster, da eine containerisierte Legacy-Anwendung neben neu gebauten Services im selben Cluster laufen kann, während Funktionalität Stück für Stück migriert. Der Tradeoff ist zusätzliche operative Fläche — Orchestrierung, Netzwerk und persistenter Speicher müssen alle verwaltet werden —, und Legacy-Anwendungen mit tiefen OS- oder Hardware-Ebenen-Abhängigkeiten können sich sauberer Containerisierung widersetzen.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Packen Sie jede Legacy-Anwendung mit ihrer exakten Laufzeit, Bibliotheken und Konfiguration in ein Container-Image
- Nutzen Sie Multi-Stage-Builds, um Container-Images klein zu halten, während alle Build-Zeit-Abhängigkeiten einbezogen werden
- Ersetzen Sie umgebungsspezifische Installationsskripte durch deklarative Dockerfiles
- Führen Sie dasselbe Container-Image über Entwicklung, Staging und Produktion aus, um Umgebungs-Drift zu eliminieren
- Führen Sie Container-Orchestrierung (z. B. Kubernetes) schrittweise ein, beginnend mit zustandslosen Services
- Nutzen Sie Container, um Legacy-Anwendungen während der Migration neben modernen Services laufen zu lassen

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Eliminiert „funktioniert auf meiner Maschine"-Probleme durch Verpackung der vollständigen Laufzeitumgebung
- Ermöglicht Legacy-Anwendungen, auf moderner Infrastruktur zu laufen, ohne neu geschrieben zu werden
- Vereinfacht Abhängigkeitsmanagement durch Isolation des Abhängigkeitsbaums jeder Anwendung
- Erleichtert schrittweise Modernisierung, indem alten und neuen Services Koexistenz erlaubt wird

**Kosten und Risiken:**
- Die Containerisierung von Legacy-Anwendungen mit spezifischen OS- oder Hardware-Abhängigkeiten kann herausfordernd sein
- Fügt operative Komplexität durch Container-Orchestrierung, Netzwerk und Speicherverwaltung hinzu
- Zustandsbehaftete Legacy-Anwendungen erfordern sorgfältige Handhabung persistenten Speichers in Containern
- Teams brauchen neue Fähigkeiten in Container-Tooling und Orchestrierungsplattformen

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Fertigungsunternehmen betrieb ein Legacy-Bestandssystem auf einer spezifischen Red-Hat-Version mit fixierten Bibliotheksversionen. Eine Server-Hardware-Erneuerung drohte, die Anwendung zu brechen. Durch die Containerisierung der Anwendung mit ihrem exakten Abhängigkeitsbaum entkoppelte das Team sie vom Host-OS, was Deployment auf moderner Infrastruktur ermöglichte. Die containerisierte Anwendung wurde außerdem zur Grundlage für eine Strangler-Fig-Migration, mit neuen Microservices, die neben dem Legacy-Container im selben Kubernetes-Cluster deployt wurden.
