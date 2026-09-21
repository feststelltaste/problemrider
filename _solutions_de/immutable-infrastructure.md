---
title: Immutable Infrastructure
description: Keine Änderung von Infrastrukturkomponenten, sondern Ersatz durch neue
  Versionen.
category:
- Operations
problems:
- configuration-drift
- deployment-environment-inconsistencies
- configuration-chaos
- deployment-risk
- complex-deployment-process
- frequent-hotfixes-and-rollbacks
- poor-system-environment
- environment-variable-issues
- inadequate-configuration-management
- legacy-configuration-management-chaos
- testing-environment-fragility
- customization-outside-version-control
layout: solution
lang: de
en_slug: immutable-infrastructure
related_solutions:
- slug: infrastructure-as-code
  similarity: 0.8
- slug: containerization
  similarity: 0.75
- slug: restore-points
  similarity: 0.75
- slug: rollback-mechanisms
  similarity: 0.7
- slug: ci-cd-pipeline
  similarity: 0.7
- slug: virtual-networks
  similarity: 0.7
---

## Description

Immutable Infrastructure ersetzt die Praxis, laufende Server vor Ort zu ändern, durch die Praxis, ein vollständiges, versioniertes Artefakt — ein Maschinenabbild, einen Container oder ein Deployment-Paket — zu bauen und es als vollständigen Ersatz für die vorherige Version zu deployen, wann immer eine Änderung nötig ist. Es werden keine manuellen Änderungen an laufenden Instanzen vorgenommen; jede Konfigurations- oder Codeänderung muss durch dieselbe Build-Pipeline fließen, die das ursprüngliche Artefakt produziert hat. Dies zielt direkt auf einen Fehlermodus, der in langlebigen Legacy-Umgebungen endemisch ist: Jahre ad hoc vorgenommener manueller Patches, punktueller Konfigurationsanpassungen und undokumentierter Fixes, die direkt auf Produktionsservern angewendet wurden, was zu Konfigurationsdrift führt, bei der keine zwei Server tatsächlich gleich konfiguriert sind, und zu Deployments, die auf einer Maschine funktionieren, aber auf einer anderen unvorhersehbar fehlschlagen. Die Übernahme von Immutable Infrastructure für eine Legacy-Anwendung erfordert im Allgemeinen zuerst, jeglichen Zustand, den die Anwendung derzeit auf der Instanz selbst hält — lokale Dateien, In-Memory-Sitzungen —, in Datenbanken oder Objektspeicher zu externalisieren, da eine so gebaute Instanz frei und vollständig ersetzbar sein muss, ohne etwas Wertvolles zu verlieren. Sobald diese Voraussetzung erfüllt ist, wird Rollback so einfach wie das erneute Deployen des vorherigen Artefakts, statt zu versuchen, eine unbekannte Menge an Änderungen manuell umzukehren, und jede deployte Version ist konstruktionsbedingt nachvollziehbar und auditierbar — der Zielkonflikt sind jedoch längere Build-Zeiten, da ganze Abbilder selbst für kleine Änderungen neu gebaut werden müssen, und ein kultureller Wandel weg von den SSH-und-Fix-Gewohnheiten, auf die sich Legacy-Betriebsteams oft verlassen.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Verpacken Sie Legacy-Anwendungen in Maschinenabbilder, Container oder Deployment-Artefakte, die alle Abhängigkeiten enthalten
- Beseitigen Sie manuelle Konfigurationsänderungen auf laufenden Servern; alle Änderungen müssen durch die Build-Pipeline fließen
- Nutzen Sie Infrastructure-as-Code-Werkzeuge, um Serverkonfigurationen neben dem Anwendungscode zu definieren und zu versionieren
- Implementieren Sie Blue-Green- oder Canary-Deployment-Strategien, bei denen neue Versionen bestehende Instanzen ersetzen statt sie zu aktualisieren
- Speichern Sie Anwendungszustand extern (Datenbanken, Objektspeicher), damit Rechen-Instanzen frei ersetzt werden können
- Automatisieren Sie die Erstellung neuer Infrastruktur von Grund auf, damit jede Umgebung identisch neu gebaut werden kann
- Markieren und archivieren Sie jedes deployte Artefakt für Auditierbarkeit und Rollback-Fähigkeit

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Beseitigt Konfigurationsdrift, die „funktioniert auf meiner Maschine"- und umgebungsspezifische Bugs verursacht
- Macht Deployments reproduzierbar und auditierbar
- Vereinfacht Rollback durch erneutes Deployen des vorherigen bekannt-guten Artefakts
- Verringert das Risiko angesammelter undokumentierter Änderungen in Produktionsumgebungen

**Kosten und Risiken:**
- Legacy-Anwendungen mit eingebettetem Zustand oder lokalen Dateiabhängigkeiten erfordern Refactoring
- Build-Zeiten steigen, da ganze Abbilder für jede Änderung neu gebaut werden müssen
- Erfordert Investition in Automatisierungs-Tooling und Container- oder Abbildverwaltungsinfrastruktur
- Teams, die an SSH-und-Fix-Workflows gewöhnt sind, brauchen kulturelle und prozessuale Anpassung

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Eine Regierungsbehörde betrieb eine Legacy-Java-Anwendung auf Servern, die über Jahre manuelle Konfigurationspatches angesammelt hatten. Keine zwei Server waren identisch konfiguriert, und Deployments schlugen häufig auf manchen Maschinen fehl. Das Team containerisierte die Anwendung und erfasste alle Abhängigkeiten und Konfigurationen in einem von CI gebauten Docker-Image. Deployments wurden zu einfachen Image-Ersetzungen, Konfigurationsdrift verschwand, und das Team konnte jede Umgebung sofort reproduzieren. Als ein Deployment Probleme verursachte, bedeutete Rollback das erneute Deployen des vorherigen Image-Tags statt den Versuch, manuelle Änderungen rückgängig zu machen.
