---
title: Datenintegration
description: Zusammenführung von Daten aus verschiedenen Quellen und einheitliche
  Bereitstellung.
category:
- Database
- Architecture
problems:
- cross-system-data-synchronization-problems
- information-fragmentation
- shared-database
- data-migration-complexities
- poor-interfaces-between-applications
- integration-difficulties
layout: solution
lang: de
en_slug: data-integration
related_solutions:
- slug: data-ecosystems
  similarity: 0.8
- slug: data-strategy
  similarity: 0.75
- slug: canonical-data-model
  similarity: 0.75
- slug: business-event-processing
  similarity: 0.7
- slug: data-integrity
  similarity: 0.7
- slug: data-replication
  similarity: 0.7
---

## Description

Datenintegration führt über mehrere Legacy-Systeme verstreute Daten zu einer kohärenten, einheitlich zugänglichen Sicht zusammen, typischerweise mittels einer dedizierten Integrationsschicht — ETL-Pipelines, Datenvirtualisierung oder ereignisbasierte Synchronisation — statt über direkte Punkt-zu-Punkt-Verbindungen zwischen jedem Systempaar. Der Ansatz beginnt damit, abzubilden, wie dieselben Geschäftsentitäten über Systeme hinweg unterschiedlich repräsentiert werden, definiert dann ein kanonisches Modell für jede gemeinsam genutzte Entität, das als der Vertrag dient, in den die Integrationsschicht jede Quelle übersetzt, wobei Daten an dieser Grenze bereinigt und validiert werden, statt die Qualitätsprobleme jeder Quelle nachgelagert zu propagieren. Dies ist besonders relevant für Legacy-Landschaften, die über Jahre durch separate Abteilungssysteme, Akquisitionen oder unkoordinierte Expansion zusammengestellt wurden, wo derselbe Kunde, Patient oder dasselbe Produkt in mehreren Systemen ohne gemeinsame Identität existiert und wo Nutzer manuell mehrere Bildschirme kreuzreferenzieren müssen, um ein einziges kohärentes Bild zu rekonstruieren. Wo eine Legacy-Datenbank nicht modifiziert werden kann, um Events direkt zu emittieren, erlaubt Change Data Capture der Integrationsschicht, Änderungen auf Datenbankebene zu beobachten und sie zu propagieren, ohne die Quellanwendung überhaupt anzufassen. Weil die Integrationsschicht zu einem Stück kritischer Infrastruktur wird, von dem jetzt jedes konsumierende System abhängt, brauchen ihre eigene Zuverlässigkeit, Überwachung und Latenzeigenschaften ebenso viel Aufmerksamkeit wie die Datenqualitätsprobleme, die sie lösen sollte.

## How to Apply ◆

- Bilden Sie Datenentitäten über Legacy-Systeme hinweg ab, um Überschneidungen, Konflikte und semantische Unterschiede zu identifizieren, wie dieselben Konzepte repräsentiert werden.
- Implementieren Sie eine Integrationsschicht (ETL-Pipelines, Datenvirtualisierung oder ereignisbasierte Synchronisation) statt Punkt-zu-Punkt-Verbindungen zwischen Legacy-Systemen.
- Definieren Sie kanonische Datenmodelle für gemeinsam genutzte Entitäten, die als Integrationsvertrag zwischen Systemen dienen.
- Behandeln Sie Datenqualitätsprobleme an der Integrationsgrenze: Validieren, bereinigen und transformieren Sie Daten, während sie zwischen Systemen fließen.
- Nutzen Sie Change Data Capture (CDC) für nahezu Echtzeit-Integration mit Legacy-Datenbanken, die nicht modifiziert werden können, um Events zu emittieren.
- Überwachen Sie Datenintegrationspipelines mit Alarmierung für Synchronisationsfehler, Datenqualitätseinbrüche und Latenzanstiege.

## Tradeoffs ⇄

**Vorteile:**
- Bietet eine einheitliche Sicht auf über Legacy-Systeme verstreute Daten, was Berichte und Analytics ermöglicht.
- Reduziert Dateninkonsistenzen, die durch manuelle Neueingabe über Systeme hinweg entstehen.
- Entkoppelt Systeme, indem Daten durch eine Integrationsschicht statt durch direkte Datenbankfreigabe geleitet werden.
- Ermöglicht schrittweisen Systemersatz, indem neuen Systemen erlaubt wird, integrierte Datenfeeds zu konsumieren.

**Kosten:**
- Der Aufbau und die Pflege von Integrationspipelines ist eine erhebliche laufende Investition.
- Die Datenabbildung über Legacy-Systeme mit inkonsistenten Schemata ist komplex und fehleranfällig.
- Integration führt Latenz ein; Echtzeitkonsistenz über Systeme hinweg ist möglicherweise nicht erreichbar.
- Die Integrationsschicht wird zu kritischer Infrastruktur; ihr Ausfall beeinträchtigt alle verbundenen Systeme.

## How It Could Be

Ein Krankenhaus betreibt separate Legacy-Systeme für Patientenregistrierung, Abrechnung, Laborergebnisse und Apotheke. Kliniker müssen sich in mehrere Systeme einloggen und Patienteninformationen manuell kreuzreferenzieren, was zu Verzögerungen und gelegentlichen Fehlern führt. Das IT-Team implementiert eine Datenintegrationsplattform mittels Apache NiFi und erstellt Pipelines, die Patientenstammdaten über Systeme hinweg synchronisieren und eine einheitliche Patientenakten-Sicht bieten. Change Data Capture auf der Datenbank des Registrierungssystems speist Aktualisierungen nahezu in Echtzeit an nachgelagerte Systeme. Die Integrationsschicht normalisiert Datenformate und löst Konflikte (wie unterschiedliche Datumsformate und Namensdarstellungen) auf, bevor Daten an Konsumenten geliefert werden. Kliniker sehen jetzt eine konsolidierte Patientensicht, und die Integrationsschicht bietet die Grundlage, um schließlich einzelne Legacy-Systeme zu ersetzen.
