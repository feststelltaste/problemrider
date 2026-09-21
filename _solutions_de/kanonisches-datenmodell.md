---
title: Kanonisches Datenmodell
description: Standardisierung eines gemeinsamen Datenmodells über Systeme hinweg
  statt Punkt-zu-Punkt-Transformationen.
category:
- Architecture
- Database
problems:
- cross-system-data-synchronization-problems
- integration-difficulties
- poor-interfaces-between-applications
- data-migration-complexities
- inconsistent-behavior
- poor-domain-model
- technology-stack-fragmentation
- master-data-ownership-gaps
layout: solution
lang: de
en_slug: canonical-data-model
related_solutions:
- slug: data-strategy
  similarity: 0.8
- slug: data-ecosystems
  similarity: 0.8
- slug: standardized-data-formats
  similarity: 0.8
- slug: data-integration
  similarity: 0.75
- slug: data-formats
  similarity: 0.75
- slug: data-modeling
  similarity: 0.7
---

## Description

Ein kanonisches Datenmodell ist eine einzige, gemeinsam genutzte Repräsentation der Kern-Geschäftsentitäten und Datenstrukturen, zu und von der alle integrierenden Systeme übersetzen, was ein Netz direkter, systemspezifischer Abbildungen durch eine Hub-and-Spoke-Anordnung ersetzt, die auf einem vereinbarten Format zentriert ist. Jedes System braucht nur einen Übersetzer, zur und von der kanonischen Form, statt eines separaten Übersetzers für jedes andere System, mit dem es Daten austauscht, was das quadratische Wachstum von Punkt-zu-Punkt-Integrationsabbildungen in ein lineares verwandelt, während Systeme hinzugefügt oder entfernt werden. In Legacy-Landschaften, die über Jahre Dutzende maßgeschneiderter Integrationen angehäuft haben, jede mit ihren eigenen subtil unterschiedlichen Feldabbildungen, ist dieses Zusammenfallen der Übersetzungslogik das, was die Integrationsfläche wieder handhabbar macht und kleine Formatdiskrepanzen davon abhält, sich zu anhaltenden Abgleichsproblemen zu summieren. Das Modell schafft außerdem ein gemeinsames Vokabular für die Geschäftskonzepte, die eine Organisation tatsächlich nutzt, was häufig fehlt, wenn Legacy-Systeme von verschiedenen Teams unter Nutzung inkonsistenter Begriffe für dieselben Entitäten gebaut wurden. Die Einführung eines kanonischen Modells in eine bestehende Landschaft ist selbst ein Modernisierungsakt: Es erfordert die Aushandlung eines Modells, das neutral genug ist, um die Bedürfnisse jedes Legacy-Systems zu bedienen, ohne zu einer Kleinste-gemeinsame-Nenner-Abstraktion zu werden, die die Nuance verwirft, auf die jedes System angewiesen ist. Weil jedes verbundene System ein Stakeholder im kanonischen Schema ist, muss seine Evolution sorgfältig gesteuert werden, da eine für eine Integration vorgenommene Änderung auf alle anderen ausstrahlt, die vom selben gemeinsamen Vertrag abhängen.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Definieren Sie ein kanonisches Datenmodell, das die gemeinsamen Geschäftskonzepte repräsentiert, die über Systeme hinweg genutzt werden
- Bauen Sie Übersetzer an der Grenze jedes Systems, die zwischen dem internen Modell des Systems und dem kanonischen Modell abbilden
- Beginnen Sie mit den am stärksten frequentierten oder fehleranfälligsten Integrationspunkten, statt alles auf einmal zu modellieren
- Versionieren Sie das kanonische Modell und verwalten Sie seine Evolution durch einen Governance-Prozess
- Speichern Sie das kanonische Schema in einem gemeinsamen Repository, zugänglich für alle Teams
- Nutzen Sie das kanonische Modell als Vertrag für ereignisgetriebene oder nachrichtenbasierte Integrationen

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Verringert die Anzahl der Integrationsabbildungen von O(n²) Punkt-zu-Punkt auf O(n) Übersetzungen
- Schafft ein gemeinsames Vokabular, das die teamübergreifende Kommunikation über Daten verbessert
- Vereinfacht das Hinzufügen neuer Systeme zur Integrationslandschaft

**Kosten und Risiken:**
- Das Design des kanonischen Modells erfordert teamübergreifenden Konsens, der langsam und politisch sein kann
- Das kanonische Modell kann zu einem Kleinste-gemeinsame-Nenner werden, der wichtige Domänennuancen verliert
- Änderungen am kanonischen Modell strahlen auf alle verbundenen Systeme aus
- Risiko, ein übermäßig generisches Modell zu schaffen, das keinem System gut dient

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Eine mittelgroße Bank hatte 12 Legacy-Systeme, die Kundendaten durch 40 Punkt-zu-Punkt-Integrationen austauschten, jede mit ihrer eigenen Feldabbildungslogik. Dateninkonsistenzen zwischen Systemen verursachten durchschnittlich fünf Abgleichsvorfälle pro Monat. Das Team definierte ein kanonisches Kundenmodell und baute über sechs Monate Übersetzer für jedes System. Die Anzahl der Integrationsabbildungen sank von 40 auf 12, Abgleichsvorfälle fielen auf weniger als einen pro Monat, und die Einbindung eines neuen CRM-Systems dauerte drei Wochen statt der zuvor geschätzten drei Monate.
