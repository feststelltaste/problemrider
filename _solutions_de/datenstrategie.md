---
title: Datenstrategie
description: Definition gemeinsamer Datenstandards, -formate und Integrationsmuster
  über Systeme hinweg.
category:
- Architecture
- Management
problems:
- cross-system-data-synchronization-problems
- poor-domain-model
- system-integration-blindness
- integration-difficulties
- data-migration-complexities
- technology-stack-fragmentation
- custom-report-sprawl
- master-data-ownership-gaps
layout: solution
lang: de
en_slug: data-strategy
related_solutions:
- slug: standardized-data-formats
  similarity: 0.85
- slug: data-ecosystems
  similarity: 0.85
- slug: canonical-data-model
  similarity: 0.8
- slug: data-integration
  similarity: 0.75
- slug: platform-independent-data-storage
  similarity: 0.75
- slug: data-formats
  similarity: 0.75
---

## Description

Eine Datenstrategie ist eine organisationsweite Definition gemeinsamer Datenstandards, -formate, Integrationsmuster und Eigentumszuweisungen, die mehrere Systeme umspannt, statt jedem Team oder Legacy-System zu überlassen, seine eigenen lokalen Entscheidungen darüber zu treffen, wie Daten strukturiert und ausgetauscht werden sollen. Sie umfasst typischerweise kanonische Modelle für über Systeme hinweg gemeinsam genutzte Entitäten, eine explizite Wahl von Integrationsmustern (ereignisgesteuert, API-basiert, Batch), die zu unterschiedlichen Anwendungsfällen passen, benannte Datenverantwortliche, die für spezifische Datendomänen zuständig sind, und eine Roadmap, die priorisiert, welche der bestehenden Ad-hoc-Integrationen der Organisation zuerst konsolidiert werden sollten. Dies ist in Legacy-Umgebungen wichtig, gerade weil das Fehlen einer solchen Strategie das ist, was die für Organisationen mit vielen Legacy-Systemen typische Fragmentierung überhaupt erst erzeugt: dasselbe Geschäftskonzept, repräsentiert in mehreren inkompatiblen Formaten über unterschiedliche Datenbanken hinweg, ohne vereinbarte einzige Quelle der Wahrheit, was Mitarbeiter zwingt, Datensätze manuell abzugleichen, die nie hätten divergieren dürfen. Eine Datenstrategie behebt kein einzelnes System für sich, aber sie gibt jeder nachfolgenden Modernisierungsentscheidung — welches Format übernommen werden soll, welches System welche Entität besitzt, welches Integrationsmuster für eine neue Verbindung genutzt werden soll — einen konsistenten Bezugsrahmen statt Ad-hoc-Improvisation, die System für System wiederholt wird. Ihr Hauptrisiko ist, zu einem Dokument zu werden, das vereinbart, aber nie umgesetzt wird, da eine Strategie nur Wert hat, sobald sie durch konkrete Integrations- und Governance-Entscheidungen durchgesetzt wird, statt als Wunschvorstellung abgelegt zu werden.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Definieren Sie eine organisationsweite Datenstrategie, die Dateneigentümerschaft, Qualitätsstandards und Integrationsmuster abdeckt
- Etablieren Sie kanonische Datenmodelle für über Systeme hinweg gemeinsam genutzte Kerngeschäftsentitäten
- Wählen und standardisieren Sie Integrationsmuster (ereignisgesteuert, API-basiert, Batch) für unterschiedliche Anwendungsfälle
- Weisen Sie Datenverantwortliche zu, die für die Qualität und Weiterentwicklung wichtiger Datendomänen zuständig sind
- Erstellen Sie eine Datenintegrations-Roadmap, die die Konsolidierung der problematischsten Legacy-Datenflüsse priorisiert
- Überprüfen und aktualisieren Sie die Datenstrategie periodisch, um Änderungen in der Systemlandschaft widerzuspiegeln

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Bietet eine kohärente Vision dafür, wie Daten durch die Organisation fließen, was Ad-hoc-Integration reduziert
- Ermöglicht informierte Entscheidungen über Datenformat- und Speicherwahl während der Legacy-Modernisierung
- Reduziert Datenqualitätsprobleme, die durch inkonsistente Standards über Systeme hinweg verursacht werden

**Kosten und Risiken:**
- Die Entwicklung einer umfassenden Datenstrategie erfordert bereichsübergreifende Abstimmung und Führungsunterstützung
- Strategie ohne Umsetzung wird zu Regalware, die die Legacy-Landschaft nicht verbessert
- Zentralisierte Data Governance kann mit Team-Autonomie in dezentralisierten Organisationen in Konflikt geraten
- Die Strategie aktuell zu halten erfordert laufende Investition

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Versicherungsunternehmen mit 20 Legacy-Systemen hatte keine Datenstrategie, was dazu führte, dass Kundendaten über sieben unterschiedliche Formate und fünf Datenbanken ohne einzige Quelle der Wahrheit verteilt waren. Schadensregulierer verbrachten durchschnittlich 30 Minuten pro Schadensfall damit, Kundeninformationen abzugleichen. Nach der Definition einer Datenstrategie mit kanonischen Modellen, zugewiesenen Datenverantwortlichen und einem ereignisgesteuerten Integrationsmuster für Kundendaten erreichte das Unternehmen innerhalb von 14 Monaten eine einheitliche Kundensicht. Die Schadensbearbeitungszeit sank um 25 Prozent.
