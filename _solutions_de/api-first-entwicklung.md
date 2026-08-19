---
title: API-First-Entwicklung
description: Entwicklung von Anwendungen mit klar definierten APIs als Grundlage.
category:
- Architecture
- Code
problems:
- poor-interfaces-between-applications
- tight-coupling-issues
- integration-difficulties
- rest-api-design-issues
- legacy-api-versioning-nightmare
- difficult-code-reuse
- poor-contract-design
layout: solution
lang: de
en_slug: api-first-development
related_solutions:
- slug: api-first-design
  similarity: 0.9
- slug: api-documentation
  similarity: 0.8
- slug: backward-compatible-apis
  similarity: 0.75
- slug: contract-testing
  similarity: 0.75
- slug: consumer-driven-contracts
  similarity: 0.75
- slug: api-gateway
  similarity: 0.7
---

## Description

API-First-Entwicklung bedeutet, den Schnittstellenvertrag eines Services — mithilfe eines Spezifikationsformats wie OpenAPI, GraphQL-Schema oder Protocol Buffers — zu definieren, bevor die dahinterliegende Implementierung geschrieben wird, und diesen Vertrag dann als die stabile, autoritative Vereinbarung zu behandeln, die sowohl die Implementierung als auch ihre Konsumenten einhalten müssen. Server-Stubs und Client-SDKs können direkt aus der Spezifikation generiert werden, und automatisierte Tests validieren, dass die laufende Implementierung tatsächlich dem entspricht, was spezifiziert wurde, sodass der Vertrag nicht still von der Realität abdriftet, wie es Ad-hoc-Dokumentation oft tut. Legacy-Systeme exponieren Funktionalität häufig durch eine inkonsistente Mischung von Schnittstellen — SOAP-Services, rohe Datenbankansichten, Batch-Datei-Importe —, die organisch ohne einen vereinheitlichenden Vertrag gewachsen sind, was jede neue Integration zu einer Übung im Reverse Engineering statt dem Lesen einer Spezifikation macht. Die Anwendung von API-First-Denken auf ein solches System bedeutet typischerweise, einen sauberen Vertrag für die meistgenutzten Fähigkeiten des Legacy-Systems zu designen und ihn vor die bestehende Implementierung zu stellen, sodass neue Konsumenten sich gegen eine stabile, gut definierte Schnittstelle integrieren, während die Legacy-Internas weiter unverändert dahinter laufen. Diese Trennung ist es, die es später erlaubt, die Legacy-Implementierung Stück für Stück zu ersetzen, da Konsumenten nur vom Vertrag abhängen und vom Austausch unberührt bleiben, solange dieser stabil bleibt. Die erforderliche Vorab-Designdisziplin und die Schwierigkeit, das reale Verhalten eines komplexen undokumentierten Legacy-Systems vollständig in einer einzigen Spezifikation zu erfassen, sind die Hauptkosten, die gegen die Integrations- und Parallelentwicklungsvorteile des Ansatzes abgewogen werden müssen.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Definieren Sie API-Verträge (OpenAPI, GraphQL-Schema oder Protocol Buffers), bevor die Backend-Logik implementiert wird
- Nutzen Sie Contract-First-Codegenerierung zur Produktion von Server-Stubs und Client-SDKs aus der API-Spezifikation
- Etablieren Sie API-Designrichtlinien, die Namenskonventionen, Versionierung, Fehlerbehandlung und Paginierung abdecken
- Implementieren Sie API-Verträge für Legacy-Systemintegrationen, selbst wenn das Legacy-System ursprünglich nicht API-getrieben war
- Nutzen Sie die API-Spezifikation als einzige Quelle der Wahrheit für Integrationsdokumentation
- Validieren Sie API-Antworten gegen die Spezifikation in automatisierten Tests, um Vertragsabdrift zu verhindern
- Veröffentlichen Sie API-Spezifikationen in einem zentralen Katalog, sodass konsumierende Teams unabhängig entdecken und sich integrieren können

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Ermöglicht Frontend- und Backend-Teams, parallel zu arbeiten, wobei der Vertrag als gemeinsame Vereinbarung dient
- Produziert sich selbst dokumentierende APIs, die Integrationsreibung verringern
- Macht Legacy-Systemfähigkeiten für neue Konsumenten über gut definierte Schnittstellen zugänglich
- Erleichtert automatisiertes Testen, Mocking und Vertragsverifikation

**Kosten und Risiken:**
- Erfordert Vorab-Designaufwand, bevor die Implementierung beginnen kann
- Das Ändern von APIs, nachdem Konsumenten sie übernommen haben, erfordert sorgfältige Versionierung und Migration
- Legacy-Systeme mit komplexem, undokumentiertem Verhalten sind schwierig vollständig in einem API-Vertrag zu erfassen
- Übermäßige API-Spezifikation kann Implementierungsflexibilität einschränken
- Die Aufrechterhaltung von Konsistenz zwischen Spezifikation und Implementierung erfordert Tooling und Disziplin

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Legacy-ERP-System exponierte Funktionalität durch eine Mischung aus SOAP-Services, Datenbankansichten und Batch-Datei-Importen, ohne konsistente Schnittstelle. Das Team definierte eine OpenAPI-Spezifikation, die die 20 meistgenutzten Operationen abdeckte, und baute ein REST-API-Gateway vor das Legacy-System. Neue Anwendungen integrierten sich ausschließlich über diese API, und die Spezifikation wurde in einem internen Entwicklerportal veröffentlicht. Als das Team später begann, ERP-Module zu ersetzen, konnten sie Implementierungen hinter der API austauschen, ohne Konsumenten zu benachrichtigen, weil der Vertrag unverändert blieb. Der API-First-Ansatz verwandelte das Legacy-System von einem Integrationsalbtraum in einen gut dokumentierten, stabilen Service.
