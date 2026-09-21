---
title: Standardisierte Schnittstellen
description: Übernahme weit akzeptierter Schnittstellenstile, sodass jeder
  Konsument ohne maßgeschneiderte Adapter integrieren kann.
category:
- Architecture
- Dependencies
problems:
- poor-interfaces-between-applications
- integration-difficulties
- rest-api-design-issues
- vendor-lock-in
- technology-lock-in
- legacy-api-versioning-nightmare
- tight-coupling-issues
- dependency-on-supplier
layout: solution
lang: de
en_slug: standardized-interfaces
related_solutions:
- slug: standardized-protocols
  similarity: 0.75
- slug: consistent-user-interface
  similarity: 0.7
- slug: api-first-design
  similarity: 0.7
- slug: standardized-data-formats
  similarity: 0.7
- slug: data-formats
  similarity: 0.7
- slug: canonical-data-model
  similarity: 0.7
---

## Description

Standardisierte Schnittstellen bedeutet, proprietäre oder Ad-hoc-Integrationsmechanismen — maßgeschneiderte TCP-Protokolle, SOAP-Endpunkte mit eigenwilligen Konventionen, Dateiablage-Integrationen — durch weit verbreitete Schnittstellenstile wie REST, GraphQL oder gRPC zu ersetzen, beschrieben mit Standard-Spezifikationsformaten wie OpenAPI oder Protocol Buffers, sodass jeder Konsument mit gängigen Werkzeugen integrieren kann statt mit einem maßgeschneiderten Adapter, der speziell für dieses eine Legacy-System gebaut wurde. Legacy-Landschaften neigen dazu, einen anderen Integrationsstil für jedes System anzusammeln, das jemals mit ihnen verbunden wurde, und jedes neue konsumierende Team muss dann Wochen investieren, um die besonderen Eigenheiten dieses Systems zu lernen und dagegen zu codieren, bevor irgendeine echte Integrationsarbeit beginnen kann. Die Einführung einer Facade oder eines API-Gateways, das eine standardisierte Schnittstelle vor der Legacy-Implementierung freilegt, lässt diese Kosten einmal, zentral, bezahlt werden, statt wiederholt von jedem neuen Konsumenten, und es entkoppelt den Integrationsaufwand des Konsumenten davon, wie das Legacy-Backend intern zufällig aussieht. Diese Entkopplung ist es, was standardisierte Schnittstellen speziell während der Modernisierung wertvoll macht: Weil Konsumenten gegen den stabilen, standardisierten Vertrag integrieren statt direkt gegen die Legacy-Implementierung, kann das Backend hinter diesem Vertrag inkrementell ersetzt werden, ohne dass Konsumenten irgendetwas ändern müssen. Die entsprechenden Kosten sind der Vorabaufwand des Baus und der Governance dieser Facade-Schicht und das Risiko, dass eine generische Standardschnittstelle nicht jede Fähigkeit perfekt ausdrücken kann, die das Legacy-System ursprünglich bot, was bewusste Kompromisse im Vertragsdesign erfordert.

## How to Apply ◆

- Ersetzen Sie proprietäre oder Ad-hoc-Schnittstellen in Legacy-Systemen durch branchenübliche Stile wie REST, GraphQL oder gRPC.
- Definieren Sie Schnittstellenverträge mit Standard-Spezifikationsformaten (OpenAPI, Protocol Buffers, AsyncAPI) und veröffentlichen Sie sie für Konsumenten.
- Führen Sie ein API-Gateway oder eine Facade vor Legacy-Systemen ein, um standardisierte Schnittstellen zu präsentieren, während die zugrunde liegende Implementierung inkrementell migriert wird.
- Etablieren Sie Schnittstellendesign-Richtlinien, denen alle Teams folgen, einschließlich Namenskonventionen, Fehlerformaten, Paginierung und Authentifizierung.
- Nutzen Sie Contract Testing, um zu verifizieren, dass sowohl Anbieter als auch Konsumenten den vereinbarten Schnittstellenspezifikationen entsprechen.
- Dokumentieren Sie alle Schnittstellen in einem zentralen API-Katalog, sodass Konsumenten ohne Ad-hoc-Kommunikation entdecken und integrieren können.

## Tradeoffs ⇄

**Vorteile:**
- Jeder Konsument kann mit bekannten Werkzeugen und Bibliotheken integrieren, was die Einarbeitungszeit reduziert.
- Entkoppelt Konsumenten- und Anbieterimplementierungen und macht unabhängige Evolution möglich.
- Reduziert die Notwendigkeit für maßgeschneiderte Adapter, Übersetzer und Integrations-Middleware.
- Macht es einfacher, Legacy-Backend-Implementierungen zu ersetzen, ohne Konsumenten zu beeinträchtigen.

**Kosten:**
- Das Umhüllen von Legacy-Systemen mit standardisierten Schnittstellen erfordert Vorab-Entwicklungsaufwand.
- Standardschnittstellen könnten sich nicht perfekt auf Legacy-Systemfähigkeiten abbilden, was Kompromiss oder Anpassung erfordert.
- Die Durchsetzung von Standards über autonome Teams hinweg erfordert Governance und Zustimmung.
- Übermäßige Standardisierung kann die Flexibilität für spezialisierte Anwendungsfälle reduzieren.

## How It Could Be

Ein Logistikunternehmen hat Dutzende interner Systeme, die über eine Mischung aus SOAP, FTP-Dateiablagen und maßgeschneiderten TCP-Protokollen kommunizieren. Neue konsumierende Teams verbringen Wochen mit dem Bau maßgeschneiderter Adapter für jede Integration. Das Architekturteam führt ein API-Gateway ein, das RESTful, OpenAPI-dokumentierte Endpunkte vor den kritischsten Legacy-Systemen freilegt. Konsumierende Teams integrieren nun mit Standard-HTTP-Clients und automatisch generierten SDKs. Im Laufe der Zeit werden die Legacy-Backends durch moderne Implementierungen hinter denselben standardisierten Schnittstellen ersetzt, und Konsumenten erleben während des Übergangs keine Störung.
