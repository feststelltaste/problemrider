---
title: API-First-Design
description: Definition und Gestaltung von Schnittstellen vor der Implementierung
  der Anwendungslogik.
category:
- Architecture
- Dependencies
quality_tactics_url: https://qualitytactics.de/en/maintainability/api-first-design/
problems:
- rest-api-design-issues
- graphql-complexity-issues
- rate-limiting-issues
- high-api-latency
- microservice-communication-overhead
- serialization-deserialization-bottlenecks
- service-discovery-failures
- shared-database
layout: solution
lang: de
en_slug: api-first-design
related_solutions:
- slug: api-first-development
  similarity: 0.9
- slug: api-documentation
  similarity: 0.8
- slug: contract-testing
  similarity: 0.8
- slug: api-security
  similarity: 0.75
- slug: strangler-fig-pattern
  similarity: 0.75
- slug: user-centered-design
  similarity: 0.7
---

## Description

API-First-Design behandelt den Schnittstellenvertrag zwischen Komponenten — seine Form, seine Datenformate, seine Ratenlimits — als das Ding, das designt und vereinbart werden soll, bevor irgendeine Implementierungsarbeit beginnt, statt als Nebenprodukt dessen, was die Implementierung zufällig produziert hat. Dies kehrt das häufige Legacy-Muster um, bei dem sich Schnittstellen über Jahre ad-hoc-Änderungen zufällig entwickelten, was Konsumenten zwang, undokumentiertes Verhalten zur Integrationszeit zurückzuentwickeln. Das Schreiben der Spezifikation zuerst, mithilfe eines Formats wie OpenAPI oder GraphQL SDL als einzige Quelle der Wahrheit, erlaubt es Konsumenten- und Anbieterteams, einen Vertrag zu überprüfen und zu vereinbaren, bevor sich eine Seite auf Code festlegt, und erlaubt es Tooling, Clients, Validierung und Vertragstests direkt aus dieser vereinbarten Form zu generieren.

## How to Apply ◆

> In Legacy-Systemen verschiebt API-First-Design das Gespräch von „was macht der Code bereits" zu „welcher Vertrag sollte zwischen Komponenten bestehen", was essentiell ist, wenn Systeme modernisiert werden, deren Schnittstellen sich über Jahre ad-hoc-Änderungen zufällig entwickelten.

- Auditieren Sie alle bestehenden API-Oberflächen im Legacy-System — REST-Endpunkte, Nachrichtenformate, datenbankebenen Integrationen, Dateiaustausche — und dokumentieren Sie die impliziten Verträge, die aktuell existieren, weil Sie nicht vorwärts designen können, ohne die aktuelle Realität zu verstehen.
- Führen Sie ein API-Spezifikationsformat wie OpenAPI für REST oder GraphQL SDL für GraphQL als einzige Quelle der Wahrheit für jede Schnittstelle ein, und verlangen Sie, dass Spezifikationsänderungen überprüft und genehmigt werden, bevor irgendeine Implementierungsarbeit beginnt.
- Etablieren Sie einen Contract-First-Workflow, bei dem neue Features mit einem Spezifikations-Pull-Request beginnen, den sowohl Konsumenten- als auch Anbieterteams überprüfen; dies verhindert das Legacy-Muster, bei dem ein Team einen Endpunkt baut und das andere seine Form erst zur Integrationszeit entdeckt.
- Nutzen Sie Codegenerierung aus API-Spezifikationen zur Produktion von Client-SDKs, Server-Stubs und Validierungs-Middleware, um sicherzustellen, dass die Implementierung nicht still vom vereinbarten Vertrag abdriften kann.
- Definieren Sie explizite Ratenbeschränkungsrichtlinien, Paginierungsstrategien und Fehlerantwortformate in der API-Spezifikation selbst, nicht als nachträgliche Gedanken, die während Lasttests entdeckt werden — Legacy-Systemen fehlen diese Beschränkungen häufig, und sie leiden unter unvorhersehbarem Verhalten unter Last.
- Spezifizieren Sie Serialisierungsformate und Payload-Beschränkungen im Voraus, einschließlich maximaler Antwortgrößen, erforderlicher Felder und Versionierungsheader, um die aufgeblähten und inkonsistenten Payloads zu verhindern, die sich üblicherweise in Legacy-APIs anhäufen.
- Richten Sie automatisiertes Vertragstesten in CI-Pipelines mit Werkzeugen wie Spectral für Linting, Prism für Mock-Server oder Pact für konsumentengetriebene Vertragstests ein, sodass Spezifikationsverletzungen vor dem Deployment statt in Produktion erfasst werden.
- Bei der Modernisierung von Microservice-Kommunikation designen Sie die Inter-Service-API-Verträge so, dass Geschwätzigkeit minimiert wird, indem grobkörnige Operationen modelliert werden, die Roundtrips verringern, statt die feingranularen internen Methodenaufrufe des Legacy-Monolithen zu spiegeln.
- Registrieren Sie API-Spezifikationen in einem zentralen API-Katalog oder Entwicklerportal, das als Service-Discovery-Mechanismus für Entwicklungsteams dient, was klarmacht, welche Services existieren, was sie bieten und wie man sich mit ihnen verbindet.

## Tradeoffs ⇄

> API-First-Design lädt Designaufwand und Koordinationskosten vor, um die Integrationsprobleme zu verhindern, die Legacy-Systeme plagen, erfordert aber Disziplin und organisatorische Unterstützung, um durchgehalten zu werden.

**Vorteile:**

- Eliminiert die inkonsistente Endpunktbenennung, Antwortformate und HTTP-Methodennutzung, die sich anhäufen, wenn APIs ad-hoc während der Implementierung designt werden.
- Ermöglicht parallele Entwicklung zwischen Konsumenten- und Anbieterteams, weil beide gegen dieselbe Spezifikation arbeiten, was die sequenziellen Abhängigkeiten verringert, die Legacy-Modernisierung verlangsamen.
- Macht Ratenbeschränkung und Ressourcenschutz von Anfang an explizit, was die falsch konfigurierte oder fehlende Drosselung verhindert, die Produktionsvorfälle in Legacy-Systemen verursacht.
- Verringert Serialisierungs-Overhead, indem bewusste Entscheidungen über Payload-Struktur und -format vor der Implementierung erzwungen werden, statt standardmäßig das zu nutzen, was das Framework generiert.
- Bietet eine Grundlage für automatisierte Kompatibilitätsprüfung, sodass Breaking Changes erkannt werden, bevor sie Produktion erreichen und Integrationsfehler über abhängige Services hinweg verursachen.
- Schafft auffindbare, gut dokumentierte Schnittstellen, die die Onboarding-Zeit für neue Entwickler verringern, die mit unvertrauten Legacy-Services arbeiten.

**Kosten und Risiken:**

- Erfordert Vorab-Designzeit, gegen die sich Teams, die an „erst bauen, später dokumentieren"-Workflows gewöhnt sind, möglicherweise wehren, besonders unter Lieferdruck.
- Spezifikationswartung wird zu einer anhaltenden Last; wenn Teams aufhören, Spezifikationen zu aktualisieren, wenn sie Implementierungen ändern, wird der Vertrag irreführend statt hilfreich.
- Übermäßig starre Vertragsdurchsetzung kann explorative Entwicklungsphasen verlangsamen, in denen die richtige API-Form noch nicht bekannt ist, was ein Gleichgewicht zwischen Disziplin und Flexibilität erfordert.
- Tooling-Investition für Codegenerierung, Vertragstesten und API-Kataloge fügt Infrastrukturkomplexität hinzu, die neben den modernisierten Legacy-Systemen gewartet werden muss.
- In Organisationen mit vielen autonomen Teams kann das Erreichen von Einigkeit über gemeinsame API-Standards und Spezifikationsformate politisch schwierig und zeitaufwendig sein.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie API-First-Design angewendet wurde, um Schnittstellenprobleme in Legacy-Systemmodernisierung anzugehen.

Ein Logistikunternehmen betrieb ein Legacy-Bestellverwaltungssystem, dessen REST-API acht Jahre lang ohne Designrichtlinien gewachsen war. Verschiedene Teams hatten Endpunkte mit inkonsistenter Benennung hinzugefügt (`/getShipments`, `/shipment/list`, `/api/v2/shipments`), inkonsistenten Antwort-Envelopes und undokumentierten Fehlercodes. Integrationspartner verbrachten Tage damit, das Verhalten jedes Endpunkts zu entschlüsseln. Das Modernisierungsteam führte eine OpenAPI-Spezifikation als obligatorischen Startpunkt für alle neuen und überarbeiteten Endpunkte ein. Bestehende Endpunkte wurden wie sie waren dokumentiert und dann inkrementell auf den Spezifikationsstandard ausgerichtet. Innerhalb von sechs Monaten sank die Integrations-Onboarding-Zeit von zwei Wochen auf zwei Tage, und Produktions-Integrationsfehler fielen um 60 %.

Ein Finanzdienstleistungsunternehmen baute eine Microservices-Plattform, bei der jedes Team sein eigenes Serialisierungsformat und Kommunikationsmuster wählte. Manche Services nutzten JSON mit tief verschachtelten Antworten, andere nutzten XML, und einige nutzten Protocol Buffers. Der Checkout-Flow erforderte sieben Inter-Service-Aufrufe, jeder mit unterschiedlichen Payload-Konventionen, und Serialisierungs-Overhead machte 35 % der Gesamtlatenz aus. Das Architekturteam schrieb API-First-Design mit einer gemeinsam genutzten Spezifikationsregistry vor. Alle Inter-Service-Verträge wurden in OpenAPI mit standardisierten Antwortformen und expliziten Payload-Größenlimits definiert. Teams generierten Client-Code aus Spezifikationen, was manuelle Serialisierungshandhabung eliminierte. Die standardisierten Verträge ermöglichten außerdem ein einheitliches Ratenbeschränkungs-Gateway, das die inkonsistente serviceweise Drosselung ersetzte, die zuvor legitimen Traffic während Spitzenzeiten blockiert hatte.

Eine Gesundheitsplattform musste ein Legacy-Patientenaktensystem mit einer neuen Telemedizin-Anwendung integrieren. Das Legacy-System exponierte eine undokumentierte SOAP-API, die vollständige Patientenakten unabhängig davon zurückgab, welche Information angefragt wurde. Das Team übernahm einen API-First-Ansatz und designte eine neue REST-Spezifikation, die präzise Ressourcenmodelle, feldweise Filterung und Paginierung definierte, bevor irgendein Code geschrieben wurde. Konsumentengetriebene Vertragstests stellten sicher, dass die neue API die tatsächlichen Datenbedürfnisse der Telemedizin-Anwendung erfüllte. Das Ergebnis war eine saubere Schnittstelle, die nur angefragte Felder zurückgab, was durchschnittliche Payload-Größen um 85 % und API-Latenz um 70 % im Vergleich zur Legacy-SOAP-Integration verringerte.
