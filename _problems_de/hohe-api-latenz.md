---
title: Hohe API-Latenz
description: Die Zeit, die eine API zum Antworten auf eine Anfrage benötigt, ist übermäßig
  lang, was zu schlechter Anwendungsperformance und einem negativen Nutzererlebnis
  führt.
category:
- Performance
related_problems:
- slug: external-service-delays
  similarity: 0.85
- slug: slow-application-performance
  similarity: 0.8
- slug: network-latency
  similarity: 0.8
- slug: upstream-timeouts
  similarity: 0.8
- slug: service-timeouts
  similarity: 0.8
- slug: slow-database-queries
  similarity: 0.75
solutions:
- api-first-design
- caching-strategy
- contract-testing
- serialization-optimization
- api-calls-optimization
- api-gateway
- api-security
- load-balancing
- optimistic-ui-updates
- predictive-loading
- predictive-prefetching
- rate-limiting
layout: problem
lang: de
en_slug: high-api-latency
---

## Description
Hohe API-Latenz ist ein verbreitetes Problem in verteilten Systemen, in denen Dienste oft voneinander abhängen, um Anfragen zu erfüllen. Wenn eine API lange braucht, um zu antworten, kann dies einen kaskadierenden Effekt haben, der Verzögerungen in nachgelagerten Diensten und ein schlechtes Nutzererlebnis verursacht. Hohe API-Latenz kann durch verschiedene Faktoren verursacht werden, von ineffizientem Code und langsamen Datenbankabfragen bis zu Netzwerkproblemen und fehlendem ordentlichen Caching. Ein systematischer Ansatz zur Performance-Analyse ist nötig, um die Grundursachen hoher API-Latenz zu identifizieren und zu beheben.

## Indicators ⟡
- Die Anwendung ist langsam, aber die Server stehen nicht unter hoher Last.
- Es zeigt sich eine hohe Anzahl an Timeout-Fehlern in den Logs.
- Die Performance der Anwendung ist inkonsistent.
- Es kommen Beschwerden von Nutzern über langsame Performance.

## Symptoms ▲

- [Langsame Anwendungsperformance](langsame-anwendungsperformance.md)
<br/>  Nutzerseitige Features, die von API-Aufrufen abhängen, fühlen sich träge an, wenn die zugrunde liegenden API-Antworten langsam sind.
- [Service-Timeouts](service-timeouts.md)
<br/>  Nachgelagerte Dienste, die die langsame API aufrufen, überschreiten ihre Timeout-Schwellenwerte, was kaskadierende Ausfälle verursacht.
- [Negatives Nutzerfeedback](negatives-nutzerfeedback.md)
<br/>  Nutzer erleben direkt langsame Seitenladezeiten und unresponsive Features, verursacht durch hohe API-Antwortzeiten.
- [Nutzerfrustration](nutzerfrustration.md)
<br/>  Durchgängig langsame API-Antworten führen zu einem schlechten Nutzererlebnis und wachsender Unzufriedenheit.
- [Kaskadierende Ausfälle](kaskadierende-ausfaelle.md)
<br/>  In verteilten Systemen kaskadiert hohe Latenz in einer API zu allen abhängigen Diensten, was weitverbreitete Verlangsamungen verursacht.

## Causes ▼

- [Performance-Probleme bei Datenbankabfragen](performance-probleme-bei-datenbankabfragen.md)
<br/>  Langsame Datenbankabfragen tragen wesentlich zur API-Latenz bei, besonders bei datenintensiven Endpunkten.
- [N+1-Abfrageproblem](n-plus-1-abfrageproblem.md)
<br/>  APIs, die N+1-Datenbankabfragen für verwandte Daten auslösen, vervielfachen Datenbank-Roundtrips und erhöhen Antwortzeiten dramatisch.
- [Verzögerungen durch externe Services](verzoegerungen-durch-externe-services.md)
<br/>  APIs, die von langsamen externen Diensten abhängen, übernehmen diese Verzögerungen in ihre eigenen Antwortzeiten.
- [Schlechte Caching-Strategie](schlechte-caching-strategie.md)
<br/>  Das Abrufen von Daten aus der Quelle bei jeder Anfrage statt Caching fügt API-Antwortzeiten unnötigen Overhead hinzu.
- [Netzwerklatenz](netzwerklatenz.md)
<br/>  Netzwerkübertragungsverzögerungen zwischen API-Komponenten und Datenquellen erhöhen direkt die API-Antwortzeiten.

## Detection Methods ○

- **Application Performance Monitoring (APM):** Nutzung von APM-Werkzeugen zur Nachverfolgung von Anfragen, Messung der Dauer jeder Operation (z. B. Datenbankaufrufe, externe Serviceaufrufe) und Lokalisierung der genauen Quelle der Verzögerung.
- **Logging:** Hinzufügen detaillierten Loggings zur Nachverfolgung der Zeit, die in unterschiedlichen Phasen des Anfrage-Lebenszyklus benötigt wird.
- **Metriken und Alerting:** Überwachung wichtiger Metriken wie p95/p99-Antwortzeiten und Einrichtung von Alerts, um über Performance-Verschlechterungen benachrichtigt zu werden.
- **Lasttests:** Nutzung von Lasttest-Werkzeugen zur Simulation von Traffic und Identifikation, wie Latenz von gleichzeitigen Nutzern beeinflusst wird.

## Examples
Der "Produktdetails"-API-Endpunkt einer E-Commerce-Website wird zunehmend langsamer, während die Anzahl der Produkte wächst. Die Untersuchung mit einem APM-Werkzeug zeigt, dass der Endpunkt eine langsame, unindizierte Abfrage macht, um Produktbewertungen abzurufen. In einem anderen Fall ist die Startzeit einer mobilen Anwendung schlecht, weil sie mehrere blockierende API-Aufrufe macht, um anfängliche Konfigurationsdaten abzurufen. Die Latenz dieser Aufrufe, besonders in langsameren mobilen Netzwerken, summiert sich erheblich. Dies ist ein verbreitetes Problem in verteilten Systemen und Microservices-Architekturen, in denen eine einzelne Nutzeraktion eine Kette von API-Aufrufen über mehrere Dienste hinweg auslösen kann.
