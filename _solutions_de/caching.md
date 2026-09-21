---
title: Caching
description: Zwischenspeicherung häufig benötigter Daten.
category:
- Performance
- Architecture
quality_tactics_url: https://qualitytactics.de/en/performance-efficiency/caching/
problems:
- poor-caching-strategy
- cache-invalidation-problems
- data-structure-cache-inefficiency
- network-latency
- external-service-delays
- high-api-latency
- excessive-disk-io
- unoptimized-file-access
- lazy-loading
- memory-swapping
- serialization-deserialization-bottlenecks
- microservice-communication-overhead
- imperative-data-fetching-logic
layout: solution
lang: de
en_slug: caching-strategy
related_solutions:
- slug: distributed-caching
  similarity: 0.8
- slug: efficient-algorithms
  similarity: 0.8
- slug: performance-optimization
  similarity: 0.8
- slug: resource-usage-optimization
  similarity: 0.75
- slug: query-optimization-process
  similarity: 0.75
- slug: lazy-loading
  similarity: 0.75
---

## Description

Caching speichert das Ergebnis einer teuren Operation — eine Datenbankabfrage, ein externer Serviceaufruf, ein Dateilesevorgang —, sodass nachfolgende Anfragen für dieselben Daten aus dem Speicher bedient werden können, statt die ursprünglichen Kosten zu wiederholen. Legacy-Systeme entwickelten sich typischerweise ohne jegliche bewusste Caching-Strategie, da die damalige Last nie eine erforderte, was bedeutet, dass dieselben Referenzdaten oder derselbe langsame externe Aufruf selbst Jahre später bei jeder einzelnen Anfrage frisch abgerufen werden. Caching an den Grenzen einzuführen, wo es sich am meisten auszahlt — Referenzdaten, externe Abhängigkeiten, wiederholte Dateilesevorgänge —, ist eine der Performance-Verbesserungen mit dem höchsten Ertrag und geringsten Risiko, die für ein Legacy-System verfügbar sind, vorausgesetzt, Invalidierung wird in jeden Pfad eingebaut, der die zugrunde liegenden Daten ändern kann, da ein verpasster Invalidierungspfad das ist, was einen Cache in eine Quelle stiller, schwer reproduzierbarer Veraltungsfehler verwandelt.

## How to Apply ◆

> Legacy-Systeme entwickelten sich typischerweise ohne bewusste Caching-Strategie. Daten werden bei jeder Anfrage aus Datenbanken, Dateisystemen oder externen Services abgerufen, weil die ursprüngliche Last nie etwas anderes erforderte. Die Einführung von Caching erfordert das Verständnis der Datenzugriffsmuster des Systems und das Hinzufügen von Cache-Schichten dort, wo sie den größten Nutzen bei akzeptablem Veraltungsrisiko liefern.

- Profilen Sie die Anwendung, um die Datenzugriffsoperationen mit der höchsten Häufigkeit und den höchsten Kosten zu identifizieren. In Legacy-Systemen sind dies oft Datenbankabfragen für Referenzdaten (Produktkataloge, Konfigurationstabellen, Nutzerrollen), die sich selten ändern, aber bei jeder Anfrage gelesen werden. Priorisieren Sie das Caching dieser Operationen zuerst für den größten Ertrag pro Aufwand.
- Führen Sie einen Anwendungsebenen-Cache ein (wie Ehcache, Caffeine oder ein einfaches In-Memory-Dictionary) für Daten, die weit häufiger gelesen als geschrieben werden. In Legacy-Systemen, wo das Hinzufügen eines externen Caches wie Redis oder Memcached Infrastrukturänderungen erfordert, die durch organisatorische Beschränkungen blockiert sein könnten, kann ein In-Process-Cache mit minimaler Störung deployt werden.
- Fügen Sie HTTP-Caching-Header (Cache-Control, ETag, Last-Modified) zu API-Antworten hinzu, die relativ stabile Daten bedienen. Viele Legacy-Systeme lassen diese Header vollständig weg, was Clients und zwischengeschaltete Proxys zwingt, unveränderte Daten bei jeder Anfrage erneut abzurufen. Dies ist besonders effektiv zur Verringerung der Netzwerklatenz für geografisch verteilte Nutzer.
- Cachen Sie Antworten von externen Services und Drittanbieter-APIs, von denen das Legacy-System abhängt. Externe Serviceverzögerungen sind ein übliches Problem in alternden Systemen, die über die Jahre Integrationen angehäuft haben. Selbst kurzlebiges Caching (30-60 Sekunden) kann die Auswirkung langsamer oder unzuverlässiger externer Abhängigkeiten dramatisch verringern und kaskadierende Ausfälle verhindern.
- Implementieren Sie einen verteilten Cache (Redis, Memcached oder Hazelcast), wenn das Legacy-System auf mehreren Anwendungsserver-Instanzen läuft. Ohne einen gemeinsamen Cache unterhält jede Instanz ihren eigenen Cache, was zu inkonsistentem Verhalten zwischen Knoten und verschwendetem Speicher für duplizierte Daten führt. Ein verteilter Cache überlebt außerdem Anwendungsneustarts, was für Systeme wertvoll ist, die während der Modernisierung häufige Neu-Deployments erfordern.
- Ersetzen Sie wiederholte Dateilesevorgänge durch In-Memory-Caches für Konfigurationsdateien, Vorlagen und Nachschlagedaten, die im Dateisystem gespeichert sind. Legacy-Systeme lesen oft dieselben Dateien bei jeder Anfrage von der Festplatte, weil der Code vor gepufferten E/A-Abstraktionen entstand oder weil Entwickler sich der Performance-Kosten nicht bewusst waren. Das Caching von Dateiinhalten im Speicher eliminiert übermäßige Festplatten-E/A für diese Zugriffsmuster.
- Cachen Sie serialisierte Repräsentationen häufig angefragter Objekte, um wiederholten Serialisierungs-/Deserialisierungs-Overhead zu vermeiden. In Legacy-Systemen, die ausführliche Formate wie XML oder SOAP nutzen, kann das Caching des serialisierten Payloads statt der erneuten Serialisierung aus Domänenobjekten bei jeder Anfrage erhebliche CPU- und Speicherressourcen zurückgewinnen.
- Designen Sie Cache-Schlüssel sorgfältig, um Granularität und Trefferrate auszubalancieren. Schlüssel, die zu spezifisch sind (einschließlich Zeitstempel oder Nutzer-IDs für gemeinsam genutzte Daten), produzieren schlechte Trefferraten, während zu breite Schlüssel veraltete oder falsche Daten bedienen. Auditieren Sie bestehende Datenzugriffsmuster, um zu bestimmen, welche Parameter die Antwort genuin variieren.
- Implementieren Sie explizite Cache-Invalidierung, gebunden an Datenänderungspfade. In Legacy-Systemen werden Daten oft durch mehrere Einstiegspunkte geändert (Admin-Oberflächen, Batch-Jobs, direkte Datenbankupdates, Integrations-APIs), und das Fehlen auch nur eines Invalidierungspfads verursacht veraltete Daten. Kartieren Sie alle Schreibpfade für gecachte Daten und fügen Sie jedem Invalidierungs-Hooks hinzu.
- Fügen Sie von Anfang an Cache-Monitoring hinzu: Verfolgen Sie Trefferraten, Fehlraten, Eviction-Zählungen und Cache-Größe. In Legacy-Systemen ohne Observability-Infrastruktur helfen selbst einfache logbasierte Metriken zu erkennen, ob der Cache effektiv ist oder zu einer Quelle veralteter Daten geworden ist. Setzen Sie Alarme für Trefferraten-Rückgänge, die auf Invalidierungsprobleme oder Zugriffsmusteränderungen hinweisen.

## Tradeoffs ⇄

> Caching ist eine der effektivsten Performance-Optimierungen für Legacy-Systeme, führt aber eine Zustandsverwaltungsschicht ein, die sorgfältig kontrolliert werden muss, um Datenkonsistenzprobleme und operative Komplexität zu vermeiden.

**Vorteile:**

- Verringert die Datenbanklast, indem wiederholte Abfragen aus dem Speicher bedient werden, was oft 60-90 % der Datenbank-Roundtrips in Legacy-Systemen eliminiert, wo dieselben Referenzdaten bei jeder Anfrage abgerufen werden.
- Senkt die API-Latenz erheblich, indem die Kosten von Netzwerkaufrufen, Datenbankabfragen und Serialisierung für Daten vermieden werden, die sich seit der letzten Anfrage nicht geändert haben.
- Absorbiert die Auswirkung langsamer oder unzuverlässiger externer Serviceabhängigkeiten und erlaubt der Anwendung, gecachte Daten weiter zu bedienen, selbst wenn Drittanbieter-Services degradiert oder vorübergehend nicht verfügbar sind.
- Verringert übermäßige Festplatten-E/A, indem häufig zugegriffene Dateidaten und Konfiguration im Speicher gehalten werden, was besonders wertvoll für Legacy-Systeme ist, die auf alternder Hardware mit langsamem Speicher laufen.
- Verringert Serialisierungs-/Deserialisierungs-Overhead durch Caching vorserialisierter Payloads und gewinnt CPU und Speicher zurück, die sonst für die wiederholte Konvertierung derselben Objekte aufgewendet würden.
- Verbessert die Nutzererfahrung, ohne Änderungen an Geschäftslogik oder Datenbankschemata zu erfordern, was es zu einer der risikoärmsten verfügbaren Performance-Verbesserungen für Legacy-Systeme unter Modernisierungsbeschränkungen macht.
- Mildert das N+1-Abfrageproblem und Lazy-Loading-Overhead, indem die Ergebnisse von Abfragen gecacht werden, die ORM-Frameworks sonst wiederholt ausführen würden, was die Anzahl der Datenbank-Roundtrips verringert, ohne die Datenzugriffsschicht zu refaktorieren.

**Kosten und Risiken:**

- Veraltete Daten sind das Hauptrisiko: gecachte Daten, die nicht ordentlich invalidiert werden, führen dazu, dass Nutzer veraltete Informationen sehen, was zu falschen Geschäftsentscheidungen, Sicherheitslücken oder Datenkorruption in nachgelagerten Prozessen führen kann.
- Cache-Invalidierung ist in Legacy-Systemen genuin schwierig, wo Daten durch viele Pfade geändert werden (Batch-Jobs, Admin-Werkzeuge, direktes SQL, Integrations-APIs) und kein einzelner Codepfad alle Schreibvorgänge kontrolliert. Das Fehlen eines Invalidierungspfads erzeugt intermittierende Bugs, die extrem schwer zu reproduzieren sind.
- Der Speicherverbrauch steigt, was Speicher-Swapping auf Systemen auslösen kann, die bereits nahe an ihren physischen Speichergrenzen sind. In Legacy-Umgebungen, die auf beschränkter Hardware laufen, kann ein falsch dimensionierter Cache die Performance verschlechtern statt verbessern.
- Fügt operative Komplexität hinzu, auf die Legacy-Teams möglicherweise nicht vorbereitet sind: Cache-Infrastruktur muss überwacht, pro Umgebung konfiguriert und zusammen mit dem bestehenden System gepflegt werden. Verteilte Caches führen zusätzliche Netzwerkabhängigkeiten und potenzielle Fehlermodi ein.
- Caching verdeckt zugrunde liegende Performance-Probleme, statt sie zu beheben. Entwickler könnten einen Cache als permanente Lösung akzeptieren und nie die Grundursache angehen (ineffiziente Abfragen, schlechte Datenmodelle, übermäßige Serialisierung), was technische Schulden schafft, die sich über die Zeit summieren.
- Das Debugging wird schwieriger, weil das Anwendungsverhalten vom Cache-Zustand abhängt. Probleme, die sich nur bei spezifischem Cache-Inhalt oder Timing manifestieren, sind in Entwicklungsumgebungen, in denen Caches häufig geleert werden, schwer zu reproduzieren.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie Caching Performance-Probleme adressiert, die üblicherweise in Legacy-Systemen zu finden sind.

Ein 15 Jahre altes Versicherungsschadensverarbeitungssystem fragt bei jeder Schadensmeldung eine Referenztabelle von Policentypen, Deckungsregeln und regulatorischen Codes ab. Die Tabelle enthält 2.000 Zeilen und ändert sich nur während vierteljährlicher regulatorischer Updates, dennoch führt das System dieselbe Abfrage 50.000 Mal pro Tag aus. Jede Abfrage dauert 15 ms einschließlich Netzwerk-Roundtrip zur Datenbank. Das Team führt einen In-Process-Caffeine-Cache mit 4-Stunden-TTL und einem ereignisgetriebenen Invalidierungs-Hook ein, der vom vierteljährlichen Import-Job ausgelöst wird. Datenbankabfragen für Referenzdaten sinken auf weniger als 10 pro Tag, die Latenz der Schadensverarbeitung verringert sich um 35 %, und die Datenbank-CPU-Auslastung sinkt um 20 %, was Kapazität für tatsächliche transaktionale Abfragen freisetzt. Der gesamte Implementierungsaufwand beträgt zwei Tage, weil die Cache-Schicht die bestehenden Datenzugriffsmethoden umschließt, ohne Geschäftslogik zu ändern.

Ein Legacy-Auftragsmanagementsystem integriert sich mit fünf externen Services für Steuerberechnung, Versandtarife, Bestandsverifikation, Betrugsprüfung und Zahlungsabwicklung. Während Spitzenzeiten verursachen externe Serviceverzögerungen Checkout-Zeiten von über 8 Sekunden, wobei Steuer- und Versandtarifabfragen zusammen 3 Sekunden beitragen. Das Team fügt einen Redis-Cache für Steuersätze hinzu (geschlüsselt nach Gerichtsbarkeit und Produktkategorie, 1-Stunden-TTL) und Versandtarife (geschlüsselt nach Ursprung, Ziel und Gewichtsklasse, 30-Minuten-TTL). Die Cache-Trefferrate erreicht 85 % für Steuerabfragen und 70 % für Versand, was die durchschnittliche Checkout-Zeit von 8 Sekunden auf 3,5 Sekunden verringert. Als der Steuerservice während eines Feiertagsverkaufs einen 20-minütigen Ausfall erlebt, bedient die Anwendung weiterhin gecachte Steuersätze ohne kundenseitig sichtbare Auswirkung.

Ein monolithisches Java-ERP-System generiert PDF-Berichte, indem es XML-Vorlagen aus dem Dateisystem liest, parst und mit Datenbankabfrageergebnissen zusammenführt. Jeder Bericht liest und parst dieselben 50 Vorlagendateien, und während der Monatsendberichterstattung verarbeitet der Server 500 Berichte in einem Batch. Profiling zeigt, dass 40 % der Batch-Verarbeitungszeit für wiederholte Datei-E/A und XML-Parsing unveränderter Vorlagen aufgewendet wird. Das Team cacht geparste Vorlagenobjekte in einer WeakHashMap mit Datei-Änderungszeitstempeln als Invalidierungsschlüssel. Die Batch-Verarbeitungszeit sinkt von 4 Stunden auf 2,5 Stunden, die Festplatten-E/A während der Berichterstattung sinkt um 75 %, und der Ansatz erfordert keine Änderungen an den Vorlagendateien oder der Berichtsgenerierungslogik selbst.
