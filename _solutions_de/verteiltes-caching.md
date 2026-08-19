---
title: Verteiltes Caching
description: Zwischenspeicherung häufig benötigter Daten auf mehreren Computern.
category:
- Performance
problems:
- slow-application-performance
- poor-caching-strategy
- cache-invalidation-problems
- high-database-resource-utilization
- scaling-inefficiencies
- slow-database-queries
layout: solution
lang: de
en_slug: distributed-caching
related_solutions:
- slug: data-replication
  similarity: 0.8
- slug: connection-pooling
  similarity: 0.8
- slug: load-balancing
  similarity: 0.8
- slug: caching-strategy
  similarity: 0.8
- slug: lazy-loading
  similarity: 0.8
- slug: denormalization
  similarity: 0.8
---

## Description

Verteiltes Caching platziert häufig gelesene, selten geänderte Daten in einem In-Memory-Speicher — wie Redis, Memcached oder Hazelcast —, der zwischen der Anwendungsebene und der Datenbank sitzt und über jede Anwendungsinstanz hinweg geteilt wird, statt dass jede Instanz ihren eigenen lokalen, inkonsistenten Cache hält. Anfragen prüfen zuerst den gemeinsamen Cache und fallen nur bei einem Miss zur Datenbank durch, was wiederholte, redundante Abfragen nach denselben Referenzdaten vollständig aus der Datenbank-Workload entfernt. Legacy-Systeme erreichen häufig die Grenzen einer einzelnen Datenbankinstanz, nicht weil die zugrunde liegenden Daten groß sind, sondern weil dieselben Abfragen — Produktkataloge, Konfigurationsabfragen, Sitzungsdaten — weit häufiger ausgeführt werden, als sich die zugrunde liegenden Daten tatsächlich ändern, und keine Caching-Schicht existiert, um diese Wiederholung zu absorbieren. Einen verteilten Cache einzuführen adressiert dieses spezifische Muster, ohne die Datenbank selbst neu architektieren zu müssen, was ihn zu einem attraktiven Hebel in Legacy-Kontexten macht, wo tiefere Datenbankänderungen teuer oder riskant sind. Weil der Cache gemeinsam genutzt wird, erlaubt er der Anwendungsebene auch, horizontal zu skalieren, ohne dass jede neue Instanz eine proportionale neue Last zur Datenbank hinzufügt — eine Eigenschaft, die individuelle, instanzbezogene Caches nicht bieten können. Der Tradeoff ist, dass Cache-Invalidierung nun aktiv verwaltet werden muss, da das Servieren veralteter Daten zu einem echten Risiko wird, sobald Datenänderungen nicht zuverlässig im Cache widergespiegelt werden.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Identifizieren Sie Daten, die häufig gelesen, aber selten geändert werden: Referenzdaten, Sitzungszustand, berechnete Ergebnisse
- Setzen Sie einen verteilten Cache (Redis, Memcached, Hazelcast) ein, zugänglich für alle Anwendungsinstanzen
- Implementieren Sie das Cache-Aside-Muster: Prüfen Sie den Cache vor der Datenbankabfrage, füllen Sie ihn bei einem Cache-Miss
- Definieren Sie angemessene TTL-Werte (Time-to-Live) basierend darauf, wie veraltet die Daten für jeden Anwendungsfall sein können
- Implementieren Sie Cache-Invalidierungsstrategien, die zu den Konsistenzanforderungen passen: ereignisgesteuert, TTL-basiert oder Write-Through
- Überwachen Sie Cache-Trefferraten und Eviction-Raten, um Cache-Größe und TTL-Konfigurationen abzustimmen
- Fügen Sie Circuit Breaker hinzu, damit die Anwendung graziös degradiert, wenn der Cache nicht verfügbar wird

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Reduziert die Datenbanklast, indem häufig abgerufene Daten aus dem Speicher bedient werden
- Bietet konsistente Performance über mehrere Anwendungsinstanzen hinweg durch einen gemeinsamen Cache
- Ermöglicht horizontale Skalierung der Anwendungsebene, ohne die Datenbanklast proportional zu erhöhen
- Verbessert die Antwortzeiten für gecachte Daten dramatisch

**Kosten und Risiken:**
- Cache-Invalidierung ist notorisch schwierig und kann dazu führen, dass veraltete Daten serviert werden
- Fügt eine Infrastrukturabhängigkeit hinzu: Cache-Ausfälle können auf die Datenbank kaskadieren, wenn nicht behandelt
- Speicherkosten für das Caching großer Datensätze können erheblich sein
- Verteilter Cache fügt im Vergleich zu lokalen In-Process-Caches Netzwerklatenz hinzu

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Eine Legacy-Reisebuchungsplattform hatte Produktkatalogabfragen, die die Datenbank bei jeder Suchanfrage trafen, wobei dieselben Zieldaten Tausende Male pro Stunde abgerufen wurden. Während der Traffic wuchs, wurde die Datenbank während Spitzenbuchungssaisons zum Engpass. Das Team setzte einen Redis-Cluster ein und implementierte ein Cache-Aside-Muster für Zieldaten, Hoteldetails und Preisstufen mit einer 15-Minuten-TTL. Die Cache-Trefferraten überstiegen 95 Prozent für Katalogabfragen, was das Datenbankabfragevolumen um eine Größenordnung reduzierte. Performance-Probleme in Spitzensaisons verschwanden, und das Team konnte ein teures Datenbank-Hardware-Upgrade aufschieben.
