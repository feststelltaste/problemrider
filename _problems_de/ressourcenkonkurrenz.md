---
title: Ressourcenkonkurrenz
description: Der Server ist überlastet, und die Anwendung konkurriert um begrenzte
  Ressourcen wie CPU, Speicher oder I/O.
category:
- Code
- Performance
related_problems:
- slug: excessive-disk-io
  similarity: 0.7
- slug: high-database-resource-utilization
  similarity: 0.65
- slug: lock-contention
  similarity: 0.65
- slug: high-connection-count
  similarity: 0.65
- slug: slow-database-queries
  similarity: 0.65
- slug: slow-application-performance
  similarity: 0.65
solutions:
- backpressure
- concurrency-control
- elastic-scaling
- resource-pooling
- resource-usage-optimization
- serialization-optimization
- bulkhead
- elastic-resource-utilization
- timeout-management
- virtualization
layout: problem
lang: de
en_slug: resource-contention
---

## Description
Ressourcenkonkurrenz tritt auf, wenn mehrere Prozesse oder Threads um dieselben begrenzten Ressourcen konkurrieren, wie CPU, Speicher oder I/O. Diese Konkurrenz kann zu Performance-Verschlechterung führen, während Prozesse gezwungen sind, darauf zu warten, dass Ressourcen verfügbar werden. In schweren Fällen kann sie zu Deadlocks oder anderen Formen der Systeminstabilität führen. Das Verständnis und die Verwaltung von Ressourcenkonkurrenz ist ein Schlüsselaspekt beim Bau skalierbarer und performanter Systeme.

## Indicators ⟡
- Der Server ist langsam, auch wenn es keine offensichtlichen Anzeichen hoher CPU-Nutzung gibt.
- Der Server nutzt viel Festplatten-I/O, auch wenn keine hohe Datenbanklast vorliegt.
- Der Server ist nicht reaktionsfähig oder träge.
- Sie erhalten Beschwerden von Nutzern über langsame Performance.

## Symptoms ▲

- [Langsame Antwortzeiten für Listen](langsame-antwortzeiten-fuer-listen.md)
<br/>  Ressourcenkonkurrenz verursacht, dass sich datenintensive Operationen wie Listenabfragen erheblich verlangsamen, während Prozesse um I/O und CPU konkurrieren.
- [Schleichende Performance-Verschlechterung](schleichende-performance-verschlechterung.md)
<br/>  Während sich die Ressourcenkonkurrenz über die Zeit verstärkt, verschlechtert sich die Gesamtsystemperformance stetig.
- [Kaskadierende Ausfälle](kaskadierende-ausfaelle.md)
<br/>  Wenn Ressourcen erschöpft sind, beginnen Komponenten in Sequenz zu scheitern, weil sie die Ressourcen, die sie zum Funktionieren benötigen, nicht erhalten können.
- [Memory Swapping](memory-swapping.md)
<br/>  Starke Speicherkonkurrenz zwingt das Betriebssystem, Speicher auf die Festplatte auszulagern, was die Systemperformance dramatisch verschlechtert.
- [Unvorhersehbares Systemverhalten](unvorhersehbares-systemverhalten.md)
<br/>  Ressourcenkonkurrenz verursacht timing-abhängiges Verhalten, bei dem die Systemperformance unvorhersehbar basierend auf gleichzeitigen Lastmustern variiert.

## Causes ▼

- [Fehler bei der Ressourcenzuweisung](fehler-bei-der-ressourcenzuweisung.md)
<br/>  Lecke Ressourcen verringern die verfügbare Kapazität, was die Konkurrenz unter Prozessen um die verbleibenden Ressourcen verstärkt.
- [N+1-Abfrageproblem](n-plus-1-abfrageproblem.md)
<br/>  Exzessive Datenbankabfragen aus N+1-Mustern verbrauchen Datenbankressourcen und schaffen I/O-Konkurrenz.
- [Kapazitäts-Fehlanpassung](kapazitaets-fehlanpassung.md)
<br/>  Infrastruktur, die nicht zu tatsächlichen Nachfragemustern passt, führt zu Ressourcenkonkurrenz während Spitzennutzungszeiten.
- [Ineffizienzen bei der Skalierung](ineffizienzen-bei-der-skalierung.md)
<br/>  Die Unfähigkeit, Komponenten unabhängig zu skalieren, bedeutet, dass Engpasskomponenten Ressourcenkonkurrenz für das gesamte System schaffen.
- [Overhead durch atomare Operationen](overhead-durch-atomare-operationen.md)
<br/>  Mehrere Threads, die um atomare Variablen konkurrieren, schaffen Ressourcenkonkurrenz auf CPU-Ebene durch Cache-Kohärenz-Verkehr.

## Detection Methods ○

- **System-Monitoring-Werkzeuge:** Nutzung von Werkzeugen wie `top`, `htop`, `vmstat`, `iostat` (Linux) oder Task-Manager (Windows) zur Überwachung von CPU-, Speicher- und I/O-Nutzung.
- **Application Performance Monitoring (APM):** APM-Werkzeuge können oft Ressourcennutzung auf Anwendungsebene zeigen und helfen zu identifizieren, welche Teile der Anwendung ressourcenintensiv sind.
- **Lasttests:** Simulation hoher Last zur Identifikation von Ressourcenengpässen und Konkurrenzpunkten.
- **Profiling:** Nutzung von Profiling-Werkzeugen zur Identifikation von Codeabschnitten, die exzessiv CPU oder Speicher verbrauchen.

## Examples
Ein Webserver erlebt während Spitzenzeiten langsame Antwortzeiten. Monitoring offenbart, dass die CPU-Auslastung konsequent bei 100 % liegt. Dies deutet darauf hin, dass der Server nicht genug CPU-Kapazität hat, um eingehende Anfragen zu handhaben. In einem anderen Fall erlebt ein Datenbankserver hohe I/O-Wartezeiten. Untersuchung offenbart, dass mehrere Anwendungen gleichzeitig große, nicht indizierte Abfragen durchführen, was zu Festplattenkonkurrenz führt. Dieses Problem ist häufig in Systemen, die nicht ordentlich skaliert sind oder wo sich Ressourcennutzungsmuster über die Zeit geändert haben. Es erfordert oft eine Kombination aus Kapazitätsplanung, Code-Optimierung und Infrastruktur-Tuning zur Lösung.
