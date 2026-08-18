---
title: Memory Swapping
description: Dem Datenbankserver geht der physische Speicher aus, und er beginnt,
  Festplatten-Swap-Speicher zu nutzen, was die Performance dramatisch verlangsamt.
category:
- Performance
related_problems:
- slug: virtual-memory-thrashing
  similarity: 0.7
- slug: resource-contention
  similarity: 0.6
- slug: memory-fragmentation
  similarity: 0.6
- slug: memory-leaks
  similarity: 0.6
- slug: excessive-disk-io
  similarity: 0.55
- slug: slow-database-queries
  similarity: 0.55
solutions:
- backpressure
- caching-strategy
- memory-management-optimization
- resource-pooling
- resource-usage-optimization
- monitoring-system-utilization
- capacity-planning
- profiling
- performance-measurements
- elastic-scaling
layout: problem
lang: de
en_slug: memory-swapping
---

## Description
Memory Swapping ist ein Prozess, bei dem das Betriebssystem einen Speicherblock (eine „Seite") von RAM auf die Festplatte verschiebt, um RAM für andere Prozesse freizugeben. Während dies dem System erlaubt, weiterzufunktionieren, wenn wenig Speicher verfügbar ist, geht es mit erheblichen Performance-Kosten einher, da der Zugriff auf Daten von der Festplatte viel langsamer ist als der Zugriff aus dem RAM. Häufiges Memory Swapping ist ein starker Indikator dafür, dass ein System nicht genug physischen Speicher für seine Arbeitslast hat, und es kann zu einem dramatischen Rückgang der Anwendungsperformance führen.

## Indicators ⟡
- Der Server ist langsam, auch wenn es keine offensichtlichen Anzeichen hoher CPU-Nutzung gibt.
- Der Server nutzt viel Festplatten-I/O, auch wenn keine hohe Datenbanklast vorliegt.
- Der Server ist nicht reaktionsfähig oder träge.
- Sie erhalten Beschwerden von Nutzern über langsame Performance.

## Symptoms ▲

- [Langsame Anwendungsperformance](langsame-anwendungsperformance.md)
<br/>  Festplattenbasiertes Swapping ist um Größenordnungen langsamer als RAM-Zugriff, was dramatische Anwendungsverlangsamungen verursacht.
- [Übermäßige Festplatten-I/O](uebermaessige-festplatten-io.md)
<br/>  Memory Swapping erzeugt erhebliche Festplatten-I/O, während Seiten zwischen RAM und Festplatten-Swap-Speicher verschoben werden.
- [Performance-Probleme bei Datenbankabfragen](performance-probleme-bei-datenbankabfragen.md)
<br/>  Wenn der Speicher des Datenbankservers auf die Festplatte ausgelagert wird, wird die Abfrageausführung extrem langsam, da Daten von der Festplatte statt aus dem Speicher gelesen werden müssen.
- [Service-Timeouts](service-timeouts.md)
<br/>  Die dramatische Verlangsamung durch Swapping führt dazu, dass Services ihre Timeout-Schwellenwerte überschreiten, was zu kaskadierenden Fehlschlägen führt.

## Causes ▼

- [Speicherlecks](speicherlecks.md)
<br/>  Speicherlecks verbrauchen allmählich physischen RAM, bis das System zum Swapping gezwungen wird, wie im Beispiel einer Java-Anwendung veranschaulicht, die MySQL ins Swapping treibt.
- [Hohe Datenbank-Ressourcennutzung](hohe-datenbank-ressourcennutzung.md)
<br/>  Überlastete Datenbankserver, die exzessiven Speicher verbrauchen, treiben das System über die physischen RAM-Grenzen hinaus ins Swapping.
- [Ressourcenkonkurrenz](ressourcenkonkurrenz.md)
<br/>  Mehrere Prozesse, die um begrenzten physischen Speicher konkurrieren, zwingen das Betriebssystem, weniger aktive Seiten auf die Festplatte auszulagern.

## Detection Methods ○

- **System-Monitoring-Werkzeuge:** Nutzung von `free -h`, `vmstat`, `top` oder `htop` (Linux) zur Beobachtung der `swap`-Nutzung und der `si`/`so`-Raten (Swap-in/-out).
- **Datenbank-Monitoring-Werkzeuge:** Viele datenbankspezifische Monitoring-Werkzeuge melden Speichernutzung und Swap-Aktivität.
- **Cloud-Anbieter-Metriken:** Bei Nutzung einer cloud-verwalteten Datenbank Überprüfung der Metriken des Cloud-Anbieters auf Swap-Nutzung.
- **Alerting:** Einrichtung von Alarmen für hohe Swap-Nutzung oder hohe I/O-Wartezeiten auf Datenbankservern.

## Examples
Ein PostgreSQL-Datenbankserver, ursprünglich mit 8 GB RAM ausgestattet, beginnt nach einem Jahr Betrieb, schwere Verlangsamungen zu erleben. Untersuchung zeigt, dass die `shared_buffers`-Einstellung auf 6 GB erhöht wurde und der `work_mem` für viele gleichzeitige Abfragen jetzt den verbleibenden physischen Speicher übersteigt, was das System zu starkem Swapping zwingt. In einem anderen Fall hat eine Java-Anwendung, die auf demselben Server wie eine MySQL-Datenbank läuft, ein Speicherleck. Über mehrere Tage verbraucht die Java-Anwendung immer mehr RAM und treibt die MySQL-Datenbank schließlich in starkes Swapping, was zu Anwendungsausfällen führt. Dieses Problem ist besonders heimtückisch, weil es sich allmählich entwickeln kann, während Datenvolumen wachsen oder die Anwendungsnutzung zunimmt. Es deutet oft auf einen fundamentalen Ressourcenengpass hin, der durch Hinzufügen von mehr RAM, Optimierung der Datenbankkonfiguration oder Reduzierung des Speicherverbrauchs behoben werden muss.
