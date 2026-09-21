---
title: False Sharing
description: Mehrere CPU-Kerne greifen auf unterschiedliche Variablen zu, die sich
  auf derselben Cache-Line befinden, was unnötigen Cache-Kohärenz-Verkehr und Performance-Einbußen
  verursacht.
category:
- Code
- Performance
related_problems:
- slug: lock-contention
  similarity: 0.6
- slug: memory-barrier-inefficiency
  similarity: 0.6
- slug: atomic-operation-overhead
  similarity: 0.6
- slug: race-conditions
  similarity: 0.6
- slug: data-structure-cache-inefficiency
  similarity: 0.55
- slug: alignment-and-padding-issues
  similarity: 0.55
solutions:
- profiling
- performance-measurements
- concurrency-control
- memory-hierarchy
- parallelization
- efficient-algorithms
- load-testing
- continuous-performance-monitoring
- static-code-analysis
- performance-modeling
layout: problem
lang: de
en_slug: false-sharing
---

## Description

False Sharing entsteht, wenn mehrere CPU-Kerne auf unterschiedliche Datenelemente zugreifen, die zufällig auf derselben Cache-Line liegen, was dazu führt, dass das Cache-Kohärenz-Protokoll Cache-Lines zwischen Kernen invalidiert und überträgt, obwohl die Kerne logisch keine Daten teilen. Dies erzeugt unnötigen Speicherverkehr und Performance-Einbußen in Multithreaded-Anwendungen, während Kerne um Cache-Lines konkurrieren, die unzusammenhängende Daten enthalten.

## Indicators ⟡

- Die Multithreaded-Performance skaliert schlecht mit steigender Threadanzahl
- Der Cache-Kohärenz-Verkehr ist hoch im Verhältnis zu den tatsächlichen Anforderungen an Daten-Sharing
- Die Performance verschlechtert sich, wenn Threads auf scheinbar unabhängige Datenstrukturen zugreifen
- Profiling zeigt übermäßige Cache-Line-Übertragungen zwischen CPU-Kernen
- Die Single-Thread-Performance ist gut, aber die Multithreaded-Performance ist schlecht

## Symptoms ▲

- [Langsame Anwendungsperformance](langsame-anwendungsperformance.md)
<br/>  False Sharing verursacht unnötigen Cache-Kohärenz-Verkehr, der die Multithreaded-Anwendungsperformance verschlechtert, was die Anwendung merklich langsamer macht.
- [Ressourcenkonkurrenz](ressourcenkonkurrenz.md)
<br/>  Kerne konkurrieren um Cache-Lines, die unzusammenhängende Daten enthalten, was künstliche Ressourcenkonkurrenz auf Hardware-Ebene schafft.
- [Ineffizienzen bei der Skalierung](ineffizienzen-bei-der-skalierung.md)
<br/>  False Sharing verhindert lineare Performance-Skalierung mit zusätzlichen Threads oder Kernen, da mehr Parallelität den Cache-Kohärenz-Overhead erhöht.
- [Schleichende Performance-Verschlechterung](schleichende-performance-verschlechterung.md)
<br/>  Während im Laufe der Zeit mehr Threads hinzugefügt werden, summieren sich False-Sharing-Effekte, was zu progressiv schlechterer Performance-Verschlechterung führt.
- [Overhead durch atomare Operationen](overhead-durch-atomare-operationen.md)
<br/>  False Sharing führt dazu, dass atomare Operationen auf unabhängigen Daten um dieselbe Cache-Line konkurrieren, was den Overhead verstärkt.

## Causes ▼

- [Ausrichtungs- und Padding-Probleme](ausrichtungs-und-padding-probleme.md)
<br/>  Schlechte Datenstrukturausrichtung platziert unabhängige Variablen auf derselben Cache-Line, was direkt False Sharing zwischen Kernen verursacht.
- [Cache-Ineffizienz von Datenstrukturen](cache-ineffizienz-von-datenstrukturen.md)
<br/>  Datenstrukturen, die ohne Berücksichtigung von Cache-Line-Grenzen organisiert sind, führen dazu, dass unzusammenhängende Daten sich Cache-Lines teilen.
- [Unerfahrene Entwickler](unerfahrene-entwickler.md)
<br/>  Entwickler ohne Kenntnisse der CPU-Cache-Architektur schaffen möglicherweise unwissentlich Datenlayouts, die False Sharing verursachen.

## Detection Methods ○

- **Cache-Performance-Profiling:** Nutzung von Profilern, die Cache-Line-Konkurrenz und False Sharing erkennen können
- **Hardware-Performance-Zähler:** Überwachung von Cache-Kohärenz-Ereignissen und Inter-Core-Verkehr
- **Speicherlayout-Analyse:** Untersuchung von Datenstrukturlayouts und Speicherausrichtung
- **Thread-Affinitätstests:** Testen der Performance mit unterschiedlichen Thread-zu-Kern-Zuweisungen
- **Padding-Experimente:** Hinzufügen von Padding zwischen Datenstrukturen, um auf False-Sharing-Effekte zu testen
- **Cache-Line-Analysewerkzeuge:** Nutzung spezialisierter Werkzeuge, die False-Sharing-Muster erkennen

## Examples

Eine Multithreaded-Zähleranwendung hat ein Array von Zählervariablen, eine pro Thread, um Synchronisation zu vermeiden. Wenn jedoch mehrere Zähler auf dieselbe 64-Byte-Cache-Line passen, verursachen Threads, die unterschiedliche Zähler aktualisieren, ein Cache-Line-Ping-Pong zwischen CPU-Kernen. Jede Aktualisierung durch einen Thread invalidiert die Cache-Line für andere Threads, was sie zwingt, die gesamte Cache-Line neu zu laden, obwohl sie völlig unterschiedliche Zähler aktualisieren. Ein weiteres Beispiel betrifft ein Producer-Consumer-System, bei dem Producer- und Consumer-Threads jeweils ihre eigenen Indexvariablen (Head und Tail) für einen zirkulären Puffer haben. Wenn diese Indizes benachbart im Speicher gespeichert werden, verursacht die Aktualisierung eines Index die Invalidierung der Cache-Line des anderen Threads, was künstliche Konkurrenz für logisch unabhängige Daten schafft.
