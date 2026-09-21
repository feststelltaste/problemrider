---
title: Lock Contention
description: Mehrere Threads konkurrieren um dieselben Locks, was Threads blockiert
  und die Effizienz paralleler Ausführung verringert.
category:
- Code
- Performance
related_problems:
- slug: resource-contention
  similarity: 0.65
- slug: race-conditions
  similarity: 0.65
- slug: deadlock-conditions
  similarity: 0.6
- slug: thread-pool-exhaustion
  similarity: 0.6
- slug: false-sharing
  similarity: 0.6
- slug: memory-barrier-inefficiency
  similarity: 0.6
solutions:
- query-optimization-process
- concurrency-control
- profiling
- transactions
- performance-measurements
- read-replicas
- asynchronous-processing
- monitoring
- stress-testing
- index-lifecycle-management
layout: problem
lang: de
en_slug: lock-contention
---

## Description

Lock Contention tritt auf, wenn mehrere Threads häufig um dieselben Synchronisationsprimitive (Mutexe, Locks, Semaphoren) konkurrieren, was dazu führt, dass Threads blockieren, während sie darauf warten, dass Locks verfügbar werden. Dies verringert die Wirksamkeit paralleler Ausführung, da Threads Zeit mit Warten statt mit nützlicher Arbeit verbringen, und kann zu einer Performance-Verschlechterung führen, die in schweren Fällen schlimmer ist als bei Single-Thread-Ausführung.

## Indicators ⟡

- Multithreaded-Anwendungen performen schlechter als Single-Thread-Äquivalente
- Thread-Profiling zeigt erhebliche Zeit, die mit Warten auf Locks verbracht wird
- Lock-Erwerbszeiten steigen bei höheren Thread-Zahlen
- System-Monitoring zeigt Threads in blockierten oder wartenden Zuständen
- CPU-Auslastung ist niedrig trotz hoher Thread-Aktivität

## Symptoms ▲

- [Langsame Anwendungsperformance](langsame-anwendungsperformance.md)
<br/>  Lock Contention führt dazu, dass Threads blockieren, statt nützliche Arbeit zu verrichten, was die Antwortzeiten und den Durchsatz der Anwendung direkt verschlechtert.
- [Erschöpfung des Thread-Pools](erschoepfung-des-thread-pools.md)
<br/>  Threads, die beim Warten auf umkämpfte Locks blockiert sind, bleiben belegt und erschöpfen schließlich den verfügbaren Thread-Pool.
- [Deadlock-Zustände](deadlock-zustaende.md)
<br/>  Hohe Lock Contention erhöht die Wahrscheinlichkeit zirkulärer Lock-Abhängigkeiten, was zu Deadlocks führt.
- [Ineffizienzen bei der Skalierung](ineffizienzen-bei-der-skalierung.md)
<br/>  Das Hinzufügen weiterer Threads oder Kerne bringt abnehmende oder negative Erträge, wenn sie alle um dieselben Locks konkurrieren.
- [Ressourcenkonkurrenz](ressourcenkonkurrenz.md)
<br/>  Lock Contention ist eine direkte Form der Ressourcenkonkurrenz, bei der Threads um Synchronisationsprimitive konkurrieren, statt produktive Arbeit zu verrichten.
- [Overhead durch atomare Operationen](overhead-durch-atomare-operationen.md)
<br/>  Entwickler, die versuchen, Lock Contention zu vermeiden, nutzen möglicherweise übermäßig atomare Operationen, was den Engpass verschiebt, statt ihn zu beseitigen.

## Causes ▼

- [God-Object-Antipattern](god-object-antipattern.md)
<br/>  Ein God Object, das Zustand zentralisiert, zwingt alle Threads, sich über einen einzigen Lock zu synchronisieren, der diesen gemeinsamen Zustand schützt.
- [Hohe Kopplung und geringe Kohäsion](hohe-kopplung-und-geringe-kohaesion.md)
<br/>  Eng gekoppelte Komponenten, die veränderlichen Zustand teilen, erfordern grobkörniges Locking, was die Contention erhöht.
- [Globaler Zustand und Nebeneffekte](globaler-zustand-und-nebeneffekte.md)
<br/>  Globaler veränderlicher Zustand erfordert Locks für Thread-Sicherheit, und weithin zugängliche globale Variablen werden zu natürlichen Contention-Hotspots.

## Detection Methods ○

- **Lock-Profiling:** Nutzung von Profiling-Werkzeugen, die Lock Contention und Wartezeiten identifizieren können
- **Thread-Zustandsanalyse:** Überwachung von Thread-Zuständen zur Identifikation von Blockierungsmustern
- **Performance-Skalierungstests:** Testen der Performance mit variierenden Thread-Zahlen zur Identifikation von Contention
- **Lock-Instrumentierung:** Hinzufügen von Instrumentierung zur Messung von Lock-Haltezeiten und Wartezeiten
- **Concurrency-Profiler:** Nutzung spezialisierter Werkzeuge, die zur Erkennung von Synchronisationsengpässen entwickelt wurden
- **CPU-Auslastungs-Überwachung:** Analyse von CPU-Nutzungsmustern während Szenarien hoher Contention

## Examples

Ein Webserver nutzt einen einzigen globalen Lock, um den Zugriff auf einen gemeinsamen Cache zu schützen, der Benutzersitzungsdaten speichert. Unter hoher Last konkurrieren Hunderte von Request-Threads um diesen Lock, was dazu führt, dass die meisten Threads blockieren, während sie auf Cache-Zugriff warten. Das Ergebnis ist, dass ein 16-Kern-Server schlechter performt als eine Single-Thread-Version, weil Threads mehr Zeit mit Warten auf den Lock verbringen als mit der Verarbeitung von Anfragen. Ein weiteres Beispiel betrifft ein paralleles Datenverarbeitungssystem, bei dem mehrere Worker-Threads Elemente aus einer gemeinsamen Warteschlange verarbeiten, die durch einen einzigen Mutex geschützt wird. Mit steigender Anzahl von Worker-Threads sinkt die Performance, weil Threads die meiste Zeit damit verbringen, auf Zugriff zur Warteschlange zu warten, statt Datenelemente zu verarbeiten.
