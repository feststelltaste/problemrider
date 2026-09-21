---
title: Speicherlecks
description: Anwendungen geben nicht mehr benötigten Speicher nicht frei, was zu
  allmählichem Speicherverbrauch und schließlich Performance-Verschlechterung oder
  Abstürzen führt.
category:
- Code
- Performance
related_problems:
- slug: slow-database-queries
  similarity: 0.7
- slug: slow-application-performance
  similarity: 0.7
- slug: unreleased-resources
  similarity: 0.7
- slug: excessive-disk-io
  similarity: 0.65
- slug: inefficient-code
  similarity: 0.65
- slug: high-database-resource-utilization
  similarity: 0.65
solutions:
- concurrency-control
- memory-management-optimization
- profiling
- resource-pooling
- resource-usage-optimization
- lazy-evaluation
- lazy-loading
- monitoring-system-utilization
- pagination
- probabilistic-data-structures
- virtualized-lists
- dynamic-code-analysis
layout: problem
lang: de
en_slug: memory-leaks
---

## Description
Ein Speicherleck ist eine Art Ressourcenleck, das auftritt, wenn ein Computerprogramm Speicherzuweisungen so inkorrekt verwaltet, dass nicht mehr benötigter Speicher nicht freigegeben wird. Über die Zeit können diese Lecks eine erhebliche Menge an Speicher verbrauchen, was zu einer Performance-Verschlechterung und schließlich zu einem Absturz der Anwendung oder des gesamten Systems führt. Speicherlecks sind ein häufiges Problem in Sprachen, die manuelles Speichermanagement erfordern, können aber auch in Sprachen mit automatischem Speichermanagement auftreten, wenn Objekte unbeabsichtigt am Leben gehalten werden.

## Indicators ⟡
- Die Speichernutzung der Anwendung steigt ständig, selbst wenn sie nicht stark belastet ist.
- Die Anwendung ist langsam, und Sie vermuten, dass dies auf ein Speicherleck zurückzuführen ist.
- Die Anwendung stürzt mit Out-of-Memory-Fehlern ab.
- Sie erhalten Beschwerden von Nutzern über langsame Performance.

## Symptoms ▲

- [Schleichende Performance-Verschlechterung](schleichende-performance-verschlechterung.md)
<br/>  Während sich durchgesickerter Speicher über die Zeit anhäuft, nutzt die Anwendung mehr Ressourcen und performt zunehmend schlechter.
- [Speicherfragmentierung](speicherfragmentierung.md)
<br/>  Durchgesickerte Speicherblöcke, verstreut über den Heap, tragen zur Fragmentierung bei und verhindern effiziente Zuweisung.
- [Memory Swapping](memory-swapping.md)
<br/>  Wachsender Speicherverbrauch durch Lecks erschöpft schließlich den physischen RAM, was das Betriebssystem zwingt, Festplatten-Swap-Speicher zu nutzen.
- [Hoher Ressourcenverbrauch auf Client-Seite](hoher-ressourcenverbrauch-auf-client-seite.md)
<br/>  Speicherlecks in clientseitigen Anwendungen verursachen exzessiven Ressourcenverbrauch auf Nutzergeräten, was deren Erfahrung verschlechtert.
- [Fehler bei der Ressourcenzuweisung](fehler-bei-der-ressourcenzuweisung.md)
<br/>  Während durchgesickerter Speicher verfügbare Ressourcen verbraucht, schlagen neue Zuweisungsanfragen schließlich aufgrund von Speichererschöpfung fehl.

## Causes ▼

- [Nicht freigegebene Ressourcen](nicht-freigegebene-ressourcen.md)
<br/>  Das Versäumnis, Ressourcen wie Event-Listener, Dateihandles oder Datenbankverbindungen ordentlich freizugeben, ist eine direkte Ursache für Speicherlecks.
- [Übermäßige Objektallokation](uebermaessige-objektallokation.md)
<br/>  Die Erstellung vieler Objekte ohne ordentliches Lebenszyklusmanagement erhöht die Wahrscheinlichkeit, dass manche nicht ordentlich freigegeben werden.

## Detection Methods ○

- **Speicherprofiler:** Nutzung sprachspezifischer Speicherprofiling-Werkzeuge (z. B. Java VisualVM, .NET Memory Profiler, Chrome DevTools Memory-Tab, Valgrind für C/C++) zur Analyse von Heap-Dumps und Nachverfolgung von Objektzuweisungen und -referenzen.
- **System-Monitoring-Werkzeuge:** Überwachung der Prozessspeichernutzung der Anwendung über die Zeit mittels Betriebssystem-Werkzeugen (`top`, `htop`, Task-Manager) oder APM-Werkzeugen.
- **Lasttest mit langer Dauer:** Durchführung von Lasttests über längere Zeiträume zur Beobachtung von Speicherwachstumsmustern.
- **Code-Review:** Suche nach häufigen Speicherleck-Antipatterns, besonders in Bereichen, die Event-Listener, Ressourcenmanagement oder globalen Zustand betreffen.
- **Automatisierte Tests:** Integration von Speichernutzungsprüfungen in automatisierte Tests, besonders für lang laufende Prozesse.

## Examples
Ein lang laufender Backend-Service, der Kundenbestellungen verarbeitet, verbraucht allmählich immer mehr RAM. Nach mehreren Tagen stürzt er ab. Profiling zeigt, dass eine `HashMap`, die zum Caching von Kundendaten genutzt wird, nie geleert wird, und neue Kundeneinträge werden kontinuierlich hinzugefügt, was zu unbegrenztem Speicherwachstum führt. In einem anderen Fall erlaubt eine Single-Page-Application (SPA) Nutzern, zwischen verschiedenen Ansichten zu navigieren. Jedes Mal, wenn ein Nutzer eine bestimmte Ansicht besucht, werden neue Event-Listener an DOM-Elemente angehängt, aber die alten Listener werden nie entfernt, wenn die Ansicht zerstört wird. Über die Zeit häuft dies Tausende nicht referenzierter DOM-Knoten und Listener an, was zu einem erheblichen Speicherleck und Browser-Verlangsamung führt. Speicherlecks sind besonders problematisch in lang laufenden Anwendungen, Services oder eingebetteten Systemen. Sie können schwierig zu diagnostizieren sein, weil ihre Symptome oft allmählich auftreten und sich möglicherweise erst nach längeren Betriebszeiträumen zeigen.
