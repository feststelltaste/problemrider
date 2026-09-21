---
title: Speicherfragmentierung
description: Verfügbarer Speicher wird in kleine, nicht zusammenhängende Blöcke aufgeteilt,
  was die Zuweisung größerer Objekte trotz ausreichenden gesamten freien Speichers
  verhindert.
category:
- Code
- Performance
related_problems:
- slug: excessive-object-allocation
  similarity: 0.65
- slug: garbage-collection-pressure
  similarity: 0.6
- slug: memory-swapping
  similarity: 0.6
- slug: resource-allocation-failures
  similarity: 0.6
- slug: index-fragmentation
  similarity: 0.6
- slug: alignment-and-padding-issues
  similarity: 0.6
solutions:
- memory-management-optimization
- profiling
- resource-pooling
- resource-usage-optimization
- memory-hierarchy
- performance-measurements
- monitoring-system-utilization
- load-testing
- continuous-performance-monitoring
layout: problem
lang: de
en_slug: memory-fragmentation
---

## Description

Speicherfragmentierung tritt auf, wenn der verfügbare Speicherplatz aufgrund wiederholter Zuweisungs- und Freigabemuster in kleine, nicht zusammenhängende Blöcke aufgeteilt wird. Selbst wenn ausreichend Gesamtspeicher verfügbar ist, können Anwendungen bei der Zuweisung größerer zusammenhängender Blöcke scheitern, was zu Zuweisungsfehlern führt oder die Nutzung langsamerer, nicht zusammenhängender Speicherzuweisungsstrategien erzwingt. Dieses Problem ist besonders schwerwiegend in lang laufenden Anwendungen mit dynamischen Speicherzuweisungsmustern.

## Indicators ⟡

- Speicherzuweisungsfehler trotz ausreichendem gesamten freien Speicher
- Zunehmende Zuweisungszeit für größere Speicherblöcke
- Der Speicherallokator meldet hohe Fragmentierungsgrade
- Performance-Verschlechterung in Anwendungen, die große zusammenhängende Zuweisungen benötigen
- Die Häufigkeit von Heap-Kompaktierung oder Garbage Collection steigt erheblich

## Symptoms ▲

- [Fehler bei der Ressourcenzuweisung](fehler-bei-der-ressourcenzuweisung.md)
<br/>  Fragmentierter Speicher verhindert die Zuweisung großer zusammenhängender Blöcke trotz ausreichendem gesamten freien Speicher, was Zuweisungsfehler verursacht.
- [Langsame Anwendungsperformance](langsame-anwendungsperformance.md)
<br/>  Fragmentierter Speicher erhöht die Zuweisungszeit und verringert die Cache-Effizienz, was die Gesamtanwendungsperformance verschlechtert.
- [Virtual-Memory-Thrashing](virtual-memory-thrashing.md)
<br/>  Speicherfragmentierung zwingt das Betriebssystem, mehr virtuelle Speicherseiten zu nutzen, was Seitenfehler erhöht und potenziell Thrashing verursacht.
- [Schleichende Performance-Verschlechterung](schleichende-performance-verschlechterung.md)
<br/>  Fragmentierung verschlimmert sich über die Zeit in lang laufenden Anwendungen, was einen stetigen Rückgang der Speicherzuweisungs- und Zugriffsperformance verursacht.

## Causes ▼

- [Speicherlecks](speicherlecks.md)
<br/>  Speicherlecks hinterlassen zugewiesene Blöcke verstreut über den Heap, was zu Fragmentierungsmustern beiträgt, während freigegebener umliegender Speicher nicht zusammenhängend wird.
- [Übermäßige Objektallokation](uebermaessige-objektallokation.md)
<br/>  Häufige Zuweisung und Freigabe vieler Objekte unterschiedlicher Größe ist ein primärer Treiber der Heap-Fragmentierung.

## Detection Methods ○

- **Speicherprofiling-Werkzeuge:** Nutzung von Heap-Analysewerkzeugen zur Visualisierung von Speicherlayout und Fragmentierungsgraden
- **Zuweisungsmusteranalyse:** Überwachung von Zuweisungsgrößen und -mustern zur Identifikation von Fragmentierungsursachen
- **Speichermanager-Statistiken:** Nachverfolgung von Fragmentierungsmetriken, die von Speichermanagementsystemen bereitgestellt werden
- **Performance-Überwachung:** Überwachung von Zuweisungszeitzunahmen, die mit Fragmentierung korrelieren
- **Virtual-Memory-Analyse:** Analyse von Seitenfehlermustern und Virtual-Memory-Nutzung
- **Heap-Dump-Analyse:** Untersuchung von Heap-Snapshots zur Identifikation von Fragmentierungsmustern

## Examples

Eine Serveranwendung weist während ihrer Laufzeit verschieden große Puffer für Netzwerkoperationen zu. Kleine Puffer werden häufig zugewiesen und freigegeben, während gelegentliche große Puffer für Dateiübertragungen über längere Zeiträume zugewiesen bleiben. Über die Zeit wird der freie Speicher stark fragmentiert mit kleinen Lücken zwischen langlebigen großen Puffern. Wenn die Anwendung einen 1-MB-Puffer für eine große Dateiübertragung zuweisen muss, schlägt die Zuweisung trotz 10 MB gesamtem freiem Speicher, verteilt über Tausende kleine Fragmente, fehl. Ein weiteres Beispiel betrifft eine Echtzeit-Grafikanwendung, die Texturdaten und Vertex-Puffer unterschiedlicher Größe zuweist. Schlechte Zuweisungsreihenfolge führt dazu, dass große Texturzuweisungen den Heap fragmentieren, was nachfolgende Zuweisungen für Animationsdaten fehlschlagen lässt, obwohl insgesamt ausreichend Speicher vorhanden ist.
