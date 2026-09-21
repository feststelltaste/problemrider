---
title: Zirkuläre Referenzen
description: Zwei oder mehr Objekte referenzieren sich gegenseitig auf eine Weise,
  die Garbage Collection verhindert, was zu Speicherlecks und Ressourcenerschöpfung
  führt.
category:
- Code
- Performance
related_problems:
- slug: circular-dependency-problems
  similarity: 0.65
- slug: unreleased-resources
  similarity: 0.6
- slug: improper-event-listener-management
  similarity: 0.55
- slug: resource-allocation-failures
  similarity: 0.55
- slug: excessive-object-allocation
  similarity: 0.55
- slug: garbage-collection-pressure
  similarity: 0.5
solutions:
- design-by-contract
- loose-coupling
- separation-of-concerns
- solid-principles
- dependency-breaking-techniques
- incremental-refactoring
- static-analysis-and-linting
- code-reviews
- modularization-and-bounded-contexts
layout: problem
lang: de
en_slug: circular-references
---

## Description

Zirkuläre Referenzen entstehen, wenn zwei oder mehr Objekte sich gegenseitig referenzieren, entweder direkt oder über eine Kette von Referenzen, was einen Zyklus erzeugt, der die automatische Garbage Collection daran hindert, den Speicher zurückzugewinnen. In Sprachen mit referenzzählender Garbage Collection können zirkuläre Referenzen verhindern, dass Objekte freigegeben werden, selbst wenn sie von den Root-Objekten der Anwendung aus nicht mehr erreichbar sind, was zu Speicherlecks und potenzieller Systeminstabilität führt.

## Indicators ⟡

- Der Speicherverbrauch wächst kontinuierlich, obwohl Objekte scheinbar außer Gültigkeitsbereich geraten
- Die Garbage Collection gibt keinen Speicher für Objekte frei, die für die Bereinigung infrage kommen sollten
- Die Anwendungsperformance verschlechtert sich im Laufe der Zeit aufgrund steigenden Speicherverbrauchs
- Speicher-Profiling zeigt Objekte, die länger als erwartet zugewiesen bleiben
- Die Referenzzählung zeigt von null abweichende Zahlen für Objekte, die unerreichbar sein sollten

## Symptoms ▲

- [Speicherlecks](speicherlecks.md)
<br/>  Zirkuläre Referenzen verhindern, dass die Garbage Collection Objekte zurückgewinnt, was dazu führt, dass Speicher verbraucht, aber nie freigegeben wird.
- [Schleichende Performance-Verschlechterung](schleichende-performance-verschlechterung.md)
<br/>  Sich anhäufender, nicht freigegebener Speicher durch zirkuläre Referenzen führt dazu, dass sich die Anwendungsperformance im Laufe der Zeit verschlechtert.
- [Garbage-Collection-Druck](garbage-collection-druck.md)
<br/>  Durch zirkuläre Referenzen zurückgehaltene Objekte vergrößern die Heap-Größe, was häufigere und längere Garbage-Collection-Zyklen erzwingt.
- [Hoher Ressourcenverbrauch auf Client-Seite](hoher-ressourcenverbrauch-auf-client-seite.md)
<br/>  In browserbasierten Anwendungen führen zirkuläre Referenzen zwischen DOM- und JavaScript-Objekten zu übermäßigem Speicherverbrauch auf dem Client.

## Causes ▼

- [Schlechte Kapselung](schlechte-kapselung.md)
<br/>  Fehlende ordentliche Datenkapselung führt dazu, dass Objekte direkte Referenzen aufeinander halten, statt ordentliche Abstraktionsschichten zu nutzen.
- [Unsachgemäße Verwaltung von Event-Listenern](unsachgemaesse-verwaltung-von-event-listenern.md)
<br/>  Event-Listener, die Referenzen auf ihre übergeordneten Objekte erfassen, erzeugen zirkuläre Referenzketten, die die Garbage Collection verhindern.
- [Unerfahrene Entwickler](unerfahrene-entwickler.md)
<br/>  Entwickler, die mit Speicherverwaltungsmustern nicht vertraut sind, schaffen unwissentlich bidirektionale Referenzen zwischen Objekten.

## Detection Methods ○

- **Speicher-Profiler:** Nutzung von Heap-Analyse-Werkzeugen zur Identifikation von Objektreferenzketten und Erkennung zirkulärer Muster
- **Referenzgraph-Analyse:** Visualisierung von Objektreferenzgraphen zur Identifikation von Zyklen in der Abhängigkeitsstruktur
- **Garbage-Collection-Monitoring:** Überwachung der GC-Wirksamkeit und Identifikation von Objekten, die mehrere Sammelzyklen überleben
- **Speicherleck-Erkennungswerkzeuge:** Nutzung sprachspezifischer Werkzeuge zur Erkennung und Analyse von Speicherlecks
- **Statische Codeanalyse:** Analyse von Code auf Muster, die häufig zirkuläre Referenzen erzeugen
- **Lasttests:** Durchführung erweiterter Tests zur Beobachtung von Speicherwachstumsmustern im Laufe der Zeit

## Examples

Eine Dokumentbearbeitungsanwendung hat Document-Objekte, die Page-Objekte enthalten, und jede Page hält eine Referenz zurück zu ihrem übergeordneten Document für Navigationszwecke. Wenn Dokumente geschlossen werden, verhindern die gegenseitigen Referenzen, dass die Garbage Collection weder das Document- noch das Page-Objekt zurückgewinnt, was dazu führt, dass sich Speicher mit jedem geöffneten und geschlossenen Dokument anhäuft. Ein weiteres Beispiel betrifft eine Webanwendung, bei der DOM-Event-Handler Referenzen auf Geschäftsobjekte erfassen, während diese Geschäftsobjekte Referenzen auf DOM-Elemente für Aktualisierungen halten. Dies erzeugt einen Zyklus zwischen den JavaScript-Objekten und DOM-Knoten, der die Browser-Garbage-Collection daran hindert, UI-Komponenten zu bereinigen, wenn sie von der Seite entfernt werden, was zu Speicheraufblähung in Single-Page-Anwendungen führt.
