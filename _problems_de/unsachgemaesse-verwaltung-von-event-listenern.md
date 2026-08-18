---
title: Unsachgemäße Verwaltung von Event-Listenern
description: Event-Listener werden hinzugefügt, aber nicht entfernt, wenn zugehörige
  Objekte zerstört werden, was Speicherlecks verursacht und Garbage Collection verhindert.
category:
- Architecture
- Code
- Performance
related_problems:
- slug: unreleased-resources
  similarity: 0.6
- slug: circular-references
  similarity: 0.55
- slug: memory-leaks
  similarity: 0.55
- slug: excessive-object-allocation
  similarity: 0.55
- slug: resource-allocation-failures
  similarity: 0.55
- slug: database-connection-leaks
  similarity: 0.5
solutions:
- memory-management-optimization
- static-analysis-and-linting
- profiling
- code-reviews
- monitoring-system-utilization
- dependency-injection
- design-by-contract
- load-testing
- error-handling
- exploratory-testing
layout: problem
lang: de
en_slug: improper-event-listener-management
---

## Description

Unsachgemäße Verwaltung von Event-Listenern tritt auf, wenn Anwendungen Event-Handler oder Observer registrieren, es aber versäumen, diese ordentlich zu deregistrieren, wenn die zugehörigen Objekte oder Komponenten zerstört werden. Dies schafft fortbestehende Referenzen, die Garbage Collection verhindern und zu Speicherlecks, unerwartetem Verhalten und Ressourcenerschöpfung führen können. Das Problem ist besonders verbreitet in GUI-Anwendungen, Webanwendungen und ereignisgesteuerten Architekturen.

## Indicators ⟡

- Der Speicherverbrauch steigt mit Interaktionen der Benutzeroberfläche oder Erstellungs-/Zerstörungszyklen von Komponenten
- Event-Handler werden weiter ausgeführt, nachdem ihre zugehörigen Komponenten inaktiv sein sollten
- Die Anwendungsperformance verschlechtert sich, während sich die Anzahl inaktiver Event-Listener anhäuft
- Memory-Profiling zeigt Objekte, die einer Garbage Collection unterzogen werden sollten, aber im Speicher verbleiben
- Unerwartete Nebeneffekte entstehen durch Event-Handler, die nicht mehr aktiv sein sollten

## Symptoms ▲

- [Speicherlecks](speicherlecks.md)
<br/>  Nicht entfernte Event-Listener behalten Referenzen zu Objekten, die einer Garbage Collection unterzogen werden sollten, was fortschreitenden Speicherverbrauch verursacht.
- [Hohe Ressourcennutzung auf dem Client](hohe-ressourcennutzung-auf-dem-client.md)
<br/>  Angehäufte inaktive Event-Listener verbrauchen sowohl Speicher als auch CPU, während sie weiterhin bei Ereignissen ausgeführt werden.
- [Langsame Anwendungsperformance](langsame-anwendungsperformance.md)
<br/>  Während sich inaktive Listener anhäufen, steigt der Event-Dispatch-Overhead, und Speicherdruck verschlechtert die Gesamtanwendungsperformance.

## Causes ▼

- [Unerfahrene Entwickler](unerfahrene-entwickler.md)
<br/>  Entwickler, die mit Komponentenlebenszyklus-Verwaltung nicht vertraut sind, verstehen möglicherweise nicht die Notwendigkeit, Event-Listener zu bereinigen.
- [Inkonsistente Coding-Standards](inkonsistente-coding-standards.md)
<br/>  Ohne Standards, die die Bereinigung von Event-Listenern in Komponentenlebenszyklus-Methoden vorschreiben, besteht dieses Muster fort.
- [Unzureichendes Code-Review](unzureichendes-code-review.md)
<br/>  Code-Reviews könnten fehlenden Bereinigungscode erfassen, aber ohne sie gelangen diese Muster unwidersprochen in die Codebasis.

## Detection Methods ○

- **Memory-Profiler:** Analyse von Heap-Dumps zur Identifikation von Event-Listener-Objekten, die einer Garbage Collection hätten unterzogen werden sollen
- **Event-System-Debugging:** Überwachung von Event-Listener-Registrierungs- und Deregistrierungsmustern
- **Komponentenlebenszyklus-Analyse:** Nachverfolgung von Komponentenerstellung und -zerstörung zur Identifikation von Bereinigungslücken
- **Referenzgraph-Analyse:** Untersuchung von Objektreferenzgraphen zur Identifikation ereignisbezogener zirkulärer Referenzen
- **Performance-Monitoring:** Überwachung der Event-Dispatch-Performance zur Identifikation von Overhead durch inaktive Listener
- **Statische Code-Analyse:** Identifikation von Mustern, bei denen Event-Listener ohne entsprechende Bereinigung registriert werden

## Examples

Eine Single-Page-Webanwendung erstellt neue Ansichtskomponenten, während Nutzer zwischen Seiten navigieren. Jede Ansicht registriert Click-Handler und andere DOM-Event-Listener, aber wenn Nutzer wegnavigieren, werden die alten Ansichtskomponenten aus dem DOM entfernt, ohne ihre Event-Listener zu deregistrieren. Die Listener behalten Referenzen sowohl zu den DOM-Elementen als auch zu den Ansichtsobjekten, was die Garbage Collection der gesamten Ansichtshierarchie verhindert. Über die Zeit häuft sich dies zu Hunderten inaktiver Event-Listener und Ansichtsobjekte an, was zu erheblicher Speicheraufblähung führt. Ein weiteres Beispiel betrifft eine Desktop-Anwendung, bei der Dialogboxen sich für anwendungsweite Ereignisse wie Konfigurationsänderungen registrieren, es aber versäumen, sich zu deregistrieren, wenn die Dialoge geschlossen werden, was dazu führt, dass Event-Handler für geschlossene Dialoge ausgeführt werden und Null-Pointer-Ausnahmen oder anderes unerwartetes Verhalten verursachen.
