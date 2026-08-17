---
title: Deadlock-Zustände
description: Mehrere Threads oder Prozesse warten unbegrenzt darauf, dass der jeweils
  andere Ressourcen freigibt, was zum Einfrieren des Systems und zur Nichtreaktion
  der Anwendung führt.
category:
- Code
- Performance
related_problems:
- slug: race-conditions
  similarity: 0.65
- slug: lock-contention
  similarity: 0.6
- slug: thread-pool-exhaustion
  similarity: 0.6
- slug: resource-contention
  similarity: 0.55
solutions:
- query-optimization-process
- timeout-management
- transactions
- concurrency-control
- monitoring
- stress-testing
- idempotency-design
- profiling
- static-analysis-and-linting
- exploratory-testing
- saga-pattern
layout: problem
lang: de
en_slug: deadlock-conditions
---

## Description

Deadlock-Zustände entstehen, wenn zwei oder mehr Threads oder Prozesse unbegrenzt blockiert sind, wobei jeder darauf wartet, dass der andere eine Ressource freigibt, die er zum Fortsetzen der Ausführung benötigt. Dies schafft eine zirkuläre Abhängigkeit, bei der kein Thread fortfahren kann, was effektiv einen Teil oder die gesamte Anwendung einfriert. Deadlocks sind ein klassisches Nebenläufigkeitsproblem, das dazu führen kann, dass Anwendungen hängen bleiben, nicht mehr reagieren oder eine erzwungene Beendigung erfordern.

## Indicators ⟡

- Die Anwendung wird plötzlich nicht mehr reagierend oder scheint einzufrieren
- Threads sind blockiert und warten auf Locks, die von anderen blockierten Threads gehalten werden
- Datenbanktransaktionen erreichen ein Timeout aufgrund von Lock-Konflikten
- Die Benutzeroberfläche reagiert während bestimmter Operationen nicht mehr
- Das System-Monitoring zeigt Threads in Wartezuständen, die nie fortschreiten

## Symptoms ▲

- [Systemausfälle](systemausfaelle.md)
<br/>  Deadlocks lassen Teile oder die gesamte Anwendung einfrieren, was effektiv Dienstausfälle erzeugt, die manuelles Eingreifen erfordern.
- [Langsame Anwendungsperformance](langsame-anwendungsperformance.md)
<br/>  Selbst wenn Deadlocks erkannt und über Timeouts aufgelöst werden, verschlechtern die wiederholten Blockier- und Wiederholungszyklen die Reaktionsfähigkeit der Anwendung.
- [Erschöpfung des Thread-Pools](erschoepfung-des-thread-pools.md)
<br/>  Deadlockte Threads bleiben dauerhaft belegt und verbrauchen schrittweise alle verfügbaren Threads im Pool.
- [Debugging-Schwierigkeiten](debugging-schwierigkeiten.md)
<br/>  Deadlocks sind notorisch schwer zu reproduzieren und zu diagnostizieren, weil sie von spezifischem Timing und der Reihenfolge nebenläufiger Operationen abhängen.
- [Nutzerfrustration](nutzerfrustration.md)
<br/>  Anwendungsstillstände durch Deadlocks schaffen eine unvorhersehbare und unzuverlässige Nutzererfahrung.

## Causes ▼

- [Race Conditions](race-conditions.md)
<br/>  Unsachgemäße Synchronisation, die zu Race Conditions führt, resultiert oft in übermäßig aggressiven Locking-Strategien, die Deadlock-Potenzial schaffen.
- [Lock Contention](lock-contention.md)
<br/>  Starke Lock Contention mit inkonsistenter Lock-Reihenfolge schafft die zirkulären Wartebedingungen, die für Deadlocks nötig sind.
- [Lang laufende Transaktionen](lang-laufende-transaktionen.md)
<br/>  Transaktionen, die Locks über längere Zeiträume halten, vergrößern das Zeitfenster, in dem sich zirkuläre Wartebedingungen bilden können.
- [Unerfahrene Entwickler](unerfahrene-entwickler.md)
<br/>  Entwickler ohne Nebenläufigkeitsexpertise verstehen möglicherweise die Disziplin der Lock-Reihenfolge oder Deadlock-Vermeidungsstrategien nicht.

## Detection Methods ○

- **Deadlock-Erkennungswerkzeuge:** Nutzung von Debugging-Werkzeugen und Profilern, die zirkuläre Wartebedingungen identifizieren können
- **Thread-Dump-Analyse:** Analyse von Thread-Dumps zur Identifikation blockierter Threads und ihrer Lock-Abhängigkeiten
- **Datenbank-Lock-Monitoring:** Überwachung von Datenbank-Lock-Tabellen zur Identifikation von Transaktions-Deadlocks
- **Anwendungsprotokollierung:** Protokollierung von Lock-Erwerb und -Freigabe zur Nachverfolgung von Deadlock-Mustern
- **Timeout-Implementierung:** Nutzung von Timeouts beim Lock-Erwerb zur Erkennung potenzieller Deadlock-Situationen
- **Statische Analyse:** Analyse von Code auf potenzielle Deadlock-Muster und Probleme bei der Lock-Reihenfolge

## Examples

Eine Banking-Anwendung hat zwei Threads, die Geldüberweisungen verarbeiten. Thread A sperrt Konto 1 und versucht, Konto 2 zu sperren, während Thread B Konto 2 sperrt und versucht, Konto 1 zu sperren. Beide Threads warten unbegrenzt darauf, dass der andere seinen Lock freigibt, was das gesamte Überweisungssystem einfrieren lässt und einen Anwendungsneustart erfordert. Ein weiteres Beispiel betrifft ein Ressourcenverwaltungssystem, bei dem Thread 1 eine Datenbankverbindung erwirbt und dann versucht, einen Datei-Lock zu erwerben, während Thread 2 den Datei-Lock erwirbt und dann versucht, eine Datenbankverbindung zu erwerben. Die zirkuläre Abhängigkeit verhindert, dass einer der Threads seine Operation abschließt, was die Anwendung zum Hängen bringt, bis der Deadlock manuell aufgelöst wird.
