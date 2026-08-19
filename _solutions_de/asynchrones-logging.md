---
title: Asynchrones Logging
description: Entkopplung des Logging-Prozesses von der Hauptanwendung.
category:
- Performance
- Operations
problems:
- excessive-logging
- slow-application-performance
- log-spam
- logging-configuration-issues
- gradual-performance-degradation
layout: solution
lang: de
en_slug: asynchronous-logging
related_solutions:
- slug: logging
  similarity: 0.75
- slug: platform-independent-logging-frameworks
  similarity: 0.75
- slug: asynchronous-processing
  similarity: 0.75
- slug: error-logging
  similarity: 0.7
- slug: distributed-tracing
  similarity: 0.65
- slug: connection-pooling
  similarity: 0.65
---

## Description

Asynchrones Logging entkoppelt das Schreiben eines Log-Eintrags vom Thread, der die Anfrage bearbeitet, die ihn erzeugt hat, indem Log-Ereignisse an einen Puffer übergeben werden — typischerweise einen Ringpuffer oder eine sperrfreie Warteschlange —, den ein separater Thread leert und auf die Festplatte schreibt, sodass der anfragebearbeitende Thread nie blockiert, während er auf den Abschluss der Log-E/A wartet. Legacy-Anwendungen loggen häufig standardmäßig synchron, da dies die einfachste verfügbare Implementierung war, als Logging-Frameworks wie Log4j erstmals konfiguriert wurden, und bei geringem Traffic ist diese Kosten unsichtbar; aber während der Traffic wächst, wird jede gleichzeitige Anfrage, die um denselben synchronen Dateischreibvorgang konkurriert, zu einer Quelle von Thread-Konkurrenz und Latenzspitzen, die wie unabhängige Performance-Probleme erscheinen, bis Profiling sie auf die Logging-Aufrufe selbst zurückführt. Das Umschalten der Logging-Konfiguration auf einen asynchronen Appender entfernt diesen Engpass, ohne dass irgendeine Änderung an den tatsächlichen Logging-Anweisungen erforderlich wäre, die über die Legacy-Codebasis verstreut sind, da sich nur die Appender-Konfiguration ändert, nicht der aufrufende Code — ein seltener Fall, in dem eine bedeutsame Performance-Korrektur in einem Legacy-System fast nichts an der Anwendungslogik selbst berühren muss. Weil Log-Ereignisse jetzt gepuffert statt sofort geschrieben werden, kann ein Anwendungsabsturz vor dem Leeren des Puffers die neuesten Einträge verlieren, sodass dieser Ansatz auch ein ordentliches Shutdown-Verfahren erfordert, das ausstehende Ereignisse leert, zusammen mit einer expliziten Overflow-Richtlinie dafür, was passiert, wenn die Warteschlange schneller voll wird, als sie geleert werden kann. Die Überwachung der Tiefe der asynchronen Warteschlange in Produktion ist notwendig, um zu erkennen, wann Logging immer noch nicht mit der Anfragerate mithalten kann, an welchem Punkt entweder die Puffergröße oder die Overflow-Richtlinie überarbeitet werden muss, statt zu synchronem Logging zurückzukehren.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Konfigurieren Sie das Logging-Framework zur Nutzung asynchroner Appender, die Log-Ereignisse puffern und auf einem separaten Thread schreiben
- Legen Sie angemessene Puffergrößen und Overflow-Richtlinien fest, um Burst-Logging zu handhaben, ohne kritische Nachrichten zu verlieren
- Nutzen Sie Ringpuffer oder sperrfreie Warteschlangen für die asynchrone Übergabe, um Konkurrenz zu minimieren
- Implementieren Sie ordentliche Shutdown-Verfahren, die ausstehende Log-Ereignisse leeren, bevor die Anwendung beendet wird
- Überwachen Sie die Tiefe der asynchronen Logging-Warteschlange, um Situationen zu erkennen, in denen Logging nicht mit der Produktionsrate mithalten kann
- Migrieren Sie schrittweise von synchronen zu asynchronen Datei-Appendern, beginnend mit den Log-Quellen mit dem höchsten Volumen

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Eliminiert Logging als Latenzquelle auf dem Anfrageverarbeitungspfad
- Verringert Thread-Konkurrenz, die durch synchrone Schreibvorgänge in gemeinsam genutzte Log-Dateien verursacht wird
- Glättet E/A-Spitzen durch Batching von Log-Schreibvorgängen
- Erhält Logging-Sichtbarkeit, ohne den Anwendungsdurchsatz zu opfern

**Kosten und Risiken:**
- Log-Ereignisse könnten bei Anwendungsabstürzen verloren gehen, wenn der Puffer nicht geleert wurde
- Fügt Komplexität zur Shutdown- und Fehlerbehandlungslogik hinzu
- Pufferüberlauf unter hoher Last könnte das Verwerfen von Log-Nachrichten oder Blockieren erfordern
- Zeitstempel in Logs spiegeln aufgrund der Pufferung möglicherweise nicht perfekt die Reihenfolge der Ereignisse wider

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Eine Legacy-Java-Anwendung, die stark frequentierte REST-Endpunkte bediente, erlebte periodische Latenzspitzen. Profiling offenbarte, dass synchrone Log4j-Datei-Appender Anfrage-Threads während der Festplatten-E/A blockierten, besonders unter hoher Last, wenn viele gleichzeitige Anfragen gleichzeitig loggten. Der Wechsel zu Log4j2 AsyncAppender mit einem LMAX-Disruptor-Ringpuffer eliminierte das E/A-Blockieren vom Anfragepfad. Die P99-Latenz sank um 40 %, und die Latenzspitzen verschwanden vollständig. Das Team konfigurierte außerdem eine Verwerfungsrichtlinie für DEBUG-Level-Nachrichten während Pufferüberlauf, um sicherzustellen, dass kritische ERROR- und WARN-Nachrichten nie verloren gingen.
