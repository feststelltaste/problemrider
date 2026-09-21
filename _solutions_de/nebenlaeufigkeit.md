---
title: Nebenläufigkeit
description: Gleichzeitige Ausführung mehrerer Aufgaben innerhalb eines einzelnen
  Prozesses.
category:
- Performance
- Code
quality_tactics_url: https://qualitytactics.de/en/performance-efficiency/concurrency/
problems:
- race-conditions
- synchronization-problems
- long-running-transactions
- thread-pool-exhaustion
- resource-contention
- memory-leaks
- atomic-operation-overhead
- deadlock-conditions
- false-sharing
- lock-contention
- memory-barrier-inefficiency
- long-running-database-transactions
layout: solution
lang: de
en_slug: concurrency-control
related_solutions:
- slug: transactions
  similarity: 0.8
- slug: parallelization
  similarity: 0.75
- slug: resource-pooling
  similarity: 0.75
- slug: batch-processing
  similarity: 0.75
- slug: asynchronous-operations
  similarity: 0.75
- slug: resource-usage-optimization
  similarity: 0.75
---

## Description

Nebenläufigkeitssteuerung restrukturiert, wie ein System auf gemeinsam genutzten veränderlichen Zustand zugreift und Arbeit über Threads oder Prozesse verteilt, sodass Operationen sicher parallel laufen können, statt der Einzelthread- oder Ad-hoc-Multithread-Muster, mit denen viele Legacy-Systeme aufgewachsen sind. Jedes Stück gemeinsam genutzten Zustands zu kartieren und wo er gelesen und geschrieben wird, ist der notwendige erste Schritt, da die Einführung von Synchronisation, unveränderlichen Datenmustern oder asynchroner Verarbeitung auf eine ungeprüfte Legacy-Codebasis, ohne zuerst ihren gemeinsamen Zustand zu verstehen, ist, wie neue Race Conditions und Deadlocks eingeführt werden. Sorgfältig durchgeführt — enge Lock-Bereiche, optimistische Nebenläufigkeit für Datenbankoperationen mit geringem Konflikt, Timeouts bei jedem blockierenden Aufruf — erhöht Nebenläufigkeitssteuerung den Durchsatz und verhindert Thread-Pool-Erschöpfung, auf Kosten einer Fehlerklasse, die notorisch schwer zu reproduzieren und zu diagnostizieren ist, sobald sie eingeführt wurde.

## How to Apply ◆

> Legacy-Systeme entwickelten sich häufig in Einzelthread- oder schlecht koordinierten Multithread-Umgebungen. Die Einführung ordentlicher Nebenläufigkeitssteuerung bedeutet, zu restrukturieren, wie auf gemeinsam genutzte Ressourcen zugegriffen wird, wie Arbeit über Threads oder Prozesse verteilt wird, und wie sich das System unter nebenläufiger Last verhält.

- Identifizieren Sie allen gemeinsam genutzten veränderlichen Zustand in der Anwendung — globale Variablen, Caches, Sitzungsspeicher, Zähler und In-Memory-Sammlungen. Kartieren Sie, welche Threads oder Prozesse auf jedes Stück gemeinsam genutzten Zustands zugreifen und ob diese Zugriffe Schreibvorgänge beinhalten. Dieses Audit ist die Grundlage für alle nachfolgenden Nebenläufigkeitsverbesserungen.
- Führen Sie angemessene Synchronisationsprimitive für den Zugriff auf gemeinsam genutzten Zustand ein. Nutzen Sie Mutexe oder synchronisierte Blöcke für einfache kritische Abschnitte, Read-Write-Locks, wo Lesevorgänge Schreibvorgänge weit übersteigen, und atomare Operationen für einfache Zähler und Flags. Bevorzugen Sie den engstmöglichen Lock-Bereich, um Konkurrenz zu minimieren.
- Ersetzen Sie grobgranulares Locking durch feingranulare oder lock-freie Datenstrukturen, wo Konkurrenz hoch ist. Ersetzen Sie zum Beispiel eine einzelne Sperre, die eine gesamte Map schützt, durch eine nebenläufige Hash-Map, die einzelne Segmente sperrt, oder nutzen Sie Compare-and-Swap-Operationen für einfache Zustandsübergänge.
- Übernehmen Sie unveränderliche Datenmuster, wo immer möglich. Unveränderliche Objekte sind inhärent threadsicher und eliminieren ganze Kategorien von Race Conditions. In Legacy-Code beginnen Sie damit, Value-Objekte und Konfigurationsdaten unveränderlich zu machen, und erweitern Sie dann schrittweise das Muster auf Domänenobjekte.
- Implementieren Sie ordentliche Transaktionsabgrenzung für Datenbankoperationen. Brechen Sie lang laufende Transaktionen in kleinere, begrenzte Arbeitseinheiten auf. Nutzen Sie optimistische Nebenläufigkeitssteuerung (Versionsspalten, ETags) statt pessimistischem Locking, wo Konflikte selten sind, was Lock-Haltezeiten und Deadlock-Risiko verringert.
- Führen Sie asynchrone Verarbeitung für Operationen ein, die keine sofortigen Ergebnisse benötigen. Verschieben Sie lang laufende Aufgaben zu Hintergrund-Workern oder Nachrichtenwarteschlangen, was anfragebearbeitende Threads freisetzt, um neue Anfragen zu bedienen, und Thread-Pool-Erschöpfung verhindert.
- Fügen Sie Timeout- und Circuit-Breaker-Muster zu allen blockierenden Operationen hinzu — Datenbankabfragen, externe Serviceaufrufe, Lock-Erwerbe. Ohne Timeouts kann eine einzelne langsame Abhängigkeit alle verfügbaren Threads verbrauchen und in einen vollständigen Systemausfall kaskadieren.
- Nutzen Sie strukturierte Nebenläufigkeits-Frameworks oder -Muster (wie Javas ExecutorService, Pythons asyncio oder Gos Goroutinen mit Channels), um Thread-Lebenszyklen explizit zu verwalten, statt Ad-hoc-Threads über die Codebasis zu erzeugen.
- Instrumentieren Sie nebenläufige Codepfade mit Metriken für Lock-Wartezeiten, Thread-Pool-Auslastung, Warteschlangentiefen und Konkurrenzraten. Diese Metriken bieten Frühwarnung vor Nebenläufigkeitsengpässen, bevor sie sich als für Nutzer sichtbare Fehler manifestieren.

## Tradeoffs ⇄

> Nebenläufigkeitssteuerung ermöglicht höheren Durchsatz und bessere Ressourcennutzung, führt aber Komplexität beim Nachdenken über Programmkorrektheit ein und kann neue Kategorien von Bugs schaffen, wenn falsch angewendet.

**Vorteile:**

- Erhöht den Durchsatz, indem mehrere Operationen gleichzeitig ausgeführt werden können, was moderne Multi-Core-Hardware besser nutzt, die Legacy-Systeme oft nicht für die Nutzung designt waren.
- Verringert Antwortzeiten, indem E/A-gebundene Operationen wie Datenbankabfragen und externe Serviceaufrufe überlappt statt sequenziell ausgeführt werden.
- Verhindert Datenkorruption und inkonsistenten Zustand, verursacht durch unsynchronisierten nebenläufigen Zugriff auf gemeinsam genutzte Ressourcen.
- Eliminiert Thread-Pool-Erschöpfung, indem blockierende Arbeit zu Hintergrund-Threads verschoben wird und anfragebearbeitende Threads für neue Anfragen verfügbar bleiben.
- Verringert Datenbank-Lock-Konkurrenz und Deadlock-Häufigkeit, indem Transaktionsdauern verkürzt und optimistische Nebenläufigkeit genutzt wird, wo angemessen.

**Kosten und Risiken:**

- Nebenläufigkeitsbugs — Race Conditions, Deadlocks, Livelocks — gehören zu den am schwierigsten zu reproduzierenden, zu diagnostizierenden und zu behebenden Defekten. Das Hinzufügen von Nebenläufigkeit zu Legacy-Code ohne umfassendes Testen kann subtile, intermittierende Fehler einführen.
- Übermäßige Synchronisation (übermäßiges Locking) kann den Durchsatz unter Einzelthread-Performance verringern, was eine Nebenläufigkeitsverbesserung in eine Regression verwandelt.
- Das Debugging und Profiling nebenläufigen Codes erfordert spezialisierte Werkzeuge und Expertise, die Legacy-Teams möglicherweise nicht besitzen, was eine Wissenslücke schafft, die durch Schulung adressiert werden muss.
- Lock-freie und Wait-freie Datenstrukturen bieten bessere Performance unter Konkurrenz, sind aber erheblich schwieriger korrekt zu implementieren und auf Korrektheit zu verifizieren.
- Der Übergang von synchroner zu asynchroner Verarbeitung ändert das Programmiermodell erheblich und erfordert Überarbeitung von Fehlerbehandlung, Transaktionsmanagement und Ergebnisweitergabemustern.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie Nebenläufigkeitssteuerung Probleme in Legacy-Systemen angeht.

Ein Gehaltsabrechnungssystem, ursprünglich designt, um 500 Mitarbeiter zu handhaben, wuchs, um 15.000 Mitarbeiter zu bedienen. Der nächtliche Gehaltsabrechnungslauf umschloss den gesamten Batch in einer einzigen Datenbanktransaktion, hielt Tabellenebenen-Sperren für über vier Stunden und blockierte die HR-Anwendung von jeglichen Schreibvorgängen während dieses Fensters. Das Team restrukturierte den Batch in Pro-Abteilung-Transaktionen von jeweils 50-200 Mitarbeitern, fügte optimistisches Locking via Versionsspalten auf Mitarbeiterdatensätzen hinzu und verarbeitete Abteilungen nebenläufig unter Nutzung eines Thread-Pools von 8 Workern. Die Gesamtverarbeitungszeit sank von vier Stunden auf 25 Minuten, und die HR-Anwendung blieb während des gesamten Laufs vollständig funktionsfähig, weil keine einzelne Transaktion Sperren für mehr als ein paar Sekunden hielt.

Ein Legacy-Dokumentenmanagementsystem bediente nebenläufige Nutzer, schützte aber seinen In-Memory-Dokumentindex mit einer einzigen globalen Sperre. Unter Spitzenlast von 80 Nutzern blockierten Lesevorgänge für Dokumentensuchen hinter Schreibvorgängen, die den Index während Uploads aktualisierten, was Suchantwortzeiten auf 12 Sekunden ansteigen ließ. Das Team ersetzte die globale Sperre durch eine Read-Write-Sperre, was unbegrenzte nebenläufige Leser erlaubte, während Schreibvorgänge weiterhin serialisiert wurden. Suchantwortzeiten unter derselben Last sanken auf 200 ms, weil sich Lesevorgänge nicht mehr gegenseitig blockierten, und nur die seltenen Schreiboperationen exklusiven Zugriff benötigten.

Eine Versicherungsschadensverarbeitungsanwendung machte synchrone Aufrufe an drei externe Validierungsservices innerhalb des Anfrage-Threads. Wenn einer dieser Services Latenzspitzen erlebte, wurden alle anfragebearbeitenden Threads blockiert, und die Anwendung reagierte überhaupt nicht mehr. Das Team führte asynchrone Aufrufe mit einem 3-Sekunden-Timeout und einem Circuit Breaker ein, der nach 5 aufeinanderfolgenden Fehlern auslöste. Ansprüche, die nicht innerhalb des Timeouts validiert werden konnten, wurden für Hintergrund-Wiederholung in eine Warteschlange gestellt, statt den Anfrage-Thread zu blockieren. Die Thread-Pool-Auslastung sank von konstant 100 % während externer Serviceverlangsamungen auf unter 30 %, und die Anwendung blieb reaktionsfähig, selbst wenn nachgelagerte Services degradierten.
