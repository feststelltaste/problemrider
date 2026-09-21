---
title: Log-Spam
description: Die Anwendungs- oder Datenbank-Logs werden mit einer großen Anzahl ähnlich
  aussehender Einträge überflutet, was es schwierig macht, andere Probleme zu identifizieren
  und zu diagnostizieren.
category:
- Code
- Operations
related_problems:
- slug: excessive-logging
  similarity: 0.75
- slug: high-number-of-database-queries
  similarity: 0.7
- slug: slow-database-queries
  similarity: 0.65
- slug: imperative-data-fetching-logic
  similarity: 0.65
- slug: n-plus-one-query-problem
  similarity: 0.65
- slug: long-running-transactions
  similarity: 0.6
solutions:
- observability-and-monitoring
- asynchronous-logging
- platform-independent-logging-frameworks
- logging-and-monitoring
- production-readiness-criteria
- logging
- error-logging
- monitoring
- code-reviews
- code-conventions
- logging-guidelines
layout: problem
lang: de
en_slug: log-spam
---

## Description
Log-Spam ist die exzessive Erzeugung von Log-Nachrichten. Dies kann aus mehreren Gründen ein großes Problem sein. Erstens kann es schwierig machen, wichtige Informationen in den Logs zu finden. Zweitens kann es viel Speicherplatz verbrauchen. Drittens kann es einen negativen Einfluss auf die Performance der Anwendung haben. Log-Spam ist oft ein Symptom eines tiefer liegenden Problems, wie des N+1-Abfrageproblems oder einer fehlenden ordentlichen Logging-Konfiguration.

## Indicators ⟡
- Die Logs wachsen mit rasanter Geschwindigkeit.
- Die Logs sind voller sich wiederholender Nachrichten.
- Es ist schwierig, wichtige Informationen in den Logs zu finden.
- Die Anwendung ist langsam, und Sie vermuten, dass Logging ein beitragender Faktor sein könnte.

## Symptoms ▲

- [Debugging-Schwierigkeiten](debugging-schwierigkeiten.md)
<br/>  Wichtige Log-Nachrichten sind im Rauschen begraben, was es extrem schwer macht, relevante Diagnoseinformationen bei der Untersuchung von Problemen zu finden.
- [Übermäßige Festplatten-I/O](uebermaessige-festplatten-io.md)
<br/>  Das Schreiben massiver Mengen an sich wiederholenden Log-Nachrichten verbraucht Festplatten-I/O-Bandbreite, was potenziell die Anwendungsperformance beeinträchtigt.
- [Langsame Anwendungsperformance](langsame-anwendungsperformance.md)
<br/>  Der Overhead der Erzeugung und des Schreibens exzessiver Log-Nachrichten kann den Durchsatz und die Antwortzeiten der Anwendung messbar verschlechtern.
- [Hohe Datenbank-Ressourcennutzung](hohe-datenbank-ressourcennutzung.md)
<br/>  Wenn Logs in Datenbanken gespeichert werden, kann Log-Spam erheblichen Speicher- und Abfrageressourcen verbrauchen.
- [Langsame Vorfallslösung](langsame-vorfallsloesung.md)
<br/>  Wenn kritische Vorfälle auftreten, verschwenden Teams Zeit damit, sich durch Rauschen zu wühlen, um relevante Log-Einträge zu finden, was die Lösung verzögert.

## Causes ▼

- [N+1-Abfrageproblem](n-plus-1-abfrageproblem.md)
<br/>  N+1-Abfragemuster erzeugen eine Flut ähnlicher Abfrage-Log-Einträge, eine klassische Ursache für datenbankbezogenen Log-Spam.
- [Probleme bei der Logging-Konfiguration](probleme-bei-der-logging-konfiguration.md)
<br/>  Unsachgemäße Log-Level-Einstellungen (z. B. DEBUG in Produktion) oder fehlende Log-Filterung verursachen direkt exzessive Log-Ausgabe.
- [Übermäßiges Logging](uebermaessiges-logging.md)
<br/>  Eine allgemeine Praxis des übermäßigen Loggens im Anwendungscode erzeugt die sich wiederholenden, hochvolumigen Nachrichten, die Log-Spam ausmachen.
- [Unzureichende Fehlerbehandlung](unzureichende-fehlerbehandlung.md)
<br/>  Schlechte Fehlerbehandlung, die denselben Fehler wiederholt in engen Schleifen oder Wiederholungszyklen protokolliert, erzeugt massive Mengen doppelter Nachrichten.

## Detection Methods ○
- **Log-Analyse:** Analyse Ihrer Logs zur Identifikation von Mustern und Trends.
- **Überwachung des Log-Volumens:** Überwachung des Volumens Ihrer Logs über die Zeit.
- **Code-Review:** Während Code-Reviews gezielt nach Code suchen, der viele Log-Nachrichten erzeugt.
- **Application Performance Monitoring (APM):** APM-Werkzeuge können Log-Spam oft erkennen und markieren.

## Examples
Eine Webanwendung nutzt eine Drittanbieter-Bibliothek, die viel Log-Spam erzeugt. Die Logs wachsen mit rasanter Geschwindigkeit, und es ist schwierig, wichtige Informationen darin zu finden. Das Team ist sich des Problems nicht bewusst, weil es seine Logs nicht überwacht. Eines Tages fällt die Anwendung aus, und das Team kann nicht herausfinden, warum, weil die Logs voller Rauschen sind. Das Problem hätte vermieden werden können, wenn das Team seine Logs überwacht und Maßnahmen ergriffen hätte, um den Log-Spam zu beheben.
