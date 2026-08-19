---
title: Asynchrone Operationen
description: Ausführung zeitintensiver Operationen im Hintergrund, ohne die UI zu
  blockieren.
category:
- Performance
- Architecture
quality_tactics_url: https://qualitytactics.de/en/usability/asynchronous-operations/
problems:
- slow-application-performance
- user-frustration
- poor-user-experience-ux-design
- slow-response-times-for-lists
- high-client-side-resource-consumption
- thread-pool-exhaustion
- external-service-delays
- negative-user-feedback
layout: solution
lang: de
en_slug: asynchronous-operations
related_solutions:
- slug: asynchronous-processing
  similarity: 0.8
- slug: performance-optimization
  similarity: 0.8
- slug: consistent-user-interface
  similarity: 0.75
- slug: concurrency-control
  similarity: 0.75
- slug: auto-save
  similarity: 0.75
- slug: optimistic-ui-updates
  similarity: 0.75
---

## Description

Eine zeitintensive Operation vom synchronen Anfragepfad zu entfernen — in einen Hintergrundjob, über dessen Abschluss der Nutzer benachrichtigt wird — hält die Schnittstelle reaktionsfähig, statt einzufrieren, während ein Bericht zusammengestellt wird oder ein Export läuft. Legacy-Systeme führen genau diese Operationen häufig synchron aus, einfach weil das das einzige verfügbare Muster war, als sie gebaut wurden, was bedeutet, dass jeder lange Aufruf die UI blockiert und, serverseitig, einen anfragebearbeitenden Thread für die gesamte Dauer der Arbeit bindet. Die Interaktion auf Absenden-und-Benachrichtigen umzustellen, unterstützt durch eine Nachrichten- oder Aufgabenwarteschlange, entlastet beide Probleme gleichzeitig, erfordert aber Infrastruktur, die das Legacy-System möglicherweise noch nicht hat, und führt eigene Komplexität bei der Verfolgung des Operationszustands und der Kommunikation von Fehlern an den Nutzer ein.

## How to Apply ◆

> Legacy-Systeme führen lang laufende Operationen oft synchron aus, was die Benutzeroberfläche einfrieren lässt und Nutzer frustriert. Diese Operationen in Hintergrundverarbeitung zu verschieben verbessert die wahrgenommene Reaktionsfähigkeit.

- Identifizieren Sie synchrone Operationen, die mehr als ein oder zwei Sekunden zur Fertigstellung benötigen, wie Berichtsgenerierung, Datenexporte, Batch-Verarbeitung und externe Serviceaufrufe. Dies sind die primären Kandidaten für die asynchrone Umstellung.
- Implementieren Sie Hintergrundjob-Verarbeitung mithilfe einer Nachrichten- oder Aufgabenwarteschlange. Ersetzen Sie synchrone Anfrage-Antwort-Muster durch einen Absenden-und-Abfragen- oder Absenden-und-Benachrichtigen-Ansatz.
- Fügen Sie Fortschrittsanzeigen und Statusbenachrichtigungen hinzu, sodass Nutzer wissen, dass ihre Operation verarbeitet wird. Legacy-Systeme, die einfach einen Spinner zeigen oder den Bildschirm einfrieren lassen, bieten kein nützliches Feedback.
- Implementieren Sie ordentliche Fehlerbehandlung für Hintergrundoperationen, einschließlich Wiederholungsmechanismen und klarer Fehlerbenachrichtigungen. Nutzer müssen informiert werden, wenn eine Hintergrundoperation fehlschlägt, nicht im Ungewissen gelassen werden.
- Nutzen Sie optimistische UI-Updates, wo angemessen: Zeigen Sie das erwartete Ergebnis sofort an, während die tatsächliche Operation im Hintergrund abgeschlossen wird, und machen Sie nur einen Rollback, wenn die Operation fehlschlägt.
- Stellen Sie sicher, dass die UI vollständig interaktiv bleibt, während Hintergrundoperationen laufen. Nutzer sollten in der Lage sein, an anderen Aufgaben weiterzuarbeiten, ohne zu warten.

## Tradeoffs ⇄

> Asynchrone Operationen verbessern die Nutzererfahrung dramatisch, führen aber Komplexität in Zustandsverwaltung und Fehlerbehandlung ein.

**Vorteile:**

- Eliminiert UI-Einfrieren während lang laufender Operationen und adressiert direkt Nutzerfrustration, die durch nicht reagierende Legacy-Schnittstellen verursacht wird.
- Verbessert die wahrgenommene Performance, selbst wenn die tatsächliche Verarbeitungszeit gleich bleibt, weil Nutzer weiterarbeiten können.
- Verringert Thread-Pool-Erschöpfung auf dem Server, indem lang laufende Arbeit auf Hintergrund-Worker ausgelagert wird, statt HTTP-Anfrage-Threads zu belegen.
- Ermöglicht bessere Handhabung externer Serviceverzögerungen, indem die Nutzerinteraktion von der nachgelagerten Verarbeitung entkoppelt wird.

**Kosten und Risiken:**

- Führt Komplexität bei der Verfolgung des Operationszustands, der Fehlerbehandlung und der Sicherstellung von Konsistenz zwischen UI-Zustand und tatsächlichem Backend-Zustand ein.
- Erfordert Infrastruktur für Hintergrundjob-Verarbeitung wie Nachrichtenwarteschlangen und Worker-Prozesse, die das Legacy-System möglicherweise aktuell nicht hat.
- Das Testen asynchroner Abläufe ist komplexer als das Testen synchroner, da Timing, Race Conditions und Fehlerszenarien berücksichtigt werden müssen.
- Nutzer, die an synchrones Feedback gewöhnt sind, könnten anfänglich durch die Änderung des Interaktionsmusters verwirrt sein und brauchen klare Kommunikation darüber, was gerade geschieht.

## How It Could Be

> Hintergrundverarbeitung kann die frustrierendsten Aspekte eines Legacy-Systems in reaktionsfähige, benutzerfreundliche Erfahrungen verwandeln.

Ein Legacy-ERP-System generiert monatliche Finanzberichte synchron und sperrt die Sitzung des Nutzers für bis zu fünfzehn Minuten, während der Bericht zusammengestellt wird. Nutzer haben gelernt, den Bericht zu starten und Kaffee holen zu gehen, aber wenn die Sitzung abläuft, müssen sie den gesamten Prozess neu starten. Das Team führt einen Hintergrund-Berichtsgenerierungsservice ein: Nutzer reichen Berichtsanfragen ein und erhalten eine Benachrichtigung, wenn der Bericht zum Download bereit ist. Die Berichtsgenerierung selbst benötigt dieselbe Zeit, aber Nutzer können weiterhin Rechnungen eingeben und Bestellungen verwalten, während sie warten. Sitzungsablauf-Probleme verschwinden vollständig, und das Support-Team erhält keine Tickets mehr über verlorene Berichte.
