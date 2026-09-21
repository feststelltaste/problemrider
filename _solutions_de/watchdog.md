---
title: Watchdog
description: Überwachungskomponente zur Erkennung und Behandlung von
  Systemfehlern oder Ausfällen.
category:
- Operations
problems:
- system-outages
- cascade-failures
- slow-incident-resolution
- monitoring-gaps
- unpredictable-system-behavior
- single-points-of-failure
- constant-firefighting
- silent-data-corruption
layout: solution
lang: de
en_slug: watchdog
related_solutions:
- slug: heartbeat
  similarity: 0.85
- slug: monitoring
  similarity: 0.8
- slug: status-monitoring
  similarity: 0.75
- slug: ping
  similarity: 0.75
- slug: self-monitoring-and-diagnosis
  similarity: 0.75
- slug: error-reporting-and-analysis
  similarity: 0.75
---

## Description

Ein Watchdog ist ein überwachender Prozess, der einen oder mehrere andere Prozesse beobachtet und handelt, wenn diese aufhören, sich korrekt zu verhalten, sei es durch Nicht-Antworten, fehlenden Fortschritt oder unerwartete Terminierung. Er funktioniert, indem er verlangt, dass der überwachte Prozess periodisch beweist, dass er lebt und funktioniert — durch ein Heartbeat-Signal, einen Fortschrittsmarker oder eine Antwort auf einen Health Check —, und indem er die Abwesenheit dieses Beweises innerhalb eines erwarteten Zeitfensters als Evidenz eines Fehlschlags behandelt. Sobald ein Fehlschlag erkannt wird, kann der Watchdog durch eine konfigurierte Kette von Maßnahmen eskalieren: den Prozess neu starten, eine hängende Instanz beenden, damit sie wiederhergestellt werden kann, oder einen Betreiber alarmieren, wenn automatisierte Wiederherstellung nicht möglich ist oder bereits zu oft versucht wurde. Dieses Muster ist besonders wertvoll für Legacy-Systeme, weil solche Systeme häufig Komponenten betreiben, die nie mit Selbstüberwachung im Sinn gebaut wurden und die auf stille, ungraziöse Weisen versagen — ein hängender Batch-Job, ein blockierter Thread-Pool, ein Message-Consumer, der still aufgehört hat zu konsumieren. Statt eine invasive Neuschreibung zu erfordern, um ordentliche Observability hinzuzufügen, kann ein Watchdog als externer Supervisor über einen bestehenden Prozess geschichtet werden, was einen undurchsichtigen Fehlermodus in einen erkennbaren und oft selbstheilenden verwandelt. Dabei wirkt er direkt dem Muster konstanten Feuerwehrlöschens und langsamer Vorfalllösung entgegen, das Legacy-Betriebsteams erleben, wenn Fehlschläge erst durch nachgelagerte Symptome oder Nutzerbeschwerden entdeckt werden, Stunden nachdem der eigentliche Fehler auftrat.

## How to Apply ◆

> Legacy-Systeme versagen oft still, ohne Mechanismus zu erkennen, wenn eine Komponente aufgehört hat, korrekt zu funktionieren. Watchdog-Prozesse bieten automatisierte Erkennung und Wiederherstellung für Fehlschläge, die sonst unbemerkt blieben, bis Nutzer Symptome melden.

- Identifizieren Sie alle kritischen Prozesse im Legacy-System, die kontinuierlich laufen müssen — Anwendungsserver, Batch-Prozessoren, Message-Consumer, geplante Jobs und Hintergrund-Worker. Jeder davon braucht Watchdog-Überwachung.
- Implementieren Sie heartbeat-basiertes Monitoring, bei dem jeder kritische Prozess periodisch seine Lebendigkeit an einen Watchdog-Dienst signalisiert. Wenn der Watchdog innerhalb des erwarteten Intervalls keinen Heartbeat empfängt, löst er einen Alarm aus und initiiert optional eine Wiederherstellungsmaßnahme.
- Nutzen Sie Prozessüberwacher auf Betriebssystemebene (systemd, supervisord, Windows Service Control Manager) als erste Verteidigungslinie, um abgestürzte Prozesse automatisch neu zu starten. Konfigurieren Sie Neustart-Grenzen, um Neustart-Schleifen zu verhindern, wenn ein Prozess wiederholt aufgrund eines anhaltenden Problems abstürzt.
- Implementieren Sie Health Checks auf Anwendungsebene, die über Prozesslebendigkeit hinausgehen. Ein Prozess könnte laufen, aber feststecken (blockierte Threads, erschöpfter Verbindungspool, Endlosschleife). Watchdog-Prüfungen sollten verifizieren, dass der Prozess tatsächlich Fortschritt macht — Nachrichten verarbeitet, Anfragen abschließt, Batch-Jobs voranbringt.
- Fügen Sie Erkennung feststeckender Zustände für Batch-Prozesse und langlaufende Operationen hinzu. Wenn ein Batch-Job seinen Fortschrittsmarker nicht innerhalb eines konfigurierbaren Zeitfensters aktualisiert hat, sollte der Watchdog alarmieren und optional den feststeckenden Prozess terminieren, damit er neu gestartet werden kann.
- Konfigurieren Sie Eskalationsrichtlinien: Versuchen Sie zuerst automatische Wiederherstellung (Neustart des Prozesses), dann alarmieren Sie den Bereitschaftsingenieur, wenn automatische Wiederherstellung fehlschlägt oder derselbe Prozess mehr als eine Schwellenanzahl von Malen innerhalb eines Zeitfensters Wiederherstellung benötigt.
- Überwachen Sie den Watchdog selbst — ein Watchdog, der still versagt, ist schlimmer als gar kein Watchdog. Nutzen Sie ein sekundäres Monitoring-System oder gegenseitige Watchdog-Anordnungen, bei denen sich zwei Watchdog-Prozesse gegenseitig überwachen.

## Tradeoffs ⇄

> Watchdog-Prozesse bieten automatisierte Fehlererkennung und -wiederherstellung, was Ausfallzeit und den Bedarf an manuellem Eingreifen reduziert, fügen aber betriebliche Komplexität hinzu und können zugrunde liegende Probleme verschleiern, wenn sie nicht ordentlich verwaltet werden.

**Vorteile:**

- Erkennt Fehlschläge innerhalb von Sekunden oder Minuten statt Stunden, was die Auswirkung von Komponentenfehlern auf Nutzer dramatisch reduziert.
- Ermöglicht automatische Wiederherstellung von vorübergehenden Fehlschlägen (Prozessabstürze, Ressourcenerschöpfung), ohne menschliches Eingreifen zu erfordern.
- Bietet eine konsistente Monitoring-Schicht für Legacy-Komponenten, die nicht leicht mit modernen Observability-Werkzeugen instrumentiert werden können.
- Reduziert die betriebliche Last für Bereitschaftsingenieure, indem gängige Fehlerszenarien automatisch gehandhabt werden.

**Kosten und Risiken:**

- Automatische Neustarts können anhaltende zugrunde liegende Probleme verschleiern, indem ein fehlschlagender Prozess kontinuierlich neu gestartet wird, statt die Grundursache zu adressieren.
- Watchdog-ausgelöste Neustarts könnten Datenverlust oder -beschädigung verursachen, wenn der fehlschlagende Prozess nicht committeten Zustand hält, der bei Terminierung nicht ordentlich bereinigt wird.
- Der Watchdog selbst ist eine Komponente, die überwacht, gepflegt und hochverfügbar gehalten werden muss — ein ausgefallener Watchdog schafft ein falsches Sicherheitsgefühl.
- Übermäßig aggressive Neustart-Richtlinien können Thrashing verursachen, bei dem ein Prozess wiederholt startet, fehlschlägt, neu startet und erneut fehlschlägt, was die Situation potenziell verschlimmert.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie Watchdog-Prozesse Fehlschläge in Legacy-Systemen erkennen und davon wiederherstellen.

Ein Legacy-Messaging-System betreibt einen Java-basierten Message-Consumer, der eingehende Aufträge aus einer Warteschlange verarbeitet. Ungefähr einmal pro Woche stößt der Consumer auf einen seltenen Deserialisierungsfehler, der eine unbehandelte Exception verursacht und den JVM-Prozess terminiert. Ohne Watchdog bleibt der tote Consumer stundenlang unbemerkt, bis der Warteschlangentiefe-Alarm auslöst, zu welchem Zeitpunkt sich Tausende von Aufträgen angesammelt haben. Das Team konfiguriert eine systemd-Service-Unit für den Consumer mit automatischem Neustart bei Fehlschlag und einer Neustartgrenze von 5 Versuchen innerhalb von 10 Minuten. Sie setzen auch einen maßgeschneiderten Watchdog ein, der die Verarbeitungsrate des Consumers überwacht — wenn 3 aufeinanderfolgende Minuten lang keine Nachrichten verarbeitet werden, während die Warteschlange nicht leer ist, tötet der Watchdog den Prozess, um einen Neustart zu erzwingen, und alarmiert den Bereitschaftsingenieur. Nach der Implementierung werden Consumer-Fehlschläge automatisch innerhalb von 30 Sekunden wiederhergestellt, und das Bereitschaftsteam wird nur benachrichtigt, wenn wiederholte Fehlschläge ein systemisches Problem nahelegen, das Untersuchung erfordert.

Ein Legacy-Finanzberichtssystem führt nächtliche Batch-Jobs aus, die regulatorische Berichte generieren. Gelegentlich hängt ein Batch-Job unbegrenzt aufgrund eines Datenbank-Lock-Konkurrenzproblems, was alle nachfolgenden Jobs in der Pipeline blockiert. Das Betriebsteam entdeckt den feststeckenden Job am nächsten Morgen, wenn Berichte fehlen. Das Team implementiert einen Watchdog, der den Fortschritt jedes Batch-Jobs überwacht, indem er eine Fortschrittstabelle prüft, die die Jobs periodisch aktualisieren. Wenn ein Job seinen Fortschrittsmarker nicht innerhalb von 15 Minuten aktualisiert hat, terminiert der Watchdog den Job, gibt seine Datenbank-Locks frei und markiert ihn zur Wiederholung. Eine Benachrichtigung wird an das Betriebsteam gesendet, aber der Rest der Batch-Pipeline läuft weiter. Die nächtliche Batch-Suite wird nun zuverlässig jede Nacht abgeschlossen, und feststeckende-Job-Vorfälle, die zuvor 2-3 Stunden manuellen Eingriff erforderten, werden automatisch gelöst.
