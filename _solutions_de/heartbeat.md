---
title: Heartbeat
description: Regelmäßige Übermittlung des Lebenszeichens einer Komponente an eine
  Monitoring-Instanz.
category:
- Operations
problems:
- monitoring-gaps
- single-points-of-failure
- slow-incident-resolution
- system-outages
- unpredictable-system-behavior
- constant-firefighting
layout: solution
lang: de
en_slug: heartbeat
related_solutions:
- slug: ping
  similarity: 0.85
- slug: watchdog
  similarity: 0.85
- slug: monitoring
  similarity: 0.8
- slug: failover-mechanisms
  similarity: 0.7
- slug: health-check-endpoints
  similarity: 0.7
- slug: self-monitoring-and-diagnosis
  similarity: 0.7
---

## Description

Ein Heartbeat ist ein Signal, das eine Komponente in regelmäßigen Abständen an ein Monitoring-System sendet, rein um zu bestätigen, dass sie noch läuft, im Unterschied zu einem Health-Check-Endpunkt dadurch, dass es von der Komponente selbst gepusht wird statt von einem externen Aufrufer bei Bedarf abgerufen zu werden. Dieses Push-basierte Modell passt besonders gut zu Legacy-Hintergrundprozessen und Batch-Jobs, die überhaupt keine anfragegetriebene Schnittstelle haben, die abgefragt werden könnte — ein nächtlicher Abgleichs-Job oder ein langlaufender Queue-Consumer kann melden „Ich lebe noch und bin an diesem Punkt meiner Arbeit", ohne dass jemand fragen muss. Die Abwesenheit eines erwarteten Heartbeats, statt der Anwesenheit eines Fehlers, wird zum handlungsleitenden Signal: Wenn ein Legacy-Job still hängt, ohne Exception und ohne Log-Ausgabe — ein häufiger Fehlermodus in älteren Systemen mit minimaler Instrumentierung —, ist der fehlende Heartbeat oft das einzige Anzeichen, dass überhaupt etwas nicht stimmt. Dies verschiebt die Erkennung von passiver Entdeckung — ein nachgelagertes Team bemerkt Stunden später schließlich fehlende Daten — zu aktiver Alarmierung innerhalb von Sekunden nach Ablauf des erwarteten Intervalls, was häufig eine der günstigsten verfügbaren Monitoring-Verbesserungen für Legacy-Systeme ist, die modernem Observability-Tooling vorausgehen. Weil eine Komponente weiterhin Heartbeats senden kann, während sie sich falsch verhält, beweist der Mechanismus nur, dass ein Prozess lebt, nicht dass er seine Arbeit korrekt macht, sodass er tieferes funktionales Monitoring ergänzt statt es zu ersetzen.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Instrumentieren Sie Legacy-Komponenten, um periodische Heartbeat-Signale an ein zentrales Monitoring-System zu senden
- Definieren Sie angemessene Heartbeat-Intervalle basierend auf der Kritikalität und erwarteten Reaktionszeit jeder Komponente
- Konfigurieren Sie Alarmierungsregeln, die auslösen, wenn Heartbeats für eine definierte Anzahl aufeinanderfolgender Intervalle ausbleiben
- Beziehen Sie grundlegende Gesundheitsmetadaten in Heartbeat-Payloads ein (Speichernutzung, Queue-Tiefe, Zeitstempel der letzten Verarbeitung)
- Implementieren Sie Heartbeat-Empfänger, die den Status über alle überwachten Komponenten hinweg in einem Dashboard aggregieren
- Nutzen Sie das Ausbleiben eines Heartbeats als Auslöser für automatisierte Wiederherstellungsmaßnahmen wie Prozessneustarts

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Erkennt Komponentenausfälle innerhalb von Sekunden statt auf Nutzerberichte zu warten
- Bietet kontinuierlichen Lebensbeweis für Hintergrundprozesse und Batch-Jobs
- Einfach umzusetzen selbst in Legacy-Systemen mit begrenzter Monitoring-Infrastruktur
- Ermöglicht automatisiertes Failover durch Erkennung nicht antwortender Komponenten

**Kosten und Risiken:**
- Netzwerkprobleme können falsch-positive Ausfallerkennungen verursachen
- Heartbeat-Mechanismen fügen geringfügigen Netzwerk- und Verarbeitungs-Overhead hinzu
- Eine Komponente kann Heartbeats senden, während sie funktional defekt ist (lebendig, aber nicht korrekt arbeitend)
- Das Monitoring-System selbst wird zu einer Abhängigkeit, die verfügbar gehalten werden muss

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Fertigungsunternehmen betrieb kritische Batch-Verarbeitungsjobs auf einem Legacy-System, das gelegentlich ohne jegliche Fehlerausgabe hängen blieb. Das Betriebsteam entdeckte stecken gebliebene Jobs oft erst, wenn nachgelagerte Systeme Stunden später fehlende Daten meldeten. Durch das Hinzufügen eines einfachen Heartbeat-Mechanismus, bei dem jeder Batch-Job alle 30 Sekunden Fortschritt meldete, konnte das Monitoring-System stecken gebliebene Jobs innerhalb einer Minute erkennen und automatisch neu starten. Dies verringerte die durchschnittliche Erkennungszeit von vier Stunden auf unter zwei Minuten.
