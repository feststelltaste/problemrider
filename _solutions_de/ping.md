---
title: Ping
description: Aktives Senden von Anfragen an eine Komponente, um ihre
  Verfügbarkeit zu prüfen.
category:
- Operations
problems:
- monitoring-gaps
- single-points-of-failure
- slow-incident-resolution
- system-outages
- service-discovery-failures
layout: solution
lang: de
en_slug: ping
related_solutions:
- slug: heartbeat
  similarity: 0.85
- slug: watchdog
  similarity: 0.75
- slug: failover-mechanisms
  similarity: 0.75
- slug: health-check-endpoints
  similarity: 0.75
- slug: monitoring
  similarity: 0.75
- slug: self-test
  similarity: 0.7
---

## Description

Ping ist die einfachste Form der Verfügbarkeitsüberwachung: Ein Monitoring-System sendet aktiv periodische Anfragen an eine Komponente und behandelt die Antwort — oder ihr Fehlen — als das Signal dafür, ob diese Komponente läuft. Die Verwendung von Anwendungsebenen-Pings, wie eine HTTP-Anfrage oder eine leichtgewichtige Datenbankabfrage, statt nur Netzwerkebenen-ICMP, erfasst Fehler aus der tatsächlichen Perspektive, die ein echter Aufrufer erlebt, was für Legacy-Komponenten zählt, die auf einen Netzwerk-Ping antworten können, während ihre Anwendungslogik still aufgehört hat zu funktionieren. Diese Technik wird in Legacy-Umgebungen gerade deshalb geschätzt, weil sie keine Instrumentierung der Komponente selbst erfordert — sie funktioniert gegen alles, das auf eine Anfrage antworten kann —, was sie zu einem kostengünstigen Weg macht, eine Überwachungslücke in Systemen zu schließen, die zu alt oder zu fragil sind, um direkt modifiziert zu werden, auf Kosten dessen, nur Fehler zu erkennen, die eine Antwort offenlegen kann, statt tiefere funktionale Korrektheit.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Implementieren Sie periodische Ping-Prüfungen von einem Monitoring-System zu allen kritischen Legacy-Dienstendpunkten
- Verwenden Sie Anwendungsebenen-Pings (HTTP-Anfragen, Datenbankabfragen) statt nur Netzwerkebenen-ICMP-Pings
- Konfigurieren Sie angemessene Timeout- und Wiederholungsschwellen, um vorübergehende Probleme von echten Fehlern zu unterscheiden
- Variieren Sie die Ping-Frequenz basierend auf der Kritikalität der Komponente: häufiger für kritische Dienste
- Beziehen Sie die Verfolgung von Ping-Antwortzeiten ein, um Verschlechterungstrends zu erkennen, bevor vollständige Ausfälle auftreten
- Integrieren Sie Ping-Ergebnisse mit Alarmierungssystemen, um Operations-Teams über Verfügbarkeitsprobleme zu informieren

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Bietet einfache, verlässliche Verfügbarkeitserkennung für Legacy-Komponenten
- Funktioniert mit jedem System, das auf Anfragen antworten kann, unabhängig von der Technologie
- Erkennt Fehler aus der Perspektive des Aufrufers, einschließlich Netzwerkprobleme
- Geringe Implementierungskosten und minimale Auswirkung auf überwachte Systeme

**Kosten und Risiken:**
- Eine Komponente kann auf Pings antworten, während sie funktional defekt ist
- Ping-Traffic fügt überwachten Diensten geringfügige Last hinzu
- Netzwerkebenen-Pings erkennen möglicherweise keine Fehler auf Anwendungsebene
- Falsch positive Ergebnisse durch Netzwerküberlastung können unnötige Alarme auslösen

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Vertriebsunternehmen betrieb mehrere Legacy-SOAP-Dienste, die gelegentlich ohne Protokollierung von Fehlern nicht mehr antworteten. Das Operations-Team erfuhr von Ausfällen nur, wenn Geschäftsanwender Probleme meldeten. Durch die Bereitstellung eines Monitoring-Agenten, der alle 15 Sekunden Anwendungsebenen-Ping-Anfragen an jeden Dienst sendete und alarmierte, wenn drei aufeinanderfolgende Pings fehlschlugen, reduzierte das Team die Fehlererkennungszeit von durchschnittlich 45 Minuten auf unter eine Minute. Der Verlauf der Ping-Antwortzeiten half zudem, ein schrittweises Performance-Verschlechterungsmuster zu identifizieren, das mit Speicherlecks zusammenhing.
