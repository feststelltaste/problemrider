---
title: Optimistische UI-Updates
description: Reduzierung wahrgenommener Latenz durch Aktualisierung der
  Oberfläche vor Server-Bestätigung.
category:
- Performance
- Code
problems:
- slow-application-performance
- high-api-latency
- poor-user-experience-ux-design
- user-frustration
- external-service-delays
- network-latency
layout: solution
lang: de
en_slug: optimistic-ui-updates
related_solutions:
- slug: performance-optimization
  similarity: 0.75
- slug: predictive-loading
  similarity: 0.75
- slug: asynchronous-operations
  similarity: 0.75
- slug: api-calls-optimization
  similarity: 0.7
- slug: progressive-loading
  similarity: 0.65
- slug: lazy-loading
  similarity: 0.65
---

## Description

Optimistische UI-Updates ändern sofort nach einer Aktion, was der Nutzer sieht, noch bevor der Server die erfolgreiche Verarbeitung der Anfrage tatsächlich bestätigt hat, wobei Rollback-Logik die Oberfläche zurücksetzt, falls die Bestätigung nie eintrifft. Diese Technik verbirgt die Latenz langsamer Backend-Roundtrips, indem sie den häufigen Fall — Erfolg — annimmt und nur im seltenen Fehlerfall die Kosten der Korrektur zahlt. Sie ist besonders nützlich für Legacy-Systeme, deren APIs sich nicht leicht beschleunigen lassen, da die wahrgenommene Reaktionsfähigkeit der Anwendung auf der Präsentationsebene erheblich verbessert werden kann, ohne das Backend überhaupt anzufassen. Der Ansatz setzt voraus, dass der Server die optimistische Aktion sicher und idempotent ablehnen kann, weshalb er am besten zuerst auf vorhersagbare Interaktionen mit hoher Erfolgsquote angewendet wird, bevor er auf riskantere oder folgenreichere Operationen ausgeweitet wird.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Identifizieren Sie Nutzerinteraktionen, bei denen die Serverantwort vorhersagbar und die Erfolgsquote hoch ist (z. B. Umschalten einer Einstellung, Hinzufügen eines Elements zu einer Liste)
- Aktualisieren Sie den UI-Zustand sofort bei der Nutzeraktion, bevor die Serveranfrage abgeschlossen ist
- Implementieren Sie Rollback-Logik, die die Oberfläche in ihren vorherigen Zustand zurücksetzt, falls der Server einen Fehler zurückgibt
- Zeigen Sie dezente, nicht blockierende Indikatoren an, um zu kommunizieren, dass die Änderung gerade gespeichert wird
- Beginnen Sie mit risikoarmen Operationen und erweitern Sie auf komplexere Interaktionen, sobald das Team Vertrauen in das Muster gewinnt
- Stellen Sie serverseitige Idempotenz sicher, um Wiederholungen bei Rollbacks und erneuten Übermittlungen sauber zu handhaben

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Verbessert die wahrgenommene Reaktionsfähigkeit dramatisch, ohne Backend-Performance-Änderungen
- Reduziert Nutzerfrustration durch das Warten auf Server-Roundtrips
- Lässt die Anwendung modern und reaktionsschnell wirken, selbst wenn sie durch eine langsame Legacy-API gestützt wird
- Kann inkrementell auf spezifische Interaktionen angewendet werden, ohne das gesamte Frontend neu zu schreiben

**Kosten und Risiken:**
- Rollback-Logik fügt Komplexität hinzu und muss Grenzfälle wie gleichzeitige Aktualisierungen behandeln
- Nutzer können verwirrt sein, wenn eine Aktion zunächst erfolgreich erscheint, aber später zurückgesetzt wird
- Erfordert serverseitige Idempotenzgarantien, die Legacy-APIs möglicherweise nicht bieten
- Vergrößert die Lücke zwischen angezeigtem Zustand und tatsächlichem Serverzustand, was subtile Inkonsistenzen verursachen kann
- Nicht geeignet für Operationen, bei denen Fehlschläge häufig sind oder die Konsequenzen erheblich sind

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Projektmanagement-Tool, das auf einer Legacy-REST-API aufbaute, hatte durchschnittliche Antwortzeiten von 800 Millisekunden für Statusaktualisierungen. Nutzer klickten häufig doppelt oder navigierten weg, bevor Aktualisierungen abgeschlossen waren, was Verwirrung und doppelte Anfragen verursachte. Das Team implementierte optimistische Updates für Aufgabenstatusänderungen und spiegelte den neuen Status sofort in der Oberfläche wider, während der API-Aufruf im Hintergrund weiterlief. In den seltenen Fällen, in denen der Server das Update ablehnte, informierte eine Toast-Benachrichtigung den Nutzer, und der Status wurde zurückgesetzt. Diese Änderung reduzierte die wahrgenommene Latenz auf nahezu null und beseitigte Beschwerden über doppelte Übermittlungen, alles ohne das Legacy-Backend zu verändern.
