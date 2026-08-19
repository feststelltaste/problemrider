---
title: Bild- und Asset-Optimierung
description: Optimierung von Bildern, Schriftarten und statischen Assets für kleinere
  Nutzlasten und schnellere Ladezeiten.
category:
- Performance
problems:
- slow-application-performance
- high-client-side-resource-consumption
- inefficient-frontend-code
- high-resource-utilization-on-client
- gradual-performance-degradation
layout: solution
lang: de
en_slug: image-and-asset-optimization
related_solutions:
- slug: tree-shaking
  similarity: 0.8
- slug: code-splitting
  similarity: 0.8
- slug: compression
  similarity: 0.75
- slug: lazy-loading
  similarity: 0.75
- slug: api-calls-optimization
  similarity: 0.75
- slug: performance-optimization
  similarity: 0.75
---

## Description

Bild- und Asset-Optimierung verringert die Größe der Bilder, Schriftarten und anderen statischen Dateien, die eine Legacy-Webanwendung an den Browser ausliefert, mittels Techniken wie der Umwandlung in moderne komprimierte Formate, der Erzeugung mehrerer Auflösungen für responsive Auslieferung, dem Subsetting von Schriftarten auf nur die tatsächlich genutzten Zeichen und dem Verzögern des Ladens von allem, was nicht sofort sichtbar ist. Legacy-Webanwendungen haben diese Assets häufig über viele Jahre angesammelt, ohne dass jemals eine Optimierungs-Pipeline vorhanden war — Fotos in voller Auflösung, unverändert an mobile Geräte ausgeliefert, vollständige Schriftfamilien geladen für eine Handvoll Glyphen, Bilder eifrig geladen unabhängig davon, ob sie jemals ins Sichtfeld gescrollt werden —, weil Asset-Auslieferung zum Zeitpunkt, als der Code geschrieben wurde, nicht als erstrangiges Performance-Anliegen behandelt wurde. Der Effekt verstärkt sich auf langsamen oder mobilen Verbindungen, wo eine aus Jahren nicht optimierter Assets zusammengesetzte Seite Dutzende Megabyte wiegen und viele Sekunden brauchen kann, um nutzbar zu werden, was direkt genau die Art gradueller Performance-Verschlechterung und hohen clientseitigen Ressourcenverbrauchs antreibt, die ein alterndes System träge wirken lässt, selbst wenn seine Backend-Logik unverändert ist. Diese Optimierung innerhalb der Build-Pipeline zu automatisieren, statt sich auf Disziplin beim Hochladen zu verlassen, stellt sicher, dass neu hinzugefügte Assets nicht still dasselbe Problem wieder einführen, das der Aufwand lösen sollte. Weil ältere Browser, die noch in der Nutzerbasis eines Legacy-Systems sind, möglicherweise die neuesten Formate nicht unterstützen, muss die Optimierungsarbeit typischerweise einen Fallback-Pfad einschließen, und aggressive Komprimierung muss sorgfältig abgestimmt werden, um nicht Ladezeit gegen sichtbar verschlechterte Qualität einzutauschen.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Prüfen Sie bestehende Assets, um überdimensionierte Bilder, nicht optimierte Schriftarten und unnötige statische Dateien zu identifizieren
- Wandeln Sie Bilder in moderne Formate (WebP, AVIF) mit angemessenen Fallbacks für ältere Browser um
- Implementieren Sie responsive Bilder mittels srcset, um für jedes Gerät angemessen große Bilder auszuliefern
- Subsetten Sie Web-Schriftarten, um nur die tatsächlich in der Anwendung genutzten Zeichensätze einzubeziehen
- Setzen Sie angemessene Cache-Header für statische Assets und nutzen Sie inhaltsgehashte Dateinamen für Cache Busting
- Implementieren Sie Lazy Loading für Bilder und Assets unterhalb der Bildschirmkante oder die nicht sofort sichtbar sind
- Automatisieren Sie Asset-Optimierung in der Build-Pipeline, damit neue Assets vor dem Deployment immer optimiert sind

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Verringert Seitenladezeiten, besonders auf langsamen mobilen Verbindungen
- Verringert Bandbreitenkosten sowohl für den Server als auch den Endnutzer
- Verbessert Core-Web-Vitals-Werte, die die Suchmaschinenplatzierung beeinflussen
- Verringert den Speicherverbrauch auf Client-Geräten

**Kosten und Risiken:**
- Moderne Bildformate werden möglicherweise nicht von allen Browsern in der Nutzerbasis der Legacy-Anwendung unterstützt
- Aggressive Komprimierung kann die visuelle Qualität inakzeptabel verschlechtern
- Änderungen an der Build-Pipeline können Tooling-Aktualisierungen im Legacy-Projekt erfordern
- Die Implementierung responsiver Bilder fügt HTML-Komplexität hinzu

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Eine Legacy-Immobilienangebotsplattform lieferte Fotos in voller Auflösung (durchschnittlich 3 MB je Bild) direkt an alle Geräte aus, einschließlich Mobiltelefone. Eine typische Angebotsseite mit 20 Bildern wog über 60 MB. Das Team implementierte eine Bildverarbeitungspipeline, die WebP-Varianten in mehreren Auflösungen erzeugte, die passende Größe basierend auf dem Geräte-Viewport auslieferte und Bilder unterhalb der Bildschirmkante lazy lud. Das mittlere Seitengewicht sank von 60 MB auf 4 MB. Die mobile Seitenladezeit verbesserte sich von 12 Sekunden auf 2,5 Sekunden bei einer typischen 4G-Verbindung, und die monatlichen CDN-Bandbreitenkosten des Unternehmens sanken erheblich.
