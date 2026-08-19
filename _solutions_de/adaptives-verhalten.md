---
title: Adaptives Verhalten
description: Anpassung des Systemverhaltens basierend auf Kontext, Präferenzen oder
  Verhalten des Nutzers.
category:
- Requirements
- Architecture
problems:
- poor-user-experience-ux-design
- customer-dissatisfaction
- user-frustration
- negative-user-feedback
- feature-bloat
- user-confusion
- declining-business-metrics
layout: solution
lang: de
en_slug: adaptive-behavior
related_solutions:
- slug: intuitive-navigation
  similarity: 0.8
- slug: a-b-testing
  similarity: 0.75
- slug: customizing
  similarity: 0.75
- slug: accessibility-concept
  similarity: 0.75
- slug: consistent-user-interface
  similarity: 0.75
- slug: cognitive-load-minimization
  similarity: 0.75
---

## Description

Adaptives Verhalten bedeutet, anzupassen, was ein System zeigt oder wie es sich verhält, basierend auf dem Kontext, der Rolle, den Präferenzen oder den beobachteten Nutzungsmustern des einzelnen Nutzers, statt jedem Nutzer eine identische, einheitliche Schnittstelle zu präsentieren. Konkret kann dies rollenbasierte Standardwerte, personalisierte Dashboards, progressive Offenlegung fortgeschrittener Funktionalität oder Navigation bedeuten, die die am häufigsten genutzten Features eines Nutzers hervorhebt, statt eines erschöpfenden, undifferenzierten Menüs. Legacy-Anwendungen wuchsen üblicherweise durch Akkretion, wobei Feature um Feature auf dieselben Bildschirme für jeden Nutzer unabhängig von der Rolle hinzugefügt wurde, bis die Schnittstelle die Vereinigung der Bedürfnisse aller statt des tatsächlichen Workflows einer Person widerspiegelt, was hohe kognitive Last und niedrige Zufriedenheit produziert, obwohl die zugrunde liegende Funktionalität solide ist. Die Einführung adaptiven Verhaltens erlaubt es, die bestehende Funktionalität eines Legacy-Systems nützlicher wieder zugänglich zu machen, ohne ein vollständiges UI-Redesign, da die zugrunde liegenden Operationen dieselben bleiben und sich nur die Präsentation und Standardwerte basierend auf Interaktionsdaten oder Rolle ändern. Dies ist ein relativ risikoarmer, inkrementeller Weg, die wahrgenommene Nutzbarkeit einer Legacy-UI zu modernisieren, weil es auf bestehende Bildschirme geschichtet und graduell an Nutzersegmente ausgerollt werden kann. Der Kompromiss ist zusätzliche Komplexität und Testfläche, da sich das System nun über viele verschiedene personalisierte Konfigurationen hinweg korrekt verhalten muss statt eines einheitlichen Pfads, und inkonsistente Anpassung kann Nutzer selbst verwirren, wenn sie nicht sorgfältig designt und kommuniziert wird.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Analysieren Sie Nutzerinteraktionsdaten, um unterschiedliche Nutzungsmuster und Nutzersegmente innerhalb der Legacy-Anwendung zu identifizieren
- Implementieren Sie Speicherung von Nutzerpräferenzen, um Personalisierung häufig genutzter Features und Workflows zu erlauben
- Fügen Sie kontextbewusste Standardwerte hinzu, die sich basierend auf Nutzerrolle, Abteilung oder vergangenem Verhalten anpassen
- Führen Sie progressive Offenlegung fortgeschrittener Features ein, um Komplexität für Gelegenheitsnutzer zu verringern
- Implementieren Sie responsives Verhalten, das sich an Gerätefähigkeiten und Bildschirmgrößen anpasst
- Erstellen Sie konfigurierbare Dashboards oder Landing-Pages, die die relevantesten Informationen pro Nutzerprofil hervorheben
- Nutzen Sie Feature-Nutzungsanalytik zur Identifikation und Priorisierung, welche adaptiven Verhaltensweisen die größte Auswirkung haben werden

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Verbessert Nutzerzufriedenheit durch Verringerung von Reibung und Hervorhebung relevanter Funktionalität
- Verringert Schulungsbedarf, indem Komplexität progressiv basierend auf Nutzerkompetenz präsentiert wird
- Erhöht Produktivität durch Anpassung von Workflows an individuelle Nutzungsmuster
- Lässt Legacy-Anwendungen moderner wirken, ohne vollständige UI-Neuschreibungen

**Kosten und Risiken:**
- Adaptives Verhalten fügt der Codebasis Komplexität hinzu und erhöht Testanforderungen
- Nutzer können verwirrt werden, wenn sich das System anders als erwartet oder inkonsistent verhält
- Personalisierungsfeatures erfordern Nutzerdatensammlung, was Datenschutzüberlegungen aufwirft
- Legacy-Systeme mit starren UI-Architekturen können sich gegen das Hinzufügen adaptiver Komponenten wehren
- Übermäßige Anpassung kann es Nutzern erschweren, Features zu entdecken, die durch die Personalisierungslogik verborgen sind

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein von 3.000 Mitarbeitern genutztes Legacy-ERP-System präsentierte jedem Nutzer unabhängig von seiner Rolle dasselbe 47-Punkte-Navigationsmenü. Power-User in der Buchhaltung nutzten täglich 12 Funktionen, während Lagerpersonal nur 4 nutzte. Das Team führte rollenbasierte Menüanpassung ein, die jedem Nutzer eine auf seine Abteilung zugeschnittene Standardansicht zeigte, mit Zugriff auf das vollständige Menü über eine „Alle Module"-Option. Sie fügten außerdem einen „häufig genutzt"-Bereich hinzu, der automatisch die meistgenutzten Funktionen jedes Nutzers hervorhob. Die Nutzerzufriedenheitswerte stiegen um 35 %, und die durchschnittliche Zeit zum Erreichen häufig genutzter Funktionen sank um 50 %, was einer Schnittstelle, über die sich Nutzer lange beschwert hatten, neues Leben einhauchte.
