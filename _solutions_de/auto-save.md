---
title: Auto-Save
description: Automatisches Speichern der Nutzerarbeit in regelmäßigen Abständen gegen
  Datenverlust.
category:
- Requirements
quality_tactics_url: https://qualitytactics.de/en/usability/auto-save/
problems:
- user-frustration
- poor-user-experience-ux-design
- negative-user-feedback
- customer-dissatisfaction
- user-trust-erosion
- increased-customer-support-load
layout: solution
lang: de
en_slug: auto-save
related_solutions:
- slug: intuitive-navigation
  similarity: 0.75
- slug: confirmation-dialogs
  similarity: 0.75
- slug: asynchronous-operations
  similarity: 0.75
- slug: undo-and-redo
  similarity: 0.75
- slug: search-function
  similarity: 0.75
- slug: user-centered-design
  similarity: 0.75
---

## Description

Auto-Save speichert periodisch die in Bearbeitung befindliche Arbeit eines Nutzers — ein Formular, ein Dokument —, ohne eine explizite Speicheraktion zu erfordern, sodass ein Sitzungs-Timeout, ein Absturz oder einfaches Vergessen, auf „Speichern" zu klicken, nicht mehr bedeutet, dass die Arbeit verloren ist. Legacy-Systeme, die sich rein auf explizites manuelles Speichern verlassen, sind dieser Fehlerart besonders ausgesetzt, oft verschärft durch aggressive Sitzungs-Timeouts, die aus Gründen konfiguriert wurden, an die sich heute niemand mehr erinnert. Es gut zu implementieren bedeutet, Entwürfe zu speichern, selbst während ein Formular unvollständig oder ungültig ist, einen klaren Indikator dafür zu zeigen, wann die letzte Speicherung stattfand, und den Konflikt zu handhaben, der entsteht, wenn ein wiederhergestellter Entwurf mit an anderer Stelle vorgenommenen Änderungen kollidiert.

## How to Apply ◆

> Legacy-Systeme verlassen sich häufig auf explizite manuelle Speicherung, was zu Datenverlust führt, wenn Sitzungen ablaufen, Browser abstürzen oder Nutzer das Speichern vergessen. Auto-Save schützt Nutzerarbeit transparent.

- Implementieren Sie periodisches Auto-Save, das den aktuellen Formular- oder Dokumentzustand in regelmäßigen Abständen erfasst, typischerweise alle dreißig bis sechzig Sekunden. Speichern Sie Entwürfe serverseitig oder im lokalen Speicher als Fallback.
- Fügen Sie visuelle Indikatoren hinzu, die zeigen, wann Daten zuletzt automatisch gespeichert wurden, was Nutzern Vertrauen gibt, dass ihre Arbeit geschützt ist, ohne dass sie über das Speichern nachdenken müssen.
- Implementieren Sie Konfliktlösung für Szenarien, in denen automatisch gespeicherte Daten mit Änderungen kollidieren, die von einem anderen Nutzer oder in einer anderen Sitzung vorgenommen wurden. Zeigen Sie eine klare Vergleichsansicht statt stillschweigend zu überschreiben.
- Fügen Sie Sitzungswiederherstellung hinzu, die ungespeicherte Arbeit aus einer vorherigen Sitzung erkennt und anbietet, sie wiederherzustellen, wenn der Nutzer zurückkehrt. Dies ist kritisch für Legacy-Systeme mit aggressiven Sitzungs-Timeouts.
- Stellen Sie sicher, dass Auto-Save die Validierung nicht beeinträchtigt. Speichern Sie Entwürfe, selbst wenn das Formular in einem unvollständigen oder ungültigen Zustand ist, und verschieben Sie Validierung auf die explizite Absenden-Aktion.
- Drosseln Sie Auto-Save-Anfragen, um den Server nicht zu überlasten, besonders in Legacy-Systemen mit begrenzter Backend-Kapazität.

## Tradeoffs ⇄

> Auto-Save eliminiert eine ganze Kategorie von Nutzerfrustration, erfordert aber sorgfältige Implementierung, um Datenkonsistenzprobleme zu vermeiden.

**Vorteile:**

- Eliminiert Datenverlust durch Sitzungs-Timeouts, Browserabstürze und versehentliche Navigation, die zu den frustrierendsten Erfahrungen in Legacy-Systemen gehören.
- Verringert Support-Tickets im Zusammenhang mit verlorener Arbeit, was direkt die Kundensupport-Last verringert.
- Baut Nutzervertrauen auf, indem demonstriert wird, dass das System die Arbeit der Nutzer schützt, statt sie bei der geringsten Störung zu verwerfen.
- Entfernt die kognitive Last, sich ans Speichern zu erinnern, und erlaubt Nutzern, sich auf ihre eigentlichen Aufgaben zu fokussieren.

**Kosten und Risiken:**

- Das automatische Speichern unvollständiger oder ungültiger Daten erfordert sorgfältige Handhabung, um zu vermeiden, dass die Datenbank mit nie finalisierten Entwurfsdatensätzen verschmutzt wird.
- Erhöht die Serverlast durch häufige Speicheranfragen, was bereits ressourcenbeschränkte Legacy-Backends belasten könnte.
- Konfliktlösung zwischen automatisch gespeicherten Entwürfen und gleichzeitigen Bearbeitungen fügt Komplexität hinzu, besonders in Legacy-Mehrnutzersystemen ohne optimistisches Sperren.
- Nutzer, die an explizites Speichern gewöhnt sind, könnten durch Auto-Save-Verhalten verwirrt sein, wenn es nicht klar über die Schnittstelle kommuniziert wird.

## How It Could Be

> Datenverlust ist eine anhaltende Frustrationsquelle in Legacy-Systemen, und Auto-Save adressiert direkt die häufigsten Szenarien.

Ein Legacy-Beschaffungssystem verlangt von Nutzern, mehrseitige Beschaffungsantragsformulare auszufüllen. Das System hat ein dreißigminütiges Sitzungs-Timeout, und Nutzer verlieren häufig ihre Arbeit, wenn sie pausieren, um Informationen aus anderen Systemen zu sammeln. Das Team implementiert Auto-Save unter Nutzung des lokalen Speichers für den Entwurfszustand, mit serverseitiger Persistenz alle sechzig Sekunden. Wenn ein Nutzer nach einem Sitzungs-Timeout zurückkehrt, erkennt das System den gespeicherten Entwurf und bietet an, ihn wiederherzustellen. Support-Tickets für verlorene Arbeit sinken von etwa zwanzig pro Woche auf nahezu null, und die Nutzerzufriedenheitswerte für das Beschaffungsmodul verbessern sich in der nächsten internen Umfrage erheblich.
