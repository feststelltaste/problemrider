---
title: Bridges
description: Abstraktionshierarchien und Implementierungshierarchien unabhängig voneinander
  weiterentwickeln lassen.
category:
- Architecture
- Code
problems:
- high-coupling-low-cohesion
- tight-coupling-issues
- monolithic-architecture-constraints
- difficult-code-reuse
- ripple-effect-of-changes
- technology-lock-in
layout: solution
lang: de
en_slug: bridges
related_solutions:
- slug: abstraction
  similarity: 0.75
- slug: abstraction-layers
  similarity: 0.75
- slug: facades
  similarity: 0.75
- slug: protocol-abstraction
  similarity: 0.75
- slug: adapter
  similarity: 0.75
- slug: database-abstraction
  similarity: 0.75
---

## Description

Das Bridge-Muster trennt eine Abstraktionshierarchie (was ein Client zu rufen glaubt) von ihrer Implementierungshierarchie (wie diese Operation tatsächlich ausgeführt wird), indem beide durch eine zur Konstruktionszeit injizierte Schnittstelle statt durch Vererbung verbunden werden, sodass sich jede Hierarchie erweitern lässt, ohne die andere zu berühren oder zu duplizieren. Sein Zweck ist es, die kombinatorische Explosion zu vermeiden, die auftritt, wenn ein System mehrere Varianten entlang zweier unabhängiger Dimensionen unterstützen muss — mehrere Ausgabeformate, mehrere Plattformen, mehrere Treiber — und vererbungsbasierte Designs darauf reagieren, indem sie eine Klasse pro Kombination erstellen. Legacy-Systeme gelangen häufig organisch genau in diesen Zustand: Eine für eine Implementierungsvariante erstellte Klassenhierarchie wird für jede neue kopiert und angepasst, weil die Einführung einer ordentlichen Abstraktionsgrenze unter Terminsdruck mehr Arbeit war als das Duplizieren einer bestehenden Klasse. Einen Bridge in eine solche Hierarchie nachzurüsten bedeutet, zu identifizieren, wo Abstraktions- und Implementierungsbelange vermischt sind, und schrittweise eine Bridge-Schnittstelle für eine Implementierungsvariante nach der anderen zu extrahieren, während die Legacy-Hierarchie für den Rest unverändert weiterfunktioniert. Der Nutzen ist eine scharfe Verringerung duplizierter Logik und weit günstigere Unterstützung für neue Varianten in Zukunft, auf Kosten einer zusätzlichen Indirektionsschicht, die sich nur lohnt, sobald ein System genuin mehr als eine Implementierungsvariante braucht.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Identifizieren Sie Stellen, an denen Klassenhierarchien Abstraktionsbelange mit Implementierungsdetails vermischen (z. B. PlatformXRenderer, PlatformYRenderer)
- Trennen Sie die Abstraktionshierarchie von der Implementierungshierarchie, indem Sie eine Bridge-Schnittstelle zwischen ihnen einführen
- Injizieren Sie die Implementierung zur Konstruktionszeit durch die Bridge, statt sie zu vererben
- Nutzen Sie dieses Muster, wenn ein Legacy-System mehrere Plattformen, Treiber oder Rendering-Backends unterstützen muss, ohne Logik zu duplizieren
- Refaktorieren Sie schrittweise, indem Sie eine Implementierungsvariante nach der anderen überbrücken, während Sie die Legacy-Hierarchie funktionsfähig halten

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Sowohl Abstraktion als auch Implementierung können unabhängig erweitert werden, ohne kombinatorische Klassenexplosion
- Vereinfacht das Hinzufügen neuer Plattform- oder Technologieunterstützung zu einem Legacy-System
- Verringert Code-Duplizierung über Implementierungsvarianten hinweg

**Kosten und Risiken:**
- Fügt strukturelle Komplexität hinzu, die für Systeme mit nur einer Implementierungsvariante übermäßig sein kann
- Erfordert sorgfältiges Schnittstellendesign an der Bridge-Grenze
- Entwickler, die mit dem Muster nicht vertraut sind, könnten die Indirektion verwirrend finden
- Die Nachrüstung des Musters in eine tief verwobene Legacy-Hierarchie kann ohne gute Testabdeckung riskant sein

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Fertigungsunternehmen hatte ein Berichtssystem mit separaten Klassenhierarchien für jedes Ausgabeformat (PDF, Excel, CSV), jede erhebliche Rendering-Logik duplizierend. Durch die Einführung eines Bridge-Musters, das die Berichtsstruktur vom Ausgabe-Rendering trennte, verringerte das Team die Codebasis um 35 % und konnte ein neues HTML-Ausgabeformat in zwei Tagen hinzufügen, statt der drei Wochen, die zuvor für das Klonen und Anpassen einer gesamten Hierarchie nötig gewesen waren.
