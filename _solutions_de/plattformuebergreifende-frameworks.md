---
title: Plattformübergreifende Frameworks
description: Nutzung von Entwicklungs-Frameworks, die plattformübergreifende Anwendungen
  ermöglichen.
category:
- Architecture
- Code
problems:
- technology-lock-in
- vendor-lock-in
- difficult-code-reuse
- high-maintenance-costs
- duplicated-effort
- scaling-inefficiencies
layout: solution
lang: de
en_slug: cross-platform-frameworks
related_solutions:
- slug: platform-independent-programming-languages
  similarity: 0.8
- slug: cross-platform-build-tools
  similarity: 0.8
- slug: platform-independence
  similarity: 0.75
- slug: platform-independent-data-storage
  similarity: 0.75
- slug: platform-independent-test-frameworks
  similarity: 0.75
- slug: emulation
  similarity: 0.75
---

## Description

Plattformübergreifende Frameworks wie Kotlin Multiplatform, Flutter oder .NET MAUI erlauben einer einzelnen Codebasis, mehrere Plattformen anzusprechen, typischerweise indem Geschäftslogik über Plattformen hinweg geteilt wird, während echt plattformspezifische Belange — UI-Rendering, Hardwarezugriff — dort nativ implementiert bleiben, wo nötig. Organisationen, die separate native Anwendungen für jede Plattform betreiben, gepflegt von separaten Teams, entdecken häufig, dass Feature-Parität zwischen ihnen ein permanenter, verlorener Kampf ist: Das Team einer Plattform liefert schneller aus als das der anderen, und die Lücke zwischen den beiden Versionen weitet sich mit jedem Release-Zyklus, statt sich zu schließen. Die gemeinsame Geschäftslogik — die Teile, die nicht inhärent von einer spezifischen Plattform abhängen, etwa Domänenregeln, Terminplanungslogik oder Offline-Synchronisation — auf ein plattformübergreifendes Framework zu migrieren entfernt den duplizierten Implementierungsaufwand, der die Paritätslücke überhaupt erst verursacht hat, ohne notwendigerweise die plattformspezifischen UI-Schichten anzufassen, die am meisten davon profitieren, nativ zu bleiben. Weil eine vollständige Neuschreibung zweier etablierter nativer Codebasen auf einmal hochriskant ist, wird dies typischerweise als schrittweise Migration angegangen, die mit der am klarsten trennbaren, nicht-plattformspezifischen Logik beginnt und sich von dort ausdehnt. Der Tradeoff ist eine neue Abhängigkeit von der eigenen Roadmap und Plattform-Feature-Abdeckung des Frameworks, ein potenzieller Performance-Nachteil für UI- oder hardwareintensive Operationen und die Realität, dass nicht jede Legacy-Codebasis überhaupt sauber in gemeinsam nutzbare und plattformspezifische Schichten getrennt werden kann.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Bewerten Sie plattformübergreifende Frameworks (React Native, Flutter, .NET MAUI, Kotlin Multiplatform, Electron) basierend auf den Anforderungen der Anwendung
- Identifizieren Sie den Teil des Legacy-Codes, der Geschäftslogik enthält, die von plattformspezifischem UI- oder Systemcode trennbar ist
- Beginnen Sie damit, gemeinsame Geschäftslogik auf das plattformübergreifende Framework zu portieren, während plattformspezifische Features nativ bleiben
- Nutzen Sie die Plattform-Kanal-Mechanismen des Frameworks für den Zugriff auf native Fähigkeiten, die das Framework nicht abdeckt
- Etablieren Sie eine Teststrategie, die sowohl gemeinsamen Code als auch plattformspezifische Anpassungen abdeckt
- Planen Sie eine schrittweise Migration, statt die gesamte Anwendung auf einmal neu zu schreiben
- Überwachen Sie plattformspezifische Performance, um sicherzustellen, dass die plattformübergreifende Schicht keinen inakzeptablen Overhead einführt

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Reduziert Entwicklungsaufwand, indem Code über Plattformen hinweg geteilt wird, statt separate Codebasen zu pflegen
- Stellt konsistentes Verhalten und Feature-Parität über Plattformen hinweg sicher
- Ermöglicht kleineren Teams, mehrere Plattformen gleichzeitig zu unterstützen
- Reduziert die Time-to-Market für Features, indem sie einmal implementiert werden

**Kosten und Risiken:**
- Plattformübergreifende Frameworks unterstützen möglicherweise nicht alle nativen Plattform-Features oder hinken Plattform-Updates hinterher
- Die Performance kann bei UI- oder hardwareintensiven Operationen niedriger sein als bei vollständig nativen Implementierungen
- Schafft Abhängigkeit von der Roadmap und dem Support-Lebenszyklus des Framework-Anbieters
- Entwickler müssen möglicherweise framework-spezifische Muster zusätzlich zum Plattformwissen lernen
- Nicht alle Legacy-Codebasen können sauber in gemeinsam nutzbare und plattformspezifische Schichten getrennt werden

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Außendienstunternehmen pflegte separate Legacy-Anwendungen für iOS (Objective-C) und Android (Java), jede mit ihrem eigenen Entwicklungsteam. Feature-Parität war ein ständiger Kampf, wobei die Android-Version typischerweise drei Monate hinter iOS lag. Das Team migrierte gemeinsame Geschäftslogik (Arbeitsauftragsverwaltung, Terminplanung, Offline-Sync) zu Kotlin Multiplatform, während die UI nativ blieb. Dies reduzierte die Codebasis um 40 Prozent, eliminierte die Feature-Paritätslücke und erlaubte einem Entwickler aus jedem Plattformteam, zu anderen Projekten zu wechseln. Kritische plattformspezifische Features wie Hintergrund-GPS-Tracking blieben nativ, was sicherstellte, dass keine Funktionalität verloren ging.
