---
title: Customizing
description: Anpassung von Software an spezifische Anforderungen und Bedürfnisse
  der Nutzer.
category:
- Requirements
- Business
problems:
- feature-gaps
- poor-user-experience-ux-design
- user-frustration
- customer-dissatisfaction
- negative-user-feedback
- vendor-lock-in
layout: solution
lang: de
en_slug: customizing
related_solutions:
- slug: adaptive-behavior
  similarity: 0.75
- slug: consistent-user-interface
  similarity: 0.75
- slug: customizable-user-interface
  similarity: 0.7
- slug: standard-software
  similarity: 0.7
- slug: a-b-testing
  similarity: 0.7
- slug: explicit-extension-points
  similarity: 0.7
---

## Description

Customizing passt das Verhalten eines Systems an die spezifischen Anforderungen unterschiedlicher Nutzergruppen an, mittels konfigurationsgetriebener Mechanismen — Feature Flags, mandantenspezifische Einstellungen, Plugin-Architekturen, Erweiterungspunkte — statt durch fest codierte, einheitliche Logik, die jeden Nutzer gleich behandelt. Legacy-Systeme sind besonders anfällig für diese Starrheit, weil ihre Kern-Workflows typischerweise um die Bedürfnisse einer einzigen ursprünglichen Nutzergruppe herum gestaltet wurden, und während die Nutzerbasis des Systems sich über die Zeit diversifizierte, wurden Gruppen, deren Workflows nicht zu diesem ursprünglichen Design passen, dazu gebracht, ihre eigenen Workarounds — Tabellenkalkulationen, manuelle Prozesse, Schattensysteme — außerhalb der Software statt innerhalb zu bauen. Erweiterungspunkte und konfigurationsgetriebenes Verhalten einzuführen erlaubt es, diese divergierenden Bedürfnisse zu erfüllen, ohne das Kernsystem für jeden einzelnen zu modifizieren, und, entscheidend, ohne die Codebasis in parallele, separat gepflegte Versionen für jede Nutzergruppe zu forken. Customizing vom Kerncode zu trennen schützt auch die Investition während Upgrades, da benutzerdefinierte Konfiguration, die außerhalb des Kernsystems liegt, nicht Gefahr läuft, beim nächsten Update der zugrunde liegenden Software still überschrieben zu werden. Unkontrolliert gelassen hat Customizing jedoch die Tendenz, seine eigene technische Schuld anzuhäufen, da jeder zusätzliche Customizing-Punkt die Testmatrix vervielfacht und eine neue Gelegenheit für konfigurationsbezogene Fehler und unerwartete Interaktionen zwischen unterschiedlichen Anpassungen schafft.

## How to Apply ◆

- Identifizieren Sie Bereiche, in denen der Einheitsansatz des Legacy-Systems spezifische Nutzergruppen im Stich lässt, und priorisieren Sie Customizing-Bemühungen entsprechend.
- Führen Sie konfigurationsgetriebenes Verhalten ein (Feature Flags, Nutzerpräferenzen, mandantenspezifische Einstellungen), statt fest codierter Logik.
- Bauen Sie Erweiterungspunkte im Legacy-System, die nutzerspezifisches Verhalten erlauben, ohne den Kerncode zu modifizieren.
- Nutzen Sie Plugin-Architekturen oder Strategy-Muster, um Geschäftsregeln ohne Codeänderungen anpassbar zu machen.
- Sammeln Sie systematisch Nutzerfeedback, um zu verstehen, welche Customizing-Optionen den meisten Wert liefern.
- Stellen Sie sicher, dass Anpassungen über Upgrades hinweg pflegbar bleiben, indem Sie benutzerdefinierten Code vom Kernsystem trennen.

## Tradeoffs ⇄

**Vorteile:**
- Erhöht die Nutzerzufriedenheit, indem das System an tatsächliche Arbeitsabläufe angepasst wird, statt Nutzer zur Anpassung zu zwingen.
- Reduziert den Bedarf an Workarounds und Schattensystemen, die Nutzer erstellen, wenn die Software nicht zu ihren Bedürfnissen passt.
- Ermöglicht demselben Legacy-System, unterschiedliche Nutzergruppen oder Mandanten zu bedienen, ohne zu forken.

**Kosten:**
- Übermäßiges Customizing kann das System schwerer pflegbar, testbar und upgradebar machen.
- Jeder Customizing-Punkt erhöht die Testmatrix und das Potenzial für konfigurationsbezogene Fehler.
- Kann zu Feature-Aufblähung führen, wenn Customizing-Anfragen nicht sorgfältig priorisiert werden.
- Benutzerdefinierte Konfigurationen können auf unerwartete Weise miteinander in Konflikt geraten.

## How It Could Be

Ein Legacy-CRM-System bedient sowohl Innendienst- als auch Außendienstteams, aber sein starrer Workflow zwingt Außendiensttechniker, sich durch für Vertriebsmitarbeiter gestaltete Bildschirme zu navigieren. Statt ein separates System zu bauen, führt das Team rollenbasierte UI-Konfigurationen und anpassbare Workflow-Vorlagen ein. Außendiensttechniker sehen nur die für ihre Arbeit relevanten Felder und Schritte, während Vertriebsmitarbeiter ihre aktuelle Erfahrung behalten. Die Konfiguration wird separat vom Kerncode gespeichert, sodass Systemupgrades Anpassungen nicht überschreiben. Nutzerzufriedenheitsumfragen zeigen deutliche Verbesserung für das Außendienstteam, und die Workaround-Tabellenkalkulationen, die sie zuvor pflegten, um die starre UI auszugleichen, werden nicht mehr benötigt.
