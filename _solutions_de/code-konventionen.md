---
title: Code-Konventionen
description: Definition und Durchsetzung einheitlicher Richtlinien für Code-Formatierung
  und -Struktur.
category:
- Code
- Process
problems:
- inconsistent-coding-standards
- inconsistent-codebase
- mixed-coding-styles
- undefined-code-style-guidelines
- style-arguments-in-code-reviews
- difficult-code-comprehension
- code-review-inefficiency
- poor-naming-conventions
- automated-tooling-ineffectiveness
- clever-code
- inconsistent-execution
- inconsistent-naming-conventions
- nitpicking-culture
- bikeshedding
- log-spam
- difficult-to-understand-code
layout: solution
lang: de
en_slug: code-conventions
related_solutions:
- slug: code-review-process-reform
  similarity: 0.8
- slug: clean-code
  similarity: 0.75
- slug: code-review-guidelines
  similarity: 0.75
- slug: code-reviews
  similarity: 0.75
- slug: style-guide
  similarity: 0.75
- slug: code-comments
  similarity: 0.75
---

## Description

Code-Konventionen sind eine explizite, gemeinsam genutzte und werkzeuggetriebene Regelmenge dafür, wie Code formatiert und strukturiert wird — Namensgebung, Einrückung, Dateiorganisation, gängige Idiome —, konsistent über eine Codebasis angewendet, statt den individuellen Präferenzen jedes Entwicklers überlassen zu werden. Einmal dokumentiert, werden sie typischerweise automatisch durch Formatter und Linter durchgesetzt, die in die Build- oder CI-Pipeline eingebunden sind, sodass Einhaltung eine mechanische Eigenschaft des Codes ist statt etwas, das Reviewer mit dem Auge überwachen müssen. Legacy-Codebasen häufen fast standardmäßig mehrere, inkonsistente Stile an, da sie von verschiedenen Entwicklern über lange Zeitspannen ohne gemeinsamen Standard geschrieben werden, und die resultierende Inkonsistenz erhöht direkt die kognitiven Kosten des Lesens und Änderns jedes Teils des Systems. Die nachträgliche Etablierung von Konventionen unterscheidet sich jedoch von ihrer Etablierung bei einem Greenfield-Projekt, weil das Nachrüsten eines einzigen Stils auf eine gesamte Legacy-Codebasis in einem Durchgang enorme, nicht überprüfbare Diffs produziert und den Wert der Blame-Historie zerstört; der praktikablere Weg ist, Konventionen nur auf Dateien anzuwenden, während sie berührt werden, sodass sich Konsistenz schrittweise zusammen mit tatsächlicher Wartungsaktivität verbreitet. Über die Lesbarkeit hinaus entfernen vereinbarte und automatisierte Konventionen außerdem eine gesamte Kategorie unproduktiver Code-Review-Diskussion — Stildebatten —, die sonst Review-Zeit verbraucht, die stattdessen in Logik und Design fließen sollte. Die Hauptkosten sind der Vorabaufwand, Übereinstimmung über ein Team mit verwurzelten individuellen Gewohnheiten zu erreichen, und das Risiko, dass mit modernem Formatierungs-Tooling inkompatibler Legacy-Code manuelle Ausnahmen erfordert.

## How to Apply ◆

> In Legacy-Systemen mit mehreren über Jahre und Entwickler angehäuften Coding-Stilen bringt die Etablierung und Durchsetzung von Konventionen Konsistenz, die die Codebasis navigierbar macht.

- Dokumentieren Sie Coding-Konventionen in einem gemeinsam genutzten, versionskontrollierten Style Guide, der Namensgebung, Formatierung, Dateiorganisation und gängige Muster für die Sprache und das Framework des Projekts abdeckt.
- Automatisieren Sie die Konventionsdurchsetzung mit Formattern (Prettier, Black, gofmt) und Lintern (ESLint, Checkstyle, RuboCop), konfiguriert, um den vereinbarten Konventionen zu entsprechen.
- Integrieren Sie automatisierte Prüfungen in die CI-Pipeline, sodass Konventionsverstöße vor dem Code-Review abgefangen werden, was Stildebatten aus Reviews eliminiert.
- Übernehmen Sie für Legacy-Codebasen mit inkonsistenten bestehenden Stilen eine „Campingplatz-Regel" — wenden Sie Konventionen auf geänderte Dateien an, statt die gesamte Codebasis auf einmal umzuformatieren, um massive, ungeprüfte Diffs zu vermeiden.
- Beziehen Sie das Team in die Definition von Konventionen durch einen kollaborativen Prozess ein, um Ownership aufzubauen und Widerstand zu verringern.
- Wählen Sie Konventionen mit starker Tooling-Unterstützung gegenüber theoretisch überlegenen Konventionen, die manuelle Durchsetzung erfordern.
- Adressieren Sie die störendsten Inkonsistenzen zuerst (Namenskonventionen, Einrückung), bevor Sie weniger wirkungsvolle Stilregeln verfeinern.

## Tradeoffs ⇄

> Konventionen verringern kognitive Last und eliminieren Stildebatten, erfordern aber anfänglichen Übereinstimmungsaufwand und könnten mit etablierten Legacy-Mustern kollidieren.

**Vorteile:**

- Eliminiert unproduktive Stilargumente in Code-Reviews, was Review-Zeit für substanzielles Feedback zu Logik und Design freisetzt.
- Verringert kognitive Last beim Lesen von Code über verschiedene Teile des Legacy-Systems hinweg, indem visuelle und strukturelle Konsistenz geboten wird.
- Beschleunigt das Onboarding, weil neue Entwickler einen Satz von Konventionen lernen können, statt den persönlichen Stil jedes Entwicklers zu entschlüsseln.
- Automatisierte Formatierungswerkzeuge machen Konventionseinhaltung nach initialer Einrichtung mühelos.

**Kosten und Risiken:**

- Die Durchsetzung neuer Konventionen auf einer Legacy-Codebasis kann große Umformatierungs-Commits erzeugen, die die Versionskontrollhistorie verschmutzen und Blame-Analyse erschweren.
- Teams könnten übermäßig viel Zeit damit verbringen, über Konventionswahlen zu debattieren, statt einen „gut genug"-Standard zu übernehmen und weiterzumachen.
- Mancher Legacy-Code könnte mit modernen Formattern nicht kompatibel sein, was Ausnahmen oder manuellen Eingriff erfordert.
- Übermäßig vorschreibende Konventionen können Entwickler in Situationen unnötig einschränken, wo die Konvention nicht gut passt.

## How It Could Be

> Das folgende Szenario veranschaulicht, wie Code-Konventionen eine Legacy-Codebasis verbessern.

Ein Softwareunternehmen, das eine 500.000-Zeilen-Java-Codebasis pflegte, hatte fünf verschiedene Namenskonventionen für ähnliche Konzepte über verschiedene Module angehäuft: `getUserById`, `get_user_by_id`, `fetchUser`, `loadUserRecord` und `retrieveUserData` erschienen alle in verschiedenen Teilen der Codebasis. Code-Reviews entarteten regelmäßig zu Stildebatten, und Entwickler berichteten, 20 % der Review-Zeit für Formatierungsprobleme aufzuwenden. Das Team übernahm Googles Java-Style-Guide, konfigurierte Checkstyle und google-java-format in der CI-Pipeline und wandte die „Campingplatz-Regel" für bestehenden Code an. Über sechs Monate wurde jede geänderte Datei automatisch umformatiert, und die für Stilprobleme aufgewendete Code-Review-Zeit sank nahezu auf null. Das Team etablierte außerdem ein Namenskonventionsglossar für gängige Operationen (get, create, update, delete), das die verwirrende Vermehrung von Synonymen eliminierte.
