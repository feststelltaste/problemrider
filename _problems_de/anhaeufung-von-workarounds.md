---
title: Anhäufung von Workarounds
description: Statt Kernprobleme zu beheben, schaffen Entwickler aufwendige Workarounds,
  die Komplexität und technische Schulden zum System hinzufügen.
category:
- Code
- Process
related_problems:
- slug: workaround-culture
  similarity: 0.9
- slug: high-technical-debt
  similarity: 0.7
- slug: increased-technical-shortcuts
  similarity: 0.7
- slug: complex-implementation-paths
  similarity: 0.7
- slug: delayed-issue-resolution
  similarity: 0.65
- slug: hidden-dependencies
  similarity: 0.65
solutions:
- incremental-refactoring
- technical-debt-backlog
- strategic-code-deletion
- domain-patterns
- functional-debt-management
- improvement-budget
- preparatory-refactoring
- workaround-registry
- debt-accrual-analysis
- debt-classification
- quality-ratchet
- technical-debt-assessment
- code-hotspot-analysis
- debt-remediation-estimation
layout: problem
lang: de
en_slug: accumulation-of-workarounds
---

## Description

Die Anhäufung von Workarounds entsteht, wenn Entwickler durchgängig temporäre Fixes und aufwendige Umgehungen wählen, statt zugrunde liegende Probleme direkt anzugehen. Diese Workarounds entstehen oft unter Zeitdruck oder wenn die Grundursache zu riskant oder zu komplex erscheint, um sie richtig zu beheben. Im Laufe der Zeit schichten sich diese Workarounds übereinander und schaffen ein komplexes Geflecht aus Abhängigkeiten und alternativen Logikpfaden, das das System zunehmend schwerer verständlich und wartbar macht.

## Indicators ⟡

- Mehrere Codepfade existieren, um dieselbe grundlegende Funktionalität zu erreichen
- Dokumentation oder Kommentare erwähnen häufig "temporärer Fix" oder "Workaround für Problem X"
- Neue Features erfordern das Verstehen und Umschiffen bestehender Workarounds
- Entwickler äußern Verwirrung darüber, warum bestimmte Code-Muster existieren
- Einfache Änderungen erfordern Anpassungen an mehreren, scheinbar unzusammenhängenden Stellen

## Symptoms ▲

- [Hohe technische Schulden](hohe-technische-schulden.md)
<br/>  Jeder Workaround fügt dem System Komplexität und technische Schulden hinzu, die sich im Laufe der Zeit summieren.
- [Schwer verständliche Codebasis](schwer-verstaendliche-codebasis.md)
<br/>  Mehrere alternative Codepfade und bedingte Workarounds machen den Code extrem schwer verständlich.
- [Erhöhtes Risiko für Fehler](erhoehtes-risiko-fuer-fehler.md)
<br/>  Übereinandergeschichtete Workarounds erzeugen unerwartete Wechselwirkungen und Grenzfälle, die die Wahrscheinlichkeit von Fehlern erhöhen.
- [Anstieg der Wartungskosten](anstieg-der-wartungskosten.md)
<br/>  Jedes neue Feature oder jeder Fix muss um bestehende Workarounds herum navigieren, was den Wartungsaufwand erheblich erhöht.
- [Langsame Feature-Entwicklung](langsame-feature-entwicklung.md)
<br/>  Neue Features brauchen länger, weil Entwickler das bestehende Geflecht aus Workarounds verstehen und umgehen müssen.

## Causes ▼

- [Termindruck](termindruck.md)
<br/>  Zeitdruck treibt Entwickler dazu, schnelle Workarounds statt richtiger Fixes umzusetzen.
- [Angst vor Breaking Changes](angst-vor-breaking-changes.md)
<br/>  Entwickler schaffen Workarounds statt Grundursachen zu beheben, weil sie befürchten, dass eine Änderung der Kernlogik das System brechen könnte.
- [Refactoring-Vermeidung](refactoring-vermeidung.md)
<br/>  Wenn Teams Refactoring vermeiden, werden Probleme mit Workarounds geflickt, statt ordentlich gelöst zu werden.
- [Workaround-Kultur](workaround-kultur.md)
<br/>  Eine Organisationskultur, die schnelle Fixes gegenüber richtigen Lösungen normalisiert und belohnt, treibt die Anhäufung von Workarounds direkt an.
- [Legacy-Code ohne Tests](legacy-code-ohne-tests.md)
<br/>  Ohne Tests als Sicherheitsnetz haben Entwickler Angst, bestehenden Code zu ändern, und greifen stattdessen auf Workarounds zurück.
- [Angehäufte Entscheidungsschulden](angehaeufte-entscheidungsschulden.md)
<br/>  Wenn wichtige Entscheidungen ungelöst bleiben, umgehen Teams die Lücke mit temporären Workarounds, statt auf eine ordentliche Entscheidung zu warten.
- [Architektonische Fehlpassung](architektonische-fehlpassung.md)
<br/>  Wenn die Architektur neue Anforderungen nicht unterstützt, schaffen Entwickler Workarounds, um die Lücke zu überbrücken.

## Detection Methods ○

- **Code-Review-Analyse:** Suche nach Mustern alternativer Logikpfade und bedingter Workarounds
- **Code-Kommentar-Audit:** Suche nach Kommentaren mit "Workaround", "Hack", "temporär" oder "TODO"
- **Komplexitätsmetriken:** Beobachtung von Anstiegen der zyklomatischen Komplexität, die nicht an Wachstum der Geschäftslogik gebunden sind
- **Entwickler-Interviews:** Befragung von Teammitgliedern zu Codebereichen, die sie als verwirrend oder übermäßig komplex empfinden
- **Änderungsauswirkungsanalyse:** Nachverfolgung, wie viele Dateien für einfache Änderungen angepasst werden müssen

## Examples

Ein Zahlungsabwicklungssystem hat drei unterschiedliche Codepfade zur Berechnung von Versandkosten, weil frühere Versuche, Fehler in der ursprünglichen Berechnung zu beheben, zu Workarounds für bestimmte Kundentypen führten. Neue Entwickler müssen alle drei Pfade verstehen, um die Versandlogik zu ändern, und jeder Pfad hat seine eigenen Grenzfälle und Ausnahmen. Ein weiteres Beispiel betrifft ein Bestandsverwaltungssystem, bei dem ein Speicherleck im ursprünglichen Lagerverfolgungsalgorithmus durch das Hinzufügen einer täglichen Neustartroutine, einer stündlich laufenden Cache-Bereinigungsfunktion und eines separaten Hintergrundprozesses zum Abgleich von Unstimmigkeiten "behoben" wurde. Diese Workarounds verdecken das zugrunde liegende Problem, während sie operative Komplexität und potenzielle Fehlerquellen hinzufügen.
