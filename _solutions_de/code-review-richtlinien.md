---
title: Code-Review-Richtlinien
description: Schriftliche Vereinbarung darüber, wofür ein Review dient, was Reviewer
  prüfen müssen, was lediglich eine Meinung ist und wann eine Änderung gut genug
  zum Mergen ist.
category:
- Code
- Process
- Team
problems:
- superficial-code-reviews
- style-arguments-in-code-reviews
- nitpicking-culture
- perfectionist-review-culture
- conflicting-reviewer-opinions
- inadequate-initial-reviews
- rushed-approvals
- reviewer-anxiety
- review-process-breakdown
- code-review-inefficiency
- reviewer-inexperience
- bikeshedding
- team-members-not-engaged-in-review-process
- review-process-avoidance
- inconsistent-execution
- author-frustration
- clever-code
- cv-driven-development
- extended-cycle-times
- extended-review-cycles
- fear-of-conflict
- large-pull-requests
- mixed-coding-styles
- perfectionist-culture
- reduced-review-participation
- review-bottlenecks
- insufficient-code-review
- long-lived-feature-branches
- merge-conflicts
- poor-naming-conventions
- reduced-code-submission-frequency
- automated-tooling-ineffectiveness
- convenience-driven-development
- inadequate-code-reviews
- inconsistent-naming-conventions
- increased-risk-of-bugs
- inexperienced-developers
- undefined-code-style-guidelines
- low-code-customization-sprawl
layout: solution
lang: de
en_slug: code-review-guidelines
related_solutions:
- slug: code-review-process-reform
  similarity: 0.8
- slug: code-conventions
  similarity: 0.75
- slug: lightweight-design-review
  similarity: 0.75
- slug: architecture-reviews
  similarity: 0.75
- slug: small-change-batches
  similarity: 0.7
- slug: code-reviews
  similarity: 0.7
---

## Description

Code-Review-Richtlinien sind eine kurze, schriftliche Vereinbarung, die die Fragen beantwortet, die jedes Review implizit aufwirft: Was soll dieses Review abfangen, was muss ein Reviewer sich ansehen, was zählt als blockierender Einwand versus als Vorschlag, und wann ist eine Änderung gut genug zum Mergen. Die meiste Review-Dysfunktion entsteht daraus, dass diese Fragen nie beantwortet werden. Ohne eine gemeinsame Definition von „Review abgeschlossen" setzt jeder Reviewer seine eigene ein: Einer winkt Änderungen in zwei Minuten durch, ein anderer blockiert wegen Variablennamen, ein dritter schreibt das Design in den Kommentaren neu. In Legacy-Systemen sind die Einsätze höher, weil der Reviewer oft keine Möglichkeit hat zu beurteilen, ob eine Änderung sicher ist, außer sie sorgfältig zu lesen — und keine Möglichkeit zu wissen, wie sorgfältig sorgfältig genug ist. Richtlinien machen Reviews nicht strenger oder nachsichtiger; sie machen sie vorhersehbar, was ist, was Review von einem sozialen Spießrutenlauf in eine Engineering-Kontrolle verwandelt.

## How to Apply ◆

> In einer Legacy-Codebasis weiß der Reviewer üblicherweise weniger über das berührte Modul als der Autor, sodass Richtlinien Reviewern sagen müssen, was sie sinnvoll prüfen können, statt Allwissenheit anzunehmen.

- Schreiben Sie den **Zweck des Reviews** in ein bis zwei Sätzen auf und stellen Sie es an den Anfang des Dokuments. Eine typische Formulierung für Legacy-Arbeit: „Reviews existieren, um Defekte, unsichere Änderungen an fragilen Bereichen und Wissenslücken abzufangen — nicht, um auf den bevorzugten Stil einer Person zu konvergieren." Jede spätere Regel sollte auf diesen Zweck zurückführbar sein.
- Definieren Sie eine **Checkliste dessen, was Reviewer prüfen müssen**: Korrektheit der Änderung gegenüber ihrer erklärten Absicht, Fehler- und Randfallbehandlung, Auswirkungen auf Aufrufer des geänderten Codes, Testabdeckung für das neue Verhalten, und jegliche Interaktion mit bekannt-fragilen Modulen. Halten Sie es auf fünf bis sieben Punkte beschränkt. Eine zu lange Checkliste wird vollständig übersprungen, was, wie oberflächliche Reviews beginnen.
- Erklären Sie explizit, wofür Reviewer **keine** Review-Zeit verwenden sollten: Formatierung, Import-Reihenfolge, Namenspräferenzen ohne Korrektheitsauswirkung, und alternative Designs, die nur anders sind statt besser. Verschieben Sie alle mechanisch prüfbaren Regeln in automatisiertes Linting und Formatierung, sodass sie nie einen menschlichen Kommentar erreichen.
- Führen Sie eine **Kommentar-Taxonomie** ein, sodass jeder Kommentar sein eigenes Gewicht formuliert. Drei Stufen sind üblicherweise genug: `blocking` (muss vor dem Mergen gelöst werden, und der Reviewer nennt, warum es unsicher oder inkorrekt ist), `consider` (ein Vorschlag, den der Autor mit einem Ein-Satz-Grund ablehnen kann), und `nit` (kosmetisch, blockiert nie). Diese einzige Konvention löst die meisten widersprüchlichen Reviewer-Situationen, weil eine Meinungsverschiedenheit zwischen einem `blocking` und einem `nit` kein Patt zwischen Gleichen mehr ist.
- Formulieren Sie die **Tiebreaker-Regel** für echte Meinungsverschiedenheiten zwischen Reviewern: wer entscheidet, und innerhalb welcher Zeit. Eine übliche Regel ist, dass der Code-Eigentümer des betroffenen Moduls entscheidet, und wenn es keinen Eigentümer gibt, wird die Meinungsverschiedenheit innerhalb eines Arbeitstages an einen benannten technischen Lead eskaliert, statt im Pull-Request-Thread ausdiskutiert zu werden.
- Setzen Sie eine explizite **Gut-genug-Messlatte**: Eine Änderung darf gemergt werden, wenn sie sicher, getestet und besser ist als das, was zuvor da war — nicht wenn sie optimal ist. Schreiben Sie dies wörtlich auf, weil perfektionistische Review-Kulturen durch den unausgesprochenen Glauben aufrechterhalten werden, dass die Genehmigung unvollkommenen Codes eine persönliche Zustimmung dazu ist.
- Definieren Sie **Reaktionszeiterwartungen** in beide Richtungen: wie schnell von einem Reviewer erwartet wird, ein Review aufzunehmen, und wie schnell von einem Autor erwartet wird, auf Kommentare zu reagieren. Ohne formulierte Erwartung wird Review zur Aufgabe, die jeder aufschiebt, und Autoren lernen, sie zu umgehen.
- Geben Sie **unerfahrenen Reviewern ein explizites Mandat**. Formulieren Sie, dass „ich verstehe nicht, was das tut" ein gültiger und wertvoller Review-Kommentar ist, und dass von einem Reviewer nicht erwartet wird, alles abzufangen. Reviewer-Angst ist üblicherweise die Angst, etwas zu übersehen und dafür beschuldigt zu werden; die Richtlinien sollten klar sagen, dass Review ein zweites Augenpaar ist, keine Garantie, und Verantwortung für einen Defekt geteilt wird.
- Überprüfen Sie die Richtlinien selbst alle paar Monate in einer Retrospektive, unter Nutzung echter Beispiele schlecht gelaufener Reviews. Richtlinien, die nie überarbeitet werden, hören auf, mit der tatsächlichen Arbeitsweise des Teams übereinzustimmen, und werden zu einem weiteren ignorierten Dokument.

## Tradeoffs ⇄

> Schriftliche Richtlinien verwandeln implizite, persönlich verhandelte Standards in einen expliziten Teamstandard — was viel Reibung entfernt, aber auch etwas von der Flexibilität entfernt, die erfahrene Reviewer gut nutzen.

**Vorteile:**

- Reviews werden vorhersehbar in Umfang und Dauer, was sie leichter zu planen und viel schwerer zu vermeiden oder aufzuschieben macht.
- Meinungsverschiedenheiten zwischen Reviewern werden durch eine formulierte Regel gelöst statt durch Dienstalter, Beharrlichkeit oder wer bereit ist, weiter zu argumentieren.
- Die Automatisierung der mechanisch prüfbaren Regeln entfernt die große Mehrheit geringwertiger Review-Kommentare, was die schnellste verfügbare Korrektur für Nitpicking und Stildebatten ist.
- Neue und weniger erfahrene Reviewer können sofort beitragen, weil die Checkliste ihnen sagt, worauf sie achten sollen, statt zu verlangen, dass sie es bereits wissen.
- Die explizite Gut-genug-Messlatte macht es sozial akzeptabel, eine unvollkommene Änderung zu genehmigen, was eine Voraussetzung für jedes Team ist, das versucht, eine Legacy-Codebasis schrittweise zu verbessern.

**Kosten und Risiken:**

- Eine Checkliste kann zu einem Ersatz für Nachdenken werden. Reviewer, die Punkte abhaken, ohne sich zu engagieren, produzieren Reviews, die gründlich aussehen, aber nichts abfangen, sodass die Checkliste periodisch gegen Defekte getestet werden muss, die trotzdem Produktion erreicht haben.
- Richtlinien, die auferlegt statt vereinbart werden, werden ignoriert. Das Team muss sie gemeinsam schreiben, oder mindestens ratifizieren, sonst hat das Dokument keine Autorität, wenn tatsächlich eine Meinungsverschiedenheit auftritt.
- Die Formalisierung von Review kann triviale Änderungen verlangsamen, wenn die Checkliste einheitlich angewendet wird. Ein separater leichtgewichtiger Pfad für risikoarme Änderungen wird üblicherweise benötigt, und die Definition von „risikoarm" in einem Legacy-System ist selbst schwierig.
- Die Kommentar-Taxonomie funktioniert nur, wenn Senior-Reviewer sie ehrlich nutzen. Ein Reviewer, der jede Präferenz als `blocking` kennzeichnet, führt das ursprüngliche Problem mit zusätzlicher Zeremonie wieder ein.

## How It Could Be

Ein Team, das eine 15 Jahre alte Versicherungspolicen-Engine pflegte, hatte Review-Threads, die routinemäßig auf vierzig Kommentare anwuchsen, fast alle über Namensgebung und Formatierung, während zwei Produktionsdefekte in einem Quartal von unbehandelten Null-Fällen kamen, zu denen kein Reviewer kommentiert hatte. Das Team schrieb eine einseitige Richtlinie: eine Fünf-Punkte-Checkliste, angeführt von „was bricht, wenn diese Eingabe unerwartet ist", eine `blocking`/`consider`/`nit`-Präfixkonvention, und eine Regel, dass Formatierung die Aufgabe des Formatters war und kein gültiger Kommentar mehr. Sie aktivierten in derselben Woche einen Auto-Formatter beim Commit. Innerhalb von zwei Monaten sank die durchschnittliche Kommentaranzahl pro Review von achtunddreißig auf neun, und der Anteil der Kommentare zu Fehlerbehandlung und Aufrufer-Auswirkung stieg von unter fünf Prozent auf ungefähr ein Drittel. Die folgenden zwei Quartale sahen keine Produktionsdefekte der Null-Behandlungs-Klasse.

Das Problem eines zweiten Teams war das Gegenteil: Reviews wurden innerhalb von Minuten genehmigt und fingen nichts ab, weil sich Reviewer unqualifiziert fühlten, Änderungen in Modulen zu beurteilen, an denen sie nie gearbeitet hatten. Die Richtlinie, die dies änderte, war ein einzelner Satz, der besagte, dass „ich kann das nicht sicher bewerten" ein legitimes Review-Ergebnis ist und das Team verpflichtet, einen zweiten Reviewer mit dem relevanten Wissen zu finden, nicht den Autor, einen nachsichtigeren zu finden. Reviewer hörten auf, blind abzunicken, drei chronisch unterprüfte Module wurden identifiziert durch die Häufigkeit, mit der dieses Ergebnis genutzt wurde, und diese Module wurden zu den ersten Zielen für bewussten Wissensaustausch.
