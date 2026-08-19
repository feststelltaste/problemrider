---
title: Code-Kommentare
description: Anreicherung von Code mit aussagekräftigen Kommentaren und Dokumentationsblöcken.
category:
- Code
- Communication
problems:
- difficult-code-comprehension
- difficult-to-understand-code
- poor-documentation
- implicit-knowledge
- tacit-knowledge
- complex-and-obscure-logic
- knowledge-gaps
layout: solution
lang: de
en_slug: code-comments
related_solutions:
- slug: code-conventions
  similarity: 0.75
- slug: code-review-process-reform
  similarity: 0.75
- slug: clean-code
  similarity: 0.7
- slug: architecture-decision-records
  similarity: 0.7
- slug: documentation-as-code
  similarity: 0.7
- slug: architecture-documentation
  similarity: 0.7
---

## Description

Code-Kommentare sind direkt in Quellcode eingebettete Anmerkungen, die Aspekte des Codes erklären, die der Code selbst nicht ausdrücken kann — am wichtigsten die Begründung hinter einer Entscheidung, statt einer Wiederholung dessen, was der Code sichtbar tut. Gut genutzt erfassen sie das „Warum": die Geschäftsregel, die ein Stück Logik kodiert, die Einschränkung, die einen bestimmten Workaround erzwang, den historischen Kontext, der ein sonst unlogisches Stück Code tatsächlich notwendig macht. In Legacy-Systemen ist diese Funktion überproportional wertvoll, weil die Menschen, die die ursprünglichen Entscheidungen getroffen haben, häufig die Organisation verlassen haben, keine externe Dokumentation überlebt, und der Code die einzige verbleibende Spur institutionellen Wissens ist, das sonst vollständig verschwinden würde. Ein Kommentar, der erklärt, dass eine seltsame Berechnung aus einer spezifischen regulatorischen Vereinbarung stammt, ist beispielsweise oft das Einzige, was zwischen dieser intakt bleibenden Logik und einem zukünftigen Betreuer steht, der sie zu einem Produktionsvorfall „repariert", weil sie wie ein offensichtlicher Bug aussah. Kommentare arbeiten neben klarer Namensgebung und Struktur, statt sie zu ersetzen — das „Was" sollte idealerweise aus dem Code selbst lesbar sein, wobei Kommentare nur den Kontext tragen, der nicht abgeleitet werden kann, egal wie der Code geschrieben ist. Ihre zentrale Schwäche ist, dass sie nie von einem Compiler oder einer Test-Suite geprüft werden, sodass ein Kommentar, der nicht mit dem von ihm beschriebenen Code synchron gehalten wird, still von unhilfreich zu aktiv irreführend wird, was ein echtes Risiko in Legacy-Code ist, der geändert wird, ohne dass jemand seinen begleitenden Kommentar überprüft.

## How to Apply ◆

> In Legacy-Systemen erklären strategische Code-Kommentare das „Warum" hinter Entscheidungen, die aus dem Code allein nicht verstanden werden können, und bewahren institutionelles Wissen, das sonst verloren ginge.

- Fokussieren Sie Kommentare darauf zu erklären, warum Code existiert und warum er so funktioniert, wie er funktioniert, nicht was er tut — der Code selbst sollte das „Was" durch klare Namensgebung und Struktur kommunizieren.
- Dokumentieren Sie nicht offensichtliche Geschäftsregeln, die im Code eingebettet sind, besonders wenn die Regel dem widerspricht, was logisch erscheint (z. B. „Rabatt wird vor Steuer für Bestellungen aus Region 3 angewendet, aufgrund einer regulatorischen Vereinbarung von 2008 mit dem Bundesstaat...").
- Fügen Sie Kommentare zu Workarounds und Hacks hinzu, die das zugrunde liegende Problem erklären, das sie adressieren, die Legacy-Einschränkung, die eine ordentliche Behebung verhindert, und alle Bedingungen, unter denen der Workaround entfernt werden könnte.
- Nutzen Sie Dokumentationsblöcke (Javadoc, JSDoc, Docstrings) für öffentliche APIs und Schnittstellen, um Verträge, Vorbedingungen und Randfallverhalten zu erklären.
- Fügen Sie „WARNUNG"- oder „VORSICHT"-Kommentare zu Code hinzu, der bekannte fragile Abhängigkeiten oder nicht offensichtliche Nebeneffekte hat, die zukünftige Betreuer in eine Falle tappen lassen könnten.
- Fügen Sie während Legacy-Code-Review oder -Wartung erklärende Kommentare hinzu, wann immer Sie erhebliche Zeit damit verbringen, ein Stück Code zu verstehen — die nächste Person wird ohne sie denselben Kampf durchmachen.

## Tradeoffs ⇄

> Kommentare bewahren institutionelles Wissen, erfordern aber Disziplin zur Pflege und können irreführen, wenn sie veralten.

**Vorteile:**

- Bewahrt die Begründung hinter Legacy-Code-Entscheidungen, die nicht aus dem Code selbst abgeleitet werden kann, und verhindert, dass zukünftige Entwickler versehentlich wichtiges Verhalten entfernen.
- Verringert die Zeit, die Entwickler mit dem Reverse-Engineering obskurer Legacy-Logik verbringen, indem Kontext am Bedarfspunkt geboten wird.
- Dokumentiert Workarounds und ihre Voraussetzungen, was es möglich macht, sie zu entfernen, wenn die zugrunde liegende Einschränkung schließlich gelöst wird.
- Dient als Wissenstransfermechanismus, wenn ursprüngliche Entwickler gehen, und erfasst Einsichten, die sonst verloren gingen.

**Kosten und Risiken:**

- Kommentare, die nicht aktualisiert werden, wenn sich Code ändert, werden irreführend und erzeugen falsches Verständnis, das zu Bugs führen kann.
- Übermäßige Kommentierung offensichtlichen Codes erzeugt Rauschen, das wirklich wichtige Kommentare schwerer auffindbar macht.
- Kommentare können nicht getestet oder kompiliert werden — es gibt keinen automatisierten Weg zu erkennen, wann ein Kommentar ungenau geworden ist.
- Sich auf Kommentare statt auf die Verbesserung der Codeklarheit durch Refaktorierung zu verlassen kann schlechte Codequalität verewigen.

## How It Could Be

> Das folgende Szenario zeigt, wie strategische Kommentare kritisches Wissen in Legacy-Systemen bewahren.

Das Legacy-Abrechnungssystem eines Telekommunikationsunternehmens enthielt eine Methode, die Nutzungsgebühren mit einem scheinbar willkürlichen Anpassungsfaktor von 0,3 % berechnete, angewendet auf Anrufe über 45 Minuten. Drei verschiedene Entwickler hatten über die Jahre versucht, diesen scheinbaren Bug zu „reparieren", jedes Mal Abrechnungsdiskrepanzen verursachend, die manuelle Korrekturen erforderten. Als ein Senior-Entwickler den Faktor schließlich auf eine Interconnect-Vereinbarung von 2005 mit einem Partner-Carrier zurückführte, fügte er einen detaillierten Kommentar hinzu, der den regulatorischen Ursprung der Anpassung, die spezifische Vereinbarungsreferenznummer und die Bedingungen erklärte, unter denen sie galt. Der Kommentar vermerkte außerdem, dass die Vereinbarung 2027 auslaufen sollte, an welchem Punkt die Anpassung entfernt werden könnte. Dieser einzelne Kommentar verhinderte zukünftige „Reparatur"-Versuche und lieferte den Geschäftskontext, der für die eventuelle Modernisierung der Abrechnungs-Engine benötigt wurde.
