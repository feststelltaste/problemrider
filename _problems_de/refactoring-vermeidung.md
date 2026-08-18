---
title: Refactoring-Vermeidung
description: Das Entwicklungsteam vermeidet aktiv Refactoring der Codebasis, selbst
  wenn es anerkennt, dass es notwendig ist, aus Angst, neue Fehler einzuführen.
category:
- Code
- Process
related_problems:
- slug: resistance-to-change
  similarity: 0.75
- slug: maintenance-paralysis
  similarity: 0.75
- slug: fear-of-change
  similarity: 0.75
- slug: feature-creep-without-refactoring
  similarity: 0.7
- slug: fear-of-breaking-changes
  similarity: 0.7
- slug: brittle-codebase
  similarity: 0.65
solutions:
- incremental-refactoring
- technical-debt-backlog
- refactoring-katas
- test-driven-development-tdd
- preparatory-refactoring
- characterization-tests
- dependency-breaking-techniques
- mikado-method
- improvement-budget
- code-hotspot-analysis
- debt-classification
- debt-remediation-estimation
- quality-ratchet
- technical-debt-assessment
- debt-accrual-analysis
- automated-code-migration
- large-scale-refactoring
layout: problem
lang: de
en_slug: refactoring-avoidance
---

## Description
Refactoring-Vermeidung ist das Phänomen, bei dem ein Entwicklungsteam konsequent die Verbesserung der internen Struktur des Codes verschiebt oder vermeidet, selbst wenn es sich seiner Mängel bewusst ist. Dies wird oft durch die Angst getrieben, dass jede Änderung, wie gut gemeint auch immer, neue Fehler einführen oder bestehende Funktionalität brechen wird. Diese Vermeidung ist ein sich selbst verstärkender Kreislauf: Je länger Refactoring verzögert wird, desto mehr technische Schulden häufen sich an, und desto riskanter werden zukünftige Änderungen. Es ist ein klares Zeichen für eine fragile und ungesunde Codebasis.

## Indicators ⟡
- Entwickler sagen Dinge wie: „Fass diesen Code nicht an, das ist ein Kartenhaus."
- Das Team wählt konsequent, neuen Code hinzuzufügen, statt bestehenden zu modifizieren.
- Es gibt eine lange Liste bekannter technischer Schulden-Elemente, die nie angegangen wird.
- Die Codebasis ist voller auskommentiertem Code, totem Code und anderen Formen von Ballast.

## Symptoms ▲

- [Hohe technische Schulden](hohe-technische-schulden.md)
<br/>  Das Vermeiden von Refactoring erlaubt es technischen Schulden, sich ungebremst anzuhäufen, da strukturelle Verbesserungen nie vorgenommen werden.
- [Zunehmende Brüchigkeit](zunehmende-bruechigkeit.md)
<br/>  Ohne Refactoring wird die Codebasis zunehmend fragiler, während sich Komplexität verstärkt.
- [Code-Duplizierung](code-duplizierung.md)
<br/>  Entwickler kopieren Code statt gemeinsam genutzte Funktionalität zu refaktorieren, was zu weit verbreiteter Duplizierung führt.
- [Anhäufung von Workarounds](anhaeufung-von-workarounds.md)
<br/>  Statt problematischen Code zu refaktorieren, bauen Entwickler Workarounds, die weitere Komplexität hinzufügen.
- [Langsame Entwicklungsgeschwindigkeit](langsame-entwicklungsgeschwindigkeit.md)
<br/>  Das Umgehen struktureller Probleme statt sie zu beheben macht jede neue Änderung fortschreitend langsamer zu implementieren.
- [Stagnierende Architektur](stagnierende-architektur.md)
<br/>  Wenn Teams Refactoring vermeiden, kann sich die Architektur nicht weiterentwickeln.
- [Brüchige Codebasis](bruechige-codebasis.md)
<br/>  Langfristige Vermeidung von Refactoring erlaubt es strukturellen Problemen, sich anzuhäufen, was die Codebasis zunehmend brüchig und fragil macht.

## Causes ▼

- [Angst vor Veränderung](angst-vor-veraenderung.md)
<br/>  Die Angst, dass Modifikationen funktionierende Funktionalität brechen, hindert Teams daran, strukturelle Verbesserungen zu versuchen.
- [Angst vor Breaking Changes](angst-vor-breaking-changes.md)
<br/>  Spezifische Angst, Breaking Changes in einem fragilen System einzuführen, lässt Entwickler es vermeiden, bestehenden Code anzufassen.
- [Testschulden](testschulden.md)
<br/>  Fehlende Testabdeckung bedeutet, dass es kein Sicherheitsnetz gibt, um Regressionen während des Refactorings zu erfassen, was es zu riskant macht, es zu versuchen.
- [Unrealistischer Zeitplan](unrealistischer-zeitplan.md)
<br/>  Enge Termine lassen keine Zeit für Refactoring-Arbeit, was Feature-Lieferung über Code-Verbesserung priorisiert.

## Detection Methods ○
- **Code-Churn-Analyse:** Analyse der Historie der Codebasis, um zu sehen, welche Dateien am häufigsten modifiziert werden. Wenn dieselben Dateien immer wieder umgewälzt werden, ohne Verbesserung ihrer Struktur, ist dies ein Zeichen für Refactoring-Vermeidung.
- **Technische-Schulden-Backlog:** Wenn das Team ein Backlog technischer Schulden-Elemente hat, das ständig wächst und nie schrumpft, ist dies ein klares Zeichen, dass sie Refactoring vermeiden.
- **Entwickler-Interviews:** Befragung von Entwicklern zu ihrer Einstellung zu Refactoring. Wenn sie Angst oder Zurückhaltung äußern, ist dies ein Zeichen eines Problems.
- **Codequalitätsmetriken:** Nachverfolgung von Codequalitätsmetriken über die Zeit. Ein stetiger Rückgang der Qualität ist ein starker Indikator für Refactoring-Vermeidung.

## Examples
Ein Team arbeitet an einem Legacy-System, das seit über einem Jahrzehnt in Produktion ist. Der Code ist ein Durcheinander, funktioniert aber. Das Team hat Angst, ihn anzufassen, aus Furcht, ihn zu brechen. Wenn sie ein neues Feature hinzufügen müssen, kopieren sie einfach bestehenden Code und modifizieren ihn leicht. Dies führt zu viel Code-Duplizierung und macht das System noch schwerer zu warten. Das Team weiß, dass es den Code refaktorieren sollte, tut es aber nie, weil es die Konsequenzen fürchtet. Dies ist ein klassisches Beispiel für Refactoring-Vermeidung.
