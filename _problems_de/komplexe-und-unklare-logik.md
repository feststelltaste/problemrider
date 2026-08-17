---
title: Komplexe und unklare Logik
description: Der Code ist aufgrund verschachtelter Logik, fehlender Kommentare oder
  schlechter Namenskonventionen schwer verständlich.
category:
- Code
related_problems:
- slug: difficult-to-understand-code
  similarity: 0.75
- slug: clever-code
  similarity: 0.7
- slug: difficult-code-comprehension
  similarity: 0.7
- slug: increased-cognitive-load
  similarity: 0.7
- slug: spaghetti-code
  similarity: 0.65
- slug: debugging-difficulties
  similarity: 0.65
solutions:
- incremental-refactoring
- business-event-processing
- business-process-automation
- code-comments
- code-metrics
- decision-tables
- rule-based-systems
- collaborative-problem-solving
- domain-patterns
- domain-specific-languages
layout: problem
lang: de
en_slug: complex-and-obscure-logic
---

## Description
Komplexe und unklare Logik ist Code, der schwer zu lesen, zu verstehen und nachzuvollziehen ist. Dies kann auf eine Vielzahl von Faktoren zurückzuführen sein, einschließlich verschachtelten Kontrollflusses, unklarer Benennung, fehlender Kommentare oder der Nutzung übermäßig cleverer oder esoterischer Sprachmerkmale. Diese Art von Code trägt erheblich zu technischen Schulden bei, da sie schwer und riskant zu warten oder zu ändern ist.

## Indicators ⟡
- Entwickler vermeiden es, an bestimmten Teilen der Codebasis zu arbeiten.
- Neue Entwickler brauchen lange, um in einem bestimmten Codebereich produktiv zu werden.
- Es gibt häufige Diskussionen und Debatten darüber, wie ein bestimmtes Codestück funktioniert.

## Symptoms ▲

- [Schwer verständliche Codebasis](schwer-verstaendliche-codebasis.md)
<br/>  Verschachtelte Logik mit schlechter Benennung und fehlenden Kommentaren macht es für Entwickler extrem schwer zu verstehen, was der Code tut.
- [Angst vor Veränderung](angst-vor-veraenderung.md)
<br/>  Entwickler vermeiden es, unklaren Code zu ändern, weil sie die Konsequenzen von Änderungen nicht sicher vorhersagen können.
- [Erhöhte kognitive Last](erhoehte-kognitive-last.md)
<br/>  Das Entschlüsseln komplexer Logik erfordert übermäßigen mentalen Aufwand, der die kognitiven Ressourcen der Entwickler erschöpft.
- [Debugging-Schwierigkeiten](debugging-schwierigkeiten.md)
<br/>  Unklare Logik macht es extrem schwer, Fehler nachzuverfolgen und den Ausführungsfluss beim Debugging zu verstehen.
- [Große Schätzungen für kleine Änderungen](grosse-schaetzungen-fuer-kleine-aenderungen.md)
<br/>  Selbst kleinere Änderungen an unklarem Code erfordern umfangreiche Analysezeit, um die Auswirkung zu verstehen.
- [Schwieriges Onboarding neuer Entwickler](schwieriges-onboarding-neuer-entwickler.md)
<br/>  Neue Entwickler brauchen viel länger, um in Bereichen mit komplexer und unklarer Logik produktiv zu werden.

## Causes ▼

- [Übertrieben cleverer Code](uebertrieben-cleverer-code.md)
<br/>  Entwickler, die Code schreiben, um technisches Können statt Klarheit zu zeigen, erzeugen komplexe und unklare Implementierungen.
- [Feature-Creep ohne Refactoring](feature-creep-ohne-refactoring.md)
<br/>  Das kontinuierliche Hinzufügen von Features ohne Umstrukturierung lässt Logik im Laufe der Zeit zunehmend verschachtelt werden.
- [Schlechte Namenskonventionen](schlechte-namenskonventionen.md)
<br/>  Kryptische Variablen- und Funktionsnamen verschleiern den Zweck des Codes, was Logik schwerer nachvollziehbar macht.
- [Bequemlichkeitsgetriebene Entwicklung](bequemlichkeitsgetriebene-entwicklung.md)
<br/>  Den einfachsten Weg zu gehen, ohne Rücksicht auf Lesbarkeit, führt zu verworrener und schlecht strukturierter Logik.

## Detection Methods ○
- **Code-Komplexitätsmetriken:** Nutzung statischer Analysewerkzeuge zur Messung von Metriken wie zyklomatischer Komplexität, die helfen können, übermäßig komplexen Code zu identifizieren.
- **Code-Reviews:** Genaue Aufmerksamkeit auf Code, der während Code-Reviews schwer verständlich ist.
- **Entwickler-Feedback:** Einholen von Feedback von Entwicklern dazu, welche Teile der Codebasis am schwierigsten zu bearbeiten sind.

## Examples
Eine Funktion, die eine einfache Berechnung durchführen soll, ist als ein einziger, massiver Block verschachtelter `if-else`-Anweisungen ohne Kommentare und mit kryptischen Variablennamen geschrieben. Ein neuer Entwickler braucht Tage, um zu verstehen, was die Funktion tut, und selbst dann ist er nicht sicher genug, um Änderungen daran vorzunehmen, aus Angst, etwas zu brechen. Dies ist ein klassisches Beispiel dafür, wie komplexe und unklare Logik eine erhebliche Wartungslast erzeugen kann.
