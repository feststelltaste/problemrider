---
title: Inkonsistente Namenskonventionen
description: Unstrukturierte oder widersprüchliche Namen erschweren das Lesen, Navigieren
  und Warten von Code.
category:
- Code
- Communication
related_problems:
- slug: poor-naming-conventions
  similarity: 0.8
- slug: inconsistent-coding-standards
  similarity: 0.8
- slug: inconsistent-codebase
  similarity: 0.8
- slug: undefined-code-style-guidelines
  similarity: 0.7
- slug: mixed-coding-styles
  similarity: 0.7
- slug: code-duplication
  similarity: 0.65
solutions:
- static-analysis-and-linting
- ubiquitous-language
- code-conventions
- style-guide
- consistent-terminology
- code-reviews
- domain-driven-design
- code-review-guidelines
- communities-of-practice
- clean-code
- automated-code-migration
- large-scale-refactoring
layout: problem
lang: de
en_slug: inconsistent-naming-conventions
---

## Description

Inkonsistente Namenskonventionen treten auf, wenn unterschiedliche Teile einer Codebasis variierende Stile, Muster oder Ansätze für die Benennung von Variablen, Funktionen, Klassen, Dateien und anderen Codeelementen nutzen. Dies schafft Verwirrung für Entwickler, die versuchen, den Code zu verstehen, zu navigieren oder zu ändern, weil sie sich nicht auf vorhersehbare Muster verlassen können, um den Zweck oder Umfang unterschiedlicher Elemente zu verstehen. Das Problem geht über einfache Stilpräferenzen hinaus und beeinträchtigt Codeverständnis, Wartungseffizienz und Teamzusammenarbeit.

## Indicators ⟡

- Code-Reviews, die häufig Korrekturen oder Vorschläge zum Benennungsstil enthalten
- Mehrere Namensmuster koexistieren innerhalb desselben Moduls oder Projekts
- Neue Teammitglieder stellen Fragen zu Namenskonventionen oder kämpfen damit, Codeelemente zu finden
- Fehlende dokumentierte Namensstandards oder Styleguides für das Projekt
- Unterschiedliche Teams oder Personen folgen ihren eigenen Namenspräferenzen
- IDE- oder Editor-Warnungen zu inkonsistenten Namensmustern über die Codebasis
- Such- und Refactoring-Operationen, die durch unvorhersehbare Benennung erschwert werden

## Symptoms ▲

- [Erhöhte kognitive Last](erhoehte-kognitive-last.md)
<br/>  Entwickler müssen eine mentale Zuordnung mehrerer Namensstile beim Lesen und Schreiben von Code aufrechterhalten, was die mentale Last erhöht.
- [Schwer verständliche Codebasis](schwer-verstaendliche-codebasis.md)
<br/>  Unvorhersehbare Namensmuster erschweren es zu verstehen, was Codeelemente repräsentieren und wie sie zueinander in Beziehung stehen.
- [Code-Duplizierung](code-duplizierung.md)
<br/>  Entwickler könnten duplizierten Code erstellen, weil inkonsistente Benennung es schwierig macht, bestehende Implementierungen durch Suche zu finden.
- [Erhöhtes Risiko für Fehler](erhoehtes-risiko-fuer-fehler.md)
<br/>  Refactoring- und Umbenennungsoperationen werden fehleranfällig, wenn mehrere Namenskonventionen berücksichtigt werden müssen, was zu übersehenen Referenzen führt.
- [Schwieriges Onboarding neuer Entwickler](schwieriges-onboarding-neuer-entwickler.md)
<br/>  Neue Entwickler haben Schwierigkeiten, Code zu navigieren und zu finden, weil sie die in unterschiedlichen Teilen der Codebasis genutzten Namensmuster nicht vorhersagen können.

## Causes ▼

- [Undefinierte Code-Stil-Richtlinien](undefinierte-code-stil-richtlinien.md)
<br/>  Ohne vereinbarte Namensstandards nutzt jeder Entwickler seine persönlichen Namenspräferenzen.
- [Widersprüchliche Reviewer-Meinungen](widerspruechliche-reviewer-meinungen.md)
<br/>  Wenn Reviewer widersprüchliche Namensanleitung geben, erhalten Entwickler gemischte Signale darüber, welchen Konventionen zu folgen ist.

## Detection Methods ○

- Nutzung statischer Analysewerkzeuge zur Identifikation von Inkonsistenzen bei Namensmustern
- Durchführung von Code-Reviews, die sich spezifisch auf die Einhaltung von Namenskonventionen konzentrieren
- Analyse der Codebasis mit Werkzeugen, die unterschiedliche Namensstile erkennen können (camelCase, snake_case usw.)
- Befragung des Entwicklungsteams zu Schwierigkeiten bei Code-Navigation und -Verständnis
- Überprüfung von Suchmustern und Häufigkeit von "In-Dateien-suchen"-Operationen für Namensvariationen
- Untersuchung der Wirksamkeit und Genauigkeit von Refactoring-Werkzeugen in der aktuellen Codebasis
- Nachverfolgung der während Code-Reviews für namensbezogene Diskussionen aufgewendeten Zeit
- Bewertung des Onboarding-Feedbacks neuer Entwickler zu Herausforderungen bei der Codebasis-Navigation

## Examples

Eine Webanwendungs-Codebasis zeigt wild inkonsistente Benennung: manche Funktionen nutzen camelCase (`getUserData()`), andere snake_case (`get_user_data()`), und wieder andere abgekürzte Formen (`getUsrDat()`). Datenbank-Tabellennamen mischen Konventionen mit `user_accounts`, `UserProfiles` und `usrPrefs`. CSS-Klassen reichen von `user-profile-header` über `UserProfileBody` bis `usr_prof_footer`. Wenn ein neuer Entwickler alle nutzerbezogene Funktionalität finden muss, muss er nach mehreren Namensvariationen suchen und übersieht oft wichtigen Code, weil er nicht alle unterschiedlichen Wege antizipiert hat, wie "user" abgekürzt oder stilisiert sein könnte. Eine einfache Aufgabe wie das Umbenennen einer Nutzereigenschaft wird zu einem komplexen Unterfangen, das umfangreiche Such- und Ersetzungsoperationen über mehrere Namensmuster hinweg erfordert, was das Risiko erhöht, durch übersehene Referenzen Fehler einzuführen.
