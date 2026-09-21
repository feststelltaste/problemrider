---
title: Schwer verständliche Codebasis
description: Eine Situation, in der Entwickler Schwierigkeiten haben, die Codebasis
  zu verstehen.
category:
- Code
related_problems:
- slug: difficult-to-understand-code
  similarity: 0.85
- slug: difficult-code-reuse
  similarity: 0.7
- slug: increased-cognitive-load
  similarity: 0.7
- slug: complex-and-obscure-logic
  similarity: 0.7
- slug: difficult-developer-onboarding
  similarity: 0.7
- slug: debugging-difficulties
  similarity: 0.7
solutions:
- clean-code
- loose-coupling
- separation-of-concerns
- architecture-documentation
- aspect-oriented-programming-aop
- code-comments
- code-conventions
- code-metrics
- code-reviews
- decision-tables
- facades
- fluent-interfaces
- high-cohesion
- layered-architecture
- object-relational-mapping-orm
- pattern-language
- rule-based-systems
- static-code-analysis
- strategic-code-deletion
- ubiquitous-language
- collaborative-problem-solving
- domain-specific-languages
- exceptions
layout: problem
lang: de
en_slug: difficult-code-comprehension
---

## Description
Eine schwer verständliche Codebasis ist eine Situation, in der Entwickler Schwierigkeiten haben, die Codebasis zu verstehen. Dies ist ein verbreitetes Problem in langlaufenden Projekten, besonders solchen, an denen über die Jahre viele unterschiedliche Personen gearbeitet haben. Eine schwer verständliche Codebasis kann zu einer Reihe von Problemen führen, einschließlich sinkender Produktivität, einer steigenden Anzahl von Fehlern und einer allgemeinen Verlangsamung der Entwicklungsgeschwindigkeit.

## Indicators ⟡
- Entwickler bitten ständig um Hilfe, um die Codebasis zu verstehen.
- Es dauert lange, bis neue Entwickler produktiv werden.
- Es gibt viel duplizierten Code.
- Die Codebasis ist eine Mischung unterschiedlicher Stile und Konventionen.

## Symptoms ▲

- [Langsame Entwicklungsgeschwindigkeit](langsame-entwicklungsgeschwindigkeit.md)
<br/>  Wenn Entwickler Schwierigkeiten haben, den Code zu verstehen, dauert jede Änderung erheblich länger.
- [Schwieriges Onboarding neuer Entwickler](schwieriges-onboarding-neuer-entwickler.md)
<br/>  Schwer verständlicher Code lässt neue Entwickler viel länger brauchen, um produktiv zu werden.
- [Erhöhtes Risiko für Fehler](erhoehtes-risiko-fuer-fehler.md)
<br/>  Entwickler, die den Code nicht vollständig verstehen, führen mit höherer Wahrscheinlichkeit Fehler ein, wenn sie Änderungen vornehmen.
- [Code-Duplizierung](code-duplizierung.md)
<br/>  Wenn Code schwer zu verstehen ist, schreiben Entwickler Funktionalität möglicherweise neu, statt bestehenden Code wiederzuverwenden, den sie nicht verstehen können.
- [Angst vor Veränderung](angst-vor-veraenderung.md)
<br/>  Entwickler vermeiden es, Code zu ändern, den sie nicht verstehen, was zu Stagnation und Workarounds führt.
- [Erhöhte kognitive Last](erhoehte-kognitive-last.md)
<br/>  Schwer verständlicher Code zwingt Entwickler dazu, übermäßigen Kontext im Gedächtnis zu behalten, was die mentale Belastung erhöht.
- [Debugging-Schwierigkeiten](debugging-schwierigkeiten.md)
<br/>  Wenn Code schwer verständlich ist, wird Debugging viel schwerer, weil Entwickler keine genauen mentalen Modelle bilden können.

## Causes ▼

- [Inkonsistente Codebasis](inkonsistente-codebasis.md)
<br/>  Gemischte Stile und Konventionen über die Codebasis hinweg erschweren es, mentale Modelle zu bilden und Muster zu verstehen.
- [Spaghetticode](spaghetticode.md)
<br/>  Verworrener, unstrukturierter Code mit verwickeltem Kontrollfluss ist von Natur aus schwer zu verstehen.
- [Schlechte Namenskonventionen](schlechte-namenskonventionen.md)
<br/>  Unklare oder irreführende Namen für Variablen, Funktionen und Klassen verschleiern die Absicht des Codes.
- [Komplexe und unklare Logik](komplexe-und-unklare-logik.md)
<br/>  Übermäßig komplexe Geschäftslogik, eingebettet in verworrene Codestrukturen, macht das Verständnis extrem schwierig.
- [Informationsverfall](informationsverfall.md)
<br/>  Veraltete oder fehlende Dokumentation bedeutet, dass sich Entwickler ausschließlich auf das Lesen von Code verlassen müssen, um die Absicht zu verstehen.
- [Anhäufung von Workarounds](anhaeufung-von-workarounds.md)
<br/>  Mehrere alternative Codepfade und bedingte Workarounds, die über die ursprüngliche Logik geschichtet sind, machen den Code extrem schwer nachvollziehbar.

## Detection Methods ○
- **Entwickler-Umfragen:** Befragung von Entwicklern, ob sie die Codebasis leicht lesbar und verständlich finden.
- **Code-Reviews:** Suche nach Code, der schwer zu verstehen und zu überprüfen ist.
- **Statische Analysewerkzeuge:** Nutzung von Werkzeugen zur Identifikation von Code Smells wie komplexem Code und langen Methoden.

## Examples
Ein Entwickler versucht, einen Fehler in einem Legacy-Modul zu beheben. Der Entwickler stellt fest, dass das Modul sehr schwer zu verstehen ist. Der Code ist eine Mischung unterschiedlicher Stile und Konventionen, und es gibt keine Dokumentation. Der Entwickler verbringt viel Zeit damit, zu versuchen, den Code zu verstehen, und kann den Fehler nicht beheben. Dies ist ein verbreitetes Problem in Unternehmen, die keine Kultur des Schreibens sauberen, lesbaren Codes haben.
