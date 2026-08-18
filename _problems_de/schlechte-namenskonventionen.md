---
title: Schlechte Namenskonventionen
description: Variablen, Funktionen, Klassen und andere Code-Elemente werden auf
  Weisen benannt, die ihren Zweck oder ihre Bedeutung nicht klar kommunizieren.
category:
- Code
- Process
related_problems:
- slug: inconsistent-naming-conventions
  similarity: 0.8
- slug: difficult-to-understand-code
  similarity: 0.65
- slug: inconsistent-coding-standards
  similarity: 0.6
- slug: inconsistent-codebase
  similarity: 0.6
- slug: undefined-code-style-guidelines
  similarity: 0.6
- slug: monolithic-functions-and-classes
  similarity: 0.6
solutions:
- static-analysis-and-linting
- code-conventions
- consistent-terminology
- fluent-interfaces
- ubiquitous-language
- code-reviews
- clean-code
- style-guide
- domain-driven-design
- code-review-guidelines
layout: problem
lang: de
en_slug: poor-naming-conventions
---

## Description

Schlechte Namenskonventionen treten auf, wenn Code-Elementen wie Variablen, Funktionen, Klassen, Modulen und Dateien Namen gegeben werden, die es versäumen, ihren Zweck, ihr Verhalten oder ihren Inhalt klar zu kommunizieren. Dies umfasst Namen, die zu kurz, zu generisch, irreführend, inkonsistent sind oder unklare Abkürzungen nutzen. Schlechte Benennung zwingt Entwickler, zusätzliche mentale Anstrengung aufzuwenden, um Code zu verstehen, erhöht die Wahrscheinlichkeit von Fehlern und erschwert die Wartung.

## Indicators ⟡

- Variablen- und Funktionsnamen erfordern zusätzliche Kommentare, um ihren Zweck zu erklären
- Code enthält Ein-Buchstaben-Variablen außerhalb von Schleifenzählern
- Methodennamen zeigen nicht klar an, was sie tun oder zurückgeben
- Klassennamen sind zu generisch oder repräsentieren keine klaren Konzepte
- Teammitglieder fragen während Code-Reviews häufig nach der Bedeutung bestimmter Namen

## Symptoms ▲

- [Schwer verständliche Codebasis](schwer-verstaendliche-codebasis.md)
<br/>  Unklare oder irreführende Namen zwingen Entwickler, umgebenden Code zu lesen, um zu verstehen, was Elemente repräsentieren.
- [Schwieriges Onboarding neuer Entwickler](schwieriges-onboarding-neuer-entwickler.md)
<br/>  Neue Teammitglieder verbringen exzessive Zeit damit, Kollegen zu fragen, was schlecht benannte Variablen und Funktionen bedeuten.
- [Erhöhte kognitive Last](erhoehte-kognitive-last.md)
<br/>  Entwickler müssen zusätzliche mentale Anstrengung aufwenden, um unklare Namen zu entschlüsseln, was ihre Kapazität für Problemlösung verringert.
- [Erhöhtes Risiko für Fehler](erhoehtes-risiko-fuer-fehler.md)
<br/>  Irreführende Namen verursachen, dass Entwickler Codeverhalten missverstehen, was zu inkorrekter Nutzung und Fehlern führt.
- [Langsame Entwicklungsgeschwindigkeit](langsame-entwicklungsgeschwindigkeit.md)
<br/>  Zeit, die für das Entziffern schlechter Namen über die Codebasis hinweg aufgewendet wird, summiert sich zu erheblichen Entwicklungsverlangsamungen.

## Causes ▼

- [Undefinierte Code-Stil-Richtlinien](undefinierte-code-stil-richtlinien.md)
<br/>  Ohne etablierte Benennungsstandards verfallen Entwickler standardmäßig auf ad-hoc, inkonsistente Benennungsmuster.
- [Unerfahrene Entwickler](unerfahrene-entwickler.md)
<br/>  Entwickler ohne Erfahrung im Schreiben lesbaren Codes wählen oft abgekürzte oder unklare Namen.
- [Termindruck](termindruck.md)
<br/>  Unter Zeitdruck wählen Entwickler schnelle, kurze Namen statt in klare, beschreibende zu investieren.
- [Oberflächliche Code-Reviews](oberflaechliche-code-reviews.md)
<br/>  Code-Reviews, die Benennung nicht genau prüfen, erlauben es schlechten Benennungsmustern, in die Codebasis einzudringen und dort zu verbleiben.

## Detection Methods ○

- **Code-Review-Musteranalyse:** Nachverfolgung, wie oft Benennungsprobleme während Code-Reviews angesprochen werden
- **Einhaltung von Namenskonventionen:** Nutzung automatisierter Werkzeuge zur Überprüfung der Einhaltung von Benennungsstandards
- **Entwicklerbefragungen:** Befragung von Teammitgliedern zu Bereichen, in denen Benennung das Verständnis von Code erschwert
- **Code-Verständnistests:** Messung, wie schnell Entwickler Code mit unterschiedlichen Benennungsmustern verstehen können
- **Namenslängen- und Klarheitsanalyse:** Analyse der Verteilung von Namenslängen und Nutzung von Abkürzungen

## Examples

Ein Zahlungsverarbeitungssystem enthält Variablen wie `amt`, `flg`, `tmp` und `data` in der gesamten Codebasis, was es nahezu unmöglich macht zu verstehen, welche Werte sie repräsentieren, ohne umgebenden Code sorgfältig zu lesen. Eine Funktion namens `process()` nimmt 15 Parameter und führt Validierung, Transformation, Persistenz und Benachrichtigungsaufgaben durch, aber ihr generischer Name gibt keinen Hinweis auf ihr komplexes Verhalten. In einem anderen System handhabt eine Klasse namens `Manager` Nutzerauthentifizierung, Session-Management und Audit-Logging – drei völlig unterschiedliche Verantwortlichkeiten, die sich nicht in ihrem Namen widerspiegeln. Das Team nutzt auch inkonsistente Benennungsmuster: manche Methoden nutzen camelCase, während andere snake_case nutzen, manche booleschen Variablen beginnen mit „is", während andere mit „has" oder „can" beginnen, und Abkürzungen werden inkonsistent genutzt („num" vs. „number" vs. „cnt" vs. „count"). Wenn ein neuer Entwickler dem Team beitritt, verbringt er den ersten Monat damit, ständig Kollegen zu fragen, was verschiedene Variablen und Funktionen tatsächlich tun, was seine Produktivität erheblich verlangsamt und anderen Teammitgliedern Zeit wegnimmt.
