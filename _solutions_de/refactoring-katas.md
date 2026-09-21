---
title: Refactoring-Katas
description: Durchführung regelmäßiger Übungen zur Verbesserung der
  Codequalität.
category:
- Code
- Team
problems:
- refactoring-avoidance
- fear-of-change
- inexperienced-developers
- skill-development-gaps
- insufficient-design-skills
- lower-code-quality
- procedural-background
- procedural-programming-in-oop-languages
layout: solution
lang: de
en_slug: refactoring-katas
related_solutions:
- slug: incremental-refactoring
  similarity: 0.8
- slug: code-review-process-reform
  similarity: 0.7
- slug: code-reviews
  similarity: 0.7
- slug: code-metrics
  similarity: 0.7
- slug: preparatory-refactoring
  similarity: 0.7
- slug: domain-patterns
  similarity: 0.65
---

## Description

Refactoring-Katas sind kurze, wiederholbare Programmierübungen — wie die Gilded-Rose-, Tennis-Refactoring- oder Trip-Service-Kata —, die Entwicklern eine sichere, risikoarme Umgebung geben, um spezifische Refactoring-Techniken wie Extract Method oder Replace Conditional with Polymorphism zu üben, bevor sie sie auf Produktionscode anwenden. Da die Übungen kleine, bekannte Codebeispiele nutzen statt das tatsächliche Legacy-System des Teams, können Entwickler frei experimentieren, Fehler machen und die Übung wiederholen, bis eine Technik automatisch wird, ohne jegliches Risiko, etwas Wichtiges kaputt zu machen. Diese Unterscheidung zählt in Legacy-Kontexten, weil ein Großteil der Zurückhaltung, alten Code zu refaktorieren, nicht aus mangelndem theoretischem Wissen stammt, sondern aus mangelnder geübter, im Muskelgedächtnis verankerter Fähigkeit, Transformationen sicher anzuwenden — Teams, die Jahre nur Features zu Legacy-Code hinzugefügt haben, statt ihn umzustrukturieren, fehlt oft das Vertrauen, ihn überhaupt anzufassen. Regelmäßige Kata-Praxis baut dieses Vertrauen schrittweise wieder auf und schafft ein gemeinsames Vokabular aus Technikbezeichnungen und Ansätzen über das Team hinweg, sodass, wenn jemand vorschlägt, während eines echten Code-Reviews ein Value Object zu extrahieren oder eine Bedingung durch Polymorphie zu ersetzen, jeder versteht, was gemeint ist und wie man es sicher macht. Über die Zeit übertragen sich die in Katas geübten Fähigkeiten direkt auf die Legacy-Codebasis und verwandeln Refactoring von einem einschüchternden, gelegentlichen Ereignis in einen routinemäßigen Teil der täglichen Arbeit.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Planen Sie regelmäßige Übungssitzungen (z. B. wöchentliche einstündige Slots), bei denen Teammitglieder Refactoring-Übungen durcharbeiten
- Nutzen Sie bekannte Katas wie die Gilded-Rose-, Tennis-Refactoring- oder Trip-Service-Kata, die Legacy-Code-Szenarien simulieren
- Üben Sie paarweise oder in kleinen Gruppen, um Techniken zu teilen und ein gemeinsames Refactoring-Vokabular aufzubauen
- Konzentrieren Sie jede Sitzung auf eine spezifische Technik: Extract Method, Replace Conditional with Polymorphism, Introduce Parameter Object
- Wenden Sie in Katas gelernte Techniken auf tatsächlichen Legacy-Code in risikoarmen Bereichen an, um Fähigkeiten zu festigen
- Verfolgen Sie, mit welchen Refactoring-Mustern das Team am wohlsten und am unwohlsten ist, um künftige Sitzungen zu leiten
- Nutzen Sie die automatisierten Refactoring-Werkzeuge der IDE während Katas, um Muskelgedächtnis für sichere Transformationen aufzubauen

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Baut Teamvertrauen auf, Produktions-Legacy-Code sicher zu refaktorieren
- Schafft gemeinsames Refactoring-Vokabular und -Ansatz im gesamten Team
- Reduziert Angst vor der Änderung von Legacy-Code durch praktische Übung in einer sicheren Umgebung
- Verbessert Codequalität schrittweise, während Entwickler gelernte Techniken täglich anwenden

**Kosten und Risiken:**
- Erfordert dedizierte Zeit, die mit dem Druck der Feature-Lieferung konkurriert
- Nutzen sind schrittweise und kurzfristig schwer zu messen
- Kann akademisch wirken, wenn Katas nicht mit echten Codebasis-Herausforderungen verbunden sind
- Risiko abnehmender Erträge, wenn dieselben Übungen ohne Fortschritt wiederholt werden

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Entwicklungsteam, das ein Legacy-ERP-System betreute, zögerte, eine 3.000-Zeilen-Bestellverarbeitungsklasse zu refaktorieren, weil sich niemand sicher genug fühlte, um strukturelle Änderungen an so kritischem Code vorzunehmen. Der Tech Lead führte zweiwöchentliche Refactoring-Kata-Sitzungen ein, beginnend mit der Gilded-Rose-Kata, um das Extrahieren von Methoden und die Einführung von Abstraktionen zu üben. Nach zwei Monaten wendete das Team das Extract-Class-Refactoring während eines geplanten Sprints auf die Bestellverarbeitungsklasse an und teilte sie in fünf fokussierte Klassen auf. Die Kata-Praxis hatte ihnen sowohl die Fähigkeiten als auch das Vertrauen gegeben, das Refactoring sicher durchzuführen, und der resultierende Code war erheblich leichter zu warten.
