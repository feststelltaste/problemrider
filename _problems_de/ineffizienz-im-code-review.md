---
title: Ineffizienz im Code-Review
description: Der Code-Review-Prozess nimmt übermäßig viel Zeit in Anspruch, bietet
  begrenzten Nutzen oder erzeugt Engpässe im Entwicklungsworkflow.
category:
- Code
- Process
- Team
related_problems:
- slug: inadequate-code-reviews
  similarity: 0.8
- slug: extended-review-cycles
  similarity: 0.8
- slug: insufficient-code-review
  similarity: 0.8
- slug: review-process-breakdown
  similarity: 0.75
- slug: review-bottlenecks
  similarity: 0.75
- slug: inadequate-initial-reviews
  similarity: 0.75
solutions:
- code-review-process-reform
- code-conventions
- static-code-analysis
- code-review-guidelines
- small-change-batches
- work-in-progress-limits
- checklists
- pair-and-mob-programming
- team-retrospectives
layout: problem
lang: de
en_slug: code-review-inefficiency
---

## Description

Ineffizienz im Code-Review entsteht, wenn der Code-Review-Prozess unverhältnismäßig viel Zeit und Aufwand im Vergleich zu dem gebotenen Nutzen verbraucht, oder wenn der Prozess selbst zu einem erheblichen Hindernis für die Entwicklungsgeschwindigkeit wird. Dies kann sich als Reviews äußern, die zu lange dauern, oberflächliches Feedback liefern, wichtige Probleme übersehen oder unnötige Hin-und-her-Diskussionen erzeugen, die die Codequalität nicht verbessern. Ineffiziente Reviews verschwenden Entwicklerzeit und können die Teammoral verringern.

## Indicators ⟡

- Code-Reviews dauern viel länger als die eigentliche Entwicklungszeit
- Reviews konzentrieren sich auf Stilpräferenzen statt auf wesentliche Fragen
- Mehrere Review-Runden sind für einfache Änderungen nötig
- Reviewer liefern widersprüchliches Feedback
- Wichtige Fehler oder Design-Probleme werden während Reviews trotz langwieriger Diskussionen übersehen

## Symptoms ▲

- [Verlängerte Review-Zyklen](verlaengerte-review-zyklen.md)
<br/>  Ineffiziente Reviews erfordern mehrere Runden trivialen Feedbacks, was die Zeit von der Einreichung bis zur Genehmigung erheblich verlängert.
- [Review-Engpässe](review-engpaesse.md)
<br/>  Langsame und umständliche Review-Prozesse erzeugen Engpässe, die die Feature-Lieferung verzögern.
- [Frustration der Autoren](frustration-der-autoren.md)
<br/>  Entwickler werden frustriert durch widersprüchliches, oberflächliches oder kleinliches Review-Feedback, das ihre Zeit verschwendet.
- [Verringerte Häufigkeit von Code-Einreichungen](verringerte-haeufigkeit-von-code-einreichungen.md)
<br/>  Entwickler bündeln Änderungen, um häufige, schmerzhafte Review-Zyklen zu vermeiden, was die Integrationshäufigkeit verringert.
- [Langsame Entwicklungsgeschwindigkeit](langsame-entwicklungsgeschwindigkeit.md)
<br/>  Unverhältnismäßig viel für Reviews aufgewendete Zeit verringert das Gesamttempo der Feature-Lieferung.

## Causes ▼

- [Undefinierte Code-Stil-Richtlinien](undefinierte-code-stil-richtlinien.md)
<br/>  Ohne klare Coding-Standards entarten Reviews zu subjektiven Stildebatten statt inhaltlicher Qualitätsdiskussionen.
- [Widersprüchliche Reviewer-Meinungen](widerspruechliche-reviewer-meinungen.md)
<br/>  Mehrere Reviewer, die widersprüchliches Feedback liefern, erzeugen Verwirrung und unnötige Überarbeitungszyklen.
- [Nitpicking-Kultur](nitpicking-kultur.md)
<br/>  Eine auf kleinliche Details fokussierte Kultur lenkt die Review-Aufmerksamkeit von wichtigen Design- und Logikfragen ab.
- [Große Pull Requests](grosse-pull-requests.md)
<br/>  Große Pull Requests sind schwerer gründlich zu überprüfen, was entweder zu oberflächlichen Reviews oder übermäßiger Review-Zeit führt.

## Detection Methods ○

- **Review-Zeit-Tracking:** Messung der für Reviews aufgewendeten Zeit im Verhältnis zur Entwicklungszeit und Änderungskomplexität
- **Review-Runden-Analyse:** Nachverfolgung, wie viele Review-Iterationen für verschiedene Arten von Änderungen nötig sind
- **Klassifikation des Review-Feedbacks:** Kategorisierung von Review-Kommentaren zur Identifikation, welche Arten von Problemen angesprochen werden
- **Entwickler-Umfragen:** Erhebung von Feedback zur Wirksamkeit und Effizienz des Review-Prozesses
- **Review-Abdeckungsanalyse:** Bewertung, ob Reviews wichtige Probleme abfangen oder sich auf triviale Belange konzentrieren

## Examples

Ein Team verbringt durchschnittlich 8 Stunden mit dem Review einer 200-Zeilen-Feature-Implementierung, deren Entwicklung 4 Stunden gedauert hat. Der Review-Prozess umfasst drei Feedback-Runden, wobei sich die meisten Kommentare auf Präferenzen bei Variablennamen, Codeformatierung und kleinere Stilfragen konzentrieren statt auf Logik, Design oder potenzielle Fehler. Trotz der umfangreichen Review-Zeit gelangt ein erheblicher Logikfehler in die Produktion, weil Reviewer durch Stildiskussionen abgelenkt waren und die Geschäftslogik nicht sorgfältig untersucht haben. Ein weiteres Beispiel betrifft ein Code-Review, bei dem fünf verschiedene Reviewer widersprüchliche Ratschläge zu demselben Codestück geben – einer schlägt vor, eine Methode zu extrahieren, ein anderer empfiehlt, sie zu inlinen, ein dritter möchte andere Variablennamen, und ein vierter stellt den gesamten Ansatz infrage. Der Autor verbringt Tage damit, alles Feedback zu adressieren, und der Review-Prozess dauert länger, als die Implementierung von drei ähnlichen Features von Grund auf gedauert hätte.
