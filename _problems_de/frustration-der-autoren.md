---
title: Frustration der Autoren
description: Entwickler werden frustriert durch unvorhersehbares, widersprüchliches
  oder willkürlich wirkendes Feedback im Code-Review-Prozess.
category:
- Culture
- Process
- Team
related_problems:
- slug: reviewer-anxiety
  similarity: 0.75
- slug: fear-of-conflict
  similarity: 0.75
- slug: conflicting-reviewer-opinions
  similarity: 0.75
- slug: developer-frustration-and-burnout
  similarity: 0.7
- slug: team-members-not-engaged-in-review-process
  similarity: 0.7
- slug: reduced-code-submission-frequency
  similarity: 0.7
solutions:
- sustainable-pace-practices
- code-review-guidelines
- psychological-safety-practices
- team-working-agreements
- small-change-batches
- blameless-postmortems
- team-retrospectives
- work-in-progress-limits
- communities-of-practice
- internal-technical-coaching
layout: problem
lang: de
en_slug: author-frustration
---

## Description

Frustration der Autoren entsteht, wenn Entwickler im Code-Review-Prozess zunehmend frustriert werden, weil sie unvorhersehbares, widersprüchliches oder willkürlich wirkendes Feedback zu ihren Code-Einreichungen erhalten. Diese Frustration entspringt inkonsistenten Review-Standards, langwierigen Hin-und-her-Diskussionen über subjektive Vorlieben oder dem Gefühl, dass Reviewer sich auf triviale Fragen konzentrieren, während wichtige Aspekte des Codes übersehen werden.

## Indicators ⟡

- Entwickler äußern Verärgerung oder Widerstand während Code-Review-Diskussionen
- Autoren stellen Review-Feedback häufig infrage oder streiten darüber
- Code-Review-Zyklen beinhalten mehrere Runden widersprüchlicher Vorschläge
- Entwickler beginnen, defensive Kommentare zu schreiben oder ihren Code übermäßig zu erklären
- Teammitglieder beginnen, das Einreichen von Code zum Review nach Möglichkeit zu vermeiden

## Symptoms ▲

- [Vermeidung des Review-Prozesses](vermeidung-des-review-prozesses.md)
<br/>  Frustrierte Autoren beginnen, den Review-Prozess zu vermeiden oder zu umgehen, um der frustrierenden Erfahrung zu entkommen.
- [Verringerte Häufigkeit von Code-Einreichungen](verringerte-haeufigkeit-von-code-einreichungen.md)
<br/>  Entwickler reichen Code seltener ein, um sich nicht mit unvorhersehbarem und frustrierendem Review-Feedback auseinandersetzen zu müssen.
- [Entwicklerfrustration und Burnout](entwicklerfrustration-und-burnout.md)
<br/>  Anhaltende Frustration mit dem Review-Prozess trägt zu allgemeinerer Entwicklerfrustration und Burnout bei.
- [Team-Dysfunktion](team-dysfunktion.md)
<br/>  Anhaltende Reibung zwischen Autoren und Reviewern schadet den Teambeziehungen und erzeugt Dysfunktion.
- [Große Pull Requests](grosse-pull-requests.md)
<br/>  Frustrierte Autoren bündeln ihre Änderungen in größeren Einreichungen, um die Anzahl der Review-Zyklen zu reduzieren, die sie durchlaufen müssen.

## Causes ▼

- [Widersprüchliche Reviewer-Meinungen](widerspruechliche-reviewer-meinungen.md)
<br/>  Widersprüchliches Feedback von verschiedenen Reviewern zu erhalten ist eine direkte Ursache für die Frustration der Autoren.
- [Nitpicking-Kultur](nitpicking-kultur.md)
<br/>  Eine Review-Kultur, die sich auf triviale Stilfragen statt auf inhaltliche Anliegen konzentriert, frustriert Autoren.
- [Undefinierte Code-Stil-Richtlinien](undefinierte-code-stil-richtlinien.md)
<br/>  Ohne definierte Stilrichtlinien wird Review-Feedback subjektiv und unvorhersehbar, was Autoren frustriert.
- [Inkonsistente Coding-Standards](inkonsistente-coding-standards.md)
<br/>  Inkonsistente Standards bedeuten, dass Autoren nicht vorhersehen können, welches Feedback sie erhalten werden, was zu Frustration führt.

## Detection Methods ○

- **Autoren-Zufriedenheitsumfragen:** Erhebung von Feedback zur Review-Erfahrung von Code-Autoren
- **Review-Zyklus-Analyse:** Nachverfolgung, wie viele Review-Runden benötigt werden und warum Überarbeitungen angefordert werden
- **Kommentartyp-Klassifikation:** Analyse, welche Arten von Problemen die meisten Hin-und-her-Diskussionen erzeugen
- **Bewertung der Teambeziehungen:** Beobachtung von Anzeichen für Spannungen oder Konflikte, die aus Review-Prozessen entstehen
- **Muster bei Code-Einreichungen:** Beobachtung von Änderungen darin, wie häufig Entwickler Code zum Review einreichen

## Examples

Ein Entwickler reicht eine gut getestete Feature-Implementierung ein und erhält Feedback von drei verschiedenen Reviewern: einer möchte den Code in kleinere Funktionen aufgeteilt haben, ein anderer schlägt vor, Funktionen zur Effizienzsteigerung zusammenzuführen, und der dritte konzentriert sich vollständig auf Namenskonventionen für Variablen. Nachdem das Feedback des ersten Reviewers berücksichtigt wurde, widerspricht der zweite Reviewer den Änderungen, und der dritte Reviewer fügt neue Stilanforderungen hinzu. Der Autor verbringt mehr Zeit damit, Review-Feedback zu adressieren, als mit dem Schreiben des ursprünglichen Features, und wird frustriert von den scheinbar willkürlichen und widersprüchlichen Anforderungen. Ein weiteres Beispiel betrifft einen Entwickler, dessen Pull Requests durchgängig Dutzende kleinerer Stilkommentare zu Abständen, Benennung und Formatierung erhalten, während logische Fehler oder Design-Probleme unbemerkt bleiben. Der Autor beginnt, übermäßige Kommentare und Dokumentation hinzuzufügen, um Kritik vorzubeugen, was den Code unnötig umständlich macht und die Entwicklung verlangsamt.
