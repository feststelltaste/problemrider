---
title: Implementierung beginnt ohne Design
description: Die Entwicklung beginnt mit unklarer Struktur, was zu unorganisiertem
  Code und architektonischer Drift führt.
category:
- Architecture
- Code
- Process
related_problems:
- slug: complex-implementation-paths
  similarity: 0.55
- slug: process-design-flaws
  similarity: 0.55
- slug: architectural-mismatch
  similarity: 0.55
- slug: insufficient-design-skills
  similarity: 0.55
- slug: feature-creep-without-refactoring
  similarity: 0.55
- slug: inexperienced-developers
  similarity: 0.55
solutions:
- evolutionary-requirements-development
- requirements-analysis
- checklists
- secure-software-development
- security-by-design
- security-requirements-definition
- technical-spike
- tracer-bullets
- walking-skeleton
- threat-modeling
- wireframing
layout: problem
lang: de
en_slug: implementation-starts-without-design
---

## Description

Implementierung beginnt ohne Design tritt auf, wenn Entwicklungsteams sofort mit dem Programmieren beginnen, ohne zuvor eine klare architektonische Vision, Systemstruktur oder ein detailliertes Design zu etablieren. Diese Eile zum Code entspringt oft Zeitdruck, Aufregung, mit dem Bauen zu beginnen, oder Missverständnissen über agile Entwicklungspraktiken. Das Ergebnis sind Systeme, die sich organisch ohne kohärente Struktur entwickeln, was zu Code führt, der schwer zu verstehen, zu warten und zu erweitern ist. Dieses Problem ist besonders schädlich in Legacy-Modernisierungsprojekten, in denen die Gelegenheit, eine bessere Architektur zu etablieren, verloren geht.

## Indicators ⟡

- Entwicklungsarbeit beginnt sofort nach der Anforderungserhebung, ohne Design-Phasen
- Architekturdiskussionen finden während der Implementierung statt, nicht davor
- Es gibt keine klar definierte Systemstruktur oder Komponentengrenzen vorab
- Datenbankschemata werden während der Entwicklung ad hoc erstellt
- API-Designs entstehen organisch, statt geplant zu werden
- Teammitglieder sind sich über die Gesamtarchitektur oder Designmuster des Systems unsicher
- Technologieentscheidungen werden individuell von Entwicklern während der Implementierung getroffen

## Symptoms ▲

- [Implementierungs-Nacharbeit](implementierungs-nacharbeit.md)
<br/>  Ohne vorheriges Design werden strukturelle Probleme während oder nach der Implementierung entdeckt, was erheblichen Neubau erfordert.
- [Probleme durch enge Kopplung](probleme-durch-enge-kopplung.md)
<br/>  Ohne geplante Komponentengrenzen wächst der Code organisch mit engen gegenseitigen Abhängigkeiten zwischen Modulen.
- [Hohe technische Schulden](hohe-technische-schulden.md)
<br/>  Ad-hoc-Design-Entscheidungen, die während des Programmierens getroffen werden, häufen sich als technische Schulden an, da ihnen eine kohärente architektonische Vision fehlt.
- [Anhäufung von Workarounds](anhaeufung-von-workarounds.md)
<br/>  Ohne ein klares Design schaffen Entwickler Workarounds, um strukturelle Probleme zu flicken, die während der Implementierung entstehen.
- [Stagnierende Architektur](stagnierende-architektur.md)
<br/>  Eine organisch entwickelte Architektur ohne klares Design wird schwer, absichtlich weiterzuentwickeln oder zu verbessern.
- [Spaghetticode](spaghetticode.md)
<br/>  Mit dem Programmieren ohne Design zu beginnen, führt direkt zu verworrenem, unstrukturiertem Code, da es keinen architektonischen Bauplan gibt, dem man folgen könnte.

## Causes ▼

- [Zeitdruck](zeitdruck.md)
<br/>  Enge Termine drängen Teams dazu, Design-Phasen zu überspringen und direkt mit dem Programmieren zu beginnen, um schnell Fortschritt zu zeigen.
- [Unzureichende Design-Fähigkeiten](unzureichende-design-faehigkeiten.md)
<br/>  Teams ohne Architektur-Expertise erkennen möglicherweise nicht den Wert vorherigen Designs oder wissen nicht, wie man es effektiv durchführt.
- [Anforderungsmehrdeutigkeit](anforderungsmehrdeutigkeit.md)
<br/>  Wenn Anforderungen vage sind, fühlen sich Teams möglicherweise unfähig, vorab zu designen, und greifen stattdessen auf exploratives Programmieren zurück.

## Detection Methods ○

- Überprüfung von Projektzeitplänen auf die Zuteilung von Design- und Architekturaktivitäten
- Untersuchung von Code-Repositories auf Nachweise konsistenter architektonischer Muster
- Durchführung von Architektur-Reviews früh im Entwicklungsprozess
- Beobachtung der Häufigkeit strukturellen Refactorings und architektonischer Änderungen
- Bewertung des Teamverständnisses der Systemstruktur durch Interviews oder Dokumentationsüberprüfungen
- Überprüfung der Datenbankschema-Entwicklung auf Anzeichen organischen, ungeplanten Wachstums
- Analyse von Code-Metriken auf Konsistenz in Designmustern und struktureller Organisation

## Examples

Ein Startup, das eine neue SaaS-Plattform baut, beginnt sofort mit der Programmierung von Features nach der Definition von User Stories, ohne die Gesamtsystemarchitektur zu entwerfen. Drei Monate in die Entwicklung stellen sie fest, dass ihr Datenmodell Multi-Tenancy nicht effizient unterstützen kann, ihr API-Design die Integration mobiler Apps erschwert und ihr Authentifizierungssystem nicht skalieren kann, um Enterprise-Kunden zu unterstützen. Was als schnelle Feature-Entwicklung begann, wird zu einer Reihe größerer Refactoring-Bemühungen, von denen jede Wochen der Arbeit erfordert und das Risiko birgt, Fehler einzuführen. Das Team verbringt mehr Zeit mit der Umstrukturierung bestehenden Codes als mit dem Bau neuer Features, und der ursprünglich enge Zeitplan verlängert sich um Monate, während sie architektonische Entscheidungen nachträglich einbauen, die von Anfang an hätten getroffen werden sollen.
