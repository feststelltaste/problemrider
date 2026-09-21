---
title: Copy-Paste-Programmierung
description: Entwickler kopieren häufig Code, statt wiederverwendbare Komponenten
  zu erstellen, was zu Wartungsalpträumen und subtilen Fehlern führt.
category:
- Code
- Process
related_problems:
- slug: code-duplication
  similarity: 0.75
- slug: inconsistent-codebase
  similarity: 0.65
- slug: increased-technical-shortcuts
  similarity: 0.65
- slug: difficult-code-reuse
  similarity: 0.65
- slug: defensive-coding-practices
  similarity: 0.65
- slug: inconsistent-coding-standards
  similarity: 0.65
solutions:
- incremental-refactoring
- aspect-oriented-programming-aop
- code-generation
- code-hotspot-analysis
- preparatory-refactoring
- clean-code
- code-reviews
- static-analysis-and-linting
- strategic-code-deletion
- code-reading-sessions
- quality-ratchet
- debt-accrual-analysis
- automated-code-migration
- duplication-detection
layout: problem
lang: de
en_slug: copy-paste-programming
---

## Description

Copy-Paste-Programmierung ist eine Entwicklungspraxis, bei der Entwickler bestehenden Code duplizieren, statt wiederverwendbare, gut entworfene Komponenten oder Abstraktionen zu erstellen. Während das Kopieren von Code wie eine schnelle Lösung erscheinen mag, schafft es langfristige Wartungsprobleme, führt zu Inkonsistenzen und macht die Codebasis brüchig. Diese Praxis wird oft durch Zeitdruck, mangelndes Verständnis bestehenden Codes oder unzureichende Erfahrung mit ordentlichen Abstraktionstechniken angetrieben.

## Indicators ⟡
- Ähnliche Codeblöcke erscheinen mit geringfügigen Variationen in der gesamten Codebasis
- Die Git-Historie zeigt häufiges Kopieren großer Codeabschnitte zwischen Dateien
- Entwickler fragen regelmäßig "wo muss ich diese gleiche Änderung noch vornehmen?"
- Fehlerbehebungen erfordern das Aufspüren mehrerer Stellen, an denen ähnlicher Code existiert
- Code-Reviews beinhalten häufig Diskussionen über bestehende ähnliche Implementierungen

## Symptoms ▲

- [Code-Duplizierung](code-duplizierung.md)
<br/>  Copy-Paste-Programmierung erzeugt direkt duplizierte Codeblöcke, die über die gesamte Codebasis verstreut sind.
- [Synchronisationsprobleme](synchronisationsprobleme.md)
<br/>  Wenn Code dupliziert wird, werden Aktualisierungen an einer Kopie nicht auf andere angewendet, was zu abweichendem Verhalten im System führt.
- [Teilweise Fehlerbehebungen](teilweise-fehlerbehebungen.md)
<br/>  Fehlerbehebungen, die an einer Kopie duplizierten Codes angewendet werden, werden bei anderen Kopien übersehen, wodurch manche Instanzen des Fehlers ungelöst bleiben.
- [Inkonsistentes Verhalten](inkonsistentes-verhalten.md)
<br/>  Duplizierter Code, der im Laufe der Zeit auseinanderdriftet, führt dazu, dass dieselbe Geschäftslogik in unterschiedlichen Teilen des Systems unterschiedliche Ergebnisse liefert.
- [Wartungs-Overhead](wartungs-overhead.md)
<br/>  Jeder duplizierte Codeblock vervielfacht die Wartungslast, da Änderungen über alle Kopien hinweg repliziert werden müssen.
- [Testkomplexität](testkomplexitaet.md)
<br/>  Die Qualitätssicherung muss dieselbe Funktionalität an mehreren Stellen verifizieren, was den Testaufwand und das Risiko übersehener Fehler erhöht.

## Causes ▼

- [Zeitdruck](zeitdruck.md)
<br/>  Unter dem Druck, schnell zu liefern, kopieren Entwickler bestehenden Code, statt Zeit in die Erstellung wiederverwendbarer Komponenten zu investieren.
- [Schwierige Code-Wiederverwendung](schwierige-code-wiederverwendung.md)
<br/>  Wenn bestehender Code nicht für Wiederverwendung entworfen ist, finden Entwickler es einfacher, ihn zu kopieren und anzupassen, als ihn in wiederverwendbare Komponenten zu refaktorieren.
- [Unerfahrene Entwickler](unerfahrene-entwickler.md)
<br/>  Entwickler ohne Erfahrung mit ordentlichen Abstraktionstechniken greifen standardmäßig auf das Kopieren von Code als einfachsten Ansatz zurück.
- [Bequemlichkeitsgetriebene Entwicklung](bequemlichkeitsgetriebene-entwicklung.md)
<br/>  Die Praxis, die einfachste Lösung zu wählen, führt natürlicherweise zum Kopieren bestehenden Codes statt zum Entwurf ordentlicher Abstraktionen.

## Detection Methods ○
- **Code-Ähnlichkeitsanalyse:** Nutzung von Werkzeugen wie PMDs Copy-Paste Detector (CPD), um duplizierte Codeblöcke zu finden
- **Versionskontrollanalyse:** Untersuchung der Commit-Historie auf Muster des Kopierens von Dateien oder großen Codeabschnitten
- **Statische Analysewerkzeuge:** Werkzeuge, die strukturelle Ähnlichkeiten zwischen Codeabschnitten erkennen können
- **Code-Review-Checklisten:** Einbeziehung von Prüfungen auf ähnliche bestehende Funktionalität während Reviews
- **Refactoring-Gelegenheiten:** Bereiche mit hoher Duplizierung sind erstklassige Kandidaten für Refactoring

## Examples

Eine Webanwendung hat Nutzerauthentifizierung auf sechs unterschiedliche Arten über verschiedene Seiten hinweg implementiert. Als ein Entwickler Login-Funktionalität zu einem neuen Feature hinzufügen musste, kopierte er, statt die bestehenden Authentifizierungskomponenten zu verstehen und wiederzuverwenden, den Login-Code von einer ähnlichen Seite. Er vergaß jedoch, die Weiterleitungs-URL nach erfolgreichem Login zu aktualisieren, was dazu führte, dass Nutzer zur falschen Seite geschickt wurden. Zusätzlich enthielt der kopierte Code einen subtilen Fehler, der später am ursprünglichen Ort behoben wurde, aber nicht in der Kopie, was inkonsistentes Sicherheitsverhalten schuf. Ein weiteres Beispiel betrifft ein E-Commerce-System, bei dem Produktpreisberechnungen über mehrere Module hinweg kopiert und eingefügt werden. Als das Unternehmen eine neue Steuerregel einführt, müssen Entwickler die Berechnung an acht unterschiedlichen Stellen aktualisieren. Sie übersehen zwei Stellen, was zu falscher Preisgestaltung auf bestimmten Seiten führt, während andere die korrekten Preise zeigen, was Kunden verwirrt und Umsatzverluste verursacht.
