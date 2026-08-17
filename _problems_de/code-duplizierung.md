---
title: Code-Duplizierung
description: Ähnlicher oder identischer Code existiert an mehreren Stellen, was
  die Wartung erschwert und Inkonsistenzrisiken schafft.
category:
- Architecture
- Code
related_problems:
- slug: synchronization-problems
  similarity: 0.8
- slug: copy-paste-programming
  similarity: 0.75
- slug: inconsistent-coding-standards
  similarity: 0.7
- slug: inconsistent-codebase
  similarity: 0.7
- slug: duplicated-work
  similarity: 0.7
- slug: duplicated-effort
  similarity: 0.65
solutions:
- incremental-refactoring
- aspect-oriented-programming-aop
- code-generation
- data-deduplication
- strategic-code-deletion
- feature-usage-measurement
- clean-code
- code-reading-sessions
- code-hotspot-analysis
- communities-of-practice
- automated-code-migration
- large-scale-refactoring
- duplication-detection
- quality-ratchet
- debt-accrual-analysis
- variant-consolidation
layout: problem
lang: de
en_slug: code-duplication
---

## Description

Code-Duplizierung entsteht, wenn ähnliche oder identische Funktionalität an mehreren Stellen in einer Codebasis implementiert ist. Während manche Duplizierung beabsichtigt oder harmlos sein mag, schafft übermäßige Duplizierung Wartungsalpträume, da Fehler an mehreren Stellen behoben werden müssen, Features an mehreren Stellen aktualisiert werden müssen und unvermeidlich Inkonsistenzen entstehen, während sich verschiedene Kopien unabhängig voneinander weiterentwickeln. Dieses Problem ist besonders verbreitet in Legacy-Systemen, in denen unterschiedliche Entwickler ähnliche Probleme isoliert voneinander gelöst haben.

## Indicators ⟡
- Ähnliche Logik erscheint in mehreren Dateien oder Funktionen
- Fehlerbehebungen müssen an mehreren unterschiedlichen Stellen angewendet werden
- Features sind über verschiedene Teile des Systems hinweg inkonsistent implementiert
- Copy-Paste-Muster sind in der Code-Historie oder -Struktur erkennbar
- Entwickler fragen häufig "wo muss ich diese Änderung noch vornehmen?"

## Symptoms ▲

- [Synchronisationsprobleme](synchronisationsprobleme.md)
<br/>  Wenn duplizierter Code an einer Stelle aktualisiert wird, aber nicht an anderen, weicht das Verhalten im System auseinander.
- [Inkonsistentes Verhalten](inkonsistentes-verhalten.md)
<br/>  Unterschiedliche Kopien duplizierter Logik entwickeln sich unabhängig weiter, was dazu führt, dass dieselbe Operation in unterschiedlichen Kontexten unterschiedliche Ergebnisse liefert.
- [Teilweise Fehlerbehebungen](teilweise-fehlerbehebungen.md)
<br/>  Fehler, die in einer Kopie duplizierten Codes behoben werden, sind möglicherweise nicht in allen anderen Kopien behoben, wodurch Schwachstellen bestehen bleiben.
- [Wartungs-Overhead](wartungs-overhead.md)
<br/>  Jede Änderung muss an mehreren Stellen angewendet werden, was den für Wartungsaufgaben erforderlichen Aufwand vervielfacht.
- [Testkomplexität](testkomplexitaet.md)
<br/>  Die Qualitätssicherung muss dieselbe Funktionalität an mehreren Stellen verifizieren, was den Testaufwand und das Risiko übersehener Fehler erhöht.
- [Erhöhtes Risiko für Fehler](erhoehtes-risiko-fuer-fehler.md)
<br/>  Dieselbe Logik an mehreren Stellen zu haben, vergrößert die Angriffsfläche für Defekte und die Wahrscheinlichkeit inkonsistenter Fixes.

## Causes ▼

- [Copy-Paste-Programmierung](copy-paste-programmierung.md)
<br/>  Entwickler kopieren und fügen bestehenden Code ein, statt wiederverwendbare Abstraktionen zu erstellen, was direkt Duplizierung erzeugt.
- [Schwierige Code-Wiederverwendung](schwierige-code-wiederverwendung.md)
<br/>  Wenn Code nicht für Wiederverwendung entworfen ist, duplizieren Entwickler ihn, weil das Extrahieren gemeinsamer Funktionalität zu kostspielig ist.
- [Team-Silos](team-silos.md)
<br/>  Teams, die isoliert arbeiten, sind sich bestehender Implementierungen nicht bewusst, was dazu führt, dass sie unabhängig voneinander ähnlichen Code schreiben.
- [Zeitdruck](zeitdruck.md)
<br/>  Unter Termindruck kopieren Entwickler bestehenden Code, statt Zeit in ordentliche Abstraktionen zu investieren.
- [Bequemlichkeitsgetriebene Entwicklung](bequemlichkeitsgetriebene-entwicklung.md)
<br/>  Bequemlichkeitsgetriebene Entwicklung führt direkt zu Code-Duplizierung, da das Kopieren bestehenden Codes der bequemste Weg ist.
- [Unerfahrene Entwickler](unerfahrene-entwickler.md)
<br/>  Unerfahrene Entwickler duplizieren häufig Code, weil sie bestehende Implementierungen nicht kennen oder nicht verstehen, wie man bestehende Logik ordentlich abstrahiert und wiederverwendet.

## Detection Methods ○
- **Statische Analysewerkzeuge:** Nutzung von Werkzeugen, die duplizierte oder ähnliche Codeblöcke in der Codebasis identifizieren können
- **Copy-Paste-Erkennung:** Werkzeuge wie CPD (Copy-Paste Detector) können duplizierte Codeabschnitte finden
- **Code-Review-Muster:** Beobachtung, ob Reviewer fragen "ist das nicht ähnlich zu Code in Modul X?"
- **Ähnlichkeitsanalyse:** Messung der Codeähnlichkeit zwischen Modulen zur Identifikation potenzieller Duplizierung
- **Fehlermuster-Analyse:** Nachverfolgung von Fehlern, die an mehreren Stellen behoben werden müssen, als Indikatoren für Duplizierung

## Examples

Ein E-Commerce-System hat drei unterschiedliche Routinen zur Nutzereingabevalidierung: eine für die Nutzerregistrierung, eine für Profilaktualisierungen und eine für Checkout-Formulare. Jede validiert E-Mail-Adressen unterschiedlich – das Registrierungsformular akzeptiert internationale Domains, die Profilaktualisierung lehnt bestimmte Sonderzeichen ab, die die Registrierung erlaubt, und das Checkout-Formular hat sein eigenes Regelwerk. Als eine Sicherheitslücke in der E-Mail-Validierung entdeckt wird, muss die Behebung an drei unterschiedlichen Stellen angewendet werden, aber der Entwickler behebt nur zwei davon. Dies führt zu inkonsistenter Nutzererfahrung und einer Sicherheitslücke, die im Checkout-Prozess bestehen bleibt. In einem anderen Fall hat eine Finanzanwendung identischen Währungsformatierungscode, der über zwölf verschiedene Reporting-Module kopiert wurde. Als sich die Geschäftsanforderungen ändern, um ein neues Währungsformat zu unterstützen, müssen Entwickler alle zwölf Instanzen aufspüren und hoffen, keine davon zu übersehen, was zu Berichten führt, die Währungen inkonsistent anzeigen.
