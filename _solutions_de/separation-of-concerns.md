---
title: Separation of Concerns
description: Aufteilung von Funktionalitäten in klar abgegrenzte, unabhängige
  Bereiche.
category:
- Architecture
- Code
quality_tactics_url: https://qualitytactics.de/en/maintainability/separation-of-concerns/
problems:
- high-coupling-low-cohesion
- ripple-effect-of-changes
- hidden-side-effects
- single-entry-point-design
- complex-implementation-paths
- cognitive-overload
- increased-cognitive-load
- uncontrolled-codebase-growth
- circular-references
- difficult-code-comprehension
- procedural-programming-in-oop-languages
- convenience-driven-development
- excessive-class-size
- over-reliance-on-utility-classes
- poor-encapsulation
- bloated-class
- circular-dependency-problems
- global-state-and-side-effects
- god-object-anti-pattern
- monolithic-functions-and-classes
- tangled-cross-cutting-concerns
layout: solution
lang: de
en_slug: separation-of-concerns
related_solutions:
- slug: modularization-and-bounded-contexts
  similarity: 0.8
- slug: solid-principles
  similarity: 0.75
- slug: clean-code
  similarity: 0.75
- slug: aspect-oriented-programming-aop
  similarity: 0.75
- slug: incremental-refactoring
  similarity: 0.75
- slug: high-cohesion
  similarity: 0.7
---

## Description

Separation of Concerns teilt ein System in klar abgegrenzte, unabhängige Verantwortungsbereiche — Berechnung getrennt von Nebenwirkungen gehalten, Querschnittsbelange wie Logging aus der Geschäftslogik herausgezogen —, sodass jeder Teil verstanden und geändert werden kann, ohne das gesamte System im Kopf zu behalten. In Legacy-Systemen ist diese Trennung selten durch das ursprüngliche Design abwesend; sie erodiert allmählich durch Jahre inkrementeller, bequemlichkeitsgetriebener Änderungen, die einst klare Grenzen verwischen, bis eine einzelne Klasse Validierung, Berechnung, Logging und Benachrichtigung gleichzeitig übernimmt und eine Änderung an einer dieser Aufgaben riskiert, eine andere still zu brechen. Co-Change-Analyse — welche Dateien in der Versionskontrolle dazu neigen, gemeinsam geändert zu werden — ist oft der schnellste Weg, um genau herauszufinden, wo diese Grenzen verletzt wurden, und ihre Wiederherstellung zahlt sich direkt in reduziertem Blast-Radius für zukünftige Änderungen aus, obwohl übermäßige Trennung in eine Explosion winziger Klassen genauso leicht dasselbe Navigationsproblem in anderer Form neu erzeugen kann.

## How to Apply ◆

> In Legacy-Systemen ist Separation of Concerns selten abwesend, weil niemand daran gedacht hat — sie ist über Jahre inkrementeller Modifikationen erodiert, die einst klare Grenzen verwischten. Ihre Wiederherstellung erfordert bewusste Identifikation und Durchsetzung von Belangsgrenzen, beginnend mit den Bereichen, die den meisten Entwicklungsschmerz verursachen.

- Identifizieren Sie zuerst die schmerzhaftesten Kopplungs-Hotspots, indem Sie untersuchen, welche Dateien am häufigsten gemeinsam in der Versionskontrolle geändert werden. Co-Change-Analyse (mit Werkzeugen wie Code Maat oder Git-Log-Mining) offenbart implizite Belangsgrenzen, die verletzt wurden — wenn eine Änderung an der Abrechnungslogik konsequent eine Änderung an der Benachrichtigungslogik erfordert, wurden diese Belange unangemessen verschmolzen.
- Trennen Sie Nebenwirkungen von Berechnung als höchstprioritäre Belangsgrenze in Legacy-Code. Funktionen, die sowohl ein Ergebnis berechnen als auch Zustand ändern (E-Mails senden, in Datenbanken schreiben, Caches aktualisieren), sollten in eine reine Berechnungsfunktion und eine explizite nebenwirkungsauslösende Funktion aufgeteilt werden. Dies beseitigt versteckte Nebenwirkungen, die Funktionen unvorhersehbar und untestbar machen.
- Extrahieren Sie Querschnittsbelange (Logging, Authentifizierung, Validierung, Fehlerbehandlung) aus der Geschäftslogik mit Middleware, Decorators oder aspektorientierten Techniken. In Legacy-Single-Entry-Point-Designs sind diese Belange typischerweise durch den Haupt-Request-Handler gewoben, wodurch er auf Tausende von Zeilen wächst. Ihre Extraktion reduziert den Einstiegspunkt auf eine dünne Routing-Schicht.
- Wenden Sie vertikale Slicing-Prinzipien an, indem Sie Code um Geschäftsfunktionen statt technischer Schichten organisieren. Statt alle Controller zusammen, alle Services zusammen und alle Repositories zusammen zu gruppieren, gruppieren Sie den gesamten Code für eine bestimmte Geschäftsfähigkeit zusammen. Dies reduziert die Anzahl der Schichten, die ein Entwickler verstehen muss, um ein Feature zu implementieren, und adressiert direkt kognitive Überlastung.
- Nutzen Sie den Strangler-Fig-Ansatz, um Belangstrennung in bestehenden Code einzuführen: statt ein verworrenes Modul neu zu schreiben, routen Sie neue Funktionalität durch ordnungsgemäß getrennte Komponenten und migrieren Sie bestehendes Verhalten schrittweise aus dem monolithischen Modul heraus.
- Führen Sie explizite Schnittstellen zwischen Belangen ein, die kommunizieren müssen. Wenn zwei Belange Daten teilen, definieren Sie einen Vertrag (eine Schnittstelle, ein DTO oder ein Event), statt ihnen zu erlauben, in die internen Details des anderen zu greifen. Dies verhindert den Kaskadeneffekt, bei dem Änderungen in einem Belang in andere übergreifen.
- Setzen Sie Belangsgrenzen durch Build-Zeit-Regeln mit Werkzeugen wie ArchUnit, Dependency-Cruiser oder ähnlichem durch. Ohne Durchsetzung lösen sich Belangsgrenzen in Legacy-Systemen innerhalb von Wochen auf, während Entwickler unter Lieferdruck bequeme Abkürzungen nehmen.
- Dokumentieren Sie die beabsichtigten Belangsgrenzen mit einem leichtgewichtigen Architecture Decision Record (ADR), das erklärt, welchen Belang jedes Modul besitzt und warum, sodass später hinzukommende Entwickler die Begründung verstehen statt nur die Regeln.

## Tradeoffs ⇄

> Separation of Concerns adressiert direkt die Grundursache vieler Legacy-System-Probleme — verworrene Verantwortlichkeiten, die jede Änderung teuer und riskant machen —, erfordert aber anhaltende Disziplin, um Grenzen zu pflegen, sobald sie etabliert sind.

**Vorteile:**

- Reduziert kognitive Last drastisch, weil Entwickler sich auf einen Belang zur Zeit konzentrieren können, statt das gesamte Systemverhalten im Arbeitsgedächtnis zu halten, was direkt die mentale Ermüdung adressiert, die Legacy-Entwicklung verlangsamt.
- Beseitigt versteckte Nebenwirkungen, indem jede Funktion für einen einzigen, klar dokumentierten Belang verantwortlich gemacht wird, sodass Entwickler vorhersagen können, was eine Funktion aus ihrer Signatur und ihrem Standort tut.
- Verkleinert den Blast-Radius von Änderungen, weil eine Modifikation an einem Belang sich nicht auf nicht verwandte Belange ausbreitet, was direkt den Kaskadeneffekt reduziert, der Legacy-Änderungen so teuer und riskant macht.
- Ermöglicht unabhängiges Testen jedes Belangs in Isolation, was besonders wertvoll in Legacy-Systemen ist, in denen End-to-End-Testing das einzige bestehende Sicherheitsnetz ist.
- Bietet eine natürliche Wachstumsstruktur für die Codebasis: Neue Features fügen neue Belangsimplementierungen hinzu, statt bestehende aufzublähen, was unkontrolliertes Wachstum verhindert.

**Kosten und Risiken:**

- Übermäßige Trennung erzeugt eine Explosion winziger Klassen und Dateien, die genauso schwer zu navigieren sein kann wie der ursprüngliche verworrene Code — Belangsgrenzen sollten sich an bedeutsamen Geschäfts- oder technischen Unterteilungen ausrichten, nicht mechanisch an jeder möglichen Nahtstelle gezogen werden.
- Kommunikation zwischen getrennten Belangen führt Indirektion ein, die Debugging erschweren kann, wenn ein Entwickler den vollständigen Ausführungspfad über mehrere Komponenten hinweg verfolgen muss.
- Legacy-Code mit tief verflochtenen Belangen widersetzt sich der Trennung: Das Extrahieren eines Belangs offenbart oft, dass die anderen Belange von seinen internen Implementierungsdetails abhängen, was sorgfältiges Entwirren erfordert, bevor Trennung möglich ist.
- Teams, die an bequemlichkeitsgetriebene Abkürzungen gewöhnt sind, könnten sich Belangstrennung widersetzen, weil sie mehr vorheriges Nachdenken darüber erfordert, wohin neuer Code gehört, obwohl sie langfristig Zeit spart.
- Belange in einem System ohne ausreichende Testabdeckung zu trennen ist riskant, weil das Refactoring selbst subtil Verhalten ändern könnte, auf eine Weise, die nicht bis zur Produktion erfasst wird.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie Separation of Concerns angewendet wurde, um Klarheit und Wartbarkeit in verworrenen Legacy-Systemen zu bringen.

Ein staatliches Steuerverarbeitungssystem hatte eine zentrale `TaxCalculationEngine`-Klasse mit 4.800 Zeilen, die Steuerregelberechnung, Steuerzahlerdatenvalidierung, Audit-Trail-Logging, Benachrichtigungsversand und Strafberechnung verflocht. Jede Steuersaison brachte neue Regeln, die eine Modifikation dieser einzigen Klasse erforderten, und jede Modifikation riskierte, Audit-Logging oder Strafberechnungen auf subtile Weise zu brechen. Das Team verbrachte drei Monate damit, jeden Belang in sein eigenes Modul zu extrahieren: `TaxRuleEvaluator` übernahm reine Berechnung ohne Nebenwirkungen, `AuditTrailRecorder` übernahm alles Logging über einen Event-Listener, und `PenaltyAssessor` operierte auf der Ausgabe des Regel-Evaluators. Die ursprüngliche Klasse wurde zu einem 90-Zeilen-Orchestrator. In der folgenden Steuersaison wurden neue Regeln durch Implementierung einer neuen `TaxRule`-Schnittstelle hinzugefügt, ohne bestehenden Code anzufassen, und der Audit-Trail funktionierte weiterhin korrekt, weil er Events abonnierte, statt in den Berechnungsfluss eingebettet zu sein.

Die Schadensbearbeitungsanwendung eines Versicherungsunternehmens leitete alle HTTP-Anfragen durch ein einziges `ClaimsServlet`, das auf 2.600 Zeilen angewachsen war. Authentifizierungsprüfungen, Eingabevalidierung, Geschäftsregelausführung, Datenbankschreibvorgänge, PDF-Generierung und E-Mail-Benachrichtigungen wurden alle sequenziell in einer einzigen Methode durchgeführt. Die Unterstützung eines neuen Schadenstyps hinzuzufügen erforderte das Verstehen der gesamten Methode, und Entwickler führten häufig Fehler in nicht verwandten Belangen ein — eine Änderung an der PDF-Formatierung brach einmal die E-Mail-Benachrichtigung, weil beide einen String-Buffer teilten. Das Team wendete Separation of Concerns an, indem es jede Verantwortlichkeit in eine Middleware-Pipeline extrahierte: Authentifizierungs-Middleware lief zuerst, dann Validierung, dann der Geschäftslogik-Handler, dann eine Nachbearbeitungs-Pipeline für Nebenwirkungen wie PDFs und E-Mails. Das Servlet wurde durch einen dünnen Dispatcher ersetzt, der die Pipeline zusammensetzte. Neue Schadenstypen konnten dann hinzugefügt werden, indem ein neuer Geschäftslogik-Handler geschrieben wurde, ohne Änderungen an Authentifizierung, Validierung oder Benachrichtigungsverhalten zu riskieren.

Eine Logistikplattform war über zwölf Jahre auf 500.000 Zeilen ohne konsistentes Organisationsprinzip gewachsen. Entwickler, die an Sendungsverfolgung arbeiteten, mussten Abrechnungscode verstehen, und Abrechnungsentwickler mussten Zollcompliance-Code verstehen, weil diese Belange über dieselben Klassen und Pakete verstreut waren. Die daraus resultierende kognitive Überlastung bedeutete, dass neue Entwickler sechs Monate brauchten, um produktiv zu werden, und erfahrene Entwickler führten weiterhin routinemäßig belangübergreifende Regressionen ein. Das Team reorganisierte die Codebasis in vertikale Slices, ausgerichtet an Geschäftsfähigkeiten — Sendungsverfolgung, Abrechnung, Zoll und Flottenmanagement — mit expliziten Schnittstellen zwischen ihnen. Innerhalb von vier Monaten sank die Einarbeitungszeit für Entwickler auf acht Wochen, und die durchschnittliche Anzahl der pro Feature modifizierten Dateien sank von dreiundzwanzig auf sieben.
