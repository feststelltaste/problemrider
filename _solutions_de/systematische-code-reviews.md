---
title: Systematische Code-Reviews
description: Systematische Überprüfung des Quellcodes durch Feedback von Kollegen.
category:
- Process
- Code
- Team
quality_tactics_url: https://qualitytactics.de/en/maintainability/code-reviews/
problems:
- code-review-inefficiency
- inadequate-code-reviews
- insufficient-code-review
- large-pull-requests
- nitpicking-culture
- conflicting-reviewer-opinions
- review-bottlenecks
- style-arguments-in-code-reviews
- superficial-code-reviews
- review-process-avoidance
- review-process-breakdown
- reviewer-anxiety
- reviewer-inexperience
- perfectionist-review-culture
- extended-review-cycles
- reduced-review-participation
- team-members-not-engaged-in-review-process
- rushed-approvals
- inadequate-initial-reviews
layout: solution
lang: de
en_slug: code-review-process-reform
related_solutions:
- slug: code-reviews
  similarity: 0.85
- slug: static-analysis-and-linting
  similarity: 0.85
- slug: architecture-reviews
  similarity: 0.85
- slug: code-review-guidelines
  similarity: 0.8
- slug: code-metrics
  similarity: 0.8
- slug: code-quality-gates
  similarity: 0.8
---

## Description

Die Reform des Code-Review-Prozesses bedeutet, das Review-Gate einzuführen oder zu verschärfen, das Legacy-Systemen häufig vollständig fehlt, sodass Hotfixes und Workarounds nicht mehr völlig ungeprüft in die Codebasis gelangen. Über das Abfangen von Defekten vor dem Mergen hinaus dient ein bewusst strukturiertes Review — kleine Änderungssätze, über unvertraute Module rotierte Reviewer, eine auf legacy-spezifische Risiken wie versteckte Nebeneffekte in geteiltem Zustand abgestimmte Checkliste — gleichzeitig als Wissensverbreitungs- und Dokumentationsmechanismus für Systeme, in denen beides sonst dünn ist. Die Review-Kommentare selbst werden oft zum Nächsten, was ein Legacy-Modul an einem Designprotokoll hat, und erklären Entscheidungen, die sonst nur im Gedächtnis einer Person leben würden.

## How to Apply ◆

> In Legacy-Systemen, wo Qualitätskontrollen schwach oder abwesend sind, ist die Reform des Code-Review-Prozesses eine der direktesten Interventionen, um weiteren Verfall zu stoppen.

- Etablieren Sie eine verpflichtende Pull-Request-Review-Richtlinie, selbst für Wartungsänderungen; Legacy-Systemen fehlt oft überhaupt jedes Review-Gate, was bedeutet, dass jeder Hotfix und Workaround ungeprüft eintritt.
- Rotieren Sie Reviewer bewusst über unvertraute Module — Legacy-Codebasen tendieren dazu, eine Person zu haben, die jede dunkle Ecke „besitzt", und Review-Rotation ist der einzige Weg, dieses Wissen zu verbreiten, bevor sie geht.
- Definieren Sie eine leichtgewichtige Review-Checkliste, zugeschnitten auf Legacy-Risiken: Prüfen auf versteckte Nebeneffekte in geteiltem globalem Zustand, undokumentierte Annahmen über externes Systemverhalten, und fehlende Rollback-Pfade für Datenbankänderungen.
- Halten Sie Änderungssätze klein, indem Sie verlangen, dass Refaktorierungs-Commits von Verhaltensänderungen getrennt sind; große gemischte Diffs in Legacy-Code sind nahezu unmöglich sinnvoll zu überprüfen.
- Nutzen Sie asynchrones Pull-Request-Review-Tooling (GitHub, GitLab, Bitbucket), selbst wenn das Team am selben Ort sitzt — dies schafft einen schriftlichen Prüfpfad von Designentscheidungen, der die fehlende Dokumentation kompensiert, die für Legacy-Systeme typisch ist.
- Automatisieren Sie alle Stil- und Formatierungsprüfungen via Linting, bevor Code menschliche Reviewer erreicht, sodass sich Reviewer auf Logik, Kopplung und architektonische Absicht statt Whitespace-Argumente fokussieren können.
- Nominieren Sie mindestens einen Reviewer pro Änderung, der kein Vorwissen über das Modul hat — seine Verwirrung offenbart undokumentierte Annahmen, die der ursprüngliche Autor längst nicht mehr bemerkt.
- Bestätigen und protokollieren Sie während des Reviews getroffene Designentscheidungen als Inline-Kommentare oder verknüpfte Entscheidungsprotokolle; in Legacy-Systemen wird dieser Kommentar oft zur einzig verfügbaren Dokumentation dafür, warum Code so strukturiert ist, wie er ist.

## Tradeoffs ⇄

> Code-Review-Reform fügt einem Entwicklungsprozess Overhead hinzu, der in Legacy-Kontexten oft bereits langsam ist, sodass die Tradeoffs im Sinne der Kosten des Nicht-Überprüfens formuliert werden müssen.

**Vorteile:**

- Stoppt die Anhäufung versteckter Workarounds, indem sie vor dem Mergen abgefangen werden, was die Rate verringert, mit der die Legacy-Codebasis neue Schulden anhäuft.
- Verbreitet Wissen über notorisch isolierte Legacy-Module über mehr Teammitglieder, was den Bus-Faktor für Systeme verringert, in denen der Abgang eines einzelnen Experten katastrophal sein könnte.
- Schafft eine schrittweise Dokumentationsschicht in Review-Kommentaren und Commit-Historien für Code, der nie formal dokumentiert wurde.
- Setzt Coding-Standards konsistent für die Zukunft durch, selbst wenn der bestehende Code sie nicht erfüllt, und verhindert weitere Divergenz des Stils über eine bereits inkonsistente Codebasis.
- Bringt wiederkehrende Muster von Fragilität ans Licht — wenn dasselbe Modul weiterhin Review-Befunde erzeugt, signalisiert es, wo tiefere Refaktorierungsinvestition benötigt wird.

**Kosten und Risiken:**

- Die Überprüfung von Legacy-Code ist langsamer als die Überprüfung von Greenfield-Code, weil Reviewer undokumentierten Kontext verstehen müssen, bevor sie Korrektheit beurteilen können; erwarten Sie, dass Reviews länger dauern als in einem gut dokumentierten System.
- Review-Engpässe sind besonders schädlich in Legacy-Kontexten, wo das Team klein ist und oft dieselben Personen die einzigen sind, die spezifische Bereiche überprüfen können.
- Ohne psychologische Sicherheit könnten Entwickler, die an schlecht geschriebenem geerbtem Code arbeiten, das Gefühl haben, dass Review-Kommentare Kritik an der Arbeit ihrer Vorgänger sind, die gegen sie gewendet wird; Review-Normen müssen dies explizit adressieren.
- Eine oberflächliche Review-Kultur — blindes Abnicken zur Erfüllung von Prozessanforderungen — bietet falsches Vertrauen, während sie Kalenderverzögerung hinzufügt, was schlimmer ist als kein Prozess.
- Die Einführung verpflichtenden Reviews in ein an direkte Commits gewöhntes Team kann Widerstand erzeugen; die schrittweise Einführung der Anforderung nach Modul oder Risikostufe verringert Reibung.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie sich Code-Review-Reform in typischen Legacy-Modernisierungskontexten entfaltet.

Ein Finanzdienstleistungsunternehmen erbte eine monolithische Java-Anwendung, gebaut über fünfzehn Jahre durch eine Abfolge von Auftragnehmern. Es existierte kein Review-Prozess; Entwickler committeten direkt in den Main-Branch. Nach drei Produktionsvorfällen, die auf ungeprüfte Hotfixes zurückgeführt wurden, die kaskadierend ineinander griffen, führte das Team Pull-Request-Reviews mit einer Zwei-Personen-Genehmigungsanforderung für Änderungen ein, die das Zahlungsverarbeitungsmodul betrafen. Innerhalb von zwei Monaten markierten Reviewer Nebeneffekte in geteiltem Transaktionszustand, die den Autoren nicht aufgefallen waren — genau die Art von Problem, das die Vorfälle verursacht hatte. Die schriftliche Review-Historie wurde außerdem zur ersten strukturierten Dokumentation des Teams über die Zahlungsflusslogik.

Eine Regierungsbehörde, die ein COBOL-basiertes Leistungssystem betrieb, musste Junior-Entwickler einbinden, um alterndes Personal zu ergänzen. Weil nur zwei Ingenieure die Kernberechnungs-Engine verstanden, wurde Review als strukturierter Wissenstransfermechanismus genutzt: Jede Änderung am Berechnungsmodul erforderte, dass die Junior-Entwickler sie mit Anleitung der Senioren überprüften, die Domänenregeln in Review-Kommentaren statt in Meetings erklärten. Über sechs Monate konnten drei zusätzliche Teammitglieder unabhängig Änderungen an der Engine bewerten — was das Single-Point-of-Failure-Risiko für ein System, das Millionen von Ansprüchen pro Jahr verarbeitete, messbar verringerte.

Die E-Commerce-Plattform eines Einzelhandelsunternehmens war durch Akquisitionen gewachsen und kombinierte drei ehemals unabhängige Codebasen mit drei unterschiedlichen Coding-Konventionen. Das Team führte eine Review-Checkliste ein, die eine Stilkonventionsprüfung beinhaltete (welcher der drei Standards genutzt wurde) und eine Kopplungsprüfung (führte die Änderung Abhängigkeiten über die ehemals separaten Subsysteme hinweg ein). Diese einfachen Checklistenpunkte fingen im ersten Quartal einen unverhältnismäßigen Anteil an Problemen ab: keine Bugs im engen Sinne, sondern architektonische Verstöße, die eine zukünftige Konsolidierung der Codebasen erheblich schwieriger gemacht hätten.
