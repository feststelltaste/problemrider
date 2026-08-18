---
title: Schlecht definierte Verantwortlichkeiten
description: Module oder Klassen sind nicht mit einer einzigen, klaren Verantwortlichkeit
  designt, was zu Verwirrung und enger Kopplung führt.
category:
- Architecture
- Code
related_problems:
- slug: monolithic-functions-and-classes
  similarity: 0.7
- slug: god-object-anti-pattern
  similarity: 0.65
- slug: high-coupling-low-cohesion
  similarity: 0.65
- slug: lack-of-ownership-and-accountability
  similarity: 0.65
- slug: difficult-to-understand-code
  similarity: 0.65
- slug: tight-coupling-issues
  similarity: 0.6
solutions:
- clear-ownership-model
- clear-roles-and-ownership
- authorization-concept
- compatibility-governance
- incident-management
- on-call-duty
- security-incident-handling
- emergency-drills
- incident-response-measures
- least-privilege
layout: problem
lang: de
en_slug: poorly-defined-responsibilities
---

## Description

Schlecht definierte Verantwortlichkeiten treten auf, wenn Softwarekomponenten keine klaren, einzelnen Zwecke haben und stattdessen mehrere nicht zusammenhängende Belange handhaben. Dies verletzt das Single-Responsibility-Prinzip und schafft Verwirrung darüber, was jede Komponente tut, was das System schwerer verständlich, testbar und wartbar macht. Wenn Verantwortlichkeiten unklar oder überlappend sind, kämpfen Entwickler damit zu wissen, wo Änderungen vorzunehmen sind, und Modifikationen in einem Bereich können unerwartete Auswirkungen auf scheinbar nicht zusammenhängende Funktionalität haben.

## Indicators ⟡
- Entwickler kämpfen damit zu erklären, was eine Klasse oder ein Modul in einem einzigen Satz tut
- Komponenten handhaben mehrere nicht zusammenhängende Geschäftsbelange oder technische Verantwortlichkeiten
- Änderungen an einem Feature erfordern Modifikationen an Komponenten, die scheinbar nicht zusammenhängen
- Ähnliche Funktionalität wird an mehreren Stellen implementiert, weil Verantwortungsgrenzen unklar sind
- Neue Features sind schwer zu implementieren, weil unklar ist, wo sie hingehören

## Symptoms ▲

- [Code-Duplizierung](code-duplizierung.md)
<br/>  Unklare Verantwortungsgrenzen führen dazu, dass ähnliche Funktionalität an mehreren Stellen implementiert wird.
- [God-Object-Antipattern](god-object-antipattern.md)
<br/>  Komponenten ohne klare einzelne Verantwortlichkeiten häufen nicht zusammenhängende Funktionalität an und werden zu God Objects.
- [Hohe Kopplung und geringe Kohäsion](hohe-kopplung-und-geringe-kohaesion.md)
<br/>  Überlappende Verantwortlichkeiten schaffen enge Kopplung zwischen Komponenten, während sie die interne Kohäsion verringern.
- [Schwer testbarer Code](schwer-testbarer-code.md)
<br/>  Komponenten, die mehrere nicht zusammenhängende Belange handhaben, sind aufgrund komplexer Abhängigkeiten schwer isoliert zu testen.
- [Wellenwirkung von Änderungen](wellenwirkung-von-aenderungen.md)
<br/>  Modifikationen an Komponenten mit mehreren Verantwortlichkeiten haben unerwartete Auswirkungen auf scheinbar nicht zusammenhängende Funktionalität.

## Causes ▼

- [Unzureichende Design-Fähigkeiten](unzureichende-design-faehigkeiten.md)
<br/>  Entwicklern ohne Design-Fähigkeiten fehlt die Erkennung und Durchsetzung klarer Single-Responsibility-Grenzen.
- [Feature-Creep ohne Refactoring](feature-creep-ohne-refactoring.md)
<br/>  Neue Features werden kontinuierlich zu bestehenden Komponenten hinzugefügt, ohne Refactoring zur Aufrechterhaltung klarer Verantwortlichkeiten.
- [Implementierung beginnt ohne Design](implementierung-beginnt-ohne-design.md)
<br/>  Der Beginn des Codings ohne vorheriges Design führt zu Ad-hoc-Verantwortungszuweisungen, die über die Zeit verschwimmen.
- [Bequemlichkeitsgetriebene Entwicklung](bequemlichkeitsgetriebene-entwicklung.md)
<br/>  Entwickler fügen Funktionalität der Bequemlichkeit halber zur nächstgelegenen verfügbaren Komponente hinzu, statt ordentlich abgegrenzte Module zu erstellen.

## Detection Methods ○
- **Verantwortlichkeitskartierung:** Dokumentation dessen, was jede Komponente tut, und Identifikation derer mit mehreren nicht zusammenhängenden Verantwortlichkeiten
- **Änderungsauswirkungsanalyse:** Nachverfolgung, welche Komponenten für verschiedene Arten von Änderungen modifiziert werden müssen
- **Kopplungsmetriken:** Messung, mit wie vielen anderen Komponenten jede Komponente interagiert
- **Code-Review-Fokus:** Spezifische Untersuchung von Komponentenverantwortlichkeiten während Reviews
- **Entwicklerbefragungen:** Befragung von Teammitgliedern, welche Komponenten am schwersten zu verstehen oder zu modifizieren sind

## Examples

Eine `UserManager`-Klasse in einer Webanwendung handhabte ursprünglich Nutzerauthentifizierung, hat aber über die Zeit Verantwortlichkeiten für Nutzerprofilverwaltung, Passwort-Reset-E-Mail-Versand, Nutzeraktivitätsprotokollierung, Berechtigungsprüfung, Nutzer-Avatar-Bildverarbeitung, Social-Media-Integration und Nutzeranalytik-Tracking angehäuft. Wenn Entwickler neue nutzerbezogene Funktionalität hinzufügen müssen, sind sie sich unsicher, ob sie in `UserManager` gehört oder eine separate Komponente sein sollte. Das Hinzufügen eines einfachen Features wie Nutzereinstellungen erfordert das Verständnis und potenzielle Modifizieren von Code im Zusammenhang mit E-Mail-Verarbeitung, Bildhandhabung und Analytik. Die Klasse ist zu einem Auffangbecken für alles Nutzerbezogene geworden, was sie schwer zu testen, zu verstehen und zu warten macht. Ein weiteres Beispiel betrifft eine `DataProcessor`-Komponente, die CSV-Datei-Parsing, Datenvalidierung, Datenbankspeicherung, Fehlerberichterstattung, E-Mail-Benachrichtigungen, Datei-Archivierung und Performance-Metrik-Sammlung handhabt. Wenn das Geschäft Unterstützung für Excel-Dateien hinzufügen möchte, müssen Entwickler all diese nicht zusammenhängenden Verantwortlichkeiten verstehen, um zu bestimmen, wie die neue Funktionalität sicher hinzugefügt werden kann. Die schlecht definierten Verantwortlichkeiten machen unklar, welche Teile der Komponente zentral für die Datenverarbeitung sind versus unterstützende Belange, die getrennt werden könnten.
