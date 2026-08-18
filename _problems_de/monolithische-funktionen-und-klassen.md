---
title: Monolithische Funktionen und Klassen
description: Einzelne Funktionen oder Klassen übernehmen zu viele nicht zusammenhängende
  Verantwortlichkeiten, was sie schwer verständlich und modifizierbar macht.
category:
- Architecture
- Code
related_problems:
- slug: god-object-anti-pattern
  similarity: 0.8
- slug: excessive-class-size
  similarity: 0.7
- slug: poorly-defined-responsibilities
  similarity: 0.7
- slug: monolithic-architecture-constraints
  similarity: 0.7
- slug: bloated-class
  similarity: 0.65
- slug: code-duplication
  similarity: 0.6
solutions:
- incremental-refactoring
- code-metrics
- high-cohesion
- code-hotspot-analysis
- dependency-breaking-techniques
- mikado-method
- solid-principles
- separation-of-concerns
- preparatory-refactoring
- characterization-tests
- code-reading-sessions
- automated-code-migration
layout: problem
lang: de
en_slug: monolithic-functions-and-classes
---

## Description

Monolithische Funktionen und Klassen sind Code-Komponenten, die gewachsen sind, um mehrere, oft nicht zusammenhängende Verantwortlichkeiten innerhalb einer einzigen Einheit zu handhaben. Diese „God Functions" oder „God Classes" verletzen das Single-Responsibility-Prinzip und werden zu zentralen Komplexitätspunkten, die schwer zu verstehen, zu modifizieren, zu testen oder wiederzuverwenden sind. Sie entstehen oft organisch, während im Laufe der Zeit Features hinzugefügt werden, wobei Entwickler bestehende Funktionen kontinuierlich erweitern statt neue, fokussierte Komponenten zu erstellen.

## Indicators ⟡
- Funktionen, die Hunderte oder Tausende Zeilen lang sind
- Klassen mit Dutzenden Methoden und Instanzvariablen
- Funktionen oder Methoden, die umfangreiches Scrollen erfordern, um sie vollständig zu überprüfen
- Code, der mehrere unterschiedliche Geschäftskonzepte oder technische Belange handhabt
- Schwierigkeit, in einem einzigen Satz zusammenzufassen, was eine Funktion oder Klasse tut

## Symptoms ▲

- [Schwer verständliche Codebasis](schwer-verstaendliche-codebasis.md)
<br/>  Funktionen und Klassen, die viele Verantwortlichkeiten handhaben, sind extrem schwer zu verstehen, da Entwickler alle Belange gleichzeitig erfassen müssen.
- [Merge-Konflikte](merge-konflikte.md)
<br/>  Große Funktionen und Klassen, die von mehreren Entwicklern für verschiedene Features modifiziert werden, führen häufig zu Merge-Konflikten.
- [Schwierige Code-Wiederverwendung](schwierige-code-wiederverwendung.md)
<br/>  Wenn Funktionalität in monolithische Einheiten gebündelt ist, wird die Extraktion und Wiederverwendung einzelner Teile unpraktikabel.
- [Hohe Rate an neu eingeführten Fehlern](hohe-rate-an-neu-eingefuehrten-fehlern.md)
<br/>  Komplexe Funktionen mit mehreren Verantwortlichkeiten sind fehleranfällig, weil Änderungen an einem Belang unbeabsichtigt andere beeinflussen können.
- [Schwer testbarer Code](schwer-testbarer-code.md)
<br/>  Das Testen monolithischer Funktionen erfordert umfangreiche Einrichtung und Mocking vieler Abhängigkeiten, was gründliches Testen unpraktikabel macht.

## Causes ▼

- [Schlecht definierte Verantwortlichkeiten](schlecht-definierte-verantwortlichkeiten.md)
<br/>  Ohne klare Verantwortungszuweisung fügen Entwickler weiter Funktionalität zu bestehenden Komponenten hinzu, statt fokussierte neue zu erstellen.
- [Unzureichende Design-Fähigkeiten](unzureichende-design-faehigkeiten.md)
<br/>  Entwicklern ohne Design-Fähigkeiten fehlt die Erkenntnis, wann eine Funktion oder Klasse in kleinere, fokussierte Einheiten zerlegt werden sollte.
- [Kurzfristiger Fokus](kurzfristiger-fokus.md)
<br/>  Unter dem Druck, schnell zu liefern, erweitern Entwickler bestehende Funktionen, statt Zeit in ordentliche Zerlegung zu investieren.
- [Angst vor Veränderung](angst-vor-veraenderung.md)
<br/>  Entwickler vermeiden es, große Funktionen aufzuteilen, aufgrund des Risikos, Fehler in bereits funktionierendem Code einzuführen, was ihnen erlaubt, weiter zu wachsen.

## Detection Methods ○
- **Code-Metrik-Werkzeuge:** Nutzung statischer Analysewerkzeuge zur Messung von Funktionslänge, zyklomatischer Komplexität und Klassengröße
- **Verantwortlichkeitsanalyse:** Identifikation von Funktionen oder Klassen, die mehrere unterschiedliche Geschäfts- oder technische Belange handhaben
- **Code-Review-Muster:** Suche nach Reviews, die Schwierigkeiten beim Verstehen oder Testen bestimmter Komponenten erwähnen
- **Änderungshäufigkeitsanalyse:** Komponenten, die häufig modifiziert werden, handhaben möglicherweise zu viele Verantwortlichkeiten
- **Testkomplexität:** Identifikation von Komponenten, die umfangreiche Einrichtung oder mehrere Testszenarien erfordern

## Examples

Eine E-Commerce-Anwendung hat eine einzige `processOrder`-Funktion, die Zahlungsverarbeitung, Bestandsaktualisierungen, Kundenbenachrichtigungen, Bestellprotokollierung, Steuerberechnungen, Versandvereinbarungen, Treuepunkt-Updates und Betrugserkennung handhabt. Diese 800-Zeilen-Funktion wird modifiziert, wann immer sich ein Aspekt der Bestellabwicklung ändert, was sie zu einer ständigen Quelle von Fehlern und Merge-Konflikten macht. Das Testen dieser Funktion erfordert das Mocking von Zahlungssystemen, Datenbanken, E-Mail-Diensten und mehreren externen APIs. Wenn eine einfache Änderung an der Steuerberechnungslogik benötigt wird, müssen Entwickler den gesamten Bestellabwicklungs-Workflow verstehen und riskieren, Zahlungsverarbeitung oder Bestandsverwaltung zu brechen. Ein weiteres Beispiel betrifft eine `UserManager`-Klasse mit 45 Methoden, die Nutzerauthentifizierung, Profilverwaltung, Berechtigungen, Passwort-Reset, E-Mail-Verifikation, Aktivitätsprotokollierung und Social-Media-Integration handhabt. Jede Änderung an der Nutzerfunktionalität erfordert das Verständnis dieser massiven Klasse, und das Testen einzelner Features wie Passwort-Reset erfordert die Initialisierung des gesamten Nutzerverwaltungssystems.
