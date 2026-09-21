---
title: Dependency-Injection-Container
description: Zentralisierte Verwaltung und Bereitstellung von Abhängigkeiten.
category:
- Code
- Architecture
problems:
- tight-coupling-issues
- hidden-dependencies
- difficult-to-test-code
- high-coupling-low-cohesion
- global-state-and-side-effects
- god-object-anti-pattern
- maintenance-overhead
layout: solution
lang: de
en_slug: dependency-injection-container
related_solutions:
- slug: dependency-injection
  similarity: 0.75
- slug: containerization
  similarity: 0.7
- slug: dependency-management-strategy
  similarity: 0.65
- slug: containerized-databases
  similarity: 0.65
- slug: modularization-and-bounded-contexts
  similarity: 0.65
- slug: ci-cd-pipeline
  similarity: 0.65
---

## Description

Ein Dependency-Injection-Container ist Infrastruktur, die die Erzeugung und Verdrahtung der Objekte einer Anwendung übernimmt — die deklarierten Abhängigkeiten jeder Komponente auflöst und den Objektgraphen nach den Bedingungen des Containers konstruiert —, und ersetzt die verstreuten `new`-Aufrufe, statischen Factories und Singletons, mittels derer Legacy-Code typischerweise seine eigene Objekterzeugung verwaltet. Statt dass eine Klasse instanziiert, was sie braucht, deklariert sie diese Bedürfnisse (meist über ihren Konstruktor), und der Container ist dafür verantwortlich, konkrete Instanzen zu liefern, ihre Lebensdauer zu verwalten (Singleton, Scoped, Transient), und dies konsistent über die gesamte Anwendung hinweg zu tun, statt über welche Ad-hoc-Konvention auch immer, die ein bestimmtes Stück Legacy-Code zufällig übernommen hatte. Diese Unterscheidung ist für Legacy-Systeme enorm wichtig, weil Code, der seine eigenen Abhängigkeiten intern konstruiert, nicht isoliert unit-getestet werden kann — jeder Versuch, eine Klasse zu testen, zieht unweigerlich alles mit sich, was sie direkt instanziiert —, und ein containerverwalteter Abhängigkeitsgraph ist das, was das Ersetzen eines Mocks oder Stubs für eine echte Abhängigkeit während des Testens unkompliziert macht, statt zu erfordern, dass die gesamte Anwendung läuft. Einen Container in eine bestehende Codebasis einzuführen ist notwendigerweise schrittweise, zuerst werden die testbarkeitsbeschränktesten Komponenten registriert, und containerverwaltete und legacy-verwaltete Objekterzeugung koexistieren während des Übergangs, und die Übung, Komponenten in einem Container zu registrieren, deckt häufig zuvor unsichtbare Probleme auf, etwa zirkuläre Abhängigkeiten, die der Container zu lösen ablehnt und die unentdeckt geblieben waren, solange Objekte von Hand konstruiert wurden. Das entsprechende Risiko ist, dass ein schlecht genutzter Container Legacy-Komplexität einfach verlagern statt entfernen kann — Konfigurationsfehler tauchen zur Laufzeit statt zur Compile-Zeit auf, und mit dem Muster nicht vertraute Teams können in das Service-Locator-Antimuster abdriften, das das Werkzeug ihnen eigentlich helfen sollte zu vermeiden.

## How to Apply ◆

> In Legacy-Systemen zentralisiert ein DI-Container Objekterzeugung und -verdrahtung, die sonst über Factories, Singletons und statische Initialisierer verstreut ist, und macht den Abhängigkeitsgraphen explizit und handhabbar.

- Wählen Sie einen zum Technologie-Stack des Legacy-Systems passenden DI-Container (Spring für Java, Autofac oder Microsoft.Extensions.DependencyInjection für .NET, InversifyJS für TypeScript).
- Migrieren Sie Objekterzeugung von verstreuten `new`-Aufrufen, statischen Factories und Service Locators zu containerverwalteter Registrierung, beginnend mit den testbarkeitsbeschränktesten Komponenten.
- Definieren Sie Komponenten-Lebensdauern (Singleton, Scoped, Transient) explizit in der Container-Konfiguration, um implizite Lebenszyklusverwaltung im Legacy-Code zu ersetzen.
- Nutzen Sie den Container, um Querschnittsbelange (Logging, Caching, Transaktionsverwaltung) mittels Decorator- oder Interceptor-Mustern zu verwalten, statt sie in Geschäftslogik einzubetten.
- Registrieren Sie Legacy-Komponenten im Container neben neuen Komponenten, um schrittweise Migration zu ermöglichen, ohne alles auf einmal refaktorieren zu müssen.
- Vermeiden Sie das Service-Locator-Antimuster, bei dem der Container selbst herumgereicht wird — injizieren Sie stattdessen spezifische Abhängigkeiten über Konstruktoren.

## Tradeoffs ⇄

> Ein DI-Container vereinfacht Abhängigkeitsverwaltung und ermöglicht Testbarkeit, fügt aber Framework-Komplexität hinzu und kann Laufzeitverhalten verschleiern.

**Vorteile:**

- Zentralisiert Abhängigkeitskonfiguration an einem Ort, was den Abhängigkeitsgraphen des Systems explizit und handhabbar macht.
- Ermöglicht das Austauschen von Implementierungen für Testing, Migration oder umgebungsspezifisches Verhalten, ohne Konsumentencode zu ändern.
- Verwaltet Objektlebensdauern automatisch, was Ressourcenlecks durch unsachgemäße Lebenszyklusbehandlung im Legacy-Code verhindert.
- Unterstützt schrittweise Legacy-Modernisierung, indem Legacy- und moderne Komponenten im selben Container koexistieren können.

**Kosten und Risiken:**

- Container-Konfigurationsfehler manifestieren sich zur Laufzeit statt zur Compile-Zeit, was potenziell schwer zu diagnostizierende Fehler verursacht.
- Große Container-Konfigurationen können komplex und schwer verständlich werden, was effektiv eine Form versteckter Komplexität durch eine andere ersetzt.
- Mit DI-Containern nicht vertraute Teams könnten sie missbrauchen und übermäßig komplexe Konfigurationen erzeugen oder in das Service-Locator-Antimuster verfallen.
- Die Container-Startzeit kann in Legacy-Systemen mit Tausenden registrierter Komponenten erheblich werden.

## How It Could Be

> Das folgende Szenario zeigt, wie ein DI-Container Testbarkeit in einer Legacy-Anwendung ermöglicht.

Die Legacy-Java-Anwendung eines Gesundheitsunternehmens hatte 800 Klassen, die ihre Abhängigkeiten mittels `new`-Operatoren und statischen Factory-Methoden erzeugten, was Unit-Testing ohne Start der gesamten Anwendung unmöglich machte. Das Team führte Springs DI-Container schrittweise ein: Sie begannen damit, die 50 kritischsten Service-Klassen zu registrieren und Schnittstellen für ihre Datenbank- und externen Service-Abhängigkeiten zu extrahieren. Innerhalb von zwei Monaten konnten diese 50 Klassen isoliert mit vom Container injizierten Mock-Implementierungen getestet werden. Der Container enthüllte auch Abhängigkeitszyklen, die unsichtbar gewesen waren — drei Service-Klassen bildeten eine zirkuläre Abhängigkeitskette, die der Container zu lösen ablehnte, was das Team zwang, den Zyklus zu entwirren. Über das folgende Jahr migrierte das Team alle 800 Klassen zur Container-Verwaltung, und die Testabdeckung stieg von 5 auf 55 Prozent.
