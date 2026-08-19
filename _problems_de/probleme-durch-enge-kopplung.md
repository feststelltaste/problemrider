---
title: Probleme durch enge Kopplung
description: Komponenten sind übermäßig voneinander abhängig, was Änderungen schwierig
  macht und die Systemflexibilität und Wartbarkeit verringert.
category:
- Architecture
- Code
related_problems:
- slug: high-coupling-low-cohesion
  similarity: 0.75
- slug: deployment-coupling
  similarity: 0.7
- slug: ripple-effect-of-changes
  similarity: 0.7
- slug: circular-dependency-problems
  similarity: 0.7
- slug: unpredictable-system-behavior
  similarity: 0.65
- slug: hidden-dependencies
  similarity: 0.65
solutions:
- event-driven-architecture
- incremental-refactoring
- modularization-and-bounded-contexts
- abstracted-file-system-access
- abstraction
- abstraction-layers
- api-first-development
- architecture-conformity-analysis
- bounded-contexts
- bridges
- business-event-processing
- event-driven-integration
- fault-containment
- hexagonal-architecture
- isolation-of-faulty-components
- layered-architecture
- mediator
- microservices
- microservices-architecture
- modulith
- protocol-abstraction
- standardized-interfaces
- tolerant-reader
- database-abstraction
- dependency-injection
- dependency-injection-container
- fitness-functions
- saga-pattern
- dependency-breaking-techniques
- explicit-extension-points
layout: problem
lang: de
en_slug: tight-coupling-issues
---

## Description

Probleme durch enge Kopplung treten auf, wenn Systemkomponenten übermäßig von den internen Implementierungen des jeweils anderen abhängen, was es schwierig macht, einzelne Komponenten zu modifizieren, zu testen oder zu ersetzen, ohne andere zu beeinflussen. Eng gekoppelte Systeme sind fragil, schwierig zu warten und widersetzen sich Veränderung, weil Modifikationen in einem Bereich oft Änderungen im gesamten System erfordern.

## Indicators ⟡

- Änderungen an einer Komponente erfordern häufig Änderungen an vielen anderen Komponenten
- Komponenten können nicht isoliert getestet werden, ohne komplexen Aufbau
- Zirkuläre Abhängigkeiten zwischen Modulen oder Klassen
- Komponenten greifen direkt auf die internen Datenstrukturen des jeweils anderen zu
- Schwierigkeiten beim Ersetzen oder Aktualisieren einzelner Komponenten

## Symptoms ▲

- [Wellenwirkung von Änderungen](wellenwirkung-von-aenderungen.md)
<br/>  Wenn Komponenten eng gekoppelt sind, erzwingt die Modifikation einer Komponente Änderungen in vielen anderen, was eine Wellenwirkung über die Codebasis schafft.
- [Angst vor Veränderung](angst-vor-veraenderung.md)
<br/>  Entwickler werden zögerlich, Code zu modifizieren, weil enge Kopplung es unmöglich macht, die vollständige Auswirkung von Änderungen vorherzusagen.
- [Regressionsfehler](regressionsfehler.md)
<br/>  Enge Kopplung bedeutet, dass Änderungen in einer Komponente still Funktionalität in abhängigen Komponenten brechen können, was Regressionen verursacht.
- [Deployment-Kopplung](deployment-kopplung.md)
<br/>  Wenn Komponenten auf Code-Ebene eng gekoppelt sind, müssen sie zusammen deployt werden, selbst wenn sich nur eine geändert hat.
- [Refactoring-Vermeidung](refactoring-vermeidung.md)
<br/>  Das hohe Risiko und der Aufwand des Refactorings eng gekoppelten Codes führt dazu, dass Teams notwendige Verbesserungen vermeiden.
- [Debugging-Schwierigkeiten](debugging-schwierigkeiten.md)
<br/>  Enge Kopplung macht es schwer, Bugs zu isolieren, weil Probleme sich auf nicht offensichtliche Weisen durch gekoppelte Komponenten fortpflanzen können.

## Causes ▼

- [Feature-Creep ohne Refactoring](feature-creep-ohne-refactoring.md)
<br/>  Das kontinuierliche Hinzufügen von Features ohne Refactoring des Designs führt dazu, dass Komponenten über die Zeit gegenseitige Abhängigkeiten anhäufen.
- [Schlecht definierte Verantwortlichkeiten](schlecht-definierte-verantwortlichkeiten.md)
<br/>  Wenn Modulen klare Einzelverantwortlichkeiten fehlen, greifen sie tendenziell in andere Komponenten für Funktionalität, was enge Kopplung schafft.
- [Einschränkungen durch monolithische Architektur](einschraenkungen-durch-monolithische-architektur.md)
<br/>  Monolithische Architekturen begünstigen naturgemäß enge Kopplung, da alle Komponenten dieselbe Deployment-Einheit und Codebasis teilen.

## Detection Methods ○

- **Abhängigkeitsanalyse:** Analyse von Komponentenabhängigkeiten und Identifikation von Mustern enger Kopplung
- **Änderungsauswirkungsanalyse:** Verfolgung, wie Änderungen in einer Komponente andere beeinflussen
- **Erkennung zyklischer Abhängigkeiten:** Identifikation zirkulärer Abhängigkeiten zwischen Komponenten
- **Analyse von Schnittstelle vs. Implementierung:** Überprüfung, wie Komponenten miteinander interagieren
- **Komponentenisolationstests:** Testen der Fähigkeit, Komponenten unabhängig auszuführen und zu testen

## Examples

Ein E-Commerce-Bestellverarbeitungssystem hat enge Kopplung zwischen den Komponenten Bestand, Zahlung und Versand. Die Bestandskomponente greift direkt auf die Zahlungsdatenbank zu, um den Zahlungsstatus zu prüfen, die Zahlungskomponente modifiziert Bestandsmengen direkt, und die Versandkomponente liest Bestelldaten direkt aus Zahlungstabellen. Wenn das Zahlungssystem aktualisiert werden muss, um neue Zahlungsmethoden zu unterstützen, erfordert dies Änderungen an allen drei Komponenten, weil sie alle eng an das spezifische Zahlungsdatenbankschema und die interne Zahlungsverarbeitungslogik gekoppelt sind. Ein weiteres Beispiel betrifft eine Benutzeroberfläche, bei der UI-Komponenten direkt Geschäftslogikmethoden aufrufen und auf Datenbankentitäten zugreifen. Wenn sich die Geschäftslogik ändern muss, bricht dies mehrere UI-Komponenten, und wenn sich das Datenbankschema ändert, benötigen sowohl Geschäftslogik als auch UI-Komponenten Updates, was jede Änderung teuer und riskant macht.
