---
title: Unzureichende Integrationstests
description: Die Interaktionen zwischen unterschiedlichen Modulen oder Diensten werden
  nicht gründlich getestet, was zu Integrationsfehlern führt.
category:
- Architecture
- Code
- Testing
related_problems:
- slug: system-integration-blindness
  similarity: 0.75
- slug: poor-interfaces-between-applications
  similarity: 0.65
- slug: missing-end-to-end-tests
  similarity: 0.65
- slug: integration-difficulties
  similarity: 0.6
- slug: insufficient-testing
  similarity: 0.6
- slug: inadequate-test-infrastructure
  similarity: 0.6
solutions:
- test-coverage-strategy
- compatibility-testing
- consumer-driven-contracts
- integration-tests
- interoperability-tests
- isolated-test-environments
- simulation-environments
- self-test
layout: problem
lang: de
en_slug: inadequate-integration-tests
---

## Description

Unzureichende Integrationstests treten auf, wenn sich die Teststrategie primär auf einzelne Komponenten konzentriert, während sie es versäumt, zu verifizieren, dass unterschiedliche Teile des Systems korrekt zusammenarbeiten. Integrationsprobleme entstehen oft an den Grenzen zwischen Modulen, Diensten oder externen Systemen, wo Annahmen über Datenformate, Timing, Fehlerbehandlung oder Kommunikationsprotokolle falsch sein können. Ohne ordentliche Integrationstests können Systeme isoliert gut funktionieren, aber fehlschlagen, wenn Komponenten in Produktionsumgebungen interagieren.

## Indicators ⟡
- Unit-Tests bestehen, aber die Anwendung schlägt fehl, wenn Module kombiniert werden
- Fehler treten häufig an den Grenzen zwischen unterschiedlichen Systemkomponenten auf
- Probleme treten nur auf, wenn mehrere Features oder Dienste zusammen genutzt werden
- Produktionsprobleme betreffen oft Datenformat-Diskrepanzen oder Kommunikationsfehler
- Das Deployment in integrierte Umgebungen offenbart Probleme, die in isoliertem Testen nicht erfasst wurden

## Symptoms ▲

- [Hohe Fehlerrate in Produktion](hohe-fehlerrate-in-produktion.md)
<br/>  Integrationsfehler, die nicht im Testen erfasst werden, entkommen in die Produktion, was die Fehlerrate erhöht.
- [Regressionsfehler](regressionsfehler.md)
<br/>  Ohne Integrationstests können Änderungen an einer Komponente stillschweigend Interaktionen mit anderen Komponenten brechen.
- [Release-Instabilität](release-instabilitaet.md)
<br/>  Releases, die Unit-Tests bestehen, aber keine Integrationsabdeckung haben, verursachen häufig Produktionsinstabilität.
- [Kaskadierende Ausfälle](kaskadierende-ausfaelle.md)
<br/>  Nicht getestete Komponenteninteraktionen können Kettenreaktionen von Ausfällen auslösen, wenn Annahmen an Servicegrenzen verletzt werden.
- [Häufige Hotfixes und Rollbacks](haeufige-hotfixes-und-rollbacks.md)
<br/>  In Produktion entdeckte Integrationsprobleme erfordern Notfall-Fixes und Rollbacks, um den Service wiederherzustellen.
- [Debugging-Schwierigkeiten](debugging-schwierigkeiten.md)
<br/>  Ohne Integrationstests, die Probleme an Komponentengrenzen erfassen, tauchen Integrationsfehler in der Produktion auf, wo sie viel schwerer zu diagnostizieren sind.
- [API-Versionierungskonflikte](api-versionierungskonflikte.md)
<br/>  Ohne Integrationstests, die mehrere API-Versionen zusammen ausüben, bleiben Inkompatibilitäten zwischen Versionen unentdeckt, bis sie die Produktion erreichen.

## Causes ▼

- [Unzureichende Testinfrastruktur](unzureichende-testinfrastruktur.md)
<br/>  Fehlende ordentliche Testumgebungen und -werkzeuge machen es schwierig oder unmöglich, aussagekräftige Integrationstests durchzuführen.
- [Unzureichendes Testdatenmanagement](unzureichendes-testdatenmanagement.md)
<br/>  Ohne realistische Testdaten, die Multi-Komponenten-Interaktionen darstellen, können Integrationstests das Systemverhalten nicht effektiv validieren.
- [Zeitdruck](zeitdruck.md)
<br/>  Integrationstests sind komplexer und zeitaufwendiger zu schreiben, daher werden sie unter Termindruck oft übersprungen.
- [Team-Silos](team-silos.md)
<br/>  Wenn Teams isoliert an einzelnen Komponenten arbeiten, übernimmt niemand die Verantwortung für das Testen komponentenübergreifender Interaktionen.

## Detection Methods ○
- **Integrationstest-Abdeckungsanalyse:** Messung, welcher Prozentsatz der Komponenteninteraktionen von Integrationstests abgedeckt ist
- **Kategorisierung von Produktionsproblemen:** Nachverfolgung, wie viele Fehler aus Integrationsproblemen im Vergleich zu komponentenspezifischen Problemen entstehen
- **Überprüfung der Schnittstellendokumentation:** Bewertung, ob Komponentenschnittstellen gut definiert und getestet sind
- **Komponentenübergreifende Fehleranalyse:** Identifikation von Fehlern, die mehrere Systemkomponenten umfassen
- **Testen von Deployment-Umgebungen:** Vergleich der Fehlerraten zwischen isolierten und integrierten Testumgebungen

## Examples

Eine microservices-basierte E-Commerce-Plattform hat umfassende Unit-Tests für jeden Dienst: Nutzerverwaltung, Bestand, Zahlungsabwicklung und Auftragsabwicklung. Jeder Dienst funktioniert isoliert perfekt und besteht alle Unit-Tests. Integrationstests sind jedoch minimal und konzentrieren sich nur auf Happy-Path-Szenarien. In Produktion, wenn ein Nutzer versucht, einen nicht vorrätigen Artikel zu kaufen, gibt der Bestandsdienst korrekt einen "Nicht vorrätig"-Status zurück, aber der Zahlungsdienst hat die Belastung bereits verarbeitet, weil er das Timing der Bestandsprüfungen nicht ordentlich handhabt. Der Auftragsabwicklungsdienst schlägt dann fehl, weil er widersprüchliche Informationen über Zahlungsstatus und Bestandsverfügbarkeit erhält. Der Integrationsfehler führt dazu, dass Kunden für Artikel belastet werden, die sie nicht erhalten können. Ein weiteres Beispiel betrifft ein Dokumentenmanagementsystem, bei dem die Upload-Komponente, die Verarbeitungs-Engine und der Speicherdienst alle einzeln korrekt funktionieren. Integrationstests übersahen jedoch die Tatsache, dass die Upload-Komponente Metadaten in einem Format erzeugt, während die Verarbeitungs-Engine ein anderes Format erwartet. In Produktion werden Dokumente erfolgreich hochgeladen und korrekt gespeichert, aber die Verarbeitungs-Engine versäumt es stillschweigend, sie zu indexieren, wodurch hochgeladene Dokumente nicht durchsuchbar werden, obwohl sie erfolgreich verarbeitet zu sein scheinen.
