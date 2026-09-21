---
title: Fehlende End-to-End-Tests
description: Vollständige Nutzerabläufe werden nicht von Anfang bis Ende getestet,
  was workflow-brechenden Fehlern erlaubt, Produktion zu erreichen.
category:
- Code
- Process
- Testing
related_problems:
- slug: inadequate-integration-tests
  similarity: 0.65
- slug: quality-blind-spots
  similarity: 0.65
- slug: poor-test-coverage
  similarity: 0.6
- slug: system-integration-blindness
  similarity: 0.6
- slug: inadequate-test-infrastructure
  similarity: 0.6
- slug: testing-environment-fragility
  similarity: 0.6
solutions:
- test-coverage-strategy
- acceptance-tests
- compatibility-testing-by-users
- integration-tests
- interoperability-tests
- simulation-environments
- smoke-testing
- tracer-bullets
- exploratory-testing
layout: problem
lang: de
en_slug: missing-end-to-end-tests
---

## Description

Fehlende End-to-End-Tests treten auf, wenn Teststrategien sich auf einzelne Komponenten oder Features fokussieren, ohne vollständige Nutzer-Workflows von Anfang bis Ende zu verifizieren. End-to-End-Tests simulieren echte Nutzerinteraktionen über das gesamte System hinweg, einschließlich Benutzeroberflächen, Geschäftslogik, Datenbanken und externen Integrationen. Ohne diese Tests können Anwendungen auf Komponentenebene korrekt funktionieren, aber scheitern, wenn Nutzer versuchen, tatsächliche Geschäftsprozesse abzuschließen, was zu kritischen Workflow-Fehlschlägen in Produktion führt.

## Indicators ⟡
- Komponenten funktionieren einzeln, aber vollständige Nutzer-Workflows scheitern
- Nutzer berichten, gängige Aufgaben nicht abschließen zu können, obwohl einzelne Features funktionieren
- Fehler treten an den Schnittstellen mehrerer Features oder Systeme auf
- Integrationsprobleme zeigen sich nur beim Durchlaufen vollständiger Nutzerreisen
- Produktionsprobleme, die in isolierten Testumgebungen schwer zu reproduzieren sind

## Symptoms ▲

- [Erhöhte Fehlerraten](erhoehte-fehlerraten.md)
<br/>  Ohne End-to-End-Tests erreichen workflow-brechende Fehler die Produktion, was die Gesamtdefektrate erhöht.
- [Langsame Vorfallslösung](langsame-vorfallsloesung.md)
<br/>  Integrationsprobleme, die nicht durch End-to-End-Tests erfasst werden, verursachen Produktionsvorfälle, wenn Nutzer vollständige Workflows durchlaufen.
- [Unzufriedenheit der Stakeholder](unzufriedenheit-der-stakeholder.md)
<br/>  Nutzer, die auf defekte Workflows treffen, obwohl einzelne Features funktionieren, führen zu Vertrauensverlust und Unzufriedenheit.

## Causes ▼

- [Testkomplexität](testkomplexitaet.md)
<br/>  Die inhärente Komplexität der Einrichtung und Pflege von End-to-End-Testumgebungen entmutigt Teams davon, umfassende Tests zu erstellen.
- [Schwer testbarer Code](schwer-testbarer-code.md)
<br/>  Eng gekoppelter oder schlecht strukturierter Code macht es unpraktikabel, End-to-End-Tests zu erstellen, die vollständige Workflows durchlaufen.
- [Termindruck](termindruck.md)
<br/>  Unter Zeitdruck sind End-to-End-Tests oft die erste Testaktivität, die gestrichen wird, da sie am zeitaufwendigsten zu erstellen sind.

## Detection Methods ○
- **User-Journey-Kartierung:** Dokumentation vollständiger Nutzer-Workflows und Bewertung der Testabdeckung für jede Journey
- **Produktionsproblemanalyse:** Nachverfolgung von Fehlern, die mehrere Systemkomponenten oder Nutzer-Workflow-Schritte umfassen
- **Nutzerfeedback-Analyse:** Überwachung von Kundenberichten über die Unfähigkeit, Aufgaben abzuschließen
- **Überwachung der Workflow-Erfolgsrate:** Nachverfolgung der Abschlussraten für kritische Geschäftsprozesse in Produktion
- **Cross-System-Fehlererkennung:** Identifikation von Problemen, die nur auftreten, wenn mehrere Komponenten sequenziell interagieren

## Examples

Eine E-Commerce-Plattform hat umfassende Unit-Tests für Produktkatalog, Warenkorb, Zahlungsverarbeitung und Bestellverwaltungskomponenten. Jede Komponente funktioniert isoliert einwandfrei und besteht alle Einzeltests. Es gibt jedoch keine End-to-End-Tests, die vollständige Kaufabläufe verifizieren. In Produktion entdecken Nutzer, dass sie Artikel in ihren Warenkorb legen und zur Kasse gehen können, aber wenn sie die Zahlungsverarbeitung abschließen, wird ihre Bestellung mit falschen Versandadressen erstellt, weil die Adressvalidierungskomponente Daten in einem anderen Format erwartet als die Zahlungskomponente liefert. Die Bestellung erscheint für den Nutzer erfolgreich, aber die Abwicklung scheitert, weil Versandadressen ungültig sind. Dieser workflow-brechende Fehler wurde nicht erfasst, weil keine Tests den vollständigen Kaufprozess von der Produktauswahl bis zur erfolgreichen Bestellabwicklung verifizierten. Ein weiteres Beispiel betrifft eine Bankanwendung, bei der einzelne Features wie Kontostandsprüfung, Geldüberweisungen und Transaktionshistorie alle korrekt funktionieren. End-to-End-Tests fehlen jedoch für den vollständigen „Geld zwischen Konten überweisen"-Workflow. In Produktion können Nutzer Überweisungen initiieren und Bestätigungsnachrichten erhalten, aber aufgrund einer Race Condition zwischen Debit- und Kredit-Operationen führen manche Überweisungen dazu, dass Geld vom Quellkonto abgebucht wird, ohne dem Zielkonto gutgeschrieben zu werden. Das Problem tritt nur unter bestimmten Timing-Bedingungen auf, die im vollständigen Workflow entstehen, aber nie beim isolierten Komponententest.
