---
title: Integrationstests
description: Durchführung von Tests zur Verifikation des Zusammenspiels verschiedener
  Systemkomponenten.
category:
- Testing
problems:
- inadequate-integration-tests
- missing-end-to-end-tests
- regression-bugs
- fear-of-change
- deployment-risk
- cascade-failures
- integration-difficulties
- poor-test-coverage
- cache-invalidation-problems
- testing-complexity
layout: solution
lang: de
en_slug: integration-tests
related_solutions:
- slug: continuous-integration
  similarity: 0.85
- slug: automated-tests
  similarity: 0.8
- slug: mutation-testing
  similarity: 0.8
- slug: dependency-injection
  similarity: 0.75
- slug: test-driven-development-tdd
  similarity: 0.75
- slug: security-tests
  similarity: 0.75
---

## Description

Integrationstests verifizieren, dass zwei oder mehr Komponenten eines Systems tatsächlich korrekt zusammenarbeiten — eine Datenbankzugriffsschicht, die mit einer echten Datenbank spricht, ein Serviceaufruf, der eine tatsächliche nachgelagerte Abhängigkeit erreicht —, im Gegensatz zu Unit-Tests, die Komponenten isoliert verifizieren, oft mit den genauen Grenzen zwischen ihnen weggemockt. Diese Unterscheidung ist es, wo Legacy-Systeme am meisten exponiert sind: Unit-Tests können vollständig bestehen, während eine Serialisierungsinkonsistenz oder ein Vertragsbruch zwischen dem Auftragsdienst und dem Bestandsdienst völlig unentdeckt bleibt, weil kein Test tatsächlich den echten Interaktionspfad zwischen ihnen ausübt. Legacy-Systeme sammeln genau diese Art von Risiko über die Zeit an, da Komponenten, die ursprünglich gemeinsam gebaut und als Ganzes validiert wurden, schrittweise als separate, individuell getestete Einheiten behandelt werden, ohne dass jemand verifiziert, dass die Nahtstellen zwischen ihnen noch halten. Eine Integrationstestsuite für ein solches System zu bauen bedeutet, die Integrationspunkte zu identifizieren, die historisch die meisten Produktionsvorfälle verursacht haben, und Testcontainer oder eingebettete Datenbanken zu nutzen, um den echten Interaktionspfad an diesen Nahtstellen wiederholbar und isoliert von den gemeinsam genutzten, oft überlasteten Staging-Umgebungen auszuüben, auf die sich Legacy-Systeme tendenziell verlassen. Dies gibt Teams ein Sicherheitsnetz speziell für die Klasse von Fehlern, die Unit-Tests nicht sehen können, und es wird unverzichtbar während Refactoring- oder Migrationsvorhaben, bei denen Komponenten zerlegt oder ersetzt werden und die Nahtstellen zwischen ihnen genau der Bereich mit der aktivsten Änderung und dem größten Risiko sind.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Identifizieren Sie die kritischsten Integrationspunkte: Datenbankzugriff, externe Serviceaufrufe, Message Queues und modulübergreifende Grenzen
- Beginnen Sie mit den Integrationsnahtstellen, die historisch die meisten Produktionsvorfälle verursacht haben
- Nutzen Sie Testcontainer oder eingebettete Datenbanken, um wiederholbare, isolierte Integrationstestumgebungen zu schaffen
- Schreiben Sie Tests, die den echten Interaktionspfad ausüben, statt die Integration wegzumocken
- Halten Sie Integrationstests fokussiert auf die Verifikation von Verträgen zwischen Komponenten, nicht auf das Testen von Geschäftslogik
- Automatisieren Sie Integrationstests als Teil der CI-Pipeline, damit sie bei jedem Commit laufen
- Pflegen Sie eine separate Integrationstestsuite mit klaren Namenskonventionen, um sie von Unit-Tests zu unterscheiden

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Fängt Bugs an Komponentengrenzen, die Unit-Tests nicht erkennen können
- Erhöht das Vertrauen bei der Modifikation von Legacy-Code, der mehrere Subsysteme berührt
- Bietet ein Sicherheitsnetz für Refactoring- und Migrationsvorhaben
- Dokumentiert, wie Komponenten erwartungsgemäß interagieren sollen

**Kosten und Risiken:**
- Integrationstests sind langsamer als Unit-Tests und können den Feedback-Loop verlangsamen, wenn nicht gut verwaltet
- Einrichtung und Pflege der Testumgebung fügt laufenden Aufwand hinzu
- Flakige Tests durch Timing-, Netzwerk- oder Zustandsprobleme können das Vertrauen in die Testsuite untergraben
- Können ein falsches Sicherheitsgefühl erzeugen, wenn nur Happy Paths getestet werden

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Einzelhandelsunternehmen hatte ein Legacy-Auftragsverarbeitungssystem, in dem Auftragsdienst, Bestandsdienst und Zahlungs-Gateway eng integriert waren, aber keine Integrationstests hatten. Jedes Release war ein Glücksspiel, weil Unit-Tests bestanden, aber Produktionsausfälle an Integrationsgrenzen häufig waren. Das Team führte Integrationstests mit Testcontainers für die Datenbank und WireMock für das Zahlungs-Gateway ein. Diese Tests fingen eine kritische Serialisierungsinkonsistenz zwischen Auftrags- und Bestandsdienst ab, die stillen Datenverlust verursacht hatte. Nach der Etablierung der Integrationstestsuite verringerte das Team Produktionsvorfälle im Zusammenhang mit Komponenteninteraktionen um über 60 %.
