---
title: Modularisierung
description: Aufteilung der Anwendung in kleine, unabhängige und wiederverwendbare
  Komponenten.
category:
- Architecture
- Code
quality_tactics_url: https://qualitytactics.de/en/maintainability/modularization/
problems:
- monolithic-architecture-constraints
- high-coupling-low-cohesion
- tight-coupling-issues
- circular-dependency-problems
- complex-domain-model
- poor-domain-model
- shared-database
- shared-dependencies
- tangled-cross-cutting-concerns
- difficult-code-reuse
- god-object-anti-pattern
- spaghetti-code
- hidden-dependencies
- system-integration-blindness
- reduced-team-flexibility
- system-stagnation
- circular-references
- merge-conflicts
- organizational-structure-mismatch
- single-entry-point-design
- team-coordination-issues
- technical-architecture-limitations
- excessive-customization
layout: solution
lang: de
en_slug: modularization-and-bounded-contexts
related_solutions:
- slug: separation-of-concerns
  similarity: 0.8
- slug: strangler-fig-pattern
  similarity: 0.75
- slug: microservices
  similarity: 0.75
- slug: incremental-refactoring
  similarity: 0.75
- slug: loose-coupling
  similarity: 0.75
- slug: static-analysis-and-linting
  similarity: 0.75
---

## Description

Modularisierung teilt ein System in kleine, unabhängige Komponenten mit expliziten, durchgesetzten Grenzen auf und legt einer Codebasis eine Struktur auf, die sich in einem Legacy-System typischerweise über Jahre ganz ohne solche Grenzen entwickelt hat. Eine Abhängigkeitsanalyse einer echt gewachsenen Legacy-Codebasis offenbart routinemäßig weit schlechtere Kopplung als angenommen, weshalb die Arbeit mit einer kleinen Zahl großer, gut verstandener Geschäftsfähigkeiten beginnen muss, statt mit einer ambitionierten, feingranularen Zerlegung, die gezeichnet wird, bevor die Domäne tatsächlich verstanden ist. Die resultierenden Grenzen mit Build-Level-Tooling und Architekturtests durchzusetzen, statt sich allein auf Paketnamenskonventionen zu verlassen, ist es, was sie davon abhält, innerhalb von Wochen erneut zu erodieren — denn ohne diese Durchsetzung, und besonders ohne die gemeinsame Datenbank zu adressieren, die Legacy-Systeme fast immer unter allem haben, bleiben die Grenzen organisatorische Fiktion statt einer echten Einschränkung dessen, wie sich der Code ändern kann.

## How to Apply ◆

> Modularisierung auf ein Legacy-System anzuwenden bedeutet, einer Codebasis, die sich ohne sie entwickelt hat, explizite Grenzen aufzuerlegen, beginnend bei den Bereichen mit dem größten Schmerz und mit Durchsetzungswerkzeugen, die verhindern, dass die Grenzen sich sofort wieder auflösen.

- Beginnen Sie mit einer Abhängigkeitsanalyse der bestehenden Codebasis mittels Werkzeugen wie Structure101, Lattix oder sprachspezifischen Analysewerkzeugen, um eine tatsächliche Karte der Interkomponentenabhängigkeiten zu produzieren — die meisten Teams sind überrascht, dass die echte Kopplung schlimmer ist als angenommen.
- Identifizieren Sie zwei oder drei große Geschäftsfähigkeiten (Auftragsverwaltung, Abrechnung, Bestand) als anfängliche Modulgrenzen; widerstehen Sie dem Drang, sofort jede Subdomäne zu modellieren, da vorzeitige feingranulare Grenzen in einem System, das Sie noch nicht verstehen, oft die Linien an falschen Stellen ziehen.
- Führen Sie Modulgrenzen auf Build-System-Ebene ein (Maven-Module, Gradle-Subprojekte, npm-Workspaces), statt sich allein auf Packaging-Konventionen zu verlassen; konventionsbasierte Grenzen erodieren unter Termindruck innerhalb von Wochen, aber Kompilierfehler nicht.
- Nutzen Sie Architekturtest-Werkzeuge wie ArchUnit oder Dependency-Cruiser, um die Grenzregeln als automatisierte Tests zu kodieren, die in CI laufen; dies macht Grenzverletzungen so sichtbar und dringend wie fehlschlagende Unit-Tests.
- Gehen Sie gemeinsam genutzte Datenbanken als Teil der Modularisierungsbemühung an, indem Sie identifizieren, welche Tabellen von welchen Geschäftsfähigkeiten zugegriffen werden, und Eigentümerschaft zuweisen; Tabellen, auf die mehr als eine Fähigkeit zugreift, sind das primäre Risiko für versteckte Kopplung und verdienen explizite Anti-Corruption-Schichten oder event-basierte Synchronisation.
- Beseitigen Sie zirkuläre Abhängigkeiten, sobald sie entdeckt werden, nicht später — extrahieren Sie gemeinsame Konzepte in ein dediziertes Utility- oder Kernel-Modul, statt Modulen zu erlauben, in beide Richtungen voneinander abzuhängen.
- Weisen Sie jedem Modul einen benannten Eigentümer zu, damit Entscheidungen über seine Schnittstelle, interne Struktur und technische Schulden eine verantwortliche Partei haben; eigentümerlose Module in Legacy-Systemen sind, wo sich der schwerste Verfall konzentriert.
- Akzeptieren Sie, dass die ersten Modulgrenzen an manchen Stellen falsch sein werden; gestalten Sie sie verfeinerbar, dokumentieren Sie die Begründung und planen Sie, sie nach dem ersten Quartal Arbeit innerhalb der neuen Struktur zu überarbeiten.

## Tradeoffs ⇄

> Modularisierung erzeugt kurzfristige Reibung — Planung, Durchsetzungs-Tooling, Refactoring-Aufwand — im Austausch für langfristige Fähigkeit, Teile des Systems unabhängig zu ändern.

**Vorteile:**

- Verringert den kognitiven Overhead der Arbeit in einer großen Legacy-Codebasis, indem Entwicklern erlaubt wird, sich auf ein Modul nach dem anderen zu konzentrieren, ohne das gesamte System verstehen zu müssen.
- Ermöglicht Teams, unabhängig an separaten Modulen zu arbeiten, ohne ständige Merge-Konflikte, was den Koordinationsaufwand verringert, der monolithische Legacy-Entwicklung plagt.
- Macht selektive Modernisierung möglich: Ein gut abgegrenztes Modul kann neu geschrieben, ersetzt oder als Service extrahiert werden, ohne Änderungen an den umgebenden Modulen zu erfordern.
- Lokalisiert den Explosionsradius einer Änderung, sodass ein Defekt oder eine Regression in einem Modul weniger wahrscheinlich als unerwarteter Ausfall in einem völlig anderen Teil des Systems zutage tritt.
- Unterstützt inkrementelle Testabdeckungsverbesserung, weil einzelne Module isoliert getestet werden können, sobald ihre Abhängigkeiten über Schnittstellen offengelegt sind.

**Kosten und Risiken:**

- Die Identifikation korrekter Modulgrenzen in einem System mit Jahren angesammelter Kopplung erfordert erheblichen Vorabanalyseaufwand, und die Analyse wird oft durch undokumentiertes Verhalten blockiert, das nur bestimmte Personen verstehen.
- Falsche Grenzen — um technische Schichten statt Geschäftsfähigkeiten gezogen, oder zu fein gezogen, bevor die Domäne verstanden ist — erzeugen Reibung, die bestehen bleibt, bis teures Refactoring sie korrigiert.
- Gemeinsam genutzte Datenbanken sind das häufigste Hindernis für echte Modularisierung in Legacy-Systemen; die Aufteilung der Datenbankeigentümerschaft ist technisch und organisatorisch herausfordernd, und Teams hören häufig bei Paket-Ebene-Modularisierung auf, während die Datenbank vollständig gemeinsam genutzt bleibt.
- Grenzdurchsetzung erfordert Governance: Ohne regelmäßige Abhängigkeitsprüfungen und eine Kultur, die Grenzverletzungen so ernst nimmt wie Bugs, erodieren die Grenzen schneller, als sie etabliert wurden.
- Performance-Overhead durch schnittstellenvermittelte Kommunikation zwischen Modulen — besonders wo das Legacy-System zuvor direkte In-Process-Aufrufe über das nutzte, was nun Modulgrenzen sind — erfordert möglicherweise Messung und Optimierung.

## How It Could Be

> Die folgenden Szenarien zeigen, wie Modularisierung in der Praxis angewendet wurde, um Struktur in Systeme zu bringen, wo zuvor keine existierte.

Ein Telekommunikationsunternehmen betrieb ein Kundenverwaltungssystem, das über vierzehn Jahre zu einer einzigen Java-Webanwendung mit über 800.000 Codezeilen und keiner Paketstruktur, die die Geschäftsdomäne widerspiegelte, angewachsen war. Eine Abhängigkeitsanalyse offenbarte 6.000 Interklassenabhängigkeiten, einschließlich Dutzender Zyklen. Das Team nutzte eine dreimonatige Architekturinitiative, um fünf Kerngeschäftsfähigkeiten zu identifizieren — Kundenidentität, Vertragsverwaltung, Abrechnung, Serviceabwicklung und Support — und reorganisierte Klassen in an diesen Grenzen ausgerichtete Maven-Module. ArchUnit-Regeln setzten die neue Struktur in CI durch. Innerhalb von sechs Monaten konnte das Abrechnungsmodulteam ein neues Preismodell einführen, ohne sich mit den Teams für Serviceabwicklung oder Identität abzustimmen.

Ein Krankenversicherungsunternehmen entdeckte während einer Modernisierungsbewertung, dass sein Schadenverarbeitungssystem drei verschiedene Interpretationen des Wortes „Mitglied" hatte, abhängig davon, welcher Teil des Legacy-Codes ausgeführt wurde. In der Einschreibungslogik war ein Mitglied ein Versicherungsnehmer. In der Schadenregulierung war ein Mitglied eine unter einer Police versicherte Einzelperson. Im Reporting war ein Mitglied eine eindeutige Kombination aus Police und Leistungszeitraum. Diese überlappenden Konzepte hatten jahrelang subtile Dateninkonsistenzen verursacht. Durch Anwendung von Bounded-Context-Analyse zog das Team explizite Grenzen um Einschreibung, Regulierung und Reporting und gab jedem sein eigenes Modell von „Mitglied" mit Übersetzungen dazwischen an den Integrationspunkten. Dies beseitigte die Inkonsistenzen und machte das Domänenmodell jedes Teams intern kohärent.

Ein Finanzdienstleistungsunternehmen musste seine Handelsreporting-Fähigkeit aus einem Legacy-C++-Monolithen extrahieren, damit sie als verwalteter Service für Partnerunternehmen angeboten werden konnte. Die Extraktion sollte sechs Monate dauern, aber der tatsächliche Aufwand offenbarte, dass der Reporting-Code über 200 direkte Funktionsaufrufe und vier gemeinsam genutzte globale Datenstrukturen mit der Handelsbuchungs-Engine verwoben war. Das Team verbrachte vier Monate damit, zuerst eine Modulgrenze innerhalb des Monolithen zu schaffen — direkte Funktionsaufrufe durch eine Schnittstelle ersetzend, die gemeinsamen Globalen durch eine Datenzugriffsschicht beseitigend —, bevor eine Extraktion zu einem separaten Service überhaupt möglich war. Die Modularisierungsarbeit war die Voraussetzung, die die architektonische Änderung überhaupt erreichbar machte.
